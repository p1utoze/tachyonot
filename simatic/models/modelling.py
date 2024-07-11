from optimum.intel.openvino.modeling_decoder import OVBaseDecoderModel
import copy
import warnings
from pathlib import Path
from typing import  Callable, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from openvino.runtime import Core, Tensor, Type
from transformers import AutoModelForCausalLM, PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerateOutput, GenerationMode
from transformers.modeling_outputs import CausalLMOutputWithPast

from optimum.intel.utils import is_nncf_available, is_transformers_version
from optimum.intel.utils.modeling_utils import MULTI_QUERY_ATTN_MODELS
from optimum.intel.openvino.configuration import _DEFAULT_4BIT_CONFIGS, OVConfig, OVWeightQuantizationConfig, _check_default_4bit_configs
from optimum.intel.openvino.utils import ONNX_WEIGHTS_NAME, OV_TO_NP_TYPE, OV_XML_FILE_NAME, STR_TO_OV_TYPE

core = Core()

logger = logging.getLogger(__name__)


class SimaticModelForCausalLM(OVBaseDecoderModel, GenerationMixin):
    export_feature = "text-generation"
    auto_model_class = AutoModelForCausalLM

    def compile(self):
        print(f"Compiling the model to {self._device} ...")
        ov_config = {**self.ov_config}
        self.request = core.compile_model(self.model, self._device, ov_config)

    def prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict:
        batch_size = input_ids.shape[0]
        if self.config.model_type == "bloom":
            batch_size *= self.config.num_attention_heads

        inputs = {}
        if not self.stateful:
            if past_key_values is not None:
                if self.config.model_type not in MULTI_QUERY_ATTN_MODELS or (
                    self.config.model_type == "falcon" and self.config.new_decoder_architecture
                ):
                    if self._pkv_precision == Type.bf16:
                        # numpy does not support bf16, pretending f16, should change to bf16
                        past_key_values = tuple(
                            Tensor(past_key_value, past_key_value.shape, Type.bf16)
                            for pkv_per_layer in past_key_values
                            for past_key_value in pkv_per_layer
                        )
                    else:
                        # Flatten the past_key_values
                        past_key_values = tuple(
                            past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
                        )

                # Add the past_key_values to the decoder inputs
                inputs = dict(zip(self.key_value_input_names, past_key_values))

            # Create empty past_key_values for decoder_with_past first generation step
            elif self.use_cache:
                for input_name in self.key_value_input_names:
                    model_inputs = self.model.input(input_name)
                    dtype = OV_TO_NP_TYPE[model_inputs.get_element_type().get_type_name()]
                    shape = model_inputs.get_partial_shape()
                    if self.config.model_type == "chatglm":
                        shape[0] = 0
                        shape[1] = batch_size
                    else:
                        shape[0] = batch_size
                        if shape[2].is_dynamic:
                            shape[2] = 0
                        else:
                            shape[1] = 0
                    inputs[input_name] = np.empty([dim.get_length() for dim in shape], dtype=dtype)
        else:
            # past_key_values are not used explicitly, instead they are handled inside the model
            if past_key_values is None:
                # This is the first iteration in a sequence, reset all states
                if self.request is not None:
                    self.request.reset_state()
                # Set initial value for the next beam_idx input that will be used at the current iteration
                # and will be optionally updated by _reorder_cache at the next iterations if beam_search is used
                self.next_beam_idx = np.arange(batch_size, dtype=int)
                self._past_length = 0
        past_len = self._get_past_length(past_key_values)
        inputs["input_ids"] = np.array(input_ids)
        # Add the attention_mask inputs when needed
        if "attention_mask" in self.input_names or "position_ids" in self.input_names:
            if attention_mask is not None:
                attention_mask = np.array(attention_mask)
            else:
                attention_mask = np.ones(
                    (input_ids.shape[0], input_ids.shape[1] + past_len), dtype=inputs["input_ids"].dtype
                )

        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        if "position_ids" in self.input_names:
            if position_ids is not None:
                position_ids = np.array(position_ids)
            else:
                position_ids = np.cumsum(attention_mask, axis=1) - 1
                position_ids[attention_mask == 0] = 1
                if past_key_values:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

            inputs["position_ids"] = position_ids

        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = (
                self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)
            )

        return inputs

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        inputs = self.prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            **kwargs,
        )
        duplication_indices = None
        if self._first_iter_beam_search:
            inputs, duplication_indices = self._deduplicate_inputs(inputs)
        # Run inference
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        logits = torch.from_numpy(self.request.get_tensor("logits").data).to(self.device)
        if self.stateful:
            # Need a marker to differentiate the first generate iteration from the others in
            # the first condition at the function beginning above.
            # It should be something that is not None and it should be True when converted to Boolean.
            past_key_values = ((),)
            self._past_length += input_ids.shape[1]

        if not self.stateful:
            if self.use_cache:
                # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
                past_key_values = tuple(self.request.get_tensor(key).data for key in self.key_value_output_names)
                if self.config.model_type not in MULTI_QUERY_ATTN_MODELS or (
                    self.config.model_type == "falcon" and self.config.new_decoder_architecture
                ):
                    # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
                    past_key_values = tuple(
                        past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv)
                    )
            else:
                past_key_values = None

        if self._first_iter_beam_search:
            logits, past_key_values = self._expand_outputs_for_generation(duplication_indices, logits, past_key_values)
            self._first_iter_beam_search = False

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    # Adapted from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        attention_mask = kwargs["attention_mask"] if "attention_mask" in kwargs else None
        use_cache = kwargs.get("use_cache", None)

        if past_key_values is not None:
            past_len = self._get_past_length(past_key_values)
            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_len) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_len < input_ids.shape[1]:
                input_ids = input_ids[:, past_len:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None and "position_ids" in self.input_names:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

        return model_inputs

    def _expand_outputs_for_generation(self, indicies, logits: torch.Tensor, past_key_values: Tuple):
        batch_size = logits.shape[0]
        if indicies.shape[0] != 1:
            logits = logits[indicies]
            if past_key_values and not self.stateful:
                if self.config.model_type not in MULTI_QUERY_ATTN_MODELS or (
                    self.config.model_type == "falcon" and self.config.new_decoder_architecture
                ):
                    past_key_values = tuple(
                        tuple(
                            past_state[indicies]
                            if not self.config.model_type == "chatglm"
                            else past_state[:, indicies, ...]
                            for past_state in layer_past
                        )
                        for layer_past in past_key_values
                    )
                else:
                    past_key_values = tuple([past_state[indicies] for past_state in past_key_values])
        if self.stateful:
            self.next_beam_idx = (
                self.next_beam_idx[indicies]
                if self.next_beam_idx is not None
                else np.arange(batch_size, dtype=int)[indicies]
            )
            self._second_iter_beam_search = True
        return logits, past_key_values

    def _deduplicate_inputs(self, model_inputs: Dict):
        input_ids = model_inputs["input_ids"]
        upd_model_inputs = {}
        unique_input_ids, indicies, reverse_indicies = np.unique(
            input_ids, axis=0, return_index=True, return_inverse=True
        )
        for input_name, input_tensor in model_inputs.items():
            if input_name not in ["input_ids", "beam_idx"]:
                if input_name not in self.key_value_input_names:
                    upd_model_inputs[input_name] = input_tensor[indicies]
                else:
                    shape = input_tensor.shape if isinstance(input_tensor, Tensor) else list(input_tensor.shape)
                    dtype = input_tensor.element_type if isinstance(input_tensor, Tensor) else Type(input_tensor.dtype)
                    upd_batch_size = indicies.shape[0]
                    if self.config.model_type == "bloom":
                        upd_batch_size *= self.config.num_attention_heads
                    shape[0 if not self.config.model_type == "chatglm" else 1] = upd_batch_size
                    upd_model_inputs[input_name] = Tensor(dtype, shape)
        upd_model_inputs["input_ids"] = unique_input_ids
        if "beam_idx" in model_inputs:
            beam_range = (
                unique_input_ids.shape[0]
                if self.config.model_type != "bloom"
                else unique_input_ids.shape[0] * self.config.num_attention_heads
            )
            beam_idx = np.arange(beam_range, dtype=int)
            upd_model_inputs["beam_idx"] = beam_idx
        return upd_model_inputs, reverse_indicies

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if is_transformers_version(">=", "4.39.0"):
            _generation_config, _ = self._prepare_generation_config(generation_config, **kwargs)
            generation_mode = _generation_config.get_generation_mode(assistant_model)
        else:
            _generation_config = generation_config or self.generation_config
            generation_mode = self._get_generation_mode(_generation_config, assistant_model)

        is_beam_search = generation_mode in [
            GenerationMode.BEAM_SEARCH,
            GenerationMode.BEAM_SAMPLE,
            GenerationMode.GROUP_BEAM_SEARCH,
            GenerationMode.CONSTRAINED_BEAM_SEARCH,
        ]
        if is_beam_search:
            self._first_iter_beam_search = True
        result = super().generate(
            inputs,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            assistant_model,
            streamer,
            negative_prompt_ids,
            negative_prompt_attention_mask,
            **kwargs,
        )
        return result

    def _get_past_length(self, past_key_values=None):
        if past_key_values is None:
            return 0
        if self.stateful:
            return self._past_length
        if self.config.model_type in MULTI_QUERY_ATTN_MODELS and not (
            self.config.model_type == "falcon" and self.config.new_decoder_architecture
        ):
            return past_key_values[0].shape[-2]
        seq_length_dim = -2
        if self.config.model_type == "chatglm":
            seq_length_dim = 0
        elif self.config.model_type == "qwen":
            seq_length_dim = 1
        # input is tuple of pairs
        if isinstance(past_key_values[0], (tuple, list)):
            return past_key_values[0][1].shape[seq_length_dim]
        # past key values comes after flattening
        return past_key_values[1].shape[seq_length_dim]

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        if self.stateful:
            # TODO: Apply it differently based on model type
            # TODO: At least for bloom we need to replicate values for each attention head
            self.next_beam_idx = (
                np.array(beam_idx) if not self._second_iter_beam_search else self.next_beam_idx
            )  # save beam_idx to be used as an input in the next iteration
            self._second_iter_beam_search = False
            return past_key_values
        else:
            if self.config.model_type not in MULTI_QUERY_ATTN_MODELS or (
                self.config.model_type == "falcon" and self.config.new_decoder_architecture
            ):
                return tuple(
                    tuple(np.take(past_state, beam_idx, 0) for past_state in layer_past)
                    for layer_past in past_key_values
                )
            return tuple(np.take(past_state, beam_idx, 0) for past_state in past_key_values)

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        file_name: Optional[str] = None,
        subfolder: str = "",
        from_onnx: bool = False,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        quantization_config: Optional[Union[OVWeightQuantizationConfig, Dict]] = None,
        **kwargs,
    ):
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        model_path = Path(model_id)
        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_XML_FILE_NAME
        file_name = file_name or default_file_name

        model_cache_path = cls._cached_file(
            model_path=model_path,
            token=token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )

        model = cls.load_model(model_cache_path)

        model_type = config.model_type.replace("_", "-")
        init_cls = cls

        if isinstance(quantization_config, dict) and quantization_config == {"bits": 4}:
            quantization_config = _DEFAULT_4BIT_CONFIGS.get(config.name_or_path, quantization_config)
        quantization_config = cls._prepare_weight_quantization_config(quantization_config, load_in_8bit)

        enable_compilation = kwargs.pop("compile", True) and not quantization_config

        try:
            generation_config = GenerationConfig.from_pretrained(
                model_id,
                token=token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )
            kwargs["generation_config"] = generation_config
        except Exception:
            pass

        causal_model = init_cls(
            model=model,
            config=config,
            model_save_dir=model_cache_path.parent,
            compile=enable_compilation,
            quantization_config=quantization_config,
            **kwargs,
        )

        if quantization_config:
            if not is_nncf_available():
                raise ImportError(
                    "Quantization of the weights requires nncf, please install it with `pip install nncf`"
                )

            from optimum.intel.openvino.quantization import OVQuantizer

            default_config = _check_default_4bit_configs(config)

            if default_config:
                logger.info(
                    f"For the given model, we recommend the following `quantization_config` : {default_config}"
                )

            quantizer = OVQuantizer(causal_model)
            quantization_config_copy = copy.deepcopy(quantization_config)
            quantization_config_copy.tokenizer = quantization_config.tokenizer or model_id
            quantizer.quantize(ov_config=OVConfig(quantization_config=quantization_config_copy))

        return causal_model