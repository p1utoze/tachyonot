import click
from time import perf_counter
from optimum.intel.openvino.modeling_decoder import OVBaseDecoderModel
from transformers import AutoTokenizer, TextIteratorStreamer
from simatic.config import repo_id, ModelConfig
from simatic.helpers import get_prompt_template
from threading import Thread
from simatic import mem as memory
from simatic.models.llm import SimaticModelForCausalLM, core


@memory.cache(ignore=["my_model"])
def cache_model(my_model: SimaticModelForCausalLM, model_status="compiled") -> bytes:
    """
    Cache the compiled model if it is not already in the cache and return the model as bytes.
    :param my_model: (SimaticModelForCausalLM) The model instance to cache
    :param model_status: (bool) The status of the model
    :return: byte
    """
    my_model.compile()
    print(f"Model status: {model_status}")
    return my_model.request.export_model()


class SimaticBaseModel(ModelConfig, OVBaseDecoderModel):
    def __init__(self):
        self.model_config = super().__init__()
        self.request = None
        self.model = None
        self.model_kwargs = {
            "model_id": repo_id,
            "repo_type": "private",
            "device_map": "auto",
            "subfolder": None,
            "ov_config": {
                "ENABLE_CPU_PINNING": True,
                "EXECUTION_MODE_HINT": "ACCURACY",
                "PERFORMANCE_HINT": "LATENCY",
                "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
                "KV_CACHE_PRECISION": "bf16",
            }
        }
        self.generate_kwargs = dict()
        self._auth_token = None
        self.tokenizer_kwargs = self.model_kwargs.copy()
        self.tokenizer_kwargs.pop("device_map")
        self.tokenizer_kwargs["pretrained_model_name_or_path"] = self.model_kwargs["model_id"]
        self.tokenizer_kwargs.pop("model_id")
        self.streamer = None
        self.chat = None
        self.tokenizer = None
        self.system_prompt = None

    def __setitem__(self, key, value):
        if key not in ["model_id", "repo_type"]:
            self.model_kwargs[key] = value
            self.tokenizer_kwargs[key] = value
            return
        raise PermissionError(f"Cannot set {key} attribute")

    def __getitem__(self, item):
        return self.model_kwargs[item]

    def init_model(self, is_compile):
        """
        Initialize the model using our custom model class.
        pass is_compile=True to compile the model.
        Import the compiled model from the cache.
        Will run the cache function if the model is not in the cache.
        :param is_compile:
        :return:
        """
        start = perf_counter()
        self.model = SimaticModelForCausalLM.from_pretrained(
            **self.model_kwargs, compile=is_compile, use_auth_token=self._auth_token
        )
        compiled_model = core.import_model(
            cache_model(self.model, "compiled"),
            self.model._device,
            self.model_kwargs["ov_config"]
        )
        self.model.request = compiled_model.create_infer_request()
        print(f"{perf_counter() - start} sec")

    def init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            **self.tokenizer_kwargs, use_auth_token=self._auth_token, return_dict=True
        )

    def validate_sampling_config(self) -> bool:
        max_new_tokens, max_length = self.generate_kwargs.get("max_new_tokens"), self.generate_kwargs.get("max_length")
        if not (bool(max_new_tokens) ^ bool(max_length)):
            raise ValueError("Please provide either max_new_tokens or max_length")
        if max_new_tokens and max_length:
            self.generate_kwargs.pop("max_length")
        return True

    def generate(self, prompt: str, out_type: str):
        """
        Generate text from the model using the prompt.
        :param prompt: (str) The prompt to generate text from. It's the user input.
        :param out_type: (str) The output type. Either "stream" or "no stream"
        :return: None
        """
        messages = get_prompt_template(prompt, self.system_prompt)
        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        generation_kwargs = dict(
            pad_token_id=self.tokenizer.eos_token_id,
            **tokenized_chat,
            **self.generate_kwargs,
        )
        if out_type != "stream":
            output = self.model.generate(**generation_kwargs)
            generated_response = self.tokenizer.decode(output[:, tokenized_chat.input_ids.shape[-1]:][0], skip_special_tokens=True)
            click.secho(generated_response, fg="bright_cyan")
            return

        self.stream(generation_kwargs)
        click.echo("\n")

    def stream(self, gen_kwargs):
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        for txt in streamer:
            click.secho(txt, nl=False, fg="bright_cyan")

        thread.join()


if __name__ == '__main__':
    """
    This entrypoint is used to test the SimaticBaseModel class and its methods.
    Also used for testing the cache_model function to check Cache HIT/MISS.
    """
    # import os
    # os.environ["OPENVINO_LOG_LEVEL"] = "3"
    simatic_text = SimaticBaseModel()
    simatic_text._auth_token = input("Please enter your Hugging Face API token: ")
    simatic_text["subfolder"] = "tinyLlama/int4"

    # < -------- Load the model (First time) ------------>
    llm = simatic_text.init_model(is_compile=False)
    print(cache_model.check_call_in_cache(llm, "compiled"))

    ## <-------- Load the model (from cache) ------------>
    llm = simatic_text.init_model(is_compile=False)
    print(cache_model.check_call_in_cache(llm, "compiled"))

    # ov_model = OVModelForCausalLM.from_pretrained(
    #     model_id=repo_id,
    #     repo_type="private",
    #     device_map="auto",
    #     subfolder=simatic_text["subfolder"],
    #     use_auth_token=simatic_text._auth_token
    # )

    ## < ------- Validate the cached model ------------>
    # infer_req = simatic_model.request.create_infer_request()
    # assert infer_req.input_tensors == ov_model.request.input_tensors, "Input tensors not equal"
    # assert infer_req.output_tensors == ov_model.request.output_tensors, "Output tensors not equal"


