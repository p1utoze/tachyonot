# distutils: language=c
# cython: language_level=3

import click
from time import perf_counter
from optimum.intel.openvino.modeling_decoder import OVBaseDecoderModel
from transformers import AutoTokenizer, TextIteratorStreamer
from simatic.config import repo_id, ModelConfig
from simatic.helpers import get_prompt_template
from threading import Thread
from simatic.models.llm import SimaticModelForCausalLM
from simatic.config import CACHE_DIR


class SimaticBaseModel(ModelConfig, OVBaseDecoderModel):
    _instance = None

    def __new__(cls):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self):
        self.model_config = super().__init__()
        self.request = None
        self.model = None
        self.model_kwargs = {
            "model_id": repo_id,
            "repo_type": "private",
            "device_map": "auto",
            "subfolder": None,
            "cache_dir": CACHE_DIR,
            "ov_config": {
                "ENABLE_CPU_PINNING": True,
                "EXECUTION_MODE_HINT": "ACCURACY",
                "PERFORMANCE_HINT": "LATENCY",
                "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
                "KV_CACHE_PRECISION": "bf16",
                "CACHE_DIR": CACHE_DIR / "ov_cache",
            }
        }
        self.generate_kwargs = dict()
        self._auth_token = None
        self.tokenizer_kwargs = self.model_kwargs.copy()
        self.tokenizer_kwargs.pop("device_map")
        self.tokenizer_kwargs["pretrained_model_name_or_path"] = self.model_kwargs["model_id"]
        self.tokenizer_kwargs.pop("model_id")
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

    def compile_model(self, model_status: str = "compiled"):
        """
        Cache the compiled model if it is not already in the cache and return the model as bytes.
        :param my_model: (SimaticModelForCausalLM) The model instance to cache
        :param model_status: (bool) The status of the model
        :return: byte
        """
        model = self.model.compile()
        print(f"Model status: {model_status}")
        return model

    def init_model(self, is_compile) -> None:
        """
        Initialize the model using our custom model class.
        pass is_compile=True to compile the model.
        Import the compiled model from the cache.
        Will run the cache function if the model is not in the cache.
        :param is_compile:
        :return:
        """
        start = perf_counter()
        self.model: SimaticModelForCausalLM = SimaticModelForCausalLM.from_pretrained(
            **self.model_kwargs, compile=is_compile, use_auth_token=self._auth_token
        )
        self.model.request = self.compile_model("compiled").create_infer_request()
        print(f"{perf_counter() - start} sec")

    def init_tokenizer(self) -> None:
        """
        Initialize the tokenizer using the AutoTokenizer class from the transformers library.
        :return:
        """
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

    def generate(self, prompt: str, out_type: str = None) -> None:
        """
        Generate text from the model using the prompt.
        :param prompt: (str) The prompt to generate text from. It's the user input.
        :param out_type: (str) The output type. Either "stream" or "no stream"
        :return: None
        """
        messages = get_prompt_template(prompt, self.system_prompt)
        tokenized_chat = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
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

    def stream(self, gen_kwargs: dict) -> None:
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
    simatic_text = SimaticBaseModel()
    simatic_text3 = SimaticBaseModel()
    print(id(simatic_text) == id(simatic_text3))
    simatic_text._auth_token = input("Please enter your Hugging Face API token: ")
    simatic_text["subfolder"] = "tinyLlama/int4"
    simatic_text2 = SimaticBaseModel()
    simatic_text2["subfolder"] = "tinyLlama/int8"
    print(id(simatic_text2) == id(simatic_text))