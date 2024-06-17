import os

import click
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, TextIteratorStreamer
from simatic.config import default_dtype, repo_id, ModelConfig
from huggingface_hub.utils._errors import RepositoryNotFoundError
from simatic.helpers import get_hf_token, get_prompt_template
from threading import Thread

class SimaticModel(ModelConfig):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_config = super().__init__()
        self.model_kwargs = {
            "model_id": repo_id,
            "repo_type": "private",
            "device_map": "auto",
            "subfolder": None,
        }
        self._auth_token = None
        self.tokenizer_kwargs = self.model_kwargs.copy()
        self.tokenizer_kwargs.pop("device_map")
        self.tokenizer_kwargs["pretrained_model_name_or_path"] = self.model_kwargs["model_id"]
        self.tokenizer_kwargs.pop("model_id")
        self.streamer = None

    def __setitem__(self, key, value):
        if key not in ["model_id", "repo_type"]:
            self.model_kwargs[key] = value
            self.tokenizer_kwargs[key] = value
            return
        raise PermissionError(f"Cannot set {key} attribute")

    @property
    def hf_auth_token(self, key):
        self._auth_token = key

    def __getitem__(self, item):
        return self.model_kwargs[item]

    def init_model(self):
        self.model = OVModelForCausalLM.from_pretrained(
            **self.model_kwargs, use_auth_token=self._auth_token
        )

    def init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            **self.tokenizer_kwargs, use_auth_token=self._auth_token
        )

    def generate(self, prompt, max_length=512, temperature=0.5, do_sample=True, stream_mode=True):
        messages = get_prompt_template(prompt)
        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)

        generation_kwargs = dict(
            **tokenized_chat,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
        )
        if not stream_mode:
            output = self.model.generate(**generation_kwargs)
            generated_response = self.tokenizer.decode(output[:, tokenized_chat.input_ids.shape[-1]:][0], skip_special_tokens=True)
            click.secho(generated_response, fg="bright_cyan")
            return

        self.stream(generation_kwargs)

    def stream(self, gen_kwargs):
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        for txt in streamer:
            click.secho(txt, nl=False, fg="bright_cyan")


simatic_obj = SimaticModel()


@click.command()
@click.option('--model', '-m', type=click.Choice(simatic_obj.models_), help='Model name', required=True)
@click.option('--hf_token', "-t", type=str, expose_value=False, envvar="HF_TOKEN")
@click.option('--dtype', "-d", type=str, default=default_dtype, help='Model data type')
@click.argument('prompt', nargs=1)
def llm_inference(model, prompt, dtype):
    click.secho(f"Model: {model}, Prompt: {prompt}, Data type: {dtype}", bg="bright_cyan", fg="black")
    simatic_obj.load_model_config(model)
    simatic_obj["subfolder"] = f"{model}/{dtype}"
    try:
        simatic_obj.init_model()
    except RepositoryNotFoundError:
        simatic_obj._auth_token = get_hf_token()
        simatic_obj.init_model()
    try:
        simatic_obj.init_tokenizer()
    except OSError:
        base_pretrained_model = simatic_obj.get_base_model_tokenizer(model)
        print(f"Fallback to default Tokenizer. Using {base_pretrained_model}")
        simatic_obj.tokenizer_kwargs["pretrained_model_name_or_path"] = base_pretrained_model
        simatic_obj.tokenizer_kwargs.pop("repo_type")
        simatic_obj.tokenizer_kwargs.pop("subfolder")
        simatic_obj.init_tokenizer()

    print()
    simatic_obj.generate(prompt)

    # tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    #
    # output = hmi_model.generate(**tokenized_chat, max_length=512, temperature=0.5, do_sample=True)
    # generated_response = tokenizer.decode(output[:, tokenized_chat.input_ids.shape[-1]:][0], skip_special_tokens=True)
    # print(generated_response)


if __name__ == '__main__':
    llm_inference()
