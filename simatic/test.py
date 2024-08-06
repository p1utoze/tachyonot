import click
from simatic.models import SimaticBaseModel
from simatic.helpers import get_hf_token
from simatic.config import default_dtype, SYSTEM_PROMPT
from huggingface_hub.utils._errors import RepositoryNotFoundError

from simatic.models.rag import RAG

simatic_text = SimaticBaseModel()


def text_gen(model, data_path, prompt, dtype, output_type, system_prompt, **kwargs):
    simatic_text.system_prompt = system_prompt
    simatic_text.generate_kwargs |= kwargs  # Update the generate_kwargs with the kwargs by concatenating them
    # rag = RAG()
    click.secho(f"Model: {model}, Prompt: {prompt}, Data type: {dtype}", bg="bright_cyan", fg="black")
    # if data_path:
    # prompt = rag.retrieve(data_path=data_path, query=prompt)

    simatic_text.load_model_config(model)
    simatic_text["subfolder"] = f"{model}/{dtype}"

    # Check if the model repo exists or is public
    try:
        simatic_text.init_model(is_compile=False)
    except RepositoryNotFoundError:
        simatic_text._auth_token = get_hf_token()
        simatic_text.init_model(is_compile=False)

    # Check if the model is cached and if not, cache it
    # assert cache_model.check_call_in_cache(llm, "compiled") == True, "Model not cached"

    # Check if the tokenizer repo exists or is public
    try:
        simatic_text.init_tokenizer()
    except OSError:
        base_pretrained_model = simatic_text.get_base_model_tokenizer(model)
        print(f"Fallback to default Tokenizer. Using {base_pretrained_model}")
        simatic_text.tokenizer_kwargs["pretrained_model_name_or_path"] = base_pretrained_model
        simatic_text.tokenizer_kwargs.pop("repo_type")
        simatic_text.tokenizer_kwargs.pop("subfolder")
        simatic_text.init_tokenizer()

    print()
    simatic_text.generate(prompt=prompt, out_type=output_type)


if __name__ == "__main__":
    text_gen("opt350M", None, "What is Augmented Reality?", "int4", "stream", SYSTEM_PROMPT,
             max_new_tokens=256, temperature=0.1, do_sample=True, max_length=None)
