import click
from simatic.models import simatic_text
from simatic.helpers import get_hf_token
from simatic.config import default_dtype, SYSTEM_PROMPT
from huggingface_hub.utils._errors import RepositoryNotFoundError


llm = None
tokenizer = None


@click.command()
@click.option('--model', '-m', type=click.Choice(simatic_text.models_), help='Model name', required=True)
@click.option('--hf_token', "-o", type=str, expose_value=False, envvar="HF_TOKEN")
@click.option('--dtype', "-d", type=str, default=default_dtype, help='Model data type')
@click.option("--temperature", "-t", type=float, default=0.5, help="Temperature for sampling")
@click.option("--do-sample", is_flag=True, help="Flag to enable sampling")
@click.option("--max-length", "-l", type=int, default=None, help="Maximum length of the generated text")
@click.option("--stream", "output_type", flag_value="stream", help="Flag to enable streaming mode")
@click.option("--no-stream", "output_type", flag_value="no_stream", help="Flag to disable streaming mode")
@click.option("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate")
@click.option("--system-prompt", type=str, default=SYSTEM_PROMPT, help="System prompt to be used")
@click.argument('prompt', nargs=1)
def text_gen(model, prompt, dtype, output_type, system_prompt, **kwargs):
    global llm, tokenizer
    simatic_text.system_prompt = system_prompt
    simatic_text.generate_kwargs |= kwargs
    click.secho(f"Model: {model}, Prompt: {prompt}, Data type: {dtype}", bg="bright_cyan", fg="black")
    simatic_text.load_model_config(model)
    simatic_text["subfolder"] = f"{model}/{dtype}"

    try:
        simatic_text.init_model(is_compile=False)
    except RepositoryNotFoundError:
        simatic_text._auth_token = get_hf_token()
        simatic_text.init_model(is_compile=False)

    # assert cache_model.check_call_in_cache(llm, "compiled") == True, "Model not cached"

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
