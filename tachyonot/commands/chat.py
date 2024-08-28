from simatic.models.llama import SimaticLLM
from simatic.utils import get_config_path

chain = SimaticLLM(config_path=get_config_path())


def invoke(parser):
    """
    Add arguments to the parser
    :param parser:
    :return:
    """
    parser.add_argument("prompt", help="Input prompt for the model")
    parser.add_argument(
        "--data-path", default=None, help="Path to the data folder for RAG"
    )
    parser.add_argument("--stream", default=True, help="Enable streaming output")


def run_chat(args):
    """
    Run the chat command
    :param args: LLM chat arugments such as prompt, data_path, stream
    """
    if args.data_path:
        chain.store_documents(args.data_path)

    generator = chain.invoke(args.prompt, stream=args.stream)

    CYAN = "\033[96m"
    RESET = "\033[0m"
    print(f"{CYAN}Question: {args.prompt}{RESET}", flush=True)
    print(f"{CYAN}Response: {RESET}", end="", flush=True)
    for message in generator:
        print(f"{CYAN}{message}{RESET}", end="", flush=True)
    print()
