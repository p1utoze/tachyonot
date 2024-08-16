import argparse

from simatic.llama import SimaticLLM


def chat():
    parser = argparse.ArgumentParser(description="Generate text using SimaticLLM.")
    parser.add_argument("prompt", help="Input prompt for the model")
    parser.add_argument("--config-path", required=True, help="Path to the configuration file for RAG")
    parser.add_argument("--data-path", default=None, help="Path to the data folder for RAG")
    parser.add_argument("--stream", default=True, help="Enable streaming output")

    args = parser.parse_args()

    chain = SimaticLLM(config_path=args.config_path)

    if args.data_path:
        chain.store_documents(args.data_path)

    generator = chain.invoke(
        args.prompt,
        stream=args.stream
    )

    CYAN = "\033[96m"
    RESET = "\033[0m"
    print(f"{CYAN}Question: {args.prompt}{RESET}", flush=True)
    print(f"{CYAN}Response: {RESET}", end="", flush=True)
    for message in generator:
        print(f"{CYAN}{message}{RESET}", end="", flush=True)
    print()


if __name__ == "__main__":
    chat()

