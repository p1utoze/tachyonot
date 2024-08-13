import argparse
from simatic.llama_cpp import SimaticLLM


def chat():
    parser = argparse.ArgumentParser(description="Generate text using SimaticLLM.")
    parser.add_argument("prompt", help="Input prompt for the model")
    parser.add_argument("--data-path", default=None, help="Path to the data folder for RAG")
    parser.add_argument("--stream", default=True, help="Enable streaming output")

    args = parser.parse_args()

    chain = SimaticLLM()

    history = [
        {"role": "system",
         "content": "You are a Friendly help-desk assistant and you have to answer to questions according to the context given. Only reply with "
                    "responses and nothing else."},
    ]

    result = chain.invoke(
        args.prompt,
        history,
        data_path=args.data_path,
        stream=args.stream
    )

    if not args.stream:
        CYAN = "\033[96m"
        RESET = "\033[0m"
        print(f"{CYAN}{result}{RESET}")


if __name__ == "__main__":
    chat()

