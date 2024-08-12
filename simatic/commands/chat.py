import argparse
import sys
from ..llm import SimaticLLM
from ..rag import RAG


def chat():
    parser = argparse.ArgumentParser(description="Generate text using SimaticLLM.")
    parser.add_argument("--data-path", default=None, help="Path to the data folder for RAG")
    parser.add_argument("prompt", help="Input prompt for the model")
    parser.add_argument("--checkpoint", default="HuggingFaceTB/SmolLM-135M-Instruct", help="Model checkpoint to use")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run the model on")
    parser.add_argument("--max_new_tokens", default=256, type=int, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", default=0.6, type=float, help="Sampling temperature")
    parser.add_argument("--top_p", default=0.92, type=float, help="Top-p sampling parameter")
    parser.add_argument("--do_sample", default=True, action="store_true", help="Whether to use sampling in generation")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output")

    args = parser.parse_args()

    llm = SimaticLLM(checkpoint=args.checkpoint, device=args.device)

    if args.data_path:
        rag = RAG()
        prompt = rag.retrieve(args.data_path, args.prompt)
    else:
        prompt = args.prompt

    messages = [
        {"role": "system", "content": "You are a Friendly help-desk assistant. You must handle any type of customer questions diligently. If unambigious, ask for more information. If you are unable to answer, escalate to a human agent. The Context will be given to you. Use it to answer the users questions."},
        {"role": "user", "content": prompt}
    ]

    if args.stream:
        streamer = llm.generate(
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            stream=True
        )
        for token in streamer:
            print(token, end="", flush=True)
        print()
    else:
        response = llm.generate(
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample
        )
        print(response)


if __name__ == "__main__":
    chat()
