import argparse
import logging
import simatic.commands as cmds

parser = argparse.ArgumentParser(description="Simatic CLI tool")
subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")


def main():
    text_parser = subparsers.add_parser("chat", help="Chat with the Simatic LLM model")
    speech_parser = subparsers.add_parser(
        "listen", help="Speech to text using the Whisper model"
    )
    cmds.listen.invoke(speech_parser)

    args = parser.parse_args()
    if args.subcommand == "chat":
        logging.info("Not implemented yet...")
    elif args.subcommand == "listen":
        cmds.listen.run_speech(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
