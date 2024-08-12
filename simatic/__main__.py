import argparse
import os

import simatic.commands as ai_commands

parser = argparse.ArgumentParser(description="Simatic CLI tool")
subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")
# listen_parser = subparsers.add_parser("listen", help="Listen to the audio and transcribe it")
# speak_parser = subparsers.add_parser("speak", help="Speak to the AI model")


def cli():

    ai_commands.chat_command(subparsers)
    args = parser.parse_args()

    if args.subcommand == "chat":
        if os.getenv("HF_TOKEN") is None:
            parser.print_help()
            parser.error("Please set the HF_TOKEN environment variable")

        from simatic.commands.text import text_gen
        print("Chatting with the AI model")
        text_gen(**vars(args))
    else:
        print("Invalid subcommand")

    # ai_commands.text_gen(args)
# A placeholder for the main function. It does nothing. The subcommands override this function.
