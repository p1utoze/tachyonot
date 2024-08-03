import click

import simatic.commands.text as text
import simatic.commands.tts as tts
import simatic.commands.stt as stt


@click.group()
def cli():
    """
    Simatic CLI tool
    :return: None
    """
    pass    # A placeholder for the main function. It does nothing. The subcommands override this function.


# Add subcommands to the main command
cli.add_command(text.text_gen, name="chat")
cli.add_command(tts.speech_gen, name="speak")
cli.add_command(stt.asr, name="listen")

cli()
