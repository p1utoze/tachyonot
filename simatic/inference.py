import click
from simatic.commands import text_gen
from simatic.commands import speech_gen
from simatic.commands import speech_rec


@click.group()
def cli():
    pass


cli.add_command(text_gen, name="chat")
cli.add_command(speech_gen, name="speak")
cli.add_command(speech_rec, name="listen")

if __name__ == '__main__':
    cli()
