import click
from simatic.commands import text_gen
from simatic.commands import speech_gen


@click.group()
def cli():
    pass


cli.add_command(text_gen)
cli.add_command(speech_gen)

if __name__ == '__main__':
    cli()
