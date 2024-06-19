import click
from simatic.commands import text_gen


@click.group()
def cli():
    pass


cli.add_command(text_gen)

if __name__ == '__main__':
    cli()
