import tempfile
import click
import asyncio
import subprocess
from pydub import AudioSegment
from pydub.playback import play
import random
import edge_tts
from edge_tts import VoicesManager


async def tts_ms(text, language="en", gender="Male", output_file="output.mp3", locale=None):
    voices = await VoicesManager.create()
    voice = voices.find(Gender=gender, Language=language, locale=locale)

    if not voice:
        print(f"No voice found for language '{language}' and gender '{gender}'. Using a random voice.")
        voice = voices.find(Language=language)
        if not voice:
            raise ValueError(f"No voice found for language '{language}'")

    communicate = edge_tts.Communicate(text, random.choice(voice)["Name"])
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", dir=temp_dir, delete=False)
        print(f"Saving audio to {temp_file.name}")
        await communicate.save(temp_file.name)
        audio = AudioSegment.from_file_using_temporary_files(temp_file.name)
        play(audio)
        temp_file.close()

    if output_file:
        save_format = output_file.split(".")[-1]
        audio.export(output_file, format=save_format if save_format in ["mp3", "wav", "ogg"] else "mp3")
        print(f"Audio saved to {output_file}")


def callback(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    if param.name == "all_voices":
        ps = subprocess.run(["edge-tts", "--list-voices"], check=True, stdout=subprocess.PIPE)
        subprocess.run(["less"], input=ps.stdout, check=True)
        ctx.exit()
    # click.echo(f"Using {value} as the output file")
    return value

@click.command()
@click.option('--language', '-l', type=str, default='en', help='Language')
@click.option('--gender', '-g', type=str, default='Male', help='Gender: Male | Female')
@click.option("--locale", "-lc", type=str, default="en-IN", help="Locale")
@click.option("--output-file", "-o", type=str, default=None, help="Output file")
@click.option("--all-voices", is_flag=True, help="List all available voices", callback=callback, is_eager=True)
@click.argument('text', nargs=1)
def speech_gen(text, language, gender, output_file, locale, all_voices):
    # if all_voices:
    #     subprocess.run(["edge-tts", "--list-voices", "|", "less"], check=True)
    #     return
    try:
        asyncio.run(tts_ms(text=text, language=language, gender=gender, output_file=output_file, locale=locale))

    except Exception as e:
        print(f"An error occurred: {str(e)}")
