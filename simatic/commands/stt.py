import click
from simatic.models import audio_recorder
from simatic.models.speech import get_list_of_audio_devices

MODELS = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "distil-small.en", "medium", "medium.en", "distil-medium.en", "large-v1", "large-v2", "large-v3", "large", "distil-large-v2", "distil-large-v3"]

devices = get_list_of_audio_devices()


def list_devices_callback(ctx, param, value):
    if not value:
        return
    print("Available audio devices:")
    for device in devices:
        print(f"{device['index']} --> {device['name']}, {device['max_input_channels']} channels, {device['default_samplerate']} Hz")
    ctx.exit()


def device_callback(ctx, param, value):
    if not value:
        return
    if isinstance(value, int):
        if value < 0 or value >= len(devices):
            raise click.BadParameter(f"Invalid device ID: Use '--list-devices' to see available devices")
        return int(value)
    if isinstance(value, str):
        for i, device in enumerate(devices):
            if value in device["name"]:
                return i
        raise click.BadParameter("Invalid device name")
    return value


@click.command()
@click.option('-f', '--filename', type=str, default=None, help='audio file to store recording to')
@click.option('-l', '--list-devices', is_flag=True, help='show list of audio@ devices and exit', callback=list_devices_callback)
@click.option('-d', '--device', type=int, required=True, help='input device (numeric ID or substring)', callback=device_callback)
@click.option('-m', '--model', type=click.Choice(MODELS), default="tiny", help='Whisper model size')
@click.option('-r', '--samplerate', type=int, default=None, help='sampling rate')
@click.option('-c', '--channels', type=int, default=1, help='number of input channels')
@click.option('-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
def asr(filename, device, list_devices, model, samplerate, channels, subtype):
    try:
        transcriptions = audio_recorder.record_and_transcribe(device=device, filename=filename, sample_rate=samplerate, channels=channels)
        for segment in transcriptions:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    except Exception as e:
        raise click.ClickException(str(e))
