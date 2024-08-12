import click
from simatic.models import AudioRecorder
from simatic.models.speech import get_list_of_audio_devices
from argparse import Action

MODELS = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "distil-small.en", "medium", "medium.en", "distil-medium.en", "large-v1", "large-v2", "large-v3", "large", "distil-large-v2", "distil-large-v3"]

devices = get_list_of_audio_devices()

audio_recorder = AudioRecorder(model_size_or_path="base.en", device="cpu", compute_type="int8_float32")


def list_devices_callback(value):
    class customAction(Action):
        def __call__(self, parser, args, values, option_string=None):
            print(value)
            print("Available audio devices:")
            print(args, values)
            setattr(args, self.dest, values)
    return customAction


def list_devices_callback(value):
    if not value:
        return
    print("Available audio devices:")
    for device in devices:
        print(f"{device['index']} --> {device['name']}, {device['max_input_channels']} channels, {device['default_samplerate']} Hz")


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


def listen_command(subparsers):
    listen_parser = subparsers.add_parser("listen", help="Listen to the microphone and transcribe speech")
    listen_parser.add_argument('-f', '--filename', type=str, default=None, help='audio file to store recording to')
    listen_parser.add_argument('-ld', '--list-devices', help='show list of audio@ devices and exit', action=list_devices_callback(False))
    listen_parser.add_argument('-d', '--device', type=int, required=True, help='input device (numeric ID or substring)', callback=device_callback)
    listen_parser.add_argument('-m', '--model', choices=MODELS, default="tiny", help='Whisper model size')
    listen_parser.add_argument('-r', '--samplerate', type=int, default=None, help='sampling rate')
    listen_parser.add_argument('-c', '--channels', type=int, default=1, help='number of input channels')
    listen_parser.add_argument('-t', '--subtype', type=str, help='`sound` file subtype (e.g. "PCM_24")')
    listen_parser.add_argument("-l", "--language", type=str, default=None, help="Enter the language in ISO Code format with 2 character")


def asr(filename, device, list_devices, model, samplerate, channels, subtype, language):
    try:
        transcriptions = audio_recorder.record_and_transcribe(device=device, filename=filename, sample_rate=samplerate, channels=channels, language=language)
        for segment in transcriptions:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    except Exception as e:
        raise click.ClickException(str(e))
