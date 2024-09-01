import logging
import os

from tachyonot.models.whipser import VoiceAssistant, VoiceTranscriber
from rich.console import Console
from rich import print
from rich.panel import Panel
from os import cpu_count
from pathlib import Path

console = Console()


def invoke(parser):
    """
    Add arguments to the parser
    :param parser:
    """
    parser.add_argument(
        "-m",
        "--model",
        default="tiny.en",
        type=str,
        help="Whisper.cpp model, default to %(default)s",
    )
    parser.add_argument(
        "-d",
        "--model-dir",
        default=None,
        type=str,
        help="The directory in which Whisper model is stored. Default is `None`"
    )
    # Positional args
    record_group = parser.add_argument_group("LIVE AUDIO MODE")
    record_group.add_argument(
        "-ind",
        "--input_device",
        type=int,
        default=None,
        help=f"Id of The input device (aka microphone)\n"
        f"available devices {VoiceAssistant.available_devices()}",
    )
    record_group.add_argument(
        "-st",
        "--silence_threshold",
        default=16,
        type=int,
        help="he duration of silence after which the inference will be running, default to %(default)s",
    )
    record_group.add_argument(
        "-bd",
        "--block_duration",
        default=30,
        type=int,
        help="minimum time audio updates in ms, default to %(default)s",
    )

    media_group = parser.add_argument_group("STATIC FILE MODE")
    media_group.add_argument(
        "media_file",
        type=str,
        nargs="*",
        help="The path of the media file or a list of files" "separated by space",
    )

    media_group.add_argument(
        "-p", "--processors", help="number of processors to use during computation"
    )
    media_group.add_argument(
        "-o",
        "--output-type",
        type=str,
        choices=["txt", "vtt", "srt", "csv"],
        help="type of the output file extension where transcriptions will be saved",
    )


def run_speech(args):
    """
    Run the speech command
    :param args: Arguments for Speech to text
    """
    _config_dir = Path(os.environ["CONFIG_PATH"]).parent / "model"
    _config_dir = _config_dir.absolute()
    if not os.path.exists(_config_dir):
        raise FileNotFoundError(
            console.log(
                "[bold red]Model directory not found, please pass the model directory location using `--model-dir` flag"
            )
        )
    model_dir = Path(args.model_dir).absolute() if args.model_dir else _config_dir
    if args.media_file:

        my_transcriber = VoiceTranscriber(
            model=args.model,
            files=args.media_file,
            processors=args.processors,
            output_type=args.output_type,
            n_threads=cpu_count() // 2 if cpu_count() else 1,
            models_dir=model_dir
        )
        with console.status("[bold blue]Transcribing...") as status:
            transcriptions = my_transcriber.generate_transcription()
            console.log("[bold green]Transcription completed")

        for segment in transcriptions:
            print(
                Panel.fit(
                    "%s" % (segment.text),
                    subtitle="[%.2fs -> %.2fs]"
                    % (segment.t0 / 100.0, segment.t1 / 100.0),
                    border_style="blue3",
                    subtitle_align="left",
                )
            )

    else:
        my_assistant = VoiceAssistant(
            model=args.model,
            input_device=args.input_device,
            silence_threshold=args.silence_threshold,
            block_duration=args.block_duration,
            commands_callback=print,
            model_log_level=logging.ERROR,
            models_dir=model_dir
        )
        my_assistant.start()
