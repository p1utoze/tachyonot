import queue
import time
from typing import Callable
import numpy as np
import sounddevice as sd
import pywhispercpp.constants as constants
import webrtcvad
import logging
from pywhispercpp.model import Model
import pywhispercpp.utils as utils


class VoiceTranscriber:
    _file_dict = {
        "csv": lambda x, f: utils.output_csv(x, f),
        "vtt": lambda x, f: utils.output_vtt(x, f),
        "srt": lambda x, f: utils.output_srt(x, f),
        "txt": lambda x, f: utils.output_txt(x, f),
    }
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(
        self,
        model="tiny",
        files: str = None,
        processors: int = None,
        output_type: str = None,
        **model_params,
    ):
        """
        Initialize the VoiceTranscriber
        :param model: whisper.cpp model name or a direct path to a`ggml` model
        :param files: audio file directory path
        :param processors: number of processors to use
        :param output_type: output file type
        """
        self.model = Model(model, **model_params)
        self.processors = processors
        self.files = files
        self.output_type = output_type

    def generate_transcription(self) -> list:
        """
        Generate transcription from the audio files
        :return: list of segments
        """
        for file in self.files:
            segs = self.model.transcribe(
                file, n_processors=self.processors if self.processors else None
            )
            if self.output_type:
                file_name = VoiceTranscriber._file_dict[self.output_type](segs, file)
                logging.info(f"{self.output_type} file saved to {file_name} ")

            return segs


class VoiceAssistant:
    """
    VoiceAssistant class

    Example usage
    ```python
    from pywhispercpp.examples.assistant import VoiceAssistant

    my_assistant = VoiceAssistant(commands_callback=print, n_threads=8)
    my_assistant.start()
    ```
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(
        self,
        model="tiny",
        input_device: int = None,
        silence_threshold: int = 8,
        q_threshold: int = 16,
        block_duration: int = 30,
        commands_callback: Callable[[str], None] = None,
        model_log_level: int = logging.INFO,
        **model_params,
    ):
        """
        :param model: whisper.cpp model name or a direct path to a`ggml` model
        :param input_device: The input device (aka microphone), keep it None to take the default
        :param silence_threshold: The duration of silence after which the inference will be running
        :param q_threshold: The inference won't be running until the data queue is having at least `q_threshold` elements
        :param block_duration: minimum time audio updates in ms
        :param commands_callback: The callback to run when a command is received
        :param model_log_level: Logging level
        :param model_params: any other parameter to pass to the whsiper.cpp model see ::: pywhispercpp.constants.PARAMS_SCHEMA
        """

        self.input_device = input_device
        self.sample_rate = constants.WHISPER_SAMPLE_RATE  # same as whisper.cpp
        self.channels = 1  # same as whisper.cpp
        self.block_duration = block_duration
        self.block_size = int(self.sample_rate * self.block_duration / 1000)
        self.q = queue.Queue()

        self.vad = webrtcvad.Vad()
        self.silence_threshold = silence_threshold
        self.q_threshold = q_threshold
        self._silence_counter = 0

        self.pwccp_model = Model(
            model,
            log_level=model_log_level,
            print_realtime=False,
            print_progress=False,
            print_timestamps=False,
            single_segment=True,
            no_context=True,
            **model_params,
        )
        self.commands_callback = commands_callback

    def _audio_callback(self, indata, frames, time, status):
        """
        This is called (from a separate thread) for each audio block.
        """
        if status:
            logging.warning(f"underlying audio stack warning:{status}")

        assert frames == self.block_size
        audio_data = map(
            lambda x: (x + 1) / 2, indata
        )  # normalize from [-1,+1] to [0,1]
        audio_data = np.fromiter(audio_data, np.float16)
        audio_data = audio_data.tobytes()
        detection = self.vad.is_speech(audio_data, self.sample_rate)
        if detection:
            self.q.put(indata.copy())
            self._silence_counter = 0
        else:
            if self._silence_counter >= self.silence_threshold:
                if self.q.qsize() > self.q_threshold:
                    self._transcribe_speech()
                    self._silence_counter = 0
            else:
                self._silence_counter += 1

    def _transcribe_speech(self):
        logging.info("Speech detected ...")
        audio_data = np.array([])
        while self.q.qsize() > 0:
            # get all the data from the q
            audio_data = np.append(audio_data, self.q.get())
        # Appending zeros to the audio data as a workaround for small audio packets (small commands)
        audio_data = np.concatenate(
            [audio_data, np.zeros((int(self.sample_rate) + 10))]
        )
        # running the inference
        self.pwccp_model.transcribe(
            audio_data, new_segment_callback=self._new_segment_callback
        )

    def _new_segment_callback(self, seg):
        """
        Callback function that process each segment from the stream
        :param seg:
        :return:
        """
        # TODO: Generate a LLM chat inference
        if self.commands_callback:
            self.commands_callback(seg[0].text)

    def start(self) -> None:
        """
        Use this function to start the assistant
        :return: None
        """
        logging.info("Starting VoiceAssistant ...")
        with sd.InputStream(
            device=self.input_device,  # the default input device
            channels=self.channels,
            samplerate=constants.WHISPER_SAMPLE_RATE,
            blocksize=self.block_size,
            callback=self._audio_callback,
        ):

            try:
                logging.info("VoiceAssistant is listening ... (CTRL+C to stop)")
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logging.info("VoiceAssistant stopped")

    @staticmethod
    def available_devices():
        return sd.query_devices()


if __name__ == "__main__":
    _main()
