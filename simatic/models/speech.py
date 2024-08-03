import queue
import sys
import tempfile

import noisereduce as nr
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from scipy import signal


class AudioRecorder:
    """
    A class to record audio from the CLI directly and transcribe it using the Whisper model.
    Initialize the AudioRecorder class with the Whisper model.
    Same as the WhisperModel class, the AudioRecorder class also accepts the same parameters.
    """
    _instance = None

    def __init__(self, **model_params):
        """
        Initialize the AudioRecorder class with the Whisper model.
        Same as the WhisperModel class, the AudioRecorder class also accepts the same parameters.
        :param model_params:
        """
        self.model: WhisperModel = WhisperModel(**model_params)
        self._q = queue.Queue()

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)
        return cls._instance

    def transcribe_audio(self, audio_data, **kwargs):
        # audio_data = np.clip(audio_data, -1, 1)
        segments, info = self.model.transcribe(audio_data, **kwargs)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        return segments

    def callback(self, indata, frames, time, status):
        """
        This is called (from a separate thread) for each audio block.
        """

        # Here the frames and time is used to calculate the duration of the audio block but not used
        if status:
            print(status, file=sys.stderr)
        self._q.put(indata.copy())

    def apply_noise_reduction(self, audio_data, sample_rate):
        sos = signal.butter(10, [300, 3000], 'bandpass', fs=sample_rate, output='sos')
        filtered_audio = signal.sosfilt(sos, audio_data)
        reduced_noise = nr.reduce_noise(y=filtered_audio, sr=sample_rate)
        return reduced_noise

    def record_and_transcribe(self, sample_rate=None, channels=1, device=None, filename=None, language=None):
        try:
            if sample_rate is None:
                device_info = sd.query_devices(device, 'input')
                # soundfile expects an int, sounddevice provides a float:
                sample_rate = int(device_info['default_samplerate'])
            if filename is None:
                filename = tempfile.mktemp(prefix='default_output_',
                                                suffix='.wav', dir='')

            # Make sure the file is opened before recording anything:
            with sf.SoundFile(filename, mode='x', samplerate=sample_rate,
                              channels=channels) as file:
                with sd.InputStream(samplerate=sample_rate, device=device,
                                    channels=channels, callback=self.callback):
                    print('#' * 80)
                    print('press Ctrl+C to stop the recording')
                    print('#' * 80)
                    while True:
                        file.write(self._q.get())
        except KeyboardInterrupt:
            print('\nRecording finished: ' + repr(filename))
            # print("\nApplying noise reduction...")
            # audio_data, sample_rate = sf.read(filename)
            # reduced_noise = self.apply_noise_reduction(audio_data, sample_rate)
            transcribe_kwargs = {
                "vad_filter": True,
                "hallucination_silence_threshold": 5,
                "language": language
            }
            transcriptions = self.transcribe_audio(filename, **transcribe_kwargs)

            return transcriptions
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return False


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


def get_list_of_audio_devices():
    return sd.query_devices()

