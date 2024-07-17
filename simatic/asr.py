import argparse
from faster_whisper import WhisperModel
import sounddevice as sd
import tempfile
from pydub import AudioSegment

def record_audio(duration):
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(
        int(16000 * duration), dtype='float32', channels=1, blocking=True)
    sd.wait()
    print("Recording finished.")
    print(audio_data)
    return audio_data

def save_audio(audio_data, filename):
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=16000,
        sample_width=audio_data.dtype.itemsize,
        channels=1
    )
    audio_segment.export("temp.wav", format="wav")
    audio_segment.export(filename, format="wav")

def transcribe_audio(model: WhisperModel, audio_file):
    segments, info = model.transcribe(audio_file, beam_size=5, word_timestamps=True)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    return segments

def main():
    device_list = ' | '.join([f"{d['index']} --> {d['name']}" for d in sd.query_devices()])
    parser = argparse.ArgumentParser(description="Offline Multilingual Automatic Speech Recognition CLI")
    parser.add_argument("--duration", type=int, default=5, help="Recording duration in seconds")
    parser.add_argument("--model", default="tiny", choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "distil-small.en", "medium", "medium.en", "distil-medium.en", "large-v1", "large-v2", "large-v3", "large", "distil-large-v2", "distil-large-v3"], help="Whisper model size")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run the model on")
    parser.add_argument("--file", help="Path to an existing audio file for transcription")
    parser.add_argument("--driver-id", type=int, help=f"Driver ID for the audio device: \n{device_list}", required=True)
    args = parser.parse_args()

    if args.driver_id is not None:
        device = sd.query_devices(args.driver_id)
        sd.default.device = device["name"]
        sd.default.samplerate = device["default_samplerate"]
        sd.default.channels = device["max_input_channels"]
        print(f"channel: {sd.default.channels}, samplerate: {sd.default.samplerate}")

    print("Loading Whisper model...")
    model = WhisperModel(args.model, device=args.device, compute_type="int8_float32")
    print(f"Model loaded on {args.device}")

    if args.file:
        print(f"Transcribing file: {args.file}")
        transcription = transcribe_audio(model, args.file)
    else:
        audio_data = record_audio(args.duration)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
            save_audio(audio_data, temp_audio.name)
            print("Transcribing audio...")
            transcription = transcribe_audio(model, temp_audio.name)

    print("\nTranscription:")

    for segment in transcription:
        for word in segment.words:
            print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))

if __name__ == "__main__":
    main()