from .text import SimaticBaseModel
from .speech import AudioRecorder

simatic_text = SimaticBaseModel()
audio_recorder = AudioRecorder(model_size_or_path="base.en", device="cpu", compute_type="int8_float32")
__all__ = ["simatic_text", "audio_recorder"]
