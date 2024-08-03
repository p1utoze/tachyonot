"""
This module is the entry point for creating a new instance of the SimaticBaseModel and AudioRecorder classes.
"""

from .text import SimaticBaseModel
from .speech import AudioRecorder


__all__ = ["SimaticBaseModel", "AudioRecorder"]
