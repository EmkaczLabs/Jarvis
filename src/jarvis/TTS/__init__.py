"""Text-to-Speech (TTS) synthesis components for Jarvis package.

This module provides a protocol-based interface for text-to-speech synthesis
and a factory function to create synthesizer instances for different voices.
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from . import tts_glados, tts_kokoro


class SpeechSynthesizerProtocol(Protocol):
    sample_rate: int

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]: ...


# Factory function
def get_speech_synthesizer(
    voice: str = "jarvis",
) -> SpeechSynthesizerProtocol:
    """
    Factory function to get an instance of an audio synthesizer based on the specified voice type.

    Parameters:
        voice (str): The type of TTS engine to use:
            - "jarvis" or "glados": Jarvis/GLaDOS voice synthesizer
            - <str>: Kokoro voice synthesizer using the specified voice <str> is available
    Returns:
        SpeechSynthesizerProtocol: An instance of the requested speech synthesizer
    Raises:
        ValueError: If the specified TTS engine type is not supported
    """
    if voice.lower() in ("glados", "jarvis"):
        return tts_glados.SpeechSynthesizer()

    available_voices = tts_kokoro.get_voices()
    if voice not in available_voices:
        raise ValueError(f"Voice '{voice}' not available. Available voices: {available_voices}")

    return tts_kokoro.SpeechSynthesizer(voice=voice)


__all__ = ["SpeechSynthesizerProtocol", "get_speech_synthesizer"]
