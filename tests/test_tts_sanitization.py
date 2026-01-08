import numpy as np
import queue
import threading

from glados.core.tts_synthesizer import TextToSpeechSynthesizer


class FakeTTSModel:
    sample_rate = 16000

    def generate_speech_audio(self, text: str) -> np.ndarray:
        # Return a short array based on text length
        return np.ones((max(1, len(text.split())),), dtype=np.float32) * 0.1


class FakeSTC:
    def text_to_spoken(self, text: str) -> str:
        return text


def _make_synth():
    return TextToSpeechSynthesizer(
        tts_input_queue=queue.Queue(),
        audio_output_queue=queue.Queue(),
        tts_model=FakeTTSModel(),
        stc_instance=FakeSTC(),
        shutdown_event=threading.Event(),
        pause_time=0.01,
    )


def test_sanitize_removes_think_tags():
    synth = _make_synth()
    s = synth._sanitize_text_for_tts("<think>internal</think> Hello there!")
    assert s is not None
    assert "think" not in s.lower()
    assert "hello there" in s.lower()


def test_sanitize_drops_only_think():
    synth = _make_synth()
    s = synth._sanitize_text_for_tts("<think>internal thoughts here</think>")
    assert s is None


def test_sanitize_strips_leading_punctuation():
    synth = _make_synth()
    s = synth._sanitize_text_for_tts("<think>stuff</think> . hello!")
    assert s is not None
    assert s.lower().startswith("hello")
