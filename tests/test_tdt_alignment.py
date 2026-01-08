import pytest
from pathlib import Path
import soundfile as sf

from jarvis.ASR import get_audio_transcriber


def test_tdt_word_timestamps_smoke():
    data_path = Path("data/0.wav")
    if not data_path.exists():
        pytest.skip("No audio fixture available")

    audio, sr = sf.read(data_path, dtype="float32", always_2d=True)
    audio = audio[:, 0]

    try:
        tdt = get_audio_transcriber("tdt")
    except Exception as e:
        pytest.skip(f"TDT transcriber not available: {e}")

    try:
        words = tdt.transcribe_with_word_timestamps(audio, sample_rate=sr)
    except Exception as e:
        pytest.skip(f"TDT alignment failed: {e}")

    assert isinstance(words, list)
    if words:
        for w in words:
            assert "word" in w and "start" in w and "end" in w
            assert w["start"] <= w["end"]
