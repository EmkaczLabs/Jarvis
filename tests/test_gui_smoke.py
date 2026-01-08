import queue
import threading
import time
import numpy as np

from jarvis.core.speech_player import SpeechPlayer
from jarvis.core.audio_data import AudioMessage


class FakeAudioIO:
    def start_speaking(self, audio_data, sample_rate=None, text=""):
        # No-op for test
        pass

    def measure_percentage_spoken(self, total_samples, sample_rate=None):
        # Simulate immediate completion
        return False, 100

    def stop_speaking(self):
        pass


def test_playback_listener_called():
    audio_queue = queue.Queue()
    audio_msg = AudioMessage(audio=np.array([0.0, 0.1, -0.1], dtype=np.float32), text="hello world", is_eos=False)
    audio_queue.put(audio_msg)

    called_q: queue.Queue[AudioMessage] = queue.Queue()

    shutdown_event = threading.Event()
    currently_speaking_event = threading.Event()
    processing_active_event = threading.Event()

    player = SpeechPlayer(
        audio_io=FakeAudioIO(),
        audio_output_queue=audio_queue,
        conversation_history=[],
        tts_sample_rate=16000,
        shutdown_event=shutdown_event,
        currently_speaking_event=currently_speaking_event,
        processing_active_event=processing_active_event,
        pause_time=0.01,
    )

    def listener(msg: AudioMessage) -> None:
        called_q.put(msg)

    player.register_playback_listener(listener)

    t = threading.Thread(target=player.run, daemon=True)
    t.start()

    # Wait for the listener to receive the callback
    try:
        received = called_q.get(timeout=1.0)
        assert isinstance(received, AudioMessage)
        assert received.text == "hello world"
    finally:
        shutdown_event.set()
        t.join(timeout=1.0)
