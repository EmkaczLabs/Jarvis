import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import queue
import threading

from jarvis.core.engine import Jarvis


class FakeAudioIO:
    def __init__(self):
        self.stopped = False

    def stop_speaking(self):
        self.stopped = True


def test_submit_text_when_interruptible_sets_queue_and_flags():
    jar = Jarvis.__new__(Jarvis)  # Bypass __init__ to avoid starting threads

    jar.interruptible = True
    jar.currently_speaking_event = threading.Event()
    jar.currently_speaking_event.set()  # Simulate that assistant is speaking
    jar.processing_active_event = threading.Event()
    jar.audio_io = FakeAudioIO()
    jar.llm_queue = queue.Queue()

    jar.submit_text("Hello typed world")

    # Text should be queued
    assert jar.llm_queue.get_nowait() == "Hello typed world"
    # Processing event should be set
    assert jar.processing_active_event.is_set()
    # stop_speaking should have been called
    assert jar.audio_io.stopped is True


def test_submit_text_when_not_interruptible_ignores_input():
    jar = Jarvis.__new__(Jarvis)

    jar.interruptible = False
    jar.currently_speaking_event = threading.Event()
    jar.currently_speaking_event.set()  # Assistant is speaking
    jar.processing_active_event = threading.Event()
    jar.audio_io = FakeAudioIO()
    jar.llm_queue = queue.Queue()

    jar.submit_text("Should be ignored")

    assert jar.llm_queue.empty()
    assert not jar.processing_active_event.is_set()
    # stop_speaking should not have been called in non-interruptible mode
    assert jar.audio_io.stopped is False
