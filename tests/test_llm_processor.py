import queue
import threading

from jarvis.core.llm_processor import LanguageModelProcessor


def _make_processor():
    llm_queue = queue.Queue()
    tts_queue = queue.Queue()
    history = []
    processing_active_event = threading.Event()
    processing_active_event.set()  # Simulate normal processing state
    shutdown_event = threading.Event()
    return LanguageModelProcessor(
        llm_input_queue=llm_queue,
        tts_input_queue=tts_queue,
        conversation_history=history,
        completion_url="http://127.0.0.1/",
        model_name="m",
        api_key=None,
        processing_active_event=processing_active_event,
        shutdown_event=shutdown_event,
    )


def test_sanitizes_inline_thinking():
    proc = _make_processor()
    chunk = {"message": {"content": "<think>this is internal</think> Hello there."}}
    out = proc._process_chunk(chunk)
    assert out is not None
    assert "think" not in out.lower()
    assert "hello there" in out.lower()


def test_sanitizes_across_chunks_and_emits_sentence():
    proc = _make_processor()
    tts_queue = proc.tts_input_queue

    # Simulate chunked stream where the thinking tag crosses chunks
    chunk1 = {"message": {"content": "<think> internal thought that continues"}}
    chunk2 = {"message": {"content": " and ends now</think> Hello!"}}

    out1 = proc._process_chunk(chunk1)
    out2 = proc._process_chunk(chunk2)

    assert out1 is None
    assert out2 is not None
    assert "hello" in out2.lower()
    assert "think" not in out2.lower()

    # Now send the sentence parts to sentence processor and ensure tts queue gets the message
    proc._process_sentence_for_tts([out2])
    sent = tts_queue.get_nowait()
    assert "think" not in sent.lower()
    assert "hello" in sent.lower()


def test_strips_leading_punctuation_after_thinking():
    proc = _make_processor()
    chunk = {"message": {"content": "<think>stuff</think> . hello!"}}
    out = proc._process_chunk(chunk)
    assert out is not None
    # Leading dot should be removed
    assert not out.startswith(".")
    assert out.strip().lower().startswith("hello")


def test_sanitizes_openai_choice_format():
    proc = _make_processor()
    # content in OpenAI 'choices' delta
    chunk = {"choices": [{"delta": {"content": "<think>internal</think> Hello there."}}]}
    out = proc._process_chunk(chunk)
    assert out is not None
    assert "think" not in out.lower()
    assert "hello there" in out.lower()


def test_rejects_openai_inner_think_across_chunks():
    proc = _make_processor()
    # Simulate OpenAI-formatted chunks crossing a <think> span
    chunk1 = {"choices": [{"delta": {"content": "<think> internal thought that continues"}}]}
    chunk2 = {"choices": [{"delta": {"content": " and ends now</think> Hello!"}}]}

    out1 = proc._process_chunk(chunk1)
    out2 = proc._process_chunk(chunk2)

    assert out1 is None
    assert out2 is not None
    assert "hello" in out2.lower()
    assert "think" not in out2.lower()
    proc._process_sentence_for_tts([out2])
    sent = proc.tts_input_queue.get_nowait()
    assert "think" not in sent.lower()
    assert "hello" in sent.lower()


def test_closing_think_with_leading_punctuation():
    proc = _make_processor()
    chunk = {"message": {"content": "<think>stuff</think> . hello!"}}
    out = proc._process_chunk(chunk)
    assert out is not None
    # Leading punctuation should be stripped
    assert not out.startswith(".")
    assert out.strip().lower().startswith("hello")
