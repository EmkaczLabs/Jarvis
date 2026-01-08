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


def test_preserves_spaces_between_chunks():
    proc = _make_processor()
    # Simulate chunks that would naturally arrive split from the LLM stream
    proc._process_sentence_for_tts(["Hello", "How can I assist you today?"])
    sent = proc.tts_input_queue.get_nowait()
    assert sent == "Hello How can I assist you today?"


def test_think_removal_preserves_spaces_across_tags():
    proc = _make_processor()

    # Simulate an inline think span that appears between words
    chunk = {"message": {"content": "Hello<think>internal</think> How can I assist you today."}}
    out = proc._process_chunk(chunk)
    assert out is not None
    assert "think" not in out.lower()
    # The space should be preserved where the tag was removed
    assert "hello how can" in out.lower()


def test_think_removal_across_chunk_boundaries_preserves_spaces():
    proc = _make_processor()
    # Simulate chunked stream where opening/closing tags split across chunks
    chunk1 = {"message": {"content": "Hello<think>internal continues"}}
    chunk2 = {"message": {"content": " and ends</think> How are you?"}}

    out1 = proc._process_chunk(chunk1)
    out2 = proc._process_chunk(chunk2)

    assert out1 is None
    assert out2 is not None
    assert "hello" in out2.lower()
    assert "how are you" in out2.lower()


def test_merges_mid_word_chunks():
    proc = _make_processor()
    # Simulate a mid-word split across chunks
    proc._process_sentence_for_tts(["clar", "ify"])
    sent = proc.tts_input_queue.get_nowait()
    assert sent == "clarify"


def test_preserves_space_when_present_between_chunks():
    proc = _make_processor()
    # Simulate a normal word boundary where the first chunk contained trailing whitespace
    proc._process_sentence_for_tts(["hello ", "world"])
    sent = proc.tts_input_queue.get_nowait()
    assert sent == "hello world"


def test_single_letter_capital_prefix_merges():
    proc = _make_processor()
    proc._process_sentence_for_tts(["N", "iger", "so"])
    sent = proc.tts_input_queue.get_nowait()
    assert sent.startswith("Niger")
    assert "Niger so" in sent


def test_does_not_merge_I_or_A_prefix():
    proc = _make_processor()
    proc._process_sentence_for_tts(["I", "am"])
    sent = proc.tts_input_queue.get_nowait()
    assert sent == "I am"


def test_does_not_merge_common_small_words():
    proc = _make_processor()
    proc._process_sentence_for_tts(["can", "you"])
    sent = proc.tts_input_queue.get_nowait()
    assert sent == "can you"


def test_inserts_period_before_sentence_starter():
    proc = _make_processor()
    sentence = (
        "As of twenty twenty-three approximately thirty-eight million people live in Poland "
        "Please check the latest data for accuracy"
    )
    proc._process_sentence_for_tts([sentence])
    sent = proc.tts_input_queue.get_nowait()
    assert "Poland. Please" in sent
