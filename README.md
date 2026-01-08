# Jarvis — Local Personality Core

Jarvis is a local voice assistant orchestrator that combines voice activity detection (VAD), automatic speech recognition (ASR), streaming language model responses (LLM), and text-to-speech (TTS) into a low-latency interactive assistant.

This README is based on the project's source code (CLI, TUI, core engine and API).

## Quick facts
- Python: requires Python >= 3.12 (see `pyproject.toml`).
- CLI entry point: `jarvis` (installed by the package; implementation in `src/jarvis/cli.py`).
- Main commands: `download`, `start`, `tui`, `say`.

## Install (development / local)
1. Create and activate a virtualenv (example PowerShell):

   python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1

2. Install the package (editable) and developer extras:

   pip install -e .[dev]

3. Optional extras for ONNX runtimes (if you need them):

   pip install -e .[cpu]    # CPU ONNX runtime
   pip install -e .[cuda]   # GPU ONNX runtime

Dependencies and extras are declared in `pyproject.toml`.

## Download required models
The project provides a convenience command to download required models (ASR / VAD / TTS etc.):

    jarvis download

(The CLI download task reads model URLs and SHA256 checksums from `src/jarvis/cli.py` and writes files into the `models/` tree.)

## Run
- Start headless assistant (blocking):

    jarvis start --config configs/jarvis_config.yaml

- Start the TUI (recommended for local interaction):

    jarvis tui

- Make Jarvis speak a short phrase from the CLI:

    jarvis say "Hello, world"

You can also run the CLI via `python -m jarvis.cli` if you don't install the package.

## TUI typing feature
- While the TUI is running, press `t` to open a typing prompt.
- Type a message and press Enter; the text will be submitted to the assistant and processed exactly like spoken input (Jarvis will respond with both voice and text).
- Typed input bypasses the wake-word requirement (if any in config) and is submitted immediately via `Jarvis.submit_text()`.
- The `interruptible` setting in the config still applies: if `interruptible: false` and Jarvis is speaking, new typed input will be ignored until speaking finishes.

## Configuration (canonical keys)
Configuration is read into the `JarvisConfig` model (see `src/glados/core/engine.py`). `configs/jarvis_config.yaml` contains an example configuration. Important fields include:

- `llm_model` — identifier/name of the LLM to use
- `completion_url` — URL for the LLM completion/streaming endpoint
- `api_key` — optional API key for the LLM
- `interruptible` — whether user input can interrupt speech
- `audio_io` — audio backend name (e.g., `sounddevice`)
- `asr_engine` — ASR engine name (e.g., `tdt`)
- `wake_word` — optional wake word string (typing bypasses this)
- `voice` — TTS voice identifier
- `announcement` — optional startup announcement

The full example config (in `configs/jarvis_config.yaml`) also contains a `personality_preprompt` list used to seed the assistant's conversation history.

## Components & architecture (high level)
- `Jarvis` (src/glados/core/engine.py): orchestrator that wires ASR, TTS, LLM, audio I/O, queues and threads.
- `SpeechListener` (src/glados/core/speech_listener.py): captures audio, runs VAD, buffers samples, and triggers ASR when speech ends.
- `LanguageModelProcessor` (src/glados/core/llm_processor.py): sends user text (from queue) to the LLM, streams responses, splits into sentences and dispatches them to the TTS queue.
  - The processor handles both OpenAI-style streaming ``data: `` format and plain JSON streaming (e.g., Ollama-like responses).
- `TextToSpeechSynthesizer` (src/glados/core/tts_synthesizer.py): converts sentences into audio and places `AudioMessage` objects onto the audio queue.
- `SpeechPlayer` (src/glados/core/speech_player.py): plays audio messages, manages interruption behavior and appends assistant messages to conversation history when an end-of-stream token is received.
- `Audio I/O` implementations (src/glados/audio_io/): e.g., `sounddevice` backend (`sounddevice_io.py`) using `sounddevice` and a VAD model to provide sample chunks and VAD confidence.

Inter-thread communication uses Python `queue.Queue` instances and threading `Event` objects such as `processing_active_event`, `currently_speaking_event`, and `shutdown_event`.

## Models
Model files are stored under `models/ASR` and `models/TTS`. The download command in `src/jarvis/cli.py` lists the models it expects and verifies checksums.

## HTTP TTS API
An API route exists in `src/glados/api/app.py`:

- POST `/v1/audio/speech` — accepts JSON `{ input, model, voice, response_format, speed }` and returns a file-like stream of generated audio.
- A helper script `scripts/serve` starts a local server running that API (uses `litestar` under the hood).

## Tests & development
- Tests use `pytest`. Install dev extras and run:

    pip install -e .[dev]
    pytest -q

- Linting / formatting: configured with `ruff` (see `pyproject.toml`).

## Troubleshooting
- Make sure required optional runtime packages (ONNX runtime variants) match your hardware and OS.
- If you encounter DLL/installation issues when running ONNX models on Windows, ensure your environment has the appropriate system dependencies for your ONNX runtime distribution.

## License
See `LICENSE.txt` in the repository for license terms.

## Contributing
Contributions are welcome: open an issue or submit a pull request with tests and a short description of the change.

---

If you'd like, I can also add a short CONTRIBUTING.md and a developer quickstart script to the repository.