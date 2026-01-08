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
Configuration is read into the `JarvisConfig` model (see `src/jarvis/core/engine.py`). `configs/jarvis_config.yaml` contains an example configuration. Important fields include:

- `llm_model` — identifier/name of the LLM to use
- `completion_url` — URL for the LLM completion/streaming endpoint
- `api_key` — optional API key for the LLM
- `interruptible` — whether user input can interrupt speech
- `audio_io` — audio backend name (e.g., `sounddevice`)
- `asr_engine` — ASR engine name (e.g., `tdt`)
- `wake_word` — optional wake word string (typing bypasses this)
- `voice` — TTS voice identifier
- `announcement` — optional startup announcement
- The full example config (in `configs/jarvis_config.yaml`) also contains a `personality_preprompt` list used to seed the assistant's conversation history.

## Components & architecture (high level)
- `Jarvis` (src/jarvis/core/engine.py): orchestrator that wires ASR, TTS, LLM, audio I/O, queues and threads.
- `SpeechListener` (src/jarvis/core/speech_listener.py): captures audio, runs VAD, buffers samples, and triggers ASR when speech ends.
- `LanguageModelProcessor` (src/jarvis/core/llm_processor.py): sends user text (from queue) to the LLM, streams responses, splits into sentences and dispatches them to the TTS queue.
  - The processor handles both OpenAI-style streaming ``data: `` format and plain JSON streaming (e.g., Ollama-like responses).
- `TextToSpeechSynthesizer` (src/jarvis/core/tts_synthesizer.py): converts sentences into audio and places `AudioMessage` objects onto the audio queue.
- `SpeechPlayer` (src/jarvis/core/speech_player.py): plays audio messages, manages interruption behavior and appends assistant messages to conversation history when an end-of-stream token is received.
- `Audio I/O` implementations (src/jarvis/audio_io/): e.g., `sounddevice` backend (`sounddevice_io.py`) using `sounddevice` and a VAD model to provide sample chunks and VAD confidence.

Inter-thread communication uses Python `queue.Queue` instances and threading `Event` objects such as `processing_active_event`, `currently_speaking_event`, and `shutdown_event`.

## Models
Model files are stored under `models/ASR` and `models/TTS`. The download command in `src/jarvis/cli.py` lists the models it expects and verifies checksums.

## HTTP TTS API
An API route exists in `src/jarvis/api/app.py`:

- POST `/v1/audio/speech` — accepts JSON `{ input, model, voice, response_format, speed }` and returns a file-like stream of generated audio.
- A helper script `scripts/serve` starts a local server running that API (uses `litestar` under the hood).

## Web UI & Electron (experimental)

A minimal React + Vite SPA and an Electron wrapper are included under `web/` to provide a desktop UI.

- Development workflow:
  1. Start the Python API server (backend):
     - In PowerShell: `uv run litestar --app jarvis.api.app:app run --host 127.0.0.1 --port 5050 --reload` (or use `scripts/serve`).
  2. Start the frontend dev server:
     - `cd web; npm install; npm run dev` (Vite dev server defaults to port 5173)
  3. The Vite dev server proxies `/v1/*` requests to the backend so the SPA can call the API using relative paths like `/v1/session`.

- Electron:
  - A minimal Electron main process file is available at `web/electron/main.js`.
  - In dev you can run Electron against the Vite dev server (see `web/package.json` scripts). Packaging a production Electron binary requires bundling the Python runtime and the ONNX/model files for the target platform — see the "Caveats" section below.

Notes & caveats:
- The frontend uses the API endpoints to create sessions, stream responses via SSE, and request TTS audio which it plays in-browser. The visualizer is driven by the WebAudio API's AnalyserNode.
- Packaging: bundling Python and ONNX runtimes for Electron is non-trivial and platform-specific. For now the Electron wrapper assumes a running backend during development.

### Talking to Jarvis from the Web UI

The web UI includes a "Record" button in the composer area. It supports two modes:

- Browser SpeechRecognition (preferred): If your browser supports the Web Speech API (e.g., Chrome), the UI will use it to transcribe your speech locally and send the transcript to the assistant immediately.
- Upload / Server ASR: If the browser does not support the Web Speech API, the UI records a short audio clip, converts it to WAV in-browser, uploads it to the server (`POST /v1/audio/asr`), and streams the resulting transcript to the assistant.

If server-side ASR is not available (missing models or runtime issues), the ASR endpoint will return a 503 with a helpful message. In that case the browser SpeechRecognition fallback (if available) is the recommended way to try local speech input.

## Optional GUI visualizer

To enable the cinematic desktop visualizer install the optional GUI extras:

    pip install -e .[gui]

Then run:

    jarvis gui

Features:
- Cinematic "string" visualizer with glowing strokes and a progress indicator (designed to feel sci‑fi/movie‑grade).
- Immediate visual feedback (coarse RMS envelope) followed by a higher‑accuracy view once a background worker finishes.
- Word‑level highlighting synchronized with the assistant's speech when the TDT ASR engine is available. Highlighting uses a blue accent color for readability.
- Optional alignment path: if the configured `asr_engine` is `tdt` the GUI will attempt to compute exact word start/end times and highlight words precisely. If alignment is not available or fails, the GUI falls back to a proportional token timing heuristic.

Notes & tuning:
- The alignment step uses the TDT ASR model and can be CPU‑intensive for long utterances. The GUI performs alignment off the UI thread so visuals start immediately.
- For the lowest possible latency / best sample‑accurate sync it is possible to drive highlights directly from the audio playback position (via the audio backend's output stream callback). This is not enabled by default, but can be added if you need sample-accurate highlighting.
- If you want to avoid the startup announcement playing automatically, set `announcement: null` in `configs/jarvis_config.yaml`.
- Run the GUI with the same Python environment you used to install the GUI extras; using different executables (e.g., `uv run ...`) can cause import differences. If you see Qt errors like "QWidget: Must construct a QApplication before a QWidget", ensure PySide6 is installed in the Python used to run the GUI and launch via `python -m jarvis.cli gui`.

The GUI is optional — you can still use `jarvis start` (headless) or `jarvis tui` (terminal UI).

---