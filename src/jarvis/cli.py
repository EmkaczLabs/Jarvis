"""CLI entry point for the Jarvis rebrand.

This module is a small wrapper around the existing `glados.cli` functionality but with
user-facing strings and default configuration names updated to the `jarvis` branding.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

from rich import print as rprint
import httpx
from hashlib import sha256
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn
import sounddevice as sd  # type: ignore
import soundfile as sf

from jarvis.core.engine import Jarvis, JarvisConfig

# Use a Jarvis-branded default config located relative to this package
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "jarvis_config.yaml"

# Model manifest copied from the legacy glados.cli
MODEL_DETAILS: dict[str, dict[str, str]] = {
    "models/ASR/nemo-parakeet_tdt_ctc_110m.onnx": {
        "url": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/nemo-parakeet_tdt_ctc_110m.onnx",
        "checksum": "313705ff6f897696ddbe0d92b5ffadad7429a47d2ddeef370e6f59248b1e8fb5",
    },
    "models/ASR/parakeet-tdt-0.6b-v2_encoder.onnx": {
        "url": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/parakeet-tdt-0.6b-v2_encoder.onnx",
        "checksum": "f133a92186e63c7d4ab5b395a8e45d49f4a7a84a1d80b66f494e8205dfd57b63",
    },
    "models/ASR/parakeet-tdt-0.6b-v2_decoder.onnx": {
        "url": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/parakeet-tdt-0.6b-v2_decoder.onnx",
        "checksum": "415b14f965b2eb9d4b0b8517f0a1bf44a014351dd43a09c3a04d26a41c877951",
    },
    "models/ASR/parakeet-tdt-0.6b-v2_joiner.onnx": {
        "url": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/parakeet-tdt-0.6b-v2_joiner.onnx",
        "checksum": "846929b668a94462f21be25c7b5a2d83526e0b92a8306f21d8e336fc98177976",
    },
    "models/ASR/silero_vad_v5.onnx": {
        "url": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/silero_vad_v5.onnx",
        "checksum": "6b99cbfd39246b6706f98ec13c7c50c6b299181f2474fa05cbc8046acc274396",
    },
    "models/TTS/glados.onnx": {
        "url": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/glados.onnx",
        "checksum": "17ea16dd18e1bac343090b8589042b4052f1e5456d42cad8842a4f110de25095",
    },
    "models/TTS/kokoro-v1.0.fp16.onnx": {
        "url": "https://github.com/dnhkng/GLaDOS/releases/download/0.1/kokoro-v1.0.fp16.onnx",
        "checksum": "c1610a859f3bdea01107e73e50100685af38fff88f5cd8e5c56df109ec880204",
    },
    "models/TTS/kokoro-voices-v1.0.bin": {
        "url": "https://github.com/dnhkng/GLaDOS/releases/download/0.1/kokoro-voices-v1.0.bin",
        "checksum": "c5adf5cc911e03b76fa5025c1c225b141310d0c4a721d6ed6e96e73309d0fd88",
    },
    "models/TTS/phomenizer_en.onnx": {
        "url": "https://github.com/dnhkng/GLaDOS/releases/download/0.1/phomenizer_en.onnx",
        "checksum": "b64dbbeca8b350927a0b6ca5c4642e0230173034abd0b5bb72c07680d700c5a0",
    },
}


async def download_with_progress(
    client: httpx.AsyncClient, url: str, file_path: Path, expected_checksum: str, progress: Progress
) -> bool:
    task_id = progress.add_task(f"Downloading {file_path}", status="")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    hash_sha256 = sha256()

    try:
        async with client.stream("GET", url) as response:
            response.raise_for_status()

            total_size = int(response.headers.get("Content-Length", 0))
            if total_size:
                progress.update(task_id, total=total_size)

            with file_path.open(mode="wb") as f:
                async for chunk in response.aiter_bytes(32768):
                    f.write(chunk)
                    hash_sha256.update(chunk)
                    progress.advance(task_id, len(chunk))

        actual_checksum = hash_sha256.hexdigest()
        if actual_checksum != expected_checksum:
            progress.update(task_id, status="[bold red]Checksum failed")
            Path.unlink(file_path)
            return False
        else:
            progress.update(task_id, status="[bold green]OK")
            return True

    except Exception as e:
        progress.update(task_id, status=f"[bold red]Error: {str(e)}")
        return False


async def download_models() -> int:
    with Progress(
        TextColumn("[grey50][progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TextColumn("  {task.fields[status]}"),
    ) as progress:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            tasks = [
                asyncio.create_task(
                    download_with_progress(client, model_info["url"], Path(path), model_info["checksum"], progress)
                )
                for path, model_info in MODEL_DETAILS.items()
            ]
            results: list[bool] = await asyncio.gather(*tasks)

    if not all(results):
        rprint("\n[bold red]Some files were not downloaded successfully")
        return 1
    rprint("\n[bold green]All files downloaded and verified successfully")
    return 0


def models_valid() -> bool:
    for path, model_info in MODEL_DETAILS.items():
        file_path = Path(path)
        if not (file_path.exists() and sha256(file_path.read_bytes()).hexdigest() == model_info["checksum"]):
            return False
    return True


def say(text: str, config_path: str | Path = DEFAULT_CONFIG) -> None:
    jarvis_tts = Jarvis.TTS.tts_glados.SpeechSynthesizer() if False else None
    # Fallback approach: reuse original implementation of running through a conversion and playback
    # Instead of replicating internal behavior (which depends on package state), call into our existing
    # TTS helper
    try:
        from jarvis.api.tts import write_jarvis_audio_file
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_name = tmp.name
        write_jarvis_audio_file(tmp_name, text, format='wav')
        data, sr = sf.read(tmp_name)
        sd.play(data, sr)
        sd.wait()
    finally:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass


def start(config_path: str | Path = DEFAULT_CONFIG) -> None:
    jarvis_config = JarvisConfig.from_yaml(str(config_path))
    jarvis = Jarvis.from_config(jarvis_config)
    if jarvis.announcement:
        jarvis.play_announcement()
    jarvis.run()


def tui(config_path: str | Path = DEFAULT_CONFIG) -> None:
    try:
        from jarvis.tui import JarvisUI

        app = JarvisUI()
        app.run()
    except KeyboardInterrupt:
        sys.exit()


def gui(config_path: str | Path = DEFAULT_CONFIG) -> None:
    try:
        from jarvis.gui import run_gui_with_config
    except ImportError:
        print("Missing GUI dependencies. Install with: 'python -m pip install PySide6 pyqtgraph' or 'pip install .[gui]'")
        return
    except Exception as e:
        print(f"Failed to import GUI runner: {e}")
        raise

    run_gui_with_config(str(config_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Jarvis Voice Assistant")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Download command
    subparsers.add_parser("download", help="Download model files")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start Jarvis voice assistant")
    start_parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to configuration file (default: {DEFAULT_CONFIG})",
    )

    # TUI command
    tui_parser = subparsers.add_parser("tui", help="Start Jarvis voice assistant with TUI")

    # GUI command
    gui_parser = subparsers.add_parser("gui", help="Start Jarvis voice assistant with GUI visualizer")
    gui_parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to configuration file (default: {DEFAULT_CONFIG})",
    )

    # Say command
    say_parser = subparsers.add_parser("say", help="Make Jarvis speak text")
    say_parser.add_argument("text", type=str, help="Text for Jarvis to speak")
    say_parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to configuration file (default: {DEFAULT_CONFIG})",
    )

    args = parser.parse_args()

    if args.command == "download":
        return asyncio.run(download_models())
    else:
        if not models_valid():
            print("Some model files are invalid or missing. Please run 'uv run jarvis download'")
            return 1
        if args.command == "say":
            say(args.text, args.config)
        elif args.command == "start":
            start(args.config)
        elif args.command == "tui":
            tui()
        elif args.command == "gui":
            gui(args.config)
        else:
            # Default to start if no command specified
            start(DEFAULT_CONFIG)
        return 0


if __name__ == "__main__":
    sys.exit(main())
