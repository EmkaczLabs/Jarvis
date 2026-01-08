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

from glados.cli import download_models as _download_models, models_valid as _models_valid, say as _say, tui as _tui
from jarvis.core.engine import Jarvis, JarvisConfig

# Use a Jarvis-branded default config located relative to this package
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "jarvis_config.yaml"


def say(text: str, config_path: str | Path = DEFAULT_CONFIG) -> None:
    return _say(text, config_path)


def start(config_path: str | Path = DEFAULT_CONFIG) -> None:
    """Start Jarvis using the JarvisConfig model and Jarvis orchestrator."""
    jarvis_config = JarvisConfig.from_yaml(str(config_path))
    jarvis = Jarvis.from_config(jarvis_config)
    if jarvis.announcement:
        jarvis.play_announcement()
    jarvis.run()


def tui(config_path: str | Path = DEFAULT_CONFIG) -> None:
    return _tui(config_path)


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
        return asyncio.run(_download_models())
    else:
        if not _models_valid():
            print("Some model files are invalid or missing. Please run 'uv run jarvis download'")
            return 1
        if args.command == "say":
            say(args.text, args.config)
        elif args.command == "start":
            start(args.config)
        elif args.command == "tui":
            tui()
        else:
            # Default to start if no command specified
            start(DEFAULT_CONFIG)
        return 0


if __name__ == "__main__":
    sys.exit(main())
