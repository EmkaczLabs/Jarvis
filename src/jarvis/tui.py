"""Jarvis TUI wrapper reusing the original Glados UI implementation.

This module adapts the original TUI to present a Jarvis-branded UI while reusing
the existing implementation for reliability.
"""

from glados.tui import GladosUI as _BaseUI


class JarvisUI(_BaseUI):
    """Jarvis-branded terminal UI. Inherits behavior from the original GladosUI."""

    TITLE = "Jarvis v 1.09"
    SUB_TITLE = "(c) 2026 Jarvis Labs"

    # Backwards-compatible alias for older code that may call "start_glados"
    def start_jarvis(self) -> None:
        return getattr(self, "start_glados")()


__all__ = ["JarvisUI"]
