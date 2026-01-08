"""Compatibility wrapper for the original `glados.audio_io` package."""

from glados.audio_io import *  # re-export everything

try:
    import glados.audio_io as _src
    __all__ = getattr(_src, "__all__", [])
except Exception:
    __all__ = []
