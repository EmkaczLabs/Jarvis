"""Compatibility wrapper for the original `glados.TTS` package.

This module re-exports the public surface of `glados.TTS` so code can import
`jarvis.TTS` while the implementation remains provided by the original package.
"""

from glados.TTS import *  # re-export everything

# Try to provide a conservative __all__ if the underlying module defines it
try:
    import glados.TTS as _src
    __all__ = getattr(_src, "__all__", [])
except Exception:
    __all__ = []
