"""Compatibility wrapper for the original `glados.ASR` package."""

from glados.ASR import *  # re-export everything

try:
    import glados.ASR as _src
    __all__ = getattr(_src, "__all__", [])
except Exception:
    __all__ = []
