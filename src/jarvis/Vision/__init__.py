"""Compatibility wrapper for the original `glados.Vision` package."""

from glados.Vision import *  # re-export everything

try:
    import glados.Vision as _src
    __all__ = getattr(_src, "__all__", [])
except Exception:
    __all__ = []
