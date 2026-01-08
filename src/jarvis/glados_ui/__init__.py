"""Compatibility wrapper for the original `glados.glados_ui` package."""

from glados.glados_ui import *  # re-export everything

try:
    import glados.glados_ui as _src
    __all__ = getattr(_src, "__all__", [])
except Exception:
    __all__ = []
