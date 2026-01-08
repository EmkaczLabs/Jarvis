"""API wrappers re-exporting functionality from the original `glados` package.

This module exposes `create_speech` and `app` at the `jarvis.api` import path so that
applications can import `jarvis.api.app` directly.
"""

from glados.api.app import create_speech, app

__all__ = ["create_speech", "app"]
