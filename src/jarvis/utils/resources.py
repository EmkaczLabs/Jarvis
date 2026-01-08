from functools import lru_cache
import os
from pathlib import Path

# Prefer the rebranded package name if available, fall back to the legacy package
try:
    import jarvis as _project_pkg  # type: ignore
except Exception:
    import glados as _project_pkg  # type: ignore


@lru_cache(maxsize=1)
def get_package_root() -> Path:
    """Get the absolute path to the package root directory (cached)."""
    # Get the directory where the package module is located
    package_dir = Path(os.path.dirname(os.path.abspath(_project_pkg.__file__)))
    # Go up to the project root (src/<package> -> src -> project_root)
    return package_dir.parent.parent


def resource_path(relative_path: str) -> Path:
    """Return absolute path to a model file."""
    return get_package_root() / relative_path
