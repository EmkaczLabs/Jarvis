"""Wrapper module to provide a Jarvis alias for the original Glados engine.

This module keeps the original implementation in `glados.core.engine` and
exposes rebranded class names for backwards compatibility of imports from
`jarvis.core.engine`.
"""

from glados.core.engine import Glados as _Glados, GladosConfig as _GladosConfig, PersonalityPrompt
from pathlib import Path
import yaml


class Jarvis(_Glados):
    """Jarvis is a lightweight alias for the original Glados orchestrator.

    It inherits from the original `Glados` class so it retains all behavior while
    providing a rebranded name for user code/tests that import `jarvis`.
    """

    pass


class JarvisConfig(_GladosConfig):
    """Configuration model alias for Jarvis."""

    @classmethod
    def from_yaml(cls, path: str | Path, key_to_config: tuple[str, ...] = ("Jarvis", "Glados")) -> "JarvisConfig":
        """Load a JarvisConfig from YAML, accepting either a 'Jarvis' or 'Glados' top-level key.

        If the YAML file contains one of the expected top-level keys it will be used; if not,
        the method will attempt to interpret the file as the config mapping directly.
        """
        path = Path(path)

        # Try different encodings
        for encoding in ["utf-8", "utf-8-sig"]:
            try:
                data = yaml.safe_load(path.read_text(encoding=encoding))
                break
            except UnicodeDecodeError:
                if encoding == "utf-8-sig":
                    raise ValueError(f"Could not decode YAML file {path} with any supported encoding")

        if data is None:
            raise ValueError(f"Empty configuration file: {path}")

        # Prefer explicit top-level keys in the order provided
        if isinstance(data, dict):
            for key in key_to_config:
                if key in data:
                    return cls.model_validate(data[key])

            # If expected top-level keys are absent, assume the YAML document is already the config mapping
            return cls.model_validate(data)

        raise ValueError("Invalid YAML configuration format")


__all__ = ["Jarvis", "JarvisConfig", "PersonalityPrompt"]
