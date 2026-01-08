"""Mel spectrogram computation and configuration module.

This module provides:
1.  `MelSpectrogramConfig`: A Pydantic model for validating and managing
    mel spectrogram computation parameters, loadable from YAML files.
2.  `MelSpectrogramCalculator`: A class to compute mel spectrograms from
    audio signals, configurable via `MelSpectrogramConfig` or direct
    parameterization. This calculator can operate without a direct
    dependency on `librosa` by using its internal Slaney-compatible
    mel filterbank generation.

The mel spectrogram implementation aims for functional equivalence with
NVIDIA NeMo's `AudioToMelSpectrogramPreprocessor`, using NumPy and Numba
for efficient computation.

Example:
    Load configuration from YAML and compute a mel spectrogram:

    >>> from pathlib import Path
    >>> import numpy as np
    >>> # Assuming "config.yaml" exists with a "preprocessor" key.
    >>> # For a runnable example, create a dummy config.yaml:
    >>> # dummy_yaml_content = '''
    # ... preprocessor:
    # ...   sample_rate: 16000
    # ...   features: 80
    # ...   window_size: 0.025
    # ...   window_stride: 0.01
    # ...   n_fft: 512
    # ...   mel_norm: "slaney" # Ensure this is set for Slaney scale
    # ... '''
    >>> # Path("config.yaml").write_text(dummy_yaml_content)
    >>>
    >>> # from your_module import MelSpectrogramConfig, MelSpectrogramCalculator # Adjust import
    >>> # pydantic_config = MelSpectrogramConfig.from_yaml("config.yaml")
    >>> # calculator = MelSpectrogramCalculator.from_config(pydantic_config)
    >>> # audio_signal = np.random.randn(16000 * 3).astype(np.float32) # 3s audio
    >>> # mel_spec = calculator.compute(audio_signal)
    >>> # print(f"Mel spectrogram shape: {mel_spec.shape}")
    >>> # Path("config.yaml").unlink() # Clean up dummy file

Internal Functions:
    _extract_windows_numba: JIT-compiled function to extract windowed frames.
    _slaney_hz_to_mel: Converts Hz to Mel using Slaney's formula.
    _slaney_mel_to_hz: Converts Mel to Hz using Slaney's formula.
"""

from pathlib import Path
from typing import Any, Literal

from numba import jit  # type: ignore
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field
import yaml  # PyYAML for YAML loading

NEMO_CONSTANT = 1e-5  # Used as dither and guard, aligned with some NeMo practices


@jit(nopython=True)  # type: ignore
def _extract_windows_numba(
    audio_padded: NDArray[np.float32],
    window_coeffs: NDArray[np.float32],
    n_fft: int,
    hop_length: int,
    n_frames: int,
) -> NDArray[np.float32]:
    """Extract and window frames from a padded audio signal. (Numba JIT-compiled)

    Args:
        audio_padded: Padded audio signal with sufficient length.
        window_coeffs: Pre-computed window function coefficients (n_fft long).
        n_fft: Size of the Fast Fourier Transform (FFT) window.
        hop_length: Number of samples between successive frames.
        n_frames: Total number of frames to extract.

    Returns:
        2D array of extracted and windowed frames, shape (n_frames, n_fft).
    """
    if n_frames > 0 and len(audio_padded) < n_fft + (n_frames - 1) * hop_length:
        raise ValueError("audio_padded is not long enough for specified framing.")
    if len(window_coeffs) != n_fft:
        raise ValueError("window_coeffs length must equal n_fft.")
    if n_frames < 0:
        raise ValueError("n_frames must be non-negative.")

    frames = np.zeros((n_frames, n_fft), dtype=np.float32)
    for t in range(n_frames):
        start = t * hop_length
        frames[t] = audio_padded[start : start + n_fft] * window_coeffs
    return frames


class MelSpectrogramConfig(BaseModel):
    """Pydantic model for mel spectrogram preprocessor configuration.

    This model defines and validates parameters for mel spectrogram computation,
    allowing for easy loading from YAML files and instantiation of
    `MelSpectrogramCalculator`.
    """

    sample_rate: int = Field(16000, description="Audio sample rate in Hz.")
    window_size: float = Field(0.025, description="Window length in seconds.")
    window_stride: float = Field(0.01, description="Window hop in seconds.")
    window: Literal["hann", "hamming", "blackman", "bartlett", "none"] = Field(
        "hann", description="Name of the window function."
    )
    features: int = Field(80, description="Number of mel frequency bins (n_mels).")
    n_fft: int = Field(512, description="Size of the Fast Fourier Transform window.")
    normalize: Literal["per_feature", "all_features"] | None = Field(
        "per_feature",
        description="Normalization method ('per_feature', 'all_features', or None to disable).",
    )
    preemph: float | None = Field(None, description="Preemphasis coefficient. If None, no preemphasis (0.0).")
    dither: float = Field(NEMO_CONSTANT, description="Strength of dithering noise.")
    log: bool = Field(True, description="If True, applies log scaling to the mel spectrogram.")
    frame_splicing: int = Field(
        1,
        ge=1,
        description="Number of frames to stack. Output features = features * frame_splicing.",
    )
    pad_to: int = Field(
        0,
        ge=0,
        description="Ensures time dimension of spectrogram is multiple of this. 0 or 1 means no padding.",
    )
    spec_pad_value: float = Field(
        0.0,
        alias="pad_value",
        description="Value for padding spectrogram time dim if pad_to > 1.",
    )
    fmin: float = Field(0.0, alias="lowfreq", description="Minimum frequency for mel filterbank.")
    fmax: float | None = Field(
        None,
        alias="highfreq",
        description="Maximum frequency for mel filterbank. Defaults to sample_rate / 2.",
    )
    mag_power: float = Field(
        2.0,
        gt=0,
        description="Power to apply to magnitude spectrogram (e.g., 2.0 for power, 1.0 for magnitude).",
    )
    log_zero_guard_type: Literal["add", "clamp"] = Field(
        "add",
        description="How to handle values close to zero before log scaling ('add' or 'clamp').",
    )
    log_zero_guard_value: float = Field(float(2**-24), description="Value used for the zero guard during log scaling.")
    mel_norm: Literal["slaney", "htk"] = Field(
        "slaney",
        description="Mel filterbank normalization type ('slaney' or 'htk').",
    )
    exact_pad: bool = Field(
        False,
        description="If True, uses a specific padding scheme before STFT similar to NeMo's exact_pad "
        "(typically corresponding to STFT center=False on a pre-padded signal). "
        "If False, uses n_fft // 2 padding (STFT center=True like behavior).",
    )

    class Config:
        populate_by_name = True  # Enables use of YAML field names like 'pad_value', 'lowfreq', 'highfreq'

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "MelSpectrogramConfig":
        """Loads mel spectrogram configuration from a YAML file.

        The YAML file is expected to have a top-level 'preprocessor' key
        containing the parameters defined in this model.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            An instance of MelSpectrogramConfig.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If YAML parsing fails or 'preprocessor' key is missing/invalid.
        """
        file_path = Path(yaml_path)
        if not file_path.exists():
            raise FileNotFoundError(f"YAML configuration file not found: {file_path}")

        with open(file_path) as f:
            try:
                full_config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file {file_path}: {e}") from e

        if not isinstance(full_config, dict) or "preprocessor" not in full_config:
            raise ValueError(f"YAML file {file_path} must contain a 'preprocessor' key.")

        preprocessor_config_dict = full_config["preprocessor"]
        if not isinstance(preprocessor_config_dict, dict):
            raise ValueError(f"'preprocessor' key in {file_path} must contain a dictionary.")

        if "_target_" in preprocessor_config_dict:
            del preprocessor_config_dict["_target_"]

        # Ensure mel_norm default if not in YAML, as it's crucial for filterbank type
        if "mel_norm" not in preprocessor_config_dict and "mel_norm" not in cls.model_fields:
            # This case should ideally be handled by Pydantic default if field exists
            pass  # Pydantic handles default from Field definition
        elif (
            "mel_norm" not in preprocessor_config_dict
            and "mel_norm" in cls.model_fields
            and cls.model_fields["mel_norm"].default is not None
        ):
            preprocessor_config_dict["mel_norm"] = cls.model_fields["mel_norm"].default

        return cls(**preprocessor_config_dict)


class MelSpectrogramCalculator:
    """Computes mel spectrograms from audio signals.

    This class implements a mel spectrogram extraction pipeline that includes
    dithering, preemphasis, STFT, mel filterbank application, log scaling,
    optional normalization, frame splicing, and padding. It uses an internal
    Slaney-compatible mel filterbank generation, removing direct dependency
    on `librosa` for this step.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        features: Number of mel bins.
        n_fft: FFT size.
        hop_length: Hop length in samples.
        win_length: Window length in samples.
        window_coeffs: Computed window coefficients for STFT.
        mel_filterbank: Computed Slaney-compatible mel filterbank matrix.
        normalize: Normalization type ("per_feature" or None).
        preemph: Preemphasis coefficient.
        dither: Dithering strength.
        log: Whether to apply log scaling.
        frame_splicing: Frame splicing factor.
        pad_to: Value to make spectrogram length a multiple of.
        spec_pad_value: Value used for spectrogram padding.
        fmin: Minimum frequency for mel filterbank.
        fmax: Maximum frequency for mel filterbank.
        mag_power: Power for magnitude spectrum.
        log_zero_guard_type: Type of guard for log scaling ("add" or "clamp").
        log_zero_guard_value: Value for log zero guard.
        mel_norm: Specifies the mel filterbank normalization ('slaney' or 'htk').
                  The internal implementation primarily targets 'slaney'.
    """

    sample_rate: int
    features: int
    n_fft: int
    hop_length: int
    win_length: int
    window_coeffs: NDArray[np.float32]
    mel_filterbank: NDArray[np.float32]
    normalize: str | None
    preemph: float
    dither: float
    log: bool
    frame_splicing: int
    pad_to: int
    spec_pad_value: float
    fmin: float
    fmax: float
    mag_power: float
    log_zero_guard_type: str
    log_zero_guard_value: float
    mel_norm: str

    def __init__(
        self,
        sample_rate: int = 16000,
        window_size: float = 0.025,
        window_stride: float = 0.01,
        window: str = "hann",
        features: int = 80,
        n_fft: int = 512,
        normalize: str | None = "per_feature",
        preemph: float | None = None,
        dither: float = NEMO_CONSTANT,
        log: bool = True,
        frame_splicing: int = 1,
        pad_to: int = 0,
        spec_pad_value: float = 0.0,
        lowfreq: float = 0.0,  # Mapped to fmin via Pydantic alias or direct use
        highfreq: float | None = None,  # Mapped to fmax
        mag_power: float = 2.0,
        log_zero_guard_type: str = "add",
        log_zero_guard_value: float = 2**-24,
        mel_norm: str = "slaney",
        exact_pad: bool = False,
        **kwargs: float,  # To absorb any other potential NeMo args from config
    ) -> None:
        """Initializes the MelSpectrogramCalculator."""
        self.sample_rate = sample_rate
        self.features = features
        self.n_fft = n_fft
        self.normalize = normalize
        self.dither = dither
        self.log = log
        self.frame_splicing = frame_splicing
        self.pad_to = pad_to
        self.spec_pad_value = spec_pad_value

        self.win_length = int(sample_rate * window_size)
        self.hop_length = int(sample_rate * window_stride)

        if self.win_length > self.n_fft:
            raise ValueError(
                f"Window length ({self.win_length}) from window_size {window_size}s "
                f"cannot be greater than n_fft ({self.n_fft})."
            )

        self.preemph = preemph if preemph is not None else 0.0
        self.mag_power = mag_power
        self.log_zero_guard_type = log_zero_guard_type
        self.log_zero_guard_value = log_zero_guard_value
        self.mel_norm = mel_norm

        self.exact_pad = exact_pad
        if self.exact_pad:
            if self.hop_length <= 0:  # hop_length must be positive for stft_pad_amount calculation
                raise ValueError("hop_length must be positive when exact_pad is True.")
            # As per NeMo's FilterbankFeatures: (n_fft - hop_length) // 2
            # This padding is applied *before* STFT when exact_pad is true.
            self.stft_pad_amount = (self.n_fft - self.hop_length) // 2
            if self.stft_pad_amount < 0:
                # This case should ideally not happen if n_fft >= hop_length,
                # which is typical. If it does, it means n_fft < hop_length.
                # NeMo doesn't explicitly handle stft_pad_amount < 0,
                # torch.nn.functional.pad might error or behave unexpectedly.
                # Forcing it to 0 or raising an error might be safer.
                # Let's raise an error for now if this unusual condition is met.
                raise ValueError(
                    f"n_fft ({self.n_fft}) must be greater than or equal to hop_length ({self.hop_length}) "
                    f"when exact_pad is True, to ensure non-negative stft_pad_amount."
                )
        else:
            self.stft_pad_amount = 0  # Not used if exact_pad is False

        # Parameter validation (essential ones)
        if not (self.features > 0 and self.n_fft > 0 and self.hop_length > 0):
            raise ValueError("features, n_fft, and hop_length must be positive.")
        if self.win_length < 0:
            raise ValueError("win_length must be non-negative.")
        if not 0 <= self.preemph <= 1:
            raise ValueError("preemph must be between 0.0 and 1.0.")

        self.fmin = lowfreq
        self.fmax = highfreq if highfreq is not None else float(sample_rate) / 2.0

        self.mel_filterbank = self._create_mel_filterbank(self.fmin, self.fmax)

        window_fn_name = window.lower()
        if self.win_length == 0:
            actual_window = np.array([], dtype=np.float32)
        elif window_fn_name == "hann":
            actual_window = np.hanning(self.win_length)
        elif window_fn_name == "hamming":
            actual_window = np.hamming(self.win_length)
        elif window_fn_name == "blackman":
            actual_window = np.blackman(self.win_length)
        elif window_fn_name == "bartlett":
            actual_window = np.bartlett(self.win_length)
        elif window_fn_name == "none" or window_fn_name is None:
            actual_window = np.ones(self.win_length, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported window: {window}")

        if self.win_length == 0:
            self.window_coeffs = np.zeros(self.n_fft, dtype=np.float32)
        else:
            pad_left = (self.n_fft - self.win_length) // 2
            pad_right = self.n_fft - self.win_length - pad_left
            self.window_coeffs = np.pad(actual_window, (pad_left, pad_right), mode="constant").astype(np.float32)

    @classmethod
    def from_config(cls, config: MelSpectrogramConfig) -> "MelSpectrogramCalculator":
        """Creates a MelSpectrogramCalculator instance from a MelSpectrogramConfig object."""
        return cls(**config.model_dump())

    def _slaney_hz_to_mel(
        self,
        frequencies: float | NDArray[np.float32],
        dtype: np.dtype[Any] | None = None,
    ) -> NDArray[np.float32]:
        """Converts Hz to Mel scale using Slaney's Auditory Toolbox formula.

        This formula is linear below 1kHz and logarithmic above.

        Args:
            frequencies: Frequencies in Hz (scalar or array).
            dtype: NumPy dtype for calculations. Defaults to np.float32.

        Returns:
            Frequencies in Mel scale.
        """
        if dtype is None:  # Ruff's B008 rule ("Do not perform function calls in argument defaults")
            dtype = np.dtype(np.float32)

        freq_arr = np.atleast_1d(np.asarray(frequencies, dtype=dtype))  # Ensure it's an array for processing

        f_min = 0.0
... (file truncated)