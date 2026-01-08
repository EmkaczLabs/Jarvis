# ruff: noqa: RUF001
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from functools import cache
from pathlib import Path
from pickle import load
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore

from ..utils.resources import resource_path

# Default OnnxRuntime is way to verbose, only show fatal errors
ort.set_default_logger_severity(4)


@dataclass
class ModelConfig:
    MODEL_PATH: Path
    PHONEME_DICT_PATH: Path
    TOKEN_TO_IDX_PATH: Path
    IDX_TO_TOKEN_PATH: Path
    CHAR_REPEATS: int = 3
    MODEL_INPUT_LENGTH: int = 64
    EXPAND_ACRONYMS: bool = False
    USE_CUDA: bool = True

    def __init__(
        self,
        model_path: Path | None = None,
        phoneme_dict_path: Path | None = None,
        token_to_idx_path: Path | None = None,
        idx_to_token_path: Path | None = None,
    ) -> None:
        # Provide default Path objects if None is passed or for defaults
        self.MODEL_PATH = model_path if model_path is not None else resource_path("models/TTS/phomenizer_en.onnx")
        self.PHONEME_DICT_PATH = (
            phoneme_dict_path if phoneme_dict_path is not None else resource_path("models/TTS/lang_phoneme_dict.pkl")
        )
        self.TOKEN_TO_IDX_PATH = (
            token_to_idx_path if token_to_idx_path is not None else resource_path("models/TTS/token_to_idx.pkl")
        )
        self.IDX_TO_TOKEN_PATH = (
            idx_to_token_path if idx_to_token_path is not None else resource_path("models/TTS/idx_to_token.pkl")
        )


class SpecialTokens(Enum):
    PAD = "_"
    START = "<start>"
    END = "<end>"
    EN_US = "<en_us>"


class Punctuation(Enum):
    PUNCTUATION = "().,:?!/–"
    HYPHEN = "-"
    SPACE = " "

    @classmethod
    @cache
    def get_punc_set(cls) -> set[str]:
        return set(cls.PUNCTUATION.value + cls.HYPHEN.value + cls.SPACE.value)

    @classmethod
    @cache
    def get_punc_pattern(cls) -> re.Pattern[str]:
        """
        Compile a regular expression pattern to match punctuation and space characters.

        Returns:
            re.Pattern[str]: A compiled regex pattern that matches any punctuation or space character.

        Example:
            pattern = Punctuation.get_punc_pattern()
            # Matches single punctuation or space characters
            matches = pattern.findall("Hello, world!")  # Returns [',', ' ']
        """
        return re.compile(f"([{cls.PUNCTUATION.value + cls.SPACE.value}])")


class Phonemizer:
    """Phonemizer class for converting text to phonemes.

    This class uses an ONNX model to predict phonemes for a given text.

    Attributes:
        phoneme_dict: Dictionary of phonemes for each word.
        token_to_idx: Mapping of tokens to indices.
        idx_to_token: Mapping of indices to tokens.
        ort_session: ONNX runtime session for the model.
        special_tokens: Set of special tokens.

    Methods:
        _load_pickle(path: Path) -> dict:
            Load a pickled dictionary from path.

        _unique_consecutive(arr: np.ndarray) -> List[np.ndarray]:
            Equivalent to torch.unique_consecutive for numpy arrays.

        _remove_padding(arr: np.ndarray, padding_value: int = 0) -> List[np.ndarray]:
            Remove padding from an array.

        _trim_to_stop(arr: np.ndarray, end_index: int = 2) -> List[np.ndarray]:
            Trim an array to the stop index.

        _process_model_output(arr: np.ndarray) -> List[np.ndarray]:
            Process the output of the model to get the phoneme indices.

        _expand_acronym(word: str) -> str:
            Expand an acronym into its subwords.

        encode(sentence: Iterable[str]) -> List[int]:
            Map a sequence of symbols to a sequence of indices.

        decode(sequence: np.ndarray) -> str:
            Map a sequence of indices to a sequence of symbols.

        _pad_sequence_fixed(v: List[np.ndarray], target_length: int = ModelConfig.MODEL_INPUT_LENGTH) -> np.ndarray:
            Pad or truncate a list of arrays to a fixed length.

        _get_dict_entry(word: str, lang: str, punc_set: set[str]) -> str | None:
            Get the phoneme entry for a word in the dictionary.

        _get_phonemes(word: str, word_phonemes: Dict[str, Union[str, None]], word_splits: Dict[str, List[str]]) -> str:
            Get the phonemes for a word.

        _clean_and_split_texts(
            texts: List[str],
            punc_set: set[str],
            punc_pattern: re.Pattern
        ) -> tuple[List[List[str]], set[str]]:
            Clean and split texts.

        convert_to_phonemes(texts: List[str], lang: str) -> List[str]:
            Convert a list of texts to phonemes using a phonemizer.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """
        Initialize a Phonemizer instance with optional configuration.

        Parameters:
            config (ModelConfig, optional): Configuration settings for the phonemizer.
                If not provided, a default ModelConfig will be used.

        Attributes:
            config (ModelConfig): Configuration for the phonemizer.
            phoneme_dict (dict[str, str]): Dictionary mapping words to their phonetic representations.
            token_to_idx (dict): Mapping of tokens to their corresponding indices.
            idx_to_token (dict): Mapping of indices back to tokens.
            ort_session (InferenceSession): ONNX runtime session for model inference.
            special_tokens (set[str]): Set of special tokens used in phonemization.

        Notes:
            - Adds a special phoneme entry for "glados"
            - Configures ONNX runtime session with available providers, excluding TensorRT
        """
        if config is None:
            config = ModelConfig()
        self.config = config
        self.phoneme_dict: dict[str, str] = self._load_pickle(self.config.PHONEME_DICT_PATH)

        # Keep the existing GLaDOS mapping for backwards compatibility and add Jarvis mapping
        self.phoneme_dict["glados"] = "ɡlˈɑːdɑːs"  # Add GLaDOS to the phoneme dictionary!
        self.phoneme_dict["jarvis"] = "dʒɑːrvɪs"

        self.token_to_idx = self._load_pickle(self.config.TOKEN_TO_IDX_PATH)
        self.idx_to_token = self._load_pickle(self.config.IDX_TO_TOKEN_PATH)

        providers = ort.get_available_providers()
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")
        if "CoreMLExecutionProvider" in providers:
            providers.remove("CoreMLExecutionProvider")

        self.ort_session = ort.InferenceSession(
            self.config.MODEL_PATH,
            sess_options=ort.SessionOptions(),
            providers=providers,
        )

        self.special_tokens: set[str] = {
            SpecialTokens.PAD.value,
            SpecialTokens.END.value,
            SpecialTokens.EN_US.value,
        }

    @staticmethod
    def _load_pickle(path: Path) -> dict[str, Any]:
        """
        Load a pickled dictionary from the specified file path.

        Parameters:
            path (Path): The file path to the pickled dictionary file.

        Returns:
            dict[str, Any]: The loaded dictionary containing key-value pairs.

        Raises:
            FileNotFoundError: If the specified file path does not exist.
            pickle.UnpicklingError: If there are issues unpickling the file.
        """
        with path.open("rb") as f:
            return load(f)  # type: ignore

    @staticmethod
    def _unique_consecutive(arr: list[NDArray[np.int64]]) -> list[NDArray[np.int64]]:
        """
        Remove consecutive duplicate elements from each array in the input list.

        This method is analogous to PyTorch's `unique_consecutive` function, but implemented for NumPy arrays.
        It filters out repeated adjacent elements, keeping only the first occurrence of consecutive duplicates.

        Args:
            arr (list[NDArray[np.int64]]): A list of NumPy integer arrays to process.

        Returns:
            list[NDArray[np.int64]]: A list of arrays with consecutive duplicates removed.

        Example:
            Input: [[1, 1, 2, 2, 3, 3, 3]]
            Output: [[1, 2, 3]]
        """

        result = []
        for row in arr:
            if len(row) == 0:
                result.append(row)
            else:
                mask = np.concatenate(([True], row[1:] != row[:-1]))
                result.append(row[mask])

        return result

    @staticmethod
    def _remove_padding(arr: list[NDArray[np.int64]], padding_value: int = 0) -> list[NDArray[np.int64]]:
        """
        Remove padding values from input arrays.

        Parameters:
            arr (list[NDArray[np.int64]]): A list of numpy arrays containing integer values
            padding_value (int, optional): The value to be considered as padding. Defaults to 0.

        Returns:
            list[NDArray[np.int64]]: A list of numpy arrays with padding values removed
        """
        return [row[row != padding_value] for row in arr]

    @staticmethod
    def _trim_to_stop(arr: list[NDArray[np.int64]], end_index: int = 2) -> list[NDArray[np.int64]]:
        """
        Trims each input array to the first occurrence of a specified stop index.

        This method searches for the first occurrence of the specified end index in each input array
        and truncates the array up to and including that index. If no such index is found, the original
        array is returned unchanged.

        Parameters:
            arr (list[NDArray[np.int64]]): List of numpy integer arrays to be trimmed.
            end_index (int, optional): The index value used as a stopping point. Defaults to 2.

        Returns:
            list[NDArray[np.int64]]: A list of trimmed numpy arrays, where each array is cut off
            at the first occurrence of the end_index (inclusive).
        """
        result = []
        for row in arr:
            stop_index = np.where(row == end_index)[0]
            if len(stop_index) > 0:
                result.append(row[: stop_index[0] + 1])
            else:
                result.append(row)
        return result

    def _process_model_output(self, arr: list[NDArray[np.int64]]) -> list[NDArray[np.int64]]:
        """
        Process the ONNX model's output to extract phoneme indices with post-processing.

        This method transforms raw model output into a clean sequence of phoneme indices by applying
        several filtering techniques:
        1. Converts model probabilities to index selections using argmax
        2. Removes consecutive duplicate indices
        3. Removes padding tokens
        4. Trims the sequence to the stop token

        Args:
            arr (list[NDArray[np.int64]]): Raw model output containing phoneme probability distributions.

        Returns:
            list[NDArray[np.int64]]: Processed phoneme indices with duplicates, padding, and excess tokens removed.
        """

        arr_processed: list[NDArray[np.int64]] = np.argmax(arr[0], axis=2)
        arr_processed = self._unique_consecutive(arr_processed)
        arr_processed = self._remove_padding(arr_processed)
        arr_processed = self._trim_to_stop(arr_processed)

        return arr_processed

    @staticmethod
    def _expand_acronym(word: str) -> str:
        """
        Expands an acronym into its subwords, with current implementation preserving the original acronym.

        This method handles acronyms by maintaining their original form. Currently, it performs two key checks:
        - If the word contains a hyphen, it returns the word unchanged
        - For other acronyms, it returns the word as-is

        Parameters:
            word (str): The input word potentially representing an acronym

        Returns:
            str: The processed word, which remains unchanged for acronyms

        Notes:
            - Designed to work with true acronyms
            - Mixed case acronym handling is delegated to SpokenTextConverter
        """
        # Only split on hyphens if they exist
        if Punctuation.HYPHEN.value in word:
            return word

        # For acronyms, just return as is - they're already preprocessed
        return word

    def encode(self, sentence: Iterable[str]) -> list[int]:
... (file continues)
