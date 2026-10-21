"""Lazy, typed contracts for optional inference libraries.

The CLI and its type checker must work without GPU or Apple packages. Keep those
libraries behind this boundary, check the required entry points when loading,
and describe the parts of their APIs that the engine adapters actually use.
"""

from collections.abc import Iterable
from contextlib import AbstractContextManager
from importlib import import_module
from types import SimpleNamespace
from typing import Any, Protocol, TypedDict, runtime_checkable


class Cuda(Protocol):
    def is_available(self) -> bool: ...


@runtime_checkable
class Torch(Protocol):
    cuda: Cuda
    float16: object
    float32: object

    def inference_mode(self) -> AbstractContextManager[None]: ...


class FasterWord(Protocol):
    word: str
    start: float
    end: float
    probability: float


class FasterSegment(Protocol):
    text: str
    start: float
    end: float
    words: list[FasterWord] | None


class FasterInfo(Protocol):
    language: str
    language_probability: float
    duration: float


class FasterModel(Protocol):
    """Opaque model owned by faster-whisper."""


class BatchedPipeline(Protocol):
    def transcribe(
        self,
        audio: str,
        *,
        language: str | None,
        task: str,
        word_timestamps: bool,
        vad_filter: bool,
        batch_size: int,
        beam_size: int,
    ) -> tuple[Iterable[FasterSegment], FasterInfo]: ...


@runtime_checkable
class FasterWhisper(Protocol):
    def WhisperModel(self, model: str, *, device: str, compute_type: str) -> FasterModel: ...
    def BatchedInferencePipeline(self, *, model: FasterModel) -> BatchedPipeline: ...


class SpeechPipeline(Protocol):
    def __call__(
        self,
        audio: str,
        *,
        chunk_length_s: int,
        batch_size: int,
        return_timestamps: str | bool,
        generate_kwargs: dict[str, str],
    ) -> dict[str, Any]: ...


class Pretrained(Protocol):
    def from_pretrained(self, model: str) -> object: ...


@runtime_checkable
class Transformers(Protocol):
    AutoModelForSpeechSeq2Seq: Pretrained
    AutoProcessor: Pretrained

    def pipeline(
        self,
        task: str,
        *,
        model: str,
        dtype: object,
        device: str,
        model_kwargs: dict[str, str],
    ) -> SpeechPipeline: ...


@runtime_checkable
class TransformersCache(Protocol):
    def cached_file(
        self,
        model: str,
        filename: str,
        *,
        _raise_exceptions_for_missing_entries: bool,
    ) -> str | None: ...


@runtime_checkable
class Hub(Protocol):
    def snapshot_download(
        self, *, repo_id: str, allow_patterns: list[str] | None = None
    ) -> str: ...
    def try_to_load_from_cache(self, repo_id: str, filename: str) -> object: ...


@runtime_checkable
class MLXWhisper(Protocol):
    def transcribe(self, audio: str, **kwargs: Any) -> dict[str, Any]: ...


class TokenTimestamp(TypedDict):
    token: str
    start: float
    end: float


class TDTModel(Protocol):
    device: Any
    dtype: Any

    def to(self, device: str) -> "TDTModel": ...
    def eval(self) -> "TDTModel": ...
    def generate(self, **kwargs: Any) -> Any: ...


class TDTFactory(Protocol):
    def from_pretrained(self, model: str, *, use_safetensors: bool) -> TDTModel: ...


class TDTProcessor(Protocol):
    def __call__(self, audio: Any, *, sampling_rate: int, return_tensors: str) -> Any: ...
    def decode(
        self,
        sequences: Any,
        *,
        durations: Any,
        skip_special_tokens: bool,
    ) -> tuple[list[str], list[list[TokenTimestamp]]]: ...


class ProcessorFactory(Protocol):
    def from_pretrained(self, model: str) -> TDTProcessor: ...


@runtime_checkable
class ParakeetTransformers(Protocol):
    AutoModelForTDT: TDTFactory
    AutoProcessor: ProcessorFactory


@runtime_checkable
class Librosa(Protocol):
    def load(self, path: str, *, sr: int, mono: bool) -> tuple[Any, int]: ...


def _module_api(name: str, *attributes: str) -> SimpleNamespace:
    # Libraries such as Transformers and librosa expose lazy __getattr__ APIs.
    # runtime_checkable uses static attribute lookup on Python 3.12+, so first
    # resolve the entry points into a small explicit view without mutating them.
    module = import_module(name)
    try:
        return SimpleNamespace(
            **{attribute: getattr(module, attribute) for attribute in attributes}
        )
    except AttributeError as error:
        raise ImportError(f"{name} is missing a required entry point: {error}") from error


def torch() -> Torch:
    module = _module_api("torch", "cuda", "float16", "float32", "inference_mode")
    if not isinstance(module, Torch):
        raise ImportError("torch is missing CUDA or dtype entry points")
    return module


def faster_whisper() -> FasterWhisper:
    module = _module_api("faster_whisper", "WhisperModel", "BatchedInferencePipeline")
    if not isinstance(module, FasterWhisper):
        raise ImportError("faster-whisper is missing model or batched inference entry points")
    return module


def transformers() -> Transformers:
    module = _module_api("transformers", "AutoModelForSpeechSeq2Seq", "AutoProcessor", "pipeline")
    if not isinstance(module, Transformers):
        raise ImportError("transformers is missing speech pipeline entry points")
    return module


def transformers_cache() -> TransformersCache:
    module = _module_api("transformers.utils", "cached_file")
    if not isinstance(module, TransformersCache):
        raise ImportError("transformers is missing cached_file")
    return module


def hub() -> Hub:
    module = _module_api("huggingface_hub", "snapshot_download", "try_to_load_from_cache")
    if not isinstance(module, Hub):
        raise ImportError("huggingface-hub is missing model cache entry points")
    return module


def mlx_whisper() -> MLXWhisper:
    module = _module_api("mlx_whisper", "transcribe")
    if not isinstance(module, MLXWhisper):
        raise ImportError("mlx-whisper is missing transcribe")
    return module


def parakeet_transformers() -> ParakeetTransformers:
    module = _module_api("transformers", "AutoModelForTDT", "AutoProcessor")
    if not isinstance(module, ParakeetTransformers):
        raise ImportError("transformers is missing the Parakeet TDT entry points")
    return module


def librosa() -> Librosa:
    module = _module_api("librosa", "load")
    if not isinstance(module, Librosa):
        raise ImportError("librosa is missing its audio loader")
    return module


def flash_attention_available() -> bool:
    try:
        import_module("flash_attn")
    except ImportError:
        return False
    return True
