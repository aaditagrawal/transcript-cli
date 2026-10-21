"""NVIDIA Parakeet TDT v3 through Transformers, without NeMo's training stack."""

import math
import re
from collections.abc import Callable
from pathlib import Path

from . import dependencies, register_engine
from .base import (
    Segment,
    TranscribeOptions,
    TranscriptionEngine,
    TranscriptionResult,
    WordTimestamp,
)


def _timestamped_words(text: str, tokens: list[dependencies.TokenTimestamp]) -> list[WordTimestamp]:
    """Align decoded token spans to words without inventing token durations."""
    spans: list[tuple[int, int, float, float]] = []
    cursor = 0
    for token in tokens:
        piece = token["token"]
        start, end = token["start"], token["end"]
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError("Parakeet returned invalid token timestamps")
        if not piece:
            continue
        position = text.find(piece, cursor)
        if position < 0 or text[cursor:position].strip():
            raise ValueError("Parakeet token timestamps do not match the decoded text")
        spans.append((position, position + len(piece), start, end))
        cursor = position + len(piece)
    if text[cursor:].strip():
        raise ValueError("Parakeet returned incomplete token timestamps")

    words = []
    span_index = 0
    for word in re.finditer(r"\S+", text):
        while span_index < len(spans) and spans[span_index][1] <= word.start():
            span_index += 1
        matching = []
        index = span_index
        while index < len(spans) and spans[index][0] < word.end():
            matching.append(spans[index])
            index += 1
        if not matching:
            raise ValueError("Parakeet returned a word without timestamps")
        words.append(
            WordTimestamp(word.group(), min(s[2] for s in matching), max(s[3] for s in matching))
        )
    return words


def _segments(words: list[WordTimestamp], include_words: bool) -> list[Segment]:
    segments = []
    current: list[WordTimestamp] = []
    for word in words:
        if current and (word.start - current[-1].end >= 1 or word.end - current[0].start > 30):
            segments.append(
                Segment(
                    " ".join(w.word for w in current),
                    current[0].start,
                    current[-1].end,
                    list(current) if include_words else [],
                )
            )
            current = []
        current.append(word)
        if word.word.endswith((".", "?", "!", "。", "？", "！")):
            segments.append(
                Segment(
                    " ".join(w.word for w in current),
                    current[0].start,
                    current[-1].end,
                    list(current) if include_words else [],
                )
            )
            current = []
    if current:
        segments.append(
            Segment(
                " ".join(w.word for w in current),
                current[0].start,
                current[-1].end,
                list(current) if include_words else [],
            )
        )
    return segments


@register_engine
class ParakeetEngine(TranscriptionEngine):
    """The NVIDIA Parakeet v3 model with duration-based token and word timestamps."""

    name = "parakeet"
    description = "NVIDIA Parakeet TDT v3 - multilingual with precise timestamps"
    supported_models = ["nvidia/parakeet-tdt-0.6b-v3"]
    supported_platforms = ["nvidia"]

    def __init__(self) -> None:
        self._model: dependencies.TDTModel | None = None
        self._processor: dependencies.TDTProcessor | None = None
        self._model_key: tuple[str, str] | None = None

    def is_available(self) -> bool:
        try:
            dependencies.parakeet_transformers()
            dependencies.librosa()
            return dependencies.torch().cuda.is_available()
        except ImportError:
            return False

    def _load_model(
        self, model: str, device: str
    ) -> tuple[dependencies.TDTModel, dependencies.TDTProcessor]:
        key = (model, device)
        if self._model is None or self._processor is None or self._model_key != key:
            transformers = dependencies.parakeet_transformers()
            self._processor = transformers.AutoProcessor.from_pretrained(model)
            self._model = (
                transformers.AutoModelForTDT.from_pretrained(model, use_safetensors=True)
                .to(device)
                .eval()
            )
            self._model_key = key
        return self._model, self._processor

    def transcribe(
        self,
        audio_path: Path,
        model: str,
        options: TranscribeOptions,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        device = "cuda" if options.device == "auto" else options.device
        asr_model, processor = self._load_model(model, device)
        audio, sample_rate = dependencies.librosa().load(str(audio_path), sr=16000, mono=True)
        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        inputs = inputs.to(asr_model.device, dtype=asr_model.dtype)
        with dependencies.torch().inference_mode():
            output = asr_model.generate(**inputs, return_dict_in_generate=True)
        texts, timestamp_batches = processor.decode(
            output.sequences, durations=output.durations, skip_special_tokens=True
        )
        if len(texts) != 1 or len(timestamp_batches) != 1:
            raise ValueError("Parakeet returned an unexpected number of transcription results")
        text = texts[0]
        words = _timestamped_words(text, timestamp_batches[0])
        return TranscriptionResult(
            text=text.strip(),
            segments=_segments(words, options.word_timestamps),
            language=options.language or "en",
            duration=len(audio) / sample_rate,
        )

    def download_model(
        self,
        model: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        if progress_callback:
            progress_callback(0.0, "Downloading Parakeet model...")
        dependencies.hub().snapshot_download(
            repo_id=model, allow_patterns=["*.json", "*.safetensors"]
        )
        if progress_callback:
            progress_callback(1.0, "Complete")

    def get_model_path(self, model: str) -> Path | None:
        try:
            if "/" not in model:
                model = f"nvidia/{model}"
            cached = dependencies.hub().try_to_load_from_cache(model, "config.json")
            return Path(cached).parent if isinstance(cached, str) else None
        except Exception:
            return None

    def list_downloaded_models(self) -> list[str]:
        return [model for model in self.supported_models if self.get_model_path(model) is not None]

    def is_model_downloaded(self, model: str) -> bool:
        return self.get_model_path(model) is not None
