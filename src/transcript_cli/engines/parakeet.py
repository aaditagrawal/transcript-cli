"""NVIDIA Parakeet TDT v3 engine using NeMo."""

from pathlib import Path
from typing import Callable

from .base import (
    Segment,
    Task,
    TranscribeOptions,
    TranscriptionEngine,
    TranscriptionResult,
    WordTimestamp,
)
from . import register_engine


@register_engine
class ParakeetEngine(TranscriptionEngine):
    """NVIDIA Parakeet TDT v3 using NeMo toolkit."""

    name = "parakeet"
    description = "NVIDIA Parakeet TDT v3 - multilingual with precise timestamps"
    supported_models = [
        "nvidia/parakeet-tdt-0.6b-v3",
    ]
    supported_platforms = ["nvidia"]

    def __init__(self) -> None:
        self._model = None
        self._model_name = None

    def is_available(self) -> bool:
        """Check if NeMo is installed and CUDA is available."""
        try:
            import torch
            import nemo.collections.asr as nemo_asr  # noqa: F401

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_model(self, model: str):
        """Load or reuse the model."""
        import nemo.collections.asr as nemo_asr

        if self._model is not None and self._model_name == model:
            return self._model

        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=model)
        self._model_name = model
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        model: str,
        options: TranscribeOptions,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        """Transcribe using NeMo Parakeet."""
        asr_model = self._load_model(model)

        # Parakeet always provides timestamps
        output = asr_model.transcribe(
            [str(audio_path)],
            timestamps=True,
        )

        if not output:
            return TranscriptionResult(
                text="",
                segments=[],
                language="en",
            )

        result = output[0]

        # Build segments from timestamps
        segments = []
        full_text = result.text if hasattr(result, "text") else str(result)

        # Extract segment and word timestamps if available
        if hasattr(result, "timestamp"):
            ts = result.timestamp

            # Segment-level timestamps
            segment_ts = ts.get("segment", [])
            word_ts = ts.get("word", [])

            for seg in segment_ts:
                # Find words belonging to this segment
                seg_words = []
                if options.word_timestamps and word_ts:
                    for w in word_ts:
                        if w["start"] >= seg["start"] and w["end"] <= seg["end"]:
                            seg_words.append(
                                WordTimestamp(
                                    word=w.get("word", w.get("char", "")),
                                    start=w["start"],
                                    end=w["end"],
                                    confidence=w.get("confidence", 1.0),
                                )
                            )

                segments.append(
                    Segment(
                        text=seg.get("segment", "").strip(),
                        start=seg["start"],
                        end=seg["end"],
                        words=seg_words,
                    )
                )
        else:
            # Fallback: single segment with full text
            segments.append(
                Segment(
                    text=full_text,
                    start=0.0,
                    end=0.0,
                    words=[],
                )
            )

        # Detect language if available
        language = "en"
        if hasattr(result, "language"):
            language = result.language

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=language,
        )

    def download_model(
        self,
        model: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        """Download model from NGC/HuggingFace."""
        import nemo.collections.asr as nemo_asr

        if progress_callback:
            progress_callback(0.0, "Downloading Parakeet model...")

        # Loading triggers download
        nemo_asr.models.ASRModel.from_pretrained(model_name=model)

        if progress_callback:
            progress_callback(1.0, "Complete")

    def get_model_path(self, model: str) -> Path | None:
        """Get the cached model path if downloaded."""
        try:
            from huggingface_hub import try_to_load_from_cache

            # Parakeet uses nvidia namespace
            if "/" not in model:
                model = f"nvidia/{model}"

            cache_result = try_to_load_from_cache(model, "model_config.yaml")
            if isinstance(cache_result, str):
                return Path(cache_result).parent
            return None
        except Exception:
            return None

    def list_downloaded_models(self) -> list[str]:
        """List locally cached models."""
        downloaded = []
        for model in self.supported_models:
            if self.get_model_path(model) is not None:
                downloaded.append(model)
        return downloaded

    def is_model_downloaded(self, model: str) -> bool:
        """Check if a specific model is downloaded."""
        return self.get_model_path(model) is not None
