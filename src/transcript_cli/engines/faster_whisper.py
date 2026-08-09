"""Faster Whisper engine using CTranslate2."""

from collections.abc import Callable
from pathlib import Path

from . import register_engine
from .base import (
    Segment,
    TranscribeOptions,
    TranscriptionEngine,
    TranscriptionResult,
    WordTimestamp,
)


@register_engine
class FasterWhisperEngine(TranscriptionEngine):
    """Faster Whisper engine using CTranslate2 for optimized inference."""

    name = "faster-whisper"
    description = "Fast transcription with CTranslate2 (GPU/CPU)"
    supported_models = [
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large-v1",
        "large-v2",
        "large-v3",
        "large-v3-turbo",
        "turbo",
        "distil-large-v2",
        "distil-large-v3",
    ]
    supported_platforms = ["nvidia", "cpu"]

    def __init__(self) -> None:
        self._model = None
        self._model_name = None

    def is_available(self) -> bool:
        """Check if faster-whisper is installed."""
        try:
            import faster_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_device_and_compute(self, options: TranscribeOptions) -> tuple[str, str]:
        """Determine device and compute type from options."""
        from ..config import Platform, detect_platform

        platform = detect_platform()

        if options.device != "auto":
            device = options.device
        elif platform == Platform.NVIDIA:
            device = "cuda"
        else:
            device = "cpu"

        # Adjust compute type for CPU
        if device == "cpu":
            compute_type = "int8"
        else:
            compute_type = options.compute_type

        return device, compute_type

    def _load_model(self, model: str, options: TranscribeOptions):
        """Load or reuse the model."""
        from faster_whisper import WhisperModel

        device, compute_type = self._get_device_and_compute(options)

        # Reuse model if already loaded with same config
        if self._model is not None and self._model_name == model:
            return self._model

        self._model = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
        )
        self._model_name = model
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        model: str,
        options: TranscribeOptions,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio using faster-whisper."""
        from faster_whisper import BatchedInferencePipeline

        whisper_model = self._load_model(model, options)

        # Use batched inference for better performance
        batched = BatchedInferencePipeline(model=whisper_model)

        segments_gen, info = batched.transcribe(
            str(audio_path),
            language=options.language,
            task=options.task.value,
            word_timestamps=options.word_timestamps,
            vad_filter=options.vad_filter,
            batch_size=options.batch_size,
            beam_size=options.beam_size,
        )

        # Convert generator to list and build result
        segments = []
        all_text = []

        for seg in segments_gen:
            words = []
            if options.word_timestamps and seg.words:
                words = [
                    WordTimestamp(
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        confidence=w.probability,
                    )
                    for w in seg.words
                ]

            segments.append(
                Segment(
                    text=seg.text.strip(),
                    start=seg.start,
                    end=seg.end,
                    words=words,
                )
            )
            all_text.append(seg.text.strip())

        return TranscriptionResult(
            text=" ".join(all_text),
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
        )

    def download_model(
        self,
        model: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        """Download model (handled automatically by faster-whisper)."""
        from faster_whisper import WhisperModel

        if progress_callback:
            progress_callback(0.0, "Downloading model...")

        # Loading the model triggers download
        WhisperModel(model, device="cpu", compute_type="int8")

        if progress_callback:
            progress_callback(1.0, "Complete")

    def get_model_path(self, model: str) -> Path | None:
        """Get the cached model path if downloaded."""
        try:
            from huggingface_hub import try_to_load_from_cache

            # faster-whisper uses Systran models on HuggingFace
            if model in ["turbo", "large-v3-turbo"]:
                model_id = "Systran/faster-whisper-large-v3-turbo"
            elif model.startswith("distil-"):
                model_id = f"Systran/faster-{model}"
            else:
                model_id = f"Systran/faster-whisper-{model}"

            cache_result = try_to_load_from_cache(model_id, "config.json")
            if isinstance(cache_result, str):
                return Path(cache_result).parent
            return None
        except Exception:
            return None

    def list_downloaded_models(self) -> list[str]:
        """List locally cached models."""
        downloaded = []
        model_names = [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large-v1", "large-v2", "large-v3", "turbo"
        ]
        for model in model_names:
            if self.get_model_path(model) is not None:
                downloaded.append(model)
        return downloaded

    def is_model_downloaded(self, model: str) -> bool:
        """Check if a specific model is downloaded."""
        return self.get_model_path(model) is not None
