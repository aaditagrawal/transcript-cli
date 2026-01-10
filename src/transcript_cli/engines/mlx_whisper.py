"""MLX Whisper engine for Apple Silicon."""

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
class MLXWhisperEngine(TranscriptionEngine):
    """MLX Whisper engine optimized for Apple Silicon."""

    name = "mlx-whisper"
    description = "Apple Silicon native using MLX framework"
    supported_models = [
        "mlx-community/whisper-tiny-mlx",
        "mlx-community/whisper-base-mlx",
        "mlx-community/whisper-small-mlx",
        "mlx-community/whisper-medium-mlx",
        "mlx-community/whisper-large-v3-mlx",
        "mlx-community/whisper-large-v3-turbo",
        "mlx-community/distil-whisper-large-v3",
    ]
    supported_platforms = ["apple_silicon"]

    def is_available(self) -> bool:
        """Check if mlx-whisper is installed and running on Apple Silicon."""
        try:
            import platform

            import mlx_whisper  # noqa: F401

            return platform.machine() == "arm64" and platform.system() == "Darwin"
        except ImportError:
            return False

    def _normalize_model_name(self, model: str) -> str:
        """Convert short model names to mlx-community format."""
        if "/" in model:
            return model

        # Map short names to MLX community model IDs
        mapping = {
            "tiny": "mlx-community/whisper-tiny-mlx",
            "tiny.en": "mlx-community/whisper-tiny.en-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "large": "mlx-community/whisper-large-v3-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "turbo": "mlx-community/whisper-large-v3-turbo",
            "distil-large-v3": "mlx-community/distil-whisper-large-v3",
        }
        return mapping.get(model, f"mlx-community/whisper-{model}-mlx")

    def transcribe(
        self,
        audio_path: Path,
        model: str,
        options: TranscribeOptions,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        """Transcribe using mlx-whisper."""
        import mlx_whisper

        model_name = self._normalize_model_name(model)

        # Prepare transcription options
        transcribe_kwargs = {
            "path_or_hf_repo": model_name,
            "word_timestamps": options.word_timestamps,
        }

        if options.language:
            transcribe_kwargs["language"] = options.language

        if options.task == Task.TRANSLATE:
            transcribe_kwargs["task"] = "translate"

        # Run transcription
        result = mlx_whisper.transcribe(
            str(audio_path),
            **transcribe_kwargs,
        )

        # Parse segments
        segments = []
        raw_segments = result.get("segments", [])

        for seg in raw_segments:
            words = []
            if options.word_timestamps and "words" in seg:
                words = [
                    WordTimestamp(
                        word=w.get("word", ""),
                        start=w.get("start", 0.0),
                        end=w.get("end", 0.0),
                        confidence=w.get("probability", 1.0),
                    )
                    for w in seg.get("words", [])
                ]

            segments.append(
                Segment(
                    text=seg.get("text", "").strip(),
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    words=words,
                )
            )

        return TranscriptionResult(
            text=result.get("text", "").strip(),
            segments=segments,
            language=result.get("language", "en"),
        )

    def download_model(
        self,
        model: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        """Download model from HuggingFace."""
        from huggingface_hub import snapshot_download

        model_name = self._normalize_model_name(model)

        if progress_callback:
            progress_callback(0.0, f"Downloading {model_name}...")

        snapshot_download(repo_id=model_name)

        if progress_callback:
            progress_callback(1.0, "Complete")

    def get_model_path(self, model: str) -> Path | None:
        """Get the cached model path if downloaded."""
        try:
            from huggingface_hub import try_to_load_from_cache

            model_name = self._normalize_model_name(model)
            cache_result = try_to_load_from_cache(model_name, "config.json")
            if isinstance(cache_result, str):
                return Path(cache_result).parent
            return None
        except Exception:
            return None

    def list_downloaded_models(self) -> list[str]:
        """List locally cached models."""
        downloaded = []
        model_names = [
            "tiny", "base", "small", "medium",
            "large", "large-v2", "large-v3", "turbo"
        ]
        for model in model_names:
            if self.get_model_path(model) is not None:
                downloaded.append(model)
        return downloaded

    def is_model_downloaded(self, model: str) -> bool:
        """Check if a specific model is downloaded."""
        return self.get_model_path(model) is not None
