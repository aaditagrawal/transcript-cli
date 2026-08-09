"""Insanely Fast Whisper engine for maximum throughput on NVIDIA GPUs."""

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
class InsanelyFastWhisperEngine(TranscriptionEngine):
    """Insanely Fast Whisper using Hugging Face Transformers with Flash Attention."""

    name = "insanely-fast-whisper"
    description = "Maximum throughput on NVIDIA GPUs with Flash Attention 2"
    supported_models = [
        "openai/whisper-tiny",
        "openai/whisper-tiny.en",
        "openai/whisper-base",
        "openai/whisper-base.en",
        "openai/whisper-small",
        "openai/whisper-small.en",
        "openai/whisper-medium",
        "openai/whisper-medium.en",
        "openai/whisper-large-v2",
        "openai/whisper-large-v3",
        "openai/whisper-large-v3-turbo",
        "distil-whisper/distil-large-v2",
        "distil-whisper/distil-large-v3",
    ]
    supported_platforms = ["nvidia"]

    def is_available(self) -> bool:
        """Check if insanely-fast-whisper dependencies are available."""
        try:
            import torch
            from transformers import pipeline  # noqa: F401

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _normalize_model_name(self, model: str) -> str:
        """Convert short model names to HuggingFace format."""
        if "/" in model:
            return model

        # Map short names to HuggingFace model IDs
        mapping = {
            "tiny": "openai/whisper-tiny",
            "tiny.en": "openai/whisper-tiny.en",
            "base": "openai/whisper-base",
            "base.en": "openai/whisper-base.en",
            "small": "openai/whisper-small",
            "small.en": "openai/whisper-small.en",
            "medium": "openai/whisper-medium",
            "medium.en": "openai/whisper-medium.en",
            "large": "openai/whisper-large-v3",
            "large-v2": "openai/whisper-large-v2",
            "large-v3": "openai/whisper-large-v3",
            "turbo": "openai/whisper-large-v3-turbo",
            "distil-large-v2": "distil-whisper/distil-large-v2",
            "distil-large-v3": "distil-whisper/distil-large-v3",
        }
        return mapping.get(model, f"openai/whisper-{model}")

    def transcribe(
        self,
        audio_path: Path,
        model: str,
        options: TranscribeOptions,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        """Transcribe using insanely-fast-whisper pipeline."""
        import torch
        from transformers import pipeline

        model_name = self._normalize_model_name(model)

        # Check for Flash Attention 2 support
        use_flash = False
        try:
            import flash_attn  # noqa: F401

            use_flash = True
        except ImportError:
            pass

        # Create pipeline
        import sys

        try:
            # Quick check - if this fails, model needs download
            from transformers.utils import cached_file

            cached_file(model_name, "config.json", _raise_exceptions_for_missing_entries=False)
        except Exception:
            print(f"\n📥 Downloading model {model_name}... (first time only)", file=sys.stderr)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            dtype=torch.float16,
            device="cuda:0",
            model_kwargs={"attn_implementation": "flash_attention_2"} if use_flash else {},
        )

        # Transcribe
        generate_kwargs = {
            "task": options.task.value,
        }
        if options.language:
            generate_kwargs["language"] = options.language

        result = pipe(
            str(audio_path),
            chunk_length_s=30,
            batch_size=options.batch_size,
            return_timestamps="word" if options.word_timestamps else True,
            generate_kwargs=generate_kwargs,
        )

        # Parse results
        segments = []
        chunks = result.get("chunks", [])

        for chunk in chunks:
            text = chunk.get("text", "").strip()
            timestamp = chunk.get("timestamp", (0.0, 0.0))
            start = timestamp[0] if timestamp[0] is not None else 0.0
            end = timestamp[1] if timestamp[1] is not None else start

            # Word timestamps come as individual chunks when requested
            words = []
            if options.word_timestamps:
                words = [
                    WordTimestamp(
                        word=text,
                        start=start,
                        end=end,
                        confidence=1.0,
                    )
                ]

            segments.append(
                Segment(
                    text=text,
                    start=start,
                    end=end,
                    words=words,
                )
            )

        # Combine all text
        full_text = result.get("text", "").strip()
        if not full_text:
            full_text = " ".join(seg.text for seg in segments)

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=options.language or "en",
            language_probability=1.0,
        )

    def download_model(
        self,
        model: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        """Download model from HuggingFace."""
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        model_name = self._normalize_model_name(model)

        if progress_callback:
            progress_callback(0.0, "Downloading model...")

        AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        AutoProcessor.from_pretrained(model_name)

        if progress_callback:
            progress_callback(1.0, "Complete")

    def get_model_path(self, model: str) -> Path | None:
        """Get the cached model path if downloaded."""
        try:
            from huggingface_hub import try_to_load_from_cache

            model_name = self._normalize_model_name(model)
            # Check if config.json is cached (indicates model is downloaded)
            cache_result = try_to_load_from_cache(model_name, "config.json")
            if cache_result is not None and isinstance(cache_result, str) is not False:
                if isinstance(cache_result, str):
                    return Path(cache_result).parent
            return None
        except Exception:
            return None

    def list_downloaded_models(self) -> list[str]:
        """List locally cached models."""
        downloaded = []
        # Check the common model names
        model_names = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"]
        for model in model_names:
            if self.get_model_path(model) is not None:
                downloaded.append(model)
        return downloaded

    def is_model_downloaded(self, model: str) -> bool:
        """Check if a specific model is downloaded."""
        return self.get_model_path(model) is not None
