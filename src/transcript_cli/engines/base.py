"""Base classes for transcription engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable


class Task(Enum):
    """Transcription task type."""

    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


@dataclass
class WordTimestamp:
    """Word-level timestamp."""

    word: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class Segment:
    """Transcription segment with timestamps."""

    text: str
    start: float
    end: float
    words: list[WordTimestamp] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Result of transcription."""

    text: str
    segments: list[Segment]
    language: str
    language_probability: float = 1.0
    duration: float = 0.0

    @property
    def has_word_timestamps(self) -> bool:
        """Check if word-level timestamps are available."""
        return any(seg.words for seg in self.segments)


@dataclass
class TranscribeOptions:
    """Options for transcription."""

    language: str | None = None
    task: Task = Task.TRANSCRIBE
    word_timestamps: bool = False
    vad_filter: bool = True
    batch_size: int = 16
    beam_size: int = 5

    # Engine-specific options
    compute_type: str = "float16"  # float16, int8, int8_float16
    device: str = "auto"  # auto, cuda, cpu, mps


class TranscriptionEngine(ABC):
    """Abstract base class for transcription engines."""

    name: str = "base"
    description: str = "Base transcription engine"
    supported_models: list[str] = []
    supported_platforms: list[str] = []  # nvidia, apple_silicon, cpu

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the engine is installed and available."""
        ...

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        model: str,
        options: TranscribeOptions,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file
            model: Model name/size to use
            options: Transcription options
            progress_callback: Optional callback for progress updates (0.0 to 1.0)

        Returns:
            TranscriptionResult with text and segments
        """
        ...

    @abstractmethod
    def download_model(
        self,
        model: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        """
        Download a model.

        Args:
            model: Model name to download
            progress_callback: Optional callback with (progress, status) updates
        """
        ...

    def get_model_path(self, model: str) -> Path | None:
        """Get the cached model path if downloaded."""
        return None

    def list_downloaded_models(self) -> list[str]:
        """List locally cached models."""
        return []

    def is_model_downloaded(self, model: str) -> bool:
        """Check if a specific model is downloaded."""
        return False
