"""Base class for output formatters."""

from abc import ABC, abstractmethod
from pathlib import Path

from ..engines.base import TranscriptionResult


class OutputFormatter(ABC):
    """Abstract base class for output formatters."""

    name: str = "base"
    extension: str = ".txt"

    @abstractmethod
    def format(self, result: TranscriptionResult) -> str:
        """
        Format transcription result as a string.

        Args:
            result: TranscriptionResult to format

        Returns:
            Formatted string
        """
        ...

    def save(self, result: TranscriptionResult, output_path: Path) -> None:
        """
        Save formatted result to a file.

        Args:
            result: TranscriptionResult to format
            output_path: Path to save to
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = self.format(result)
        output_path.write_text(content, encoding="utf-8")

    def get_output_path(self, input_path: Path, output_dir: Path | None = None) -> Path:
        """
        Generate output path for a given input file.

        Args:
            input_path: Original input file path
            output_dir: Optional output directory

        Returns:
            Output path with appropriate extension
        """
        output_name = input_path.stem + self.extension
        if output_dir:
            return output_dir / output_name
        return input_path.parent / output_name
