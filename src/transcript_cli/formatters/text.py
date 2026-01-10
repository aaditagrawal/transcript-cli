"""Plain text formatter."""

from ..engines.base import TranscriptionResult
from .base import OutputFormatter


class TextFormatter(OutputFormatter):
    """Plain text formatter - just the transcript text."""

    name = "text"
    extension = ".txt"

    def format(self, result: TranscriptionResult) -> str:
        """Format as plain text."""
        return result.text
