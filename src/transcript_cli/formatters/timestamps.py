"""Timestamped text formatter."""

from ..engines.base import TranscriptionResult
from .base import OutputFormatter


class TimestampFormatter(OutputFormatter):
    """Text with timestamps at the start of each segment."""

    name = "timestamps"
    extension = ".txt"

    def format(self, result: TranscriptionResult) -> str:
        """Format as text with [HH:MM:SS] timestamps."""
        lines = []

        for segment in result.segments:
            timestamp = self._format_timestamp(segment.start)
            lines.append(f"[{timestamp}] {segment.text}")

        return "\n".join(lines)

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
