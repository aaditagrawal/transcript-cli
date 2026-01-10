"""WebVTT subtitle formatter."""

from ..engines.base import TranscriptionResult
from .base import OutputFormatter


class VTTFormatter(OutputFormatter):
    """WebVTT subtitle format."""

    name = "vtt"
    extension = ".vtt"

    def format(self, result: TranscriptionResult) -> str:
        """Format as WebVTT subtitles."""
        lines = ["WEBVTT", ""]  # VTT header

        for segment in result.segments:
            start = self._format_timestamp(segment.start)
            end = self._format_timestamp(segment.end)

            lines.append(f"{start} --> {end}")
            lines.append(segment.text)
            lines.append("")  # Blank line between entries

        return "\n".join(lines)

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS.mmm (VTT format uses period)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
