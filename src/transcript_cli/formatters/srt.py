"""SRT subtitle formatter."""

from ..engines.base import TranscriptionResult
from .base import OutputFormatter


class SRTFormatter(OutputFormatter):
    """SubRip (SRT) subtitle format."""

    name = "srt"
    extension = ".srt"

    def format(self, result: TranscriptionResult) -> str:
        """Format as SRT subtitles."""
        lines = []

        for i, segment in enumerate(result.segments, 1):
            start = self._format_timestamp(segment.start)
            end = self._format_timestamp(segment.end)

            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(segment.text)
            lines.append("")  # Blank line between entries

        return "\n".join(lines)

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS,mmm (SRT format uses comma)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
