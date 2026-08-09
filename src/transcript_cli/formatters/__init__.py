"""Output formatters for transcription results."""

from .base import OutputFormatter
from .json_output import JSONFormatter
from .srt import SRTFormatter
from .text import TextFormatter
from .timestamps import TimestampFormatter
from .vtt import VTTFormatter

# Available output formats
FORMATTERS: dict[str, type[OutputFormatter]] = {
    "text": TextFormatter,
    "timestamps": TimestampFormatter,
    "srt": SRTFormatter,
    "vtt": VTTFormatter,
    "json": JSONFormatter,
}

OUTPUT_FORMATS = list(FORMATTERS.keys())


def get_formatter(format_name: str) -> OutputFormatter:
    """Get a formatter by name."""
    if format_name not in FORMATTERS:
        raise ValueError(f"Unknown format: {format_name}. Available: {OUTPUT_FORMATS}")
    return FORMATTERS[format_name]()


__all__ = [
    "OutputFormatter",
    "TextFormatter",
    "TimestampFormatter",
    "SRTFormatter",
    "VTTFormatter",
    "JSONFormatter",
    "FORMATTERS",
    "OUTPUT_FORMATS",
    "get_formatter",
]
