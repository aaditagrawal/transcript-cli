"""JSON output formatter with full details."""

import json

from ..engines.base import TranscriptionResult
from .base import OutputFormatter


class JSONFormatter(OutputFormatter):
    """JSON format with full transcription details including word-level timestamps."""

    name = "json"
    extension = ".json"

    def format(self, result: TranscriptionResult) -> str:
        """Format as JSON with all available data."""
        data = {
            "text": result.text,
            "language": result.language,
            "language_probability": result.language_probability,
            "duration": result.duration,
            "segments": [
                {
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                    "words": [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "confidence": w.confidence,
                        }
                        for w in seg.words
                    ]
                    if seg.words
                    else [],
                }
                for seg in result.segments
            ],
        }

        return json.dumps(data, indent=2, ensure_ascii=False)
