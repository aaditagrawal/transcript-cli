"""Tests for output formatters."""

import pytest

from transcript_cli.engines.base import Segment, TranscriptionResult, WordTimestamp
from transcript_cli.formatters import (
    JSONFormatter,
    SRTFormatter,
    TextFormatter,
    TimestampFormatter,
    VTTFormatter,
)


@pytest.fixture
def sample_result():
    """Create a sample transcription result for testing."""
    return TranscriptionResult(
        text="Hello world. This is a test.",
        segments=[
            Segment(
                text="Hello world.",
                start=0.0,
                end=2.5,
                words=[
                    WordTimestamp(word="Hello", start=0.0, end=0.5, confidence=0.99),
                    WordTimestamp(word="world.", start=0.6, end=1.2, confidence=0.98),
                ],
            ),
            Segment(
                text="This is a test.",
                start=3.0,
                end=5.5,
                words=[],
            ),
        ],
        language="en",
        language_probability=0.99,
        duration=5.5,
    )


class TestTextFormatter:
    def test_format_plain_text(self, sample_result):
        formatter = TextFormatter()
        output = formatter.format(sample_result)
        assert output == "Hello world. This is a test."

    def test_extension(self):
        formatter = TextFormatter()
        assert formatter.extension == ".txt"


class TestTimestampFormatter:
    def test_format_with_timestamps(self, sample_result):
        formatter = TimestampFormatter()
        output = formatter.format(sample_result)
        lines = output.split("\n")
        assert lines[0] == "[00:00:00] Hello world."
        assert lines[1] == "[00:00:03] This is a test."


class TestSRTFormatter:
    def test_format_srt(self, sample_result):
        formatter = SRTFormatter()
        output = formatter.format(sample_result)
        lines = output.split("\n")

        # First subtitle
        assert lines[0] == "1"
        assert lines[1] == "00:00:00,000 --> 00:00:02,500"
        assert lines[2] == "Hello world."
        assert lines[3] == ""

        # Second subtitle
        assert lines[4] == "2"
        assert lines[5] == "00:00:03,000 --> 00:00:05,500"

    def test_extension(self):
        formatter = SRTFormatter()
        assert formatter.extension == ".srt"


class TestVTTFormatter:
    def test_format_vtt(self, sample_result):
        formatter = VTTFormatter()
        output = formatter.format(sample_result)
        lines = output.split("\n")

        # Header
        assert lines[0] == "WEBVTT"
        assert lines[1] == ""

        # First cue
        assert lines[2] == "00:00:00.000 --> 00:00:02.500"
        assert lines[3] == "Hello world."

    def test_extension(self):
        formatter = VTTFormatter()
        assert formatter.extension == ".vtt"


class TestJSONFormatter:
    def test_format_json(self, sample_result):
        import json

        formatter = JSONFormatter()
        output = formatter.format(sample_result)
        data = json.loads(output)

        assert data["text"] == "Hello world. This is a test."
        assert data["language"] == "en"
        assert data["duration"] == 5.5
        assert len(data["segments"]) == 2
        assert data["segments"][0]["text"] == "Hello world."
        assert len(data["segments"][0]["words"]) == 2

    def test_extension(self):
        formatter = JSONFormatter()
        assert formatter.extension == ".json"
