"""Exercise command dispatch without downloading or transcribing audio."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from transcript_cli import main as cli

runner = CliRunner()


@pytest.mark.parametrize(
    ("command", "heading"),
    [
        ("list-engines", "Transcription Engines"),
        ("list-formats", "Output Formats"),
        ("list-models", "Whisper Models"),
    ],
)
def test_listing_commands_dispatch(command: str, heading: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli, "get_best_engine", lambda: None)
    result = runner.invoke(cli.app, [command])

    assert result.exit_code == 0, result.output
    assert heading in result.output
    assert "File not found" not in result.output


@pytest.mark.parametrize("command", ["transcribe", "download"])
def test_subcommand_help(command: str):
    result = runner.invoke(cli.app, [command, "--help"])

    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output
    assert "File not found" not in result.output


@pytest.mark.parametrize("options_first", [False, True])
def test_file_shorthand_preserves_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options_first: bool
):
    audio = tmp_path / "meeting notes.wav"
    audio.touch()
    calls = []
    monkeypatch.setattr(cli, "get_best_engine", lambda: None)
    monkeypatch.setattr(cli, "transcribe", lambda **kwargs: calls.append(kwargs))
    options = ["--format", "srt", "--model", "base", "--recursive"]
    args = [*options, str(audio)] if options_first else [str(audio), *options]

    result = runner.invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    assert calls == [{"path": audio, "output_format": "srt", "model": "base", "recursive": True}]


def test_explicit_path_supports_a_filename_that_matches_a_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    audio = Path("list-engines")
    audio.touch()
    calls = []
    monkeypatch.setattr(cli, "get_best_engine", lambda: None)
    monkeypatch.setattr(cli, "transcribe", lambda **kwargs: calls.append(kwargs))

    result = runner.invoke(cli.app, ["--path", str(audio), "-f", "text", "-m", "base"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["path"] == audio


def test_leading_dash_path_after_separator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    audio = Path("-recording.wav")
    audio.touch()
    calls = []
    monkeypatch.setattr(cli, "get_best_engine", lambda: None)
    monkeypatch.setattr(cli, "transcribe", lambda **kwargs: calls.append(kwargs))

    result = runner.invoke(cli.app, ["-f", "text", "-m", "base", "--", str(audio)])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["path"] == audio


def test_unknown_shorthand_reports_missing_file():
    result = runner.invoke(cli.app, ["missing-recording.wav", "-f", "text", "-m", "base"])

    assert result.exit_code == 1
    assert "File not found" in result.output


def test_version_does_not_start_transcription():
    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0, result.output
    assert "transcript-cli v" in result.output
