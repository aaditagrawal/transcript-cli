from pathlib import Path

import pytest
from typer.testing import CliRunner

from transcript_cli import audio, main
from transcript_cli.engines.base import Segment, TranscriptionResult, WordTimestamp


@pytest.fixture
def engine(monkeypatch):
    class Engine:
        supported_models = ["nvidia/parakeet-tdt-0.6b-v3"]

        def transcribe(self, *args, **kwargs):
            return TranscriptionResult(
                "hello", [Segment("hello", 1, 2, [WordTimestamp("hello", 1, 2)])], "en"
            )

    fake = Engine()
    monkeypatch.setattr(main, "get_available_engines", lambda: {"fake": fake})
    monkeypatch.setattr(main, "get_engine", lambda name: fake)
    monkeypatch.setattr(main, "get_audio_duration", lambda path: 10.0)
    return fake


@pytest.mark.parametrize("command", ["list-engines", "list-formats", "list-models"])
def test_named_commands_remain_commands(command):
    result = CliRunner().invoke(main.app, [command])
    assert result.exit_code == 0, result.output


def test_shortcut_accepts_options_after_path(tmp_path, engine):
    media = tmp_path / "sample.wav"
    media.touch()
    result = CliRunner().invoke(main.app, [str(media), "-m", "fake", "-f", "srt"])
    assert result.exit_code == 0, result.output
    assert media.with_suffix(".srt").exists()


def test_recursive_outputs_preserve_folders(tmp_path, engine):
    for directory in ["a", "b"]:
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "meeting.wav").touch()
    output = tmp_path / "transcripts"
    result = CliRunner().invoke(
        main.app, ["transcribe", str(tmp_path), "-m", "fake", "-r", "-o", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert (output / "a/meeting.txt").exists()
    assert (output / "b/meeting.txt").exists()


def test_colliding_stems_fail_before_transcription(tmp_path, engine, monkeypatch):
    for suffix in [".wav", ".mp3"]:
        (tmp_path / f"meeting{suffix}").touch()
    calls = []
    monkeypatch.setattr(engine, "transcribe", lambda *a, **k: calls.append(a))
    result = CliRunner().invoke(main.app, ["transcribe", str(tmp_path), "-m", "fake"])
    assert result.exit_code == 1
    assert "colliding" in result.output
    assert not calls


def test_concatenation_shifts_words_and_probed_duration(tmp_path, engine, monkeypatch):
    for name in ["a.wav", "b.wav"]:
        (tmp_path / name).touch()
    saved = []
    formatter = main.get_formatter("json")
    monkeypatch.setattr(formatter, "save", lambda result, path: saved.append(result))
    monkeypatch.setattr(main, "get_formatter", lambda name: formatter)
    result = CliRunner().invoke(
        main.app, ["transcribe", str(tmp_path), "-m", "fake", "-b", "concatenate"]
    )
    assert result.exit_code == 0, result.output
    assert [segment.start for segment in saved[0].segments] == [1, 11]
    assert [segment.words[0].start for segment in saved[0].segments] == [1, 11]
    assert saved[0].duration == 20


def test_engine_failure_cleans_extracted_audio(tmp_path, engine, monkeypatch):
    media = tmp_path / "sample.mp4"
    media.touch()
    extracted = tmp_path / "temporary.wav"
    extracted.touch()
    monkeypatch.setattr(main, "prepare_audio", lambda path: (extracted, True))

    def fail(*args, **kwargs):
        raise RuntimeError("engine failed")

    monkeypatch.setattr(engine, "transcribe", fail)
    result = CliRunner().invoke(main.app, ["transcribe", str(media), "-m", "fake"])
    assert result.exit_code == 1
    assert not extracted.exists()
    assert media.exists()


def test_extraction_failure_cleans_only_owned_file(tmp_path, monkeypatch):
    created = []
    original = audio.tempfile.NamedTemporaryFile

    def create_temp(**kwargs):
        temp = original(dir=tmp_path, **kwargs)
        created.append(Path(temp.name))
        return temp

    monkeypatch.setattr(audio.tempfile, "NamedTemporaryFile", create_temp)
    monkeypatch.setattr(
        audio.ffmpeg, "input", lambda path: (_ for _ in ()).throw(RuntimeError("ffmpeg failed"))
    )
    with pytest.raises(RuntimeError):
        audio.extract_audio(tmp_path / "sample.mp4")
    assert created and not created[0].exists()
    supplied = tmp_path / "supplied.wav"
    supplied.write_text("preserve")
    with pytest.raises(RuntimeError):
        audio.extract_audio(tmp_path / "sample.mp4", supplied)
    assert supplied.read_text() == "preserve"


def test_model_prompt_uses_selected_engine_models(engine, monkeypatch):
    captured = []
    monkeypatch.setattr(
        main, "prompt_model_choice", lambda models, **kwargs: captured.append((models, kwargs))
    )
    main.choose_model(engine)
    assert captured[0][0] == engine.supported_models
    assert captured[0][1]["default"] == engine.supported_models[0]


def test_registry_works_in_fresh_process():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from transcript_cli.engines import get_all_engines; assert 'parakeet' in get_all_engines()",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_existing_dotted_output_directory_is_supported(tmp_path, engine):
    for name in ["a.wav", "b.wav"]:
        (tmp_path / name).touch()
    output = tmp_path / "transcripts.v1"
    output.mkdir()
    result = CliRunner().invoke(
        main.app, ["transcribe", str(tmp_path), "-m", "fake", "-o", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert (output / "a.txt").exists()
    assert (output / "b.txt").exists()
