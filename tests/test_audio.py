"""Exercise media conversion against FFmpeg without inference or model downloads."""

import shutil
import struct
import subprocess
import wave
from pathlib import Path
from unittest.mock import Mock

import pytest

from transcript_cli import audio


@pytest.fixture
def stereo_wav(tmp_path):
    source = tmp_path / "stereo.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(struct.pack("<hh", 1000, -1000) * 800)
    return source


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools unavailable"
)
def test_real_conversion_overwrites_resamples_and_mixes(stereo_wav, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = Path("-音声 ; $(touch unexpected).wav")
    stereo_wav.rename(source)
    output = Path("-output ; audio.wav")
    output.write_text("old output")
    assert audio.extract_audio(source, output) == output
    with wave.open(str(output), "rb") as converted:
        assert converted.getnchannels() == 1
        assert converted.getsampwidth() == 2
        assert converted.getframerate() == 16000
        assert converted.getnframes() == 1600
        assert set(converted.readframes(1600)) == {0}
    assert audio.get_audio_duration(output) == pytest.approx(0.1)
    assert not Path("unexpected").exists()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg unavailable")
def test_real_conversion_preserves_default_audio_stream_selection(stereo_wav, tmp_path):
    video = tmp_path / "multiple.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(stereo_wav),
            "-f",
            "lavfi",
            "-i",
            "aevalsrc=0.25:s=8000:d=0.1",
            "-map",
            "1:a",
            "-map",
            "0:a",
            "-c:a",
            "pcm_s16le",
            "-disposition:a:0",
            "0",
            "-disposition:a:1",
            "0",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    output, temporary = audio.prepare_audio(video)
    try:
        assert temporary
        with wave.open(str(output), "rb") as converted:
            # Automatic selection chooses the stereo stream over the first mono stream.
            assert set(converted.readframes(converted.getnframes())) == {0}
    finally:
        output.unlink()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg unavailable")
def test_output_symlink_preserves_requested_format(stereo_wav, tmp_path):
    target = tmp_path / "target.dat"
    target.write_bytes(b"old output")
    output = tmp_path / "output.wav"
    output.symlink_to(target)

    assert audio.extract_audio(stereo_wav, output) == output
    assert output.is_symlink()
    with wave.open(str(target), "rb") as converted:
        assert converted.getnchannels() == 1
        assert converted.getframerate() == 16000
        assert converted.getnframes() == 1600


@pytest.mark.parametrize("failure", [RuntimeError("conversion failed"), KeyboardInterrupt()])
def test_failed_extraction_cleans_only_owned_tempfile(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(audio.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(audio, "_run_media_command", Mock(side_effect=failure))
    with pytest.raises(type(failure)):
        audio.extract_audio(Path("broken.mp4"))
    assert list(tmp_path.iterdir()) == []
    output = tmp_path / "chosen.wav"
    output.write_bytes(b"existing")
    with pytest.raises(type(failure)):
        audio.extract_audio(Path("broken.mp4"), output)
    assert output.read_bytes() == b"existing"


def test_failed_conversion_exposes_diagnostics(monkeypatch):
    error = subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"Invalid media \xff")
    monkeypatch.setattr(audio.subprocess, "run", Mock(side_effect=error))
    with pytest.raises(RuntimeError, match="Invalid media") as caught:
        audio.extract_audio(Path("broken.mp4"))
    assert caught.value.__cause__ is error


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        b"{}",
        b"[]",
        b'{"format": null}',
        b'{"format": {"duration": "N/A"}}',
        b'{"format": {"duration": "NaN"}}',
        b'{"format": {"duration": "Infinity"}}',
        b'{"format": {"duration": -1}}',
    ],
)
def test_unavailable_duration_returns_zero(monkeypatch, payload):
    monkeypatch.setattr(audio, "_run_media_command", Mock(return_value=payload))
    assert audio.get_audio_duration(Path("audio.wav")) == 0


@pytest.mark.parametrize("error", [FileNotFoundError("ffprobe"), RuntimeError("invalid media")])
def test_probe_command_failure_returns_zero(monkeypatch, error):
    monkeypatch.setattr(audio, "_run_media_command", Mock(side_effect=error))
    assert audio.get_audio_duration(Path("audio.wav")) == 0


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffprobe unavailable")
def test_real_failed_probe(tmp_path):
    source = tmp_path / "invalid.wav"
    source.write_text("not audio")
    assert audio.get_audio_duration(source) == 0
