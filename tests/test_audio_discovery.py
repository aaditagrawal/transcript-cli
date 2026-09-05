"""Media discovery preserves results while avoiding unrelated file metadata."""

from pathlib import Path

from transcript_cli.audio import discover_files


def test_discovery_filters_before_checking_file_metadata(tmp_path, monkeypatch):
    for name in ("z.WAV", "a.mp4", "notes.txt", "cover.png"):
        (tmp_path / name).touch()
    (tmp_path / "directory.mp3").mkdir()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "voice.flac").touch()
    (nested / "notes.txt").touch()
    checked = []
    original = Path.is_file

    def record_is_file(path):
        checked.append(path)
        return original(path)

    monkeypatch.setattr(Path, "is_file", record_is_file)

    assert discover_files(tmp_path) == [tmp_path / "a.mp4", tmp_path / "z.WAV"]
    assert tmp_path / "directory.mp3" in checked
    assert tmp_path / "notes.txt" not in checked
    assert tmp_path / "cover.png" not in checked

    checked.clear()
    assert discover_files(tmp_path, recursive=True) == sorted(
        [tmp_path / "a.mp4", tmp_path / "z.WAV", nested / "voice.flac"]
    )
    assert nested / "notes.txt" not in checked


def test_discovery_handles_direct_paths_and_symlinks(tmp_path):
    media = tmp_path / "audio.wav"
    media.touch()
    link = tmp_path / "linked.WAV"
    link.symlink_to(media)
    (tmp_path / "broken.wav").symlink_to(tmp_path / "missing")
    text = tmp_path / "notes.txt"
    text.touch()

    assert discover_files(media) == [media]
    assert discover_files(link) == [link]
    assert discover_files(text) == []
    assert discover_files(tmp_path / "missing") == []
    assert discover_files(tmp_path) == [media, link]
