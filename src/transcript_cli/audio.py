"""Audio file handling and extraction from video."""

import tempfile
from pathlib import Path

import ffmpeg

from .config import AUDIO_EXTENSIONS, SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS


def is_audio_file(path: Path) -> bool:
    """Check if path is a supported audio file."""
    return path.suffix.lower() in AUDIO_EXTENSIONS


def is_video_file(path: Path) -> bool:
    """Check if path is a supported video file."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_supported_file(path: Path) -> bool:
    """Check if path is a supported audio or video file."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def discover_files(path: Path, recursive: bool = False) -> list[Path]:
    """
    Discover audio/video files from a path.

    Args:
        path: File or directory path
        recursive: If True, search subdirectories

    Returns:
        List of supported media files
    """
    if path.is_file():
        if is_supported_file(path):
            return [path]
        return []

    if path.is_dir():
        pattern = "**/*" if recursive else "*"
        files = [f for f in path.glob(pattern) if is_supported_file(f) and f.is_file()]
        return sorted(files)

    return []


def extract_audio(video_path: Path, output_path: Path | None = None) -> Path:
    """
    Extract audio from video file to WAV format.

    Args:
        video_path: Path to video file
        output_path: Optional output path, uses temp file if not provided

    Returns:
        Path to extracted audio file
    """
    if output_path is None:
        # Create temp file that will persist
        temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = Path(temp.name)
        temp.close()

    # Extract audio using ffmpeg
    (
        ffmpeg.input(str(video_path))
        .output(
            str(output_path),
            acodec="pcm_s16le",  # 16-bit PCM
            ar=16000,  # 16kHz sample rate (optimal for Whisper)
            ac=1,  # Mono
        )
        .overwrite_output()
        .run(quiet=True)
    )

    return output_path


def get_audio_duration(path: Path) -> float:
    """Get audio/video duration in seconds."""
    try:
        probe = ffmpeg.probe(str(path))
        return float(probe["format"]["duration"])
    except (ffmpeg.Error, KeyError):
        return 0.0


def prepare_audio(path: Path) -> tuple[Path, bool]:
    """
    Prepare audio for transcription.

    If the input is video, extracts audio to a temp file.
    If the input is audio, returns the original path.

    Args:
        path: Input file path

    Returns:
        Tuple of (audio_path, is_temp) where is_temp indicates if cleanup is needed
    """
    if is_video_file(path):
        return extract_audio(path), True
    return path, False
