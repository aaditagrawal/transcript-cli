"""Configuration, paths, and platform detection."""

import platform
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from platformdirs import user_cache_dir


class Platform(Enum):
    """Detected hardware platform."""

    NVIDIA = "nvidia"
    APPLE_SILICON = "apple_silicon"
    CPU = "cpu"


@dataclass
class AppConfig:
    """Application configuration."""

    cache_dir: Path
    models_dir: Path
    platform: Platform

    @classmethod
    def create(cls) -> "AppConfig":
        """Create configuration with detected platform."""
        cache_base = Path(user_cache_dir("transcript-cli"))
        return cls(
            cache_dir=cache_base,
            models_dir=cache_base / "models",
            platform=detect_platform(),
        )


def detect_platform() -> Platform:
    """Detect the best available compute platform."""
    system = platform.system()
    machine = platform.machine()

    # Apple Silicon detection
    if system == "Darwin" and machine == "arm64":
        return Platform.APPLE_SILICON

    # NVIDIA GPU detection
    if _has_nvidia_gpu():
        return Platform.NVIDIA

    return Platform.CPU


def _has_nvidia_gpu() -> bool:
    """Check if NVIDIA GPU is available via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Audio/video file extensions
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".wma", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".flv", ".m4v"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

# Whisper model sizes
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"]
