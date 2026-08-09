"""Engine registry and base classes."""

from .base import TranscribeOptions, TranscriptionEngine, TranscriptionResult

# Engine registry - populated on import
_engines: dict[str, type[TranscriptionEngine]] = {}


def register_engine(engine_class: type[TranscriptionEngine]) -> type[TranscriptionEngine]:
    """Decorator to register an engine class."""
    _engines[engine_class.name] = engine_class
    return engine_class


def get_engine(name: str) -> TranscriptionEngine:
    """Get an engine instance by name."""
    if name not in _engines:
        raise ValueError(f"Unknown engine: {name}. Available: {list(_engines.keys())}")
    return _engines[name]()


def get_available_engines() -> dict[str, TranscriptionEngine]:
    """Get all available (installed) engines."""
    return {name: cls() for name, cls in _engines.items() if cls().is_available()}


def get_all_engines() -> dict[str, TranscriptionEngine]:
    """Get all registered engines."""
    return {name: cls() for name, cls in _engines.items()}


def get_best_engine() -> TranscriptionEngine | None:
    """Get the best available engine for the current platform."""
    from ..config import Platform, detect_platform

    platform = detect_platform()
    available = get_available_engines()

    if not available:
        return None

    # Priority order per platform
    priority = {
        Platform.APPLE_SILICON: ["mlx-whisper", "faster-whisper"],
        Platform.NVIDIA: ["insanely-fast-whisper", "faster-whisper", "parakeet"],
        Platform.CPU: ["faster-whisper"],
    }

    for engine_name in priority.get(platform, []):
        if engine_name in available:
            return available[engine_name]

    # Fallback to first available
    return next(iter(available.values()))


# Import engines to trigger registration
def _load_engines() -> None:
    """Load all engine modules."""
    try:
        from . import faster_whisper  # noqa: F401
    except ImportError:
        pass

    try:
        from . import insanely_fast_whisper  # noqa: F401
    except ImportError:
        pass

    try:
        from . import parakeet  # noqa: F401
    except ImportError:
        pass

    try:
        from . import mlx_whisper  # noqa: F401
    except ImportError:
        pass


__all__ = [
    "TranscriptionEngine",
    "TranscriptionResult",
    "TranscribeOptions",
    "register_engine",
    "get_engine",
    "get_available_engines",
    "get_all_engines",
    "get_best_engine",
]
