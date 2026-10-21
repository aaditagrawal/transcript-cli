"""Check the real optional decoder without downloading an ASR model."""

import struct
import tempfile
import wave
from importlib import import_module
from pathlib import Path


def main() -> None:
    audio = import_module("faster_whisper.audio")
    av = import_module("av")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "stereo.wav"
        with wave.open(str(path), "wb") as source:
            source.setnchannels(2)
            source.setsampwidth(2)
            source.setframerate(8000)
            # Opposite channels should cancel when the decoder mixes to mono.
            source.writeframes(struct.pack("<hh", 1000, -1000) * 800)
        decoded = audio.decode_audio(str(path), sampling_rate=16000)
        assert decoded.shape == (1600,), decoded.shape
        assert decoded.dtype.name == "float32", decoded.dtype
        assert abs(decoded).max() < 1e-8
        print(f"PASS real Faster Whisper stereo decoding/resampling with PyAV {av.__version__}")


if __name__ == "__main__":
    main()
