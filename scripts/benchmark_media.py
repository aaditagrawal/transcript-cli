"""Compare media helpers with a committed baseline on the same host and data.

Run with the baseline wrapper available only in the benchmark environment:
uv run --frozen --with ffmpeg-python==0.2.0 --with future==1.0.0 python scripts/benchmark_media.py
"""

import argparse
import importlib.util
import json
import math
import os
import platform
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import wave
from functools import partial
from pathlib import Path
from types import ModuleType

from transcript_cli import audio


def load_baseline(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("transcript_cli._baseline_audio", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load baseline module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(baseline, candidate, repeats: int, warmups: int) -> dict:
    samples: dict[str, list[float]] = {"baseline": [], "candidate": []}
    operations = {"baseline": baseline, "candidate": candidate}
    for iteration in range(warmups + repeats):
        order = ["baseline", "candidate"]
        if iteration % 2:
            order.reverse()
        for name in order:
            start = time.perf_counter()
            operations[name]()
            elapsed = time.perf_counter() - start
            if iteration >= warmups:
                samples[name].append(elapsed)
    result = {}
    for name, values in samples.items():
        ordered = sorted(values)
        result[name] = {
            "median_ms": statistics.median(values) * 1000,
            "p95_ms": ordered[math.ceil(len(ordered) * 0.95) - 1] * 1000,
            "min_ms": ordered[0] * 1000,
            "max_ms": ordered[-1] * 1000,
            "samples_ms": [value * 1000 for value in values],
        }
    result["median_ratio"] = result["candidate"]["median_ms"] / result["baseline"]["median_ms"]
    result["p95_ratio"] = result["candidate"]["p95_ms"] / result["baseline"]["p95_ms"]
    return result


def write_fixture(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        block = b"".join(
            struct.pack(
                "<hh",
                round(5000 * math.sin(2 * math.pi * 440 * sample / 48000)),
                round(2000 * math.sin(2 * math.pi * 880 * sample / 48000)),
            )
            for sample in range(4800)
        )
        for _ in range(round(seconds * 10)):
            stream.writeframesraw(block)


def make_video_fixture(root: Path, name: str, seconds: float, compressed: bool) -> Path:
    source = root / "source.wav"
    write_fixture(source, seconds)
    output = root / f"{name} 音声 ; file.{'mp4' if compressed else 'mkv'}"
    command = ["ffmpeg", "-v", "error", "-y", "-i", str(source)]
    if compressed:
        command += [
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=16x16:r=1:d={seconds}",
            "-map",
            "0:a",
            "-map",
            "1:v",
            "-c:a",
            "aac",
            "-c:v",
            "mpeg4",
            "-threads",
            "1",
        ]
    else:
        command += ["-c:a", "copy"]
    subprocess.run([*command, str(output)], check=True, capture_output=True)
    return output


def fresh_import(path: Path) -> None:
    # Give each module the same loader and package context in a fresh process.
    code = (
        "import importlib.util; "
        f"s=importlib.util.spec_from_file_location('transcript_cli._bench', {str(path)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def checkpoint(report: dict, output: Path | None, completed: str) -> None:
    report["load_average_after"] = os.getloadavg() if hasattr(os, "getloadavg") else None
    if output:
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(output)
    print(f"Completed {completed}", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="aeae6dee2563a36a21a23e66f1edeac54e4c8d5d")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output", type=Path, help="save JSON after every completed workload")
    parser.add_argument(
        "--case",
        action="append",
        choices=[
            "small",
            "typical",
            "large",
            "compressed_video",
            "probe_invalid",
            "fresh_process_import",
        ],
        help="limit the run to selected fixtures or startup/error cases; repeat to combine",
    )
    arguments = parser.parse_args()
    if arguments.repeats < 2 or arguments.warmups < 1:
        parser.error("use at least two measured repeats and one warmup")
    repository = Path(__file__).resolve().parents[1]
    source = subprocess.run(
        ["git", "show", f"{arguments.baseline}:src/transcript_cli/audio.py"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    results = {}
    report = {
        "baseline": arguments.baseline,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "load_average_before": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "ffmpeg": subprocess.run(
            ["ffmpeg", "-version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0],
        "repeats": arguments.repeats,
        "warmups": arguments.warmups,
        "fixture": "48 kHz stereo dual tones in PCM MKV, 0.1/30/300 seconds; 30s AAC + MPEG4 video MP4",
        "compatibility": "byte-identical extracted WAV and equal probed duration for every fixture",
        "timing": "wall time; alternating order; extraction/probe include tool subprocess; fresh import includes Python startup",
        "results": results,
    }
    with tempfile.TemporaryDirectory(prefix="transcript-benchmark-") as directory:
        root = Path(directory)
        baseline_path = root / "baseline_audio.py"
        baseline_path.write_bytes(source)
        baseline = load_baseline(baseline_path)
        cases = [
            ("small", 0.1, False),
            ("typical", 30, False),
            ("large", 300, False),
            ("compressed_video", 30, True),
        ]
        for name, seconds, compressed in cases:
            if arguments.case and name not in arguments.case:
                continue
            path = make_video_fixture(root, name, seconds, compressed)
            output = root / "output.wav"
            reference = root / "reference.wav"
            baseline.extract_audio(path, reference)
            audio.extract_audio(path, output)
            assert output.read_bytes() == reference.read_bytes(), f"PCM mismatch: {name}"
            assert audio.get_audio_duration(path) == baseline.get_audio_duration(path)
            results[f"extract_{name}"] = compare(
                partial(baseline.extract_audio, path, output),
                partial(audio.extract_audio, path, output),
                arguments.repeats,
                arguments.warmups,
            )
            checkpoint(report, arguments.output, f"extract_{name}")
            results[f"probe_{name}"] = compare(
                partial(baseline.get_audio_duration, path),
                partial(audio.get_audio_duration, path),
                arguments.repeats,
                arguments.warmups,
            )
            checkpoint(report, arguments.output, f"probe_{name}")
        invalid = root / "invalid.wav"
        invalid.write_text("not audio")
        if not arguments.case or "probe_invalid" in arguments.case:
            results["probe_invalid"] = compare(
                lambda: baseline.get_audio_duration(invalid),
                lambda: audio.get_audio_duration(invalid),
                arguments.repeats,
                arguments.warmups,
            )
            checkpoint(report, arguments.output, "probe_invalid")
        if not arguments.case or "fresh_process_import" in arguments.case:
            results["fresh_process_import"] = compare(
                lambda: fresh_import(baseline_path),
                lambda: fresh_import(Path(audio.__file__)),
                arguments.repeats,
                arguments.warmups,
            )
            checkpoint(report, arguments.output, "fresh_process_import")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
