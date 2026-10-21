# 🎙️ Transcript CLI

Fast audio/video transcription using Whisper, optimized for NVIDIA GPU and Apple Silicon.

## Install (NVIDIA Linux)

```bash
curl -sSL https://raw.githubusercontent.com/aaditagrawal/transcript-cli/main/install.sh | bash
```

Restart terminal, then use.

## Install (Apple Silicon)

```bash
uv pip install git+https://github.com/aaditagrawal/transcript-cli.git
uv pip install mlx-whisper
```

## Optional engines

Install only the inference libraries you need in your environment:

```bash
uv pip install '.[faster]'    # CTranslate2 on NVIDIA or CPU
uv pip install '.[insanely]'  # Transformers Whisper with optional Flash Attention
uv pip install '.[parakeet]' # NVIDIA Parakeet TDT v3 through Transformers
uv pip install '.[apple]'    # MLX Whisper on Apple Silicon
```

The `parakeet` engine keeps the `nvidia/parakeet-tdt-0.6b-v3` model and produces
segment and word timestamps from the model's token durations. It now uses the
[NVIDIA/Hugging Face Transformers implementation](https://huggingface.co/docs/transformers/v5.20.2/model_doc/parakeet)
instead of NeMo's training stack. Model downloads use safetensors. There is no
NeMo, Hydra, or Lightning requirement for transcription. Subtitle segments end
at sentence punctuation, pauses of at least one second, or thirty seconds.

The Faster Whisper dependency is pinned to [its upstream PyAV 19 compatibility fix](https://github.com/SYSTRAN/faster-whisper/commit/2ce7f9d7a9fbe315a5804a33bf7224d42e101174)
until a package release includes it. Git is needed when installing this extra.
The older decoder metadata handling remains in place on PyAV versions before 19.

The `insanely-fast-whisper` engine uses Transformers directly. Its old CLI
metapackage is unnecessary for our adapter and pulled in image dependencies.
The installer uses the same audio dependencies as the optional extra.

For contributors, `uv sync --frozen --python 3.12 --group dev` installs the base
and development tools. `uv run --frozen ty check` checks the adapters even when
optional libraries are absent. Their lazy loading contracts and result handling
are covered by `uv run --frozen pytest`.

## Usage

List engines, formats, or models without starting a transcription:

```bash
transcript list-engines
transcript list-formats
transcript list-models
```

File shorthand still works. If a filename matches a command, use an explicit path,
such as `transcript --path list-engines`, or prefix it with `./`.

### Interactive Mode (Recommended for First Use)

Just run with a file - you'll be prompted for options:

```bash
transcript video.mp4
```

You'll see:
```
Select output format [text/timestamps/srt/vtt/json]: srt
Select model size [tiny/base/small/medium/large-v3/turbo]: base
```

### Quick Mode (Skip Prompts)

Specify options directly:

```bash
transcript video.mp4 -f srt -m base      # SRT subtitles with base model
transcript video.mp4 -f json -m large-v3 # JSON with large model
transcript ./folder -r -f vtt            # Batch process folder
```

## Output Formats

| Format | Flag | Description |
|--------|------|-------------|
| `text` | `-f text` | Plain text transcript |
| `timestamps` | `-f timestamps` | Text with `[HH:MM:SS]` markers |
| `srt` | `-f srt` | SubRip subtitles |
| `vtt` | `-f vtt` | WebVTT subtitles |
| `json` | `-f json` | Full metadata + word timestamps |

## Models

| Model | Flag | Speed | Quality |
|-------|------|-------|---------|
| `tiny` | `-m tiny` | Fastest | Basic |
| `base` | `-m base` | Fast | Good (default) |
| `small` | `-m small` | Medium | Better |
| `medium` | `-m medium` | Slow | Great |
| `large-v3` | `-m large-v3` | Slowest | Best |
| `turbo` | `-m turbo` | Fast | Great |

## Examples

```bash
# Quick subtitle generation
transcript movie.mp4 -f srt -m base

# High-quality transcription
transcript interview.mp3 -f json -m large-v3

# Process all videos in folder
transcript ~/Videos -r -f srt -m turbo
```

## License

MIT
