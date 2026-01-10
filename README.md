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

## Usage

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
