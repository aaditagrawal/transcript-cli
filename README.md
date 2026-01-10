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

```bash
transcript video.mp4              # Text output
transcript video.mp4 -f srt       # Subtitles
transcript video.mp4 -f json      # JSON with word timestamps
transcript ./folder -r            # Batch process
```

## Output Formats

| Format | Description |
|--------|-------------|
| `text` | Plain text |
| `timestamps` | Text with `[HH:MM:SS]` |
| `srt` | SubRip subtitles |
| `vtt` | WebVTT subtitles |
| `json` | Full metadata + word timestamps |

## Engines

| Engine | Platform | Install |
|--------|----------|---------|
| `mlx-whisper` | Apple Silicon | `uv pip install mlx-whisper` |
| `faster-whisper` | NVIDIA/CPU | `uv pip install faster-whisper` |
| `insanely-fast-whisper` | NVIDIA | Via install script |

## License

MIT
