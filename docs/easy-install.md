# Easy Install

## NVIDIA Linux

```bash
curl -sSL https://raw.githubusercontent.com/aaditagrawal/transcript-cli/main/install.sh | bash
```

Restart terminal, then:

```bash
transcript video.mp4              # Text
transcript video.mp4 -f srt       # Subtitles
```

## Requirements

- NVIDIA GPU with drivers (`nvidia-smi` should work)
- FFmpeg (`sudo apt install ffmpeg`)

## Uninstall

```bash
rm -rf ~/.local/share/transcript-cli ~/.local/bin/transcript
```
