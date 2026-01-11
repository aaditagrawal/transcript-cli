# Easy Install

## NVIDIA Linux

```bash
curl -sSL https://raw.githubusercontent.com/aaditagrawal/transcript-cli/main/install.sh | bash
```

Restart terminal.

## Usage

### Just Run It

```bash
transcript video.mp4
```

You'll be asked:
- **Output format**: text, srt (subtitles), vtt, json
- **Model size**: base (fast), large-v3 (best quality), turbo (fast + good)

### Skip Prompts

```bash
transcript video.mp4 -f srt -m base
```

## Quick Reference

| What You Want | Command |
|---------------|---------|
| Subtitles | `transcript video.mp4 -f srt` |
| Best quality | `transcript video.mp4 -m large-v3` |
| Fast + good | `transcript video.mp4 -m turbo` |
| All files in folder | `transcript ~/Videos -r` |

## Uninstall

```bash
rm -rf ~/.local/share/transcript-cli ~/.local/bin/transcript
```
