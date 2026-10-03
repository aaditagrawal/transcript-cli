"""Main CLI application for transcript-cli."""

import time
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer
from typer.core import TyperGroup

from . import __version__
from .audio import discover_files, get_audio_duration, prepare_audio
from .config import WHISPER_MODELS, AppConfig
from .engines import (
    get_all_engines,
    get_available_engines,
    get_best_engine,
    get_engine,
)
from .engines.base import Task, TranscribeOptions
from .formatters import OUTPUT_FORMATS, get_formatter
from .ui import (
    console,
    create_progress,
    print_banner,
    print_error,
    print_file_list,
    print_info,
    print_platform_info,
    print_success,
    print_transcription_summary,
    prompt_batch_mode,
    prompt_engine_choice,
    prompt_model_choice,
    prompt_output_format,
)


class ShortcutGroup(TyperGroup):
    """Route a leading media path to transcribe without consuming command names."""

    def parse_args(self, ctx, args):
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            args = ["transcribe", *args]
        return super().parse_args(ctx, args)


ENGINE_EXTRAS = {
    "faster-whisper": "faster",
    "insanely-fast-whisper": "insanely",
    "mlx-whisper": "apple",
    "parakeet": "parakeet",
}


def choose_model(engine):
    models = engine.supported_models
    default = "base" if "base" in models else models[0]
    return prompt_model_choice(models, default=default, engine=engine)


# Initialize Typer app
app = typer.Typer(
    cls=ShortcutGroup,
    name="transcript",
    help="🎙️ Multi-engine audio/video transcription CLI",
    no_args_is_help=True,
    rich_markup_mode="rich",
    invoke_without_command=True,
)


@app.callback(invoke_without_command=True)
def main_callback(
    version: Annotated[
        bool,
        typer.Option("--version", "-v", help="Show version and exit"),
    ] = False,
):
    """Transcribe media with `transcript FILE [options]` or a named command."""
    if version:
        console.print(f"transcript-cli v{__version__}")
        raise typer.Exit()


@app.command()
def transcribe(
    path: Annotated[
        Path,
        typer.Argument(
            help="Path to audio/video file or directory",
            exists=True,
            resolve_path=True,
        ),
    ],
    engine: Annotated[
        str | None,
        typer.Option("--engine", "-e", help="Transcription engine to use"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model name or size"),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format",
        ),
    ] = "text",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file or directory"),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", "-l", help="Force language (default: auto-detect)"),
    ] = None,
    task: Annotated[
        str,
        typer.Option("--task", "-t", help="Task: transcribe or translate"),
    ] = "transcribe",
    batch_mode: Annotated[
        str | None,
        typer.Option("--batch", "-b", help="Batch mode: individual or concatenate"),
    ] = None,
    word_timestamps: Annotated[
        bool,
        typer.Option("--word-timestamps", "-w", help="Include word-level timestamps"),
    ] = False,
    recursive: Annotated[
        bool,
        typer.Option("--recursive", "-r", help="Recursively search directories"),
    ] = False,
    interactive: Annotated[
        bool,
        typer.Option("--interactive", "-i", help="Interactive mode with prompts"),
    ] = False,
):
    """
    Transcribe audio or video files.

    Examples:
        transcript transcribe audio.mp3
        transcript transcribe ./recordings --recursive --format srt
        transcript transcribe video.mp4 --engine faster-whisper --model large-v3
    """
    print_banner()
    config = AppConfig.create()
    print_platform_info(config.platform)
    console.print()

    # Discover files
    files = discover_files(path, recursive=recursive)
    if not files:
        print_error(f"No supported audio/video files found in: {path}")
        raise typer.Exit(1)

    print_file_list(files)
    console.print()

    # Get available engines
    available = get_available_engines()
    if not available:
        print_error("No transcription engines installed!")
        print_info("Install an engine with: uv pip install transcript-cli[faster]")
        raise typer.Exit(1)

    # Interactive mode: prompt for missing engine
    if interactive or engine is None:
        if len(available) > 1:
            engine = prompt_engine_choice(list(available.keys()))
        else:
            engine = list(available.keys())[0]
            print_info(f"Using engine: {engine}")

    # Get engine instance early so we can show model download status
    try:
        engine_instance = get_engine(engine) if engine else get_best_engine()
        if engine_instance is None:
            print_error("No engine available")
            raise typer.Exit(1)
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    # Prompt for model with download status
    if interactive or model is None:
        model = choose_model(engine_instance)

    if interactive and output_format == "text":
        output_format = prompt_output_format(OUTPUT_FORMATS)

    if interactive and len(files) > 1 and batch_mode is None:
        batch_mode = prompt_batch_mode()

    # Validate format
    if output_format not in OUTPUT_FORMATS:
        print_error(f"Unknown format: {output_format}. Available: {OUTPUT_FORMATS}")
        raise typer.Exit(1)

    # Prepare options
    options = TranscribeOptions(
        language=language,
        task=Task.TRANSLATE if task == "translate" else Task.TRANSCRIBE,
        word_timestamps=word_timestamps or output_format == "json",
    )

    # Get formatter
    formatter = get_formatter(output_format)

    # Plan every destination before inference so collisions cannot overwrite a batch.
    output_paths = {}
    if batch_mode == "concatenate":
        if output is None:
            output = path if path.is_dir() else path.parent
    elif len(files) > 1:
        destination = output or (path if path.is_dir() else path.parent)
        if (destination.exists() and not destination.is_dir()) or (
            not destination.exists()
            and destination.suffix in {".txt", ".srt", ".vtt", ".json", ".md"}
        ):
            print_error("Multiple inputs require an output directory, not a shared file")
            raise typer.Exit(1)
        root = path if path.is_dir() else path.parent
        for file in files:
            relative = file.relative_to(root).with_suffix(formatter.extension)
            output_paths[file] = destination / relative
        if len(set(output_paths.values())) != len(files):
            print_error("Inputs have colliding transcript names; rename same-stem media files")
            raise typer.Exit(1)
        output = destination
    else:
        output = output or formatter.get_output_path(files[0])
        output_paths[files[0]] = (
            formatter.get_output_path(files[0], output) if output.is_dir() else output
        )

    # Process files
    start_time = time.time()
    total_duration = 0.0
    all_results = []
    errors = []

    with create_progress() as progress:
        task_id = progress.add_task("Transcribing...", total=len(files))

        for file in files:
            progress.update(task_id, description=f"Processing: {file.name}")

            audio_path = None
            is_temp = False
            try:
                # Prepare audio (extract from video if needed)
                audio_path, is_temp = prepare_audio(file)

                # Get duration
                duration = get_audio_duration(audio_path)
                total_duration += duration

                # Transcribe
                result = engine_instance.transcribe(
                    audio_path,
                    model=model or "base",
                    options=options,
                )

                result.duration = duration or result.duration
                all_results.append((file, result))

            except Exception as e:
                errors.append((file.name, str(e)))
            finally:
                if is_temp and audio_path is not None:
                    audio_path.unlink(missing_ok=True)
                progress.advance(task_id)

    # Show any errors that occurred
    for filename, error in errors:
        print_error(f"Failed to transcribe {filename}: {error}")

    if not all_results:
        if errors:
            print_error("All transcriptions failed. See errors above.")
        raise typer.Exit(1)

    # Save results
    if batch_mode == "concatenate":
        # Combine all results
        from .engines.base import Segment, TranscriptionResult

        combined_segments = []
        combined_text = []
        offset = 0.0

        for file, result in all_results:
            # Add file marker
            combined_text.append(f"\n--- {file.name} ---\n")
            combined_text.append(result.text)

            # Adjust segment timestamps
            for seg in result.segments:
                combined_segments.append(
                    Segment(
                        text=seg.text,
                        start=seg.start + offset,
                        end=seg.end + offset,
                        words=[
                            replace(word, start=word.start + offset, end=word.end + offset)
                            for word in seg.words
                        ],
                    )
                )

            offset += result.duration

        combined_result = TranscriptionResult(
            text="\n".join(combined_text),
            segments=combined_segments,
            language=all_results[0][1].language if all_results else "en",
            duration=total_duration,
        )

        output_path = output if output.suffix else output / f"combined{formatter.extension}"
        formatter.save(combined_result, output_path)
        print_success(f"Saved combined transcript to: {output_path}")
    else:
        # Individual files
        for file, result in all_results:
            output_path = output_paths[file]

            formatter.save(result, output_path)
            print_success(f"Saved: {output_path}")

    # Print summary
    elapsed_time = time.time() - start_time
    print_transcription_summary(
        files_processed=len(all_results),
        total_duration=total_duration,
        elapsed_time=elapsed_time,
        output_path=output,
    )


@app.command()
def download(
    engine: Annotated[
        str,
        typer.Argument(help="Engine to download model for"),
    ],
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model name or size"),
    ] = None,
):
    """
    Download a model for an engine.

    Examples:
        transcript download faster-whisper --model large-v3
        transcript download mlx-whisper --model turbo
    """
    print_banner()

    try:
        engine_instance = get_engine(engine)
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    if not engine_instance.is_available():
        print_error(f"Engine '{engine}' is not installed")
        print_info(
            f"Install with: uv pip install transcript-cli[{ENGINE_EXTRAS.get(engine, engine)}]"
        )
        raise typer.Exit(1)

    model = model or (
        "base"
        if "base" in engine_instance.supported_models
        else engine_instance.supported_models[0]
    )

    console.print(f"Downloading model [cyan]{model}[/cyan] for [cyan]{engine}[/cyan]...")

    with create_progress() as progress:
        task_id = progress.add_task(f"Downloading {model}...", total=100)

        def progress_callback(pct: float, status: str):
            progress.update(task_id, completed=int(pct * 100), description=status)

        try:
            engine_instance.download_model(model, progress_callback)
            print_success(f"Model {model} downloaded successfully!")
        except Exception as e:
            print_error(f"Download failed: {e}")
            raise typer.Exit(1)


@app.command("list-engines")
def list_engines():
    """List available transcription engines."""
    print_banner()

    config = AppConfig.create()
    print_platform_info(config.platform)
    console.print()

    all_engines = get_all_engines()

    from rich.table import Table

    table = Table(title="Transcription Engines", show_header=True)
    table.add_column("Engine", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Platforms")
    table.add_column("Description")

    for name, engine in all_engines.items():
        if engine.is_available():
            status = "[green]✓ Ready[/green]"
        else:
            status = "[dim]Not installed[/dim]"

        platforms = ", ".join(engine.supported_platforms)
        table.add_row(name, status, platforms, engine.description)

    console.print(table)

    # Installation hints
    console.print()
    print_info("To install an engine:")
    console.print("  [dim]uv pip install 'transcript-cli\\[faster]'[/dim]     # Faster Whisper")
    console.print(
        "  [dim]uv pip install 'transcript-cli\\[insanely]'[/dim]   # Insanely Fast Whisper"
    )
    console.print("  [dim]uv pip install 'transcript-cli\\[parakeet]'[/dim]   # NVIDIA Parakeet")
    console.print(
        "  [dim]uv pip install 'transcript-cli\\[apple]'[/dim]      # MLX Whisper (Apple Silicon)"
    )


@app.command("list-formats")
def list_formats():
    """List available output formats."""
    print_banner()

    from rich.table import Table

    table = Table(title="Output Formats", show_header=True)
    table.add_column("Format", style="cyan")
    table.add_column("Extension")
    table.add_column("Description")

    formats_info = [
        ("text", ".txt", "Plain text transcript"),
        ("timestamps", ".txt", "Text with [HH:MM:SS] timestamps per segment"),
        ("srt", ".srt", "SubRip subtitle format"),
        ("vtt", ".vtt", "WebVTT subtitle format"),
        ("json", ".json", "JSON with full metadata and word-level timestamps"),
    ]

    for fmt, ext, desc in formats_info:
        table.add_row(fmt, ext, desc)

    console.print(table)


@app.command("list-models")
def list_models():
    """List available Whisper models and their download status."""
    print_banner()

    config = AppConfig.create()
    print_platform_info(config.platform)
    console.print()

    from rich.table import Table

    # Get the best available engine to check download status
    engine = get_best_engine()

    # Model size estimates (approximate, varies by engine)
    model_sizes = {
        "tiny": "~75 MB",
        "base": "~145 MB",
        "small": "~465 MB",
        "medium": "~1.5 GB",
        "large": "~3 GB",
        "large-v2": "~3 GB",
        "large-v3": "~3 GB",
        "turbo": "~1.5 GB",
    }

    table = Table(title="Whisper Models", show_header=True)
    table.add_column("Model", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Size", justify="right")
    table.add_column("Description")

    model_descriptions = {
        "tiny": "Fastest, lowest accuracy",
        "base": "Fast, good for quick transcription",
        "small": "Balanced speed and accuracy",
        "medium": "High accuracy, slower",
        "large": "Highest accuracy (alias for large-v3)",
        "large-v2": "High accuracy, legacy model",
        "large-v3": "Best accuracy, multilingual",
        "turbo": "Fast + accurate, recommended",
    }

    for model in WHISPER_MODELS:
        if engine and engine.is_model_downloaded(model):
            status = "[green]✓ Downloaded[/green]"
        else:
            status = "[dim]Not cached[/dim]"

        size = model_sizes.get(model, "~1+ GB")
        desc = model_descriptions.get(model, "")
        table.add_row(model, status, size, desc)

    console.print(table)

    # Show engine-specific info
    if engine:
        console.print()
        print_info(f"Status shown for engine: {engine.name}")
        downloaded = engine.list_downloaded_models()
        if downloaded:
            console.print(f"  [dim]Downloaded: {', '.join(downloaded)}[/dim]")

    console.print()
    print_info("To download a model:")
    console.print("  [dim]transcript download <engine> --model <model>[/dim]")
    console.print("  [dim]Example: transcript download faster-whisper --model large-v3[/dim]")


if __name__ == "__main__":
    app()
