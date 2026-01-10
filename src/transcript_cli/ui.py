"""Rich UI components for the CLI."""

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from .config import Platform

console = Console()


def print_banner() -> None:
    """Print application banner."""
    banner = Text()
    banner.append("🎙️ ", style="bold")
    banner.append("Transcript CLI", style="bold cyan")
    banner.append(" - Multi-engine transcription tool", style="dim")
    console.print(banner)
    console.print()


def print_platform_info(platform: Platform) -> None:
    """Print detected platform information."""
    icons = {
        Platform.NVIDIA: "🟢 NVIDIA GPU",
        Platform.APPLE_SILICON: "🍎 Apple Silicon",
        Platform.CPU: "💻 CPU",
    }
    console.print(f"Platform: {icons[platform]}", style="dim")


def print_success(message: str) -> None:
    """Print success message."""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """Print error message."""
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_info(message: str) -> None:
    """Print info message."""
    console.print(f"[blue]ℹ[/blue] {message}")


def create_progress() -> Progress:
    """Create a progress bar for transcription."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def create_download_progress() -> Progress:
    """Create a progress bar for model downloads."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green"),
        TaskProgressColumn(),
        TextColumn("[dim]{task.fields[status]}"),
        console=console,
    )


def prompt_engine_choice(available_engines: list[str]) -> str:
    """Prompt user to select an engine."""
    console.print("\n[bold]Available engines:[/bold]")
    for i, engine in enumerate(available_engines, 1):
        console.print(f"  {i}. {engine}")
    console.print()

    choice = Prompt.ask(
        "Select engine",
        choices=[str(i) for i in range(1, len(available_engines) + 1)],
        default="1",
    )
    return available_engines[int(choice) - 1]


def prompt_model_choice(
    models: list[str],
    default: str = "base",
    engine=None,
) -> str:
    """Prompt user to select a model size, showing download status if engine provided."""
    if engine is not None:
        # Show which models are downloaded
        console.print("\n[bold]Available models:[/bold]")
        for model in models:
            if engine.is_model_downloaded(model):
                console.print(f"  [green]✓[/green] {model} [dim](cached)[/dim]")
            else:
                console.print(f"  [dim]○[/dim] {model}")
        console.print()
    
    return Prompt.ask(
        "Select model size",
        choices=models,
        default=default,
    )


def prompt_output_format(formats: list[str], default: str = "text") -> str:
    """Prompt user to select output format."""
    return Prompt.ask(
        "Output format",
        choices=formats,
        default=default,
    )


def prompt_batch_mode() -> str:
    """Prompt for batch processing mode."""
    return Prompt.ask(
        "Batch output mode",
        choices=["individual", "concatenate"],
        default="individual",
    )


def confirm_action(message: str, default: bool = True) -> bool:
    """Confirm an action with the user."""
    return Confirm.ask(message, default=default)


def print_file_list(files: list[Path], title: str = "Files to process") -> None:
    """Print a list of files in a table."""
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=4)
    table.add_column("File", style="cyan")
    table.add_column("Size", justify="right", style="green")

    for i, file in enumerate(files, 1):
        size = file.stat().st_size
        size_str = _format_size(size)
        table.add_row(str(i), file.name, size_str)

    console.print(table)


def print_transcription_summary(
    files_processed: int,
    total_duration: float,
    elapsed_time: float,
    output_path: Path,
) -> None:
    """Print transcription summary."""
    speed_ratio = total_duration / elapsed_time if elapsed_time > 0 else 0

    content = (
        f"[green]Files processed:[/green] {files_processed}\n"
        f"[green]Audio duration:[/green] {_format_duration(total_duration)}\n"
        f"[green]Elapsed time:[/green] {_format_duration(elapsed_time)}\n"
        f"[green]Speed:[/green] {speed_ratio:.1f}x realtime\n"
        f"[green]Output:[/green] {output_path}"
    )

    console.print(Panel(content, title="✅ Transcription Complete", border_style="green"))


def print_available_engines(engines: dict[str, Any]) -> None:
    """Print table of available engines."""
    table = Table(title="Available Engines", show_header=True)
    table.add_column("Engine", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Platform")
    table.add_column("Description")

    for name, info in engines.items():
        status = "[green]✓ Installed[/green]" if info["available"] else "[dim]Not installed[/dim]"
        table.add_row(name, status, info["platform"], info["description"])

    console.print(table)


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"
