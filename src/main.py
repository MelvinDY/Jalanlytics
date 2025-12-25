"""
Jalanlytics CLI - Street Income Intelligence Tool

Analyze CCTV footage to estimate income levels by detecting and classifying vehicles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .config import Config
from .pipeline import AnalysisPipeline

console = Console()


def print_banner() -> None:
    """Print the Jalanlytics banner."""
    banner = """
       ██╗ █████╗ ██╗      █████╗ ███╗   ██╗██╗  ██╗   ██╗████████╗██╗ ██████╗███████╗
       ██║██╔══██╗██║     ██╔══██╗████╗  ██║██║  ╚██╗ ██╔╝╚══██╔══╝██║██╔════╝██╔════╝
       ██║███████║██║     ███████║██╔██╗ ██║██║   ╚████╔╝    ██║   ██║██║     ███████╗
  ██   ██║██╔══██║██║     ██╔══██║██║╚██╗██║██║    ╚██╔╝     ██║   ██║██║     ╚════██║
  ╚█████╔╝██║  ██║███████╗██║  ██║██║ ╚████║███████╗██║      ██║   ██║╚██████╗███████║
   ╚════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝      ╚═╝   ╚═╝ ╚═════╝╚══════╝
    """
    console.print(banner, style="bold green")
    console.print("  Street Income Intelligence for Indonesian Markets\n", style="dim")


@click.group()
@click.version_option(version="0.1.0", prog_name="jalanlytics")
def cli() -> None:
    """Jalanlytics - Street Income Intelligence Tool

    Analyze CCTV footage of Indonesian streets to estimate area income levels
    by detecting, tracking, and classifying vehicles.
    """
    pass


@cli.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path for the report",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json", "csv"]),
    default="text",
    help="Output format for the report",
)
@click.option(
    "--frame-interval",
    "-i",
    type=int,
    default=10,
    help="Process every Nth frame (default: 10)",
)
@click.option(
    "--confidence",
    "-c",
    type=float,
    default=0.25,
    help="Minimum detection confidence (default: 0.25)",
)
@click.option(
    "--device",
    "-d",
    type=click.Choice(["cuda", "cpu", "auto"]),
    default="auto",
    help="Device to use for inference (default: auto)",
)
@click.option(
    "--no-commercial",
    is_flag=True,
    help="Disable commercial vehicle detection",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to YAML configuration file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def analyze(
    video_path: Path,
    output: Optional[Path],
    output_format: str,
    frame_interval: int,
    confidence: float,
    device: str,
    no_commercial: bool,
    config: Optional[Path],
    verbose: bool,
) -> None:
    """Analyze a video file to estimate area income level.

    VIDEO_PATH: Path to the video file (mp4, avi, mov)

    Example:
        jalanlytics analyze street_footage.mp4
        jalanlytics analyze video.mp4 --format json --output report.json
        jalanlytics analyze video.mp4 --frame-interval 5 --confidence 0.3
    """
    print_banner()

    # Load configuration
    if config:
        cfg = Config.from_yaml(config)
        console.print(f"[dim]Loaded configuration from {config}[/dim]\n")
    else:
        cfg = Config()

    # Override with CLI options
    cfg.video.frame_interval = frame_interval
    cfg.detection.confidence_threshold = confidence
    cfg.device = device
    cfg.commercial.enabled = not no_commercial
    cfg.report.output_format = output_format

    # Print configuration summary
    console.print(Panel.fit(
        f"[bold]Video:[/bold] {video_path.name}\n"
        f"[bold]Device:[/bold] {cfg.torch_device}\n"
        f"[bold]Frame Interval:[/bold] Every {frame_interval} frames\n"
        f"[bold]Confidence:[/bold] {confidence:.0%}\n"
        f"[bold]Commercial Detection:[/bold] {'Enabled' if cfg.commercial.enabled else 'Disabled'}",
        title="Analysis Configuration",
        border_style="blue",
    ))

    # Create and run pipeline
    try:
        pipeline = AnalysisPipeline(cfg, verbose=verbose)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            results = pipeline.analyze(video_path, progress=progress)

        # Generate and display report
        report = pipeline.generate_report(results)

        if output:
            output.write_text(report)
            console.print(f"\n[green]Report saved to {output}[/green]")
        else:
            console.print("\n" + report)

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise SystemExit(1)


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="jalanlytics.yaml",
    help="Output path for the config file",
)
def init_config(output: Path) -> None:
    """Generate a default configuration file.

    Creates a YAML configuration file with all available options
    and their default values.

    Example:
        jalanlytics init-config
        jalanlytics init-config --output my_config.yaml
    """
    config = Config()
    config.to_yaml(output)
    console.print(f"[green]Configuration file created:[/green] {output}")
    console.print("[dim]Edit this file to customize the analysis parameters.[/dim]")


@cli.command()
def check_gpu() -> None:
    """Check GPU availability and CUDA status.

    Displays information about available GPU devices and
    whether CUDA is properly configured.
    """
    import torch

    console.print("\n[bold]GPU Status Check[/bold]\n")

    if torch.cuda.is_available():
        console.print("[green]CUDA is available![/green]")
        console.print(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            console.print(f"  GPU {i}: {props.name}")
            console.print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
            console.print(f"    Compute Capability: {props.major}.{props.minor}")
    else:
        console.print("[yellow]CUDA is not available.[/yellow]")
        console.print("  Analysis will run on CPU (slower).")
        console.print("\n[dim]To enable GPU support, install PyTorch with CUDA:[/dim]")
        console.print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")


@cli.command()
def list_models() -> None:
    """List available detection and classification models.

    Shows the models that can be used for vehicle detection
    and classification.
    """
    console.print("\n[bold]Available Models[/bold]\n")

    console.print("[underline]Detection Models (YOLOv8)[/underline]")
    models = [
        ("yolov8n.pt", "Nano", "Fastest, lowest accuracy"),
        ("yolov8s.pt", "Small", "Good balance of speed/accuracy"),
        ("yolov8m.pt", "Medium", "Higher accuracy, slower"),
        ("yolov8l.pt", "Large", "High accuracy, requires more GPU memory"),
        ("yolov8x.pt", "Extra Large", "Highest accuracy, slowest"),
    ]
    for model, size, desc in models:
        console.print(f"  [cyan]{model:<15}[/cyan] ({size}) - {desc}")

    console.print("\n[underline]Classification Models (CLIP)[/underline]")
    clip_models = [
        ("openai/clip-vit-base-patch32", "Base", "Default, good balance"),
        ("openai/clip-vit-large-patch14", "Large", "Higher accuracy, slower"),
    ]
    for model, size, desc in clip_models:
        console.print(f"  [cyan]{model}[/cyan]")
        console.print(f"    ({size}) - {desc}")


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
