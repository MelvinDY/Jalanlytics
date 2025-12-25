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
    default=0.05,
    help="Minimum detection confidence (default: 0.05, optimized for CCTV)",
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


@cli.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=Path))
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
    "--speed",
    "-s",
    type=float,
    default=1.0,
    help="Playback speed multiplier (default: 1.0)",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]),
    default="yolov8n.pt",
    help="YOLO model to use (larger = better small object detection)",
)
@click.option(
    "--imgsz",
    type=int,
    default=640,
    help="Input image size for YOLO (larger = better small object detection, default: 640)",
)
@click.option(
    "--iou",
    type=float,
    default=0.45,
    help="NMS IOU threshold (lower = more detections in clusters, default: 0.45)",
)
def debug(
    video_path: Path,
    confidence: float,
    device: str,
    speed: float,
    model: str,
    imgsz: int,
    iou: float,
) -> None:
    """Run detection with real-time visualization for debugging.

    Shows bounding boxes, track IDs, and detection info live.
    Press 'q' to quit, 'p' to pause, 'n' for next frame when paused.

    VIDEO_PATH: Path to the video file (mp4, avi, mov)

    Example:
        jalanlytics debug street_footage.mp4
        jalanlytics debug video.mp4 --confidence 0.3 --speed 0.5
    """
    import cv2
    import time

    from .config import Config
    from .detection.detector import VehicleDetector, VehicleClass
    from .tracking.tracker import VehicleTracker

    print_banner()

    # Load configuration
    cfg = Config()
    cfg.detection.confidence_threshold = confidence
    cfg.device = device

    console.print(Panel.fit(
        f"[bold]Video:[/bold] {video_path.name}\n"
        f"[bold]Device:[/bold] {cfg.torch_device}\n"
        f"[bold]Model:[/bold] {model}\n"
        f"[bold]Image Size:[/bold] {imgsz}px\n"
        f"[bold]Confidence:[/bold] {confidence:.0%}\n"
        f"[bold]IOU (NMS):[/bold] {iou:.0%}\n"
        f"[bold]Speed:[/bold] {speed}x\n\n"
        "[dim]Controls: q=quit, p=pause, n=next frame (when paused)[/dim]",
        title="Debug Mode",
        border_style="yellow",
    ))

    # Initialize components with custom model
    cfg.detection.model_name = model
    cfg.detection.iou_threshold = iou  # Lower = more detections in clusters
    cfg.tracking.track_thresh = confidence  # Match tracker threshold to detection confidence
    detector = VehicleDetector(cfg.detection, cfg.torch_device)
    tracker = VehicleTracker(cfg.tracking)
    detector.load_model()
    detector.warmup()

    # Store imgsz for inference
    inference_imgsz = imgsz

    # Colors for different vehicle classes (BGR)
    # Note: Trucks are merged into cars (SUVs often misclassified as trucks)
    colors = {
        VehicleClass.MOTORCYCLE: (0, 255, 255),  # Yellow
        VehicleClass.CAR: (0, 255, 0),           # Green (includes SUVs/trucks)
        VehicleClass.BICYCLE: (255, 255, 0),     # Cyan
        VehicleClass.BUS: (0, 165, 255),         # Orange
    }

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        console.print(f"[red]Error: Could not open video {video_path}[/red]")
        raise SystemExit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_delay = int(1000 / (fps * speed)) if fps > 0 else 33

    console.print(f"\n[cyan]Processing {total_frames} frames at {fps:.1f} FPS...[/cyan]")
    console.print("[dim]Opening visualization window...[/dim]\n")

    frame_num = 0
    paused = False
    total_detections = 0

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_num += 1

                # Detect vehicles
                result = detector.detect(frame, frame_num, imgsz=inference_imgsz)
                total_detections += len(result.detections)

                # Update tracker
                tracking_result = tracker.update(result)

                # Draw detections and tracking info
                for track_id, detection in tracking_result.active_tracks:
                    x1, y1, x2, y2 = map(int, detection.bbox)
                    color = colors.get(detection.vehicle_class, (255, 255, 255))

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label background
                    label = f"#{track_id} {detection.vehicle_class.display_name} {detection.confidence:.0%}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), color, -1)

                    # Draw label text
                    cv2.putText(frame, label, (x1 + 2, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Draw stats overlay
                stats = [
                    f"Frame: {frame_num}/{total_frames}",
                    f"Detections: {len(result.detections)}",
                    f"Tracked: {len(tracker.get_all_tracked_vehicles())}",
                    f"Inference: {result.inference_time_ms:.1f}ms",
                ]
                for i, stat in enumerate(stats):
                    cv2.putText(frame, stat, (10, 25 + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, stat, (10, 25 + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                # Draw legend
                y_offset = frame.shape[0] - 120
                for vc, color in colors.items():
                    cv2.rectangle(frame, (10, y_offset), (25, y_offset + 15), color, -1)
                    cv2.putText(frame, vc.display_name, (30, y_offset + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 20

            # Show frame
            cv2.imshow("Jalanlytics Debug - Press Q to quit", frame)

            # Handle key presses
            key = cv2.waitKey(frame_delay if not paused else 0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                status = "PAUSED" if paused else "PLAYING"
                console.print(f"[yellow]{status}[/yellow]")
            elif key == ord('n') and paused:
                paused = False  # Process next frame
                ret, frame = cap.read()
                if ret:
                    frame_num += 1
                    result = detector.detect(frame, frame_num, imgsz=inference_imgsz)
                    tracking_result = tracker.update(result)
                    # Redraw with detections (same code as above)
                    for track_id, detection in tracking_result.active_tracks:
                        x1, y1, x2, y2 = map(int, detection.bbox)
                        color = colors.get(detection.vehicle_class, (255, 255, 255))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"#{track_id} {detection.vehicle_class.display_name} {detection.confidence:.0%}"
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), color, -1)
                        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                paused = True

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Print summary
    tracked_vehicles = tracker.get_all_tracked_vehicles()
    console.print(f"\n[green]Debug session complete![/green]")
    console.print(f"  Frames processed: {frame_num}")
    console.print(f"  Total detections: {total_detections}")
    console.print(f"  Unique vehicles tracked: {len(tracked_vehicles)}")

    # Breakdown by class
    class_counts = {}
    for v in tracked_vehicles.values():
        name = v.vehicle_class.display_name
        class_counts[name] = class_counts.get(name, 0) + 1
    for cls, count in sorted(class_counts.items()):
        console.print(f"    {cls}: {count}")


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
