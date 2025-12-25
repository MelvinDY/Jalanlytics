"""
Configuration management for Jalanlytics.

Handles all configurable parameters for the analysis pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import yaml


@dataclass
class VideoConfig:
    """Configuration for video processing."""

    frame_interval: int = 10  # Process every Nth frame
    max_frames: int | None = None  # Maximum frames to process (None = all)
    resize_width: int | None = None  # Resize frame width (None = original)
    resize_height: int | None = None  # Resize frame height (None = original)


@dataclass
class DetectionConfig:
    """Configuration for vehicle detection."""

    model_name: str = "yolov8s.pt"  # YOLOv8 model variant (n/s/m/l/x) - 's' better for small objects
    confidence_threshold: float = 0.05  # Minimum detection confidence (low for motorcycles from CCTV)
    iou_threshold: float = 0.2  # NMS IoU threshold (low for motorcycle clusters)
    imgsz: int = 1280  # Input image size (larger = better small object detection)
    # COCO class IDs for vehicles
    vehicle_classes: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 7])
    # 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck (trucks merged to cars in detector)


@dataclass
class TrackingConfig:
    """Configuration for vehicle tracking."""

    track_thresh: float = 0.05  # Detection threshold for tracking (must match detection confidence)
    track_buffer: int = 30  # Frames to keep lost tracks
    match_thresh: float = 0.8  # Threshold for matching detections to tracks
    frame_rate: int = 30  # Expected video frame rate


@dataclass
class ClassificationConfig:
    """Configuration for vehicle classification."""

    model_name: str = "openai/clip-vit-base-patch32"  # CLIP model
    top_k: int = 3  # Number of top predictions to return
    min_confidence: float = 0.1  # Minimum confidence for classification
    batch_size: int = 16  # Batch size for classification


@dataclass
class CommercialConfig:
    """Configuration for commercial vehicle detection."""

    # Ojol (ride-hailing) detection colors (HSV ranges)
    gojek_green_lower: tuple[int, int, int] = (35, 100, 100)
    gojek_green_upper: tuple[int, int, int] = (85, 255, 255)
    grab_green_lower: tuple[int, int, int] = (35, 100, 100)
    grab_green_upper: tuple[int, int, int] = (85, 255, 255)
    shopee_orange_lower: tuple[int, int, int] = (5, 100, 100)
    shopee_orange_upper: tuple[int, int, int] = (25, 255, 255)

    # Minimum percentage of color pixels to classify as commercial
    color_ratio_threshold: float = 0.15

    # Enable/disable commercial detection
    enabled: bool = True


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    output_format: Literal["text", "json", "csv"] = "text"
    include_timestamps: bool = True
    include_confidence_scores: bool = True
    currency_symbol: str = "Rp"
    locale: str = "id_ID"


@dataclass
class Config:
    """Main configuration container."""

    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    commercial: CommercialConfig = field(default_factory=CommercialConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    # Device configuration
    device: Literal["cuda", "cpu", "auto"] = "auto"

    @property
    def torch_device(self) -> torch.device:
        """Get the PyTorch device to use."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            video=VideoConfig(**data.get("video", {})),
            detection=DetectionConfig(**data.get("detection", {})),
            tracking=TrackingConfig(**data.get("tracking", {})),
            classification=ClassificationConfig(**data.get("classification", {})),
            commercial=CommercialConfig(**data.get("commercial", {})),
            report=ReportConfig(**data.get("report", {})),
            device=data.get("device", "auto"),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        from dataclasses import asdict

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "video": asdict(self.video),
            "detection": asdict(self.detection),
            "tracking": asdict(self.tracking),
            "classification": asdict(self.classification),
            "commercial": asdict(self.commercial),
            "report": asdict(self.report),
            "device": self.device,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Default configuration instance
DEFAULT_CONFIG = Config()
