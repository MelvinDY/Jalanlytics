"""
Vehicle detection module using YOLOv8.

Detects vehicles (motorcycles, cars, bicycles, buses, trucks) in video frames
using the YOLOv8 object detection model.
"""

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from ultralytics import YOLO

from ..config import DetectionConfig


class VehicleClass(IntEnum):
    """COCO class IDs for vehicle types."""

    BICYCLE = 1
    CAR = 2
    MOTORCYCLE = 3
    BUS = 5
    TRUCK = 7

    @classmethod
    def from_coco_id(cls, coco_id: int) -> Optional["VehicleClass"]:
        """Convert COCO class ID to VehicleClass."""
        try:
            return cls(coco_id)
        except ValueError:
            return None

    @property
    def display_name(self) -> str:
        """Get human-readable name."""
        return self.name.capitalize()


@dataclass
class Detection:
    """Represents a single vehicle detection."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    vehicle_class: VehicleClass
    class_id: int  # Original COCO class ID

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        """Get center point of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        """Get area of the bounding box."""
        return self.width * self.height

    def to_xyxy(self) -> np.ndarray:
        """Convert to xyxy format numpy array."""
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def to_xywh(self) -> np.ndarray:
        """Convert to xywh format (center x, center y, width, height)."""
        cx, cy = self.center
        return np.array([cx, cy, self.width, self.height])


@dataclass
class DetectionResult:
    """Container for all detections in a single frame."""

    detections: list[Detection]
    frame_number: int
    inference_time_ms: float

    @property
    def motorcycle_count(self) -> int:
        return sum(1 for d in self.detections if d.vehicle_class == VehicleClass.MOTORCYCLE)

    @property
    def car_count(self) -> int:
        return sum(1 for d in self.detections if d.vehicle_class == VehicleClass.CAR)

    @property
    def bicycle_count(self) -> int:
        return sum(1 for d in self.detections if d.vehicle_class == VehicleClass.BICYCLE)

    @property
    def bus_count(self) -> int:
        return sum(1 for d in self.detections if d.vehicle_class == VehicleClass.BUS)

    @property
    def truck_count(self) -> int:
        return sum(1 for d in self.detections if d.vehicle_class == VehicleClass.TRUCK)

    @property
    def total_vehicles(self) -> int:
        return len(self.detections)

    def get_by_class(self, vehicle_class: VehicleClass) -> list[Detection]:
        """Get all detections of a specific class."""
        return [d for d in self.detections if d.vehicle_class == vehicle_class]

    def get_bboxes_xyxy(self) -> np.ndarray:
        """Get all bounding boxes in xyxy format."""
        if not self.detections:
            return np.empty((0, 4))
        return np.array([d.to_xyxy() for d in self.detections])

    def get_confidences(self) -> np.ndarray:
        """Get all confidence scores."""
        if not self.detections:
            return np.empty(0)
        return np.array([d.confidence for d in self.detections])

    def get_class_ids(self) -> np.ndarray:
        """Get all class IDs."""
        if not self.detections:
            return np.empty(0, dtype=int)
        return np.array([d.class_id for d in self.detections])


class VehicleDetector:
    """
    YOLOv8-based vehicle detector.

    Detects motorcycles, cars, bicycles, buses, and trucks in video frames.
    Optimized for Indonesian traffic conditions with high motorcycle density.
    """

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the vehicle detector.

        Args:
            config: Detection configuration. Uses defaults if not provided.
            device: PyTorch device for inference. Auto-detected if not provided.
        """
        self.config = config or DetectionConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[YOLO] = None
        self._vehicle_class_ids = set(self.config.vehicle_classes)

    def load_model(self) -> None:
        """Load the YOLOv8 model."""
        if self._model is not None:
            return

        model_path = self.config.model_name

        # Check if it's a path or a model name
        if Path(model_path).exists():
            self._model = YOLO(model_path)
        else:
            # Download from ultralytics hub
            self._model = YOLO(model_path)

        # Move to device
        self._model.to(self.device)

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    def detect(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
    ) -> DetectionResult:
        """
        Detect vehicles in a single frame.

        Args:
            frame: BGR image as numpy array.
            frame_number: Frame number for tracking.

        Returns:
            DetectionResult containing all vehicle detections.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self.is_loaded:
            self.load_model()

        import time

        start_time = time.perf_counter()

        # Run inference
        results = self._model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=list(self._vehicle_class_ids),
            verbose=False,
        )

        inference_time = (time.perf_counter() - start_time) * 1000  # ms

        # Parse results
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for bbox, conf, cls_id in zip(boxes, confidences, class_ids):
                    vehicle_class = VehicleClass.from_coco_id(cls_id)
                    if vehicle_class is not None:
                        detections.append(
                            Detection(
                                bbox=tuple(bbox),
                                confidence=float(conf),
                                vehicle_class=vehicle_class,
                                class_id=cls_id,
                            )
                        )

        return DetectionResult(
            detections=detections,
            frame_number=frame_number,
            inference_time_ms=inference_time,
        )

    def detect_batch(
        self,
        frames: list[np.ndarray],
        frame_numbers: Optional[list[int]] = None,
    ) -> list[DetectionResult]:
        """
        Detect vehicles in multiple frames (batch processing).

        Args:
            frames: List of BGR images as numpy arrays.
            frame_numbers: Optional list of frame numbers for tracking.

        Returns:
            List of DetectionResult, one per frame.
        """
        if not self.is_loaded:
            self.load_model()

        if frame_numbers is None:
            frame_numbers = list(range(len(frames)))

        import time

        start_time = time.perf_counter()

        # Run batch inference
        results = self._model(
            frames,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=list(self._vehicle_class_ids),
            verbose=False,
        )

        total_time = (time.perf_counter() - start_time) * 1000
        time_per_frame = total_time / len(frames) if frames else 0

        # Parse results for each frame
        detection_results = []
        for i, result in enumerate(results):
            detections = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for bbox, conf, cls_id in zip(boxes, confidences, class_ids):
                    vehicle_class = VehicleClass.from_coco_id(cls_id)
                    if vehicle_class is not None:
                        detections.append(
                            Detection(
                                bbox=tuple(bbox),
                                confidence=float(conf),
                                vehicle_class=vehicle_class,
                                class_id=cls_id,
                            )
                        )

            detection_results.append(
                DetectionResult(
                    detections=detections,
                    frame_number=frame_numbers[i],
                    inference_time_ms=time_per_frame,
                )
            )

        return detection_results

    def warmup(self, input_size: tuple[int, int] = (640, 640)) -> None:
        """
        Warm up the model with a dummy inference.

        Args:
            input_size: Size of the dummy input (width, height).
        """
        if not self.is_loaded:
            self.load_model()

        dummy = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        self.detect(dummy)

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_name": self.config.model_name,
            "device": str(self.device),
            "vehicle_classes": [VehicleClass(c).display_name for c in self._vehicle_class_ids],
            "confidence_threshold": self.config.confidence_threshold,
            "iou_threshold": self.config.iou_threshold,
        }
