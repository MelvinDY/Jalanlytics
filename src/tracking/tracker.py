"""
Vehicle tracking module using ByteTrack.

Tracks unique vehicles across video frames to prevent double-counting
and enable vehicle-level analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import supervision as sv

from ..config import TrackingConfig
from ..detection.detector import Detection, DetectionResult, VehicleClass


@dataclass
class TrackedVehicle:
    """Represents a tracked vehicle with its history."""

    track_id: int
    vehicle_class: VehicleClass
    first_seen_frame: int
    last_seen_frame: int
    detections: list[Detection] = field(default_factory=list)
    is_commercial: bool = False
    classification: Optional[dict] = None  # Make/model classification

    @property
    def frame_count(self) -> int:
        """Number of frames this vehicle was tracked."""
        return len(self.detections)

    @property
    def best_detection(self) -> Optional[Detection]:
        """Get the detection with highest confidence."""
        if not self.detections:
            return None
        return max(self.detections, key=lambda d: d.confidence)

    @property
    def average_confidence(self) -> float:
        """Get average confidence across all detections."""
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)

    @property
    def average_area(self) -> float:
        """Get average bounding box area."""
        if not self.detections:
            return 0.0
        return sum(d.area for d in self.detections) / len(self.detections)

    def add_detection(self, detection: Detection, frame_number: int) -> None:
        """Add a new detection to this track."""
        self.detections.append(detection)
        self.last_seen_frame = frame_number


@dataclass
class TrackingResult:
    """Container for tracking results from a single frame."""

    frame_number: int
    active_tracks: list[tuple[int, Detection]]  # (track_id, detection) pairs
    new_tracks: list[int]  # Track IDs that appeared in this frame
    lost_tracks: list[int]  # Track IDs that were lost in this frame


class VehicleTracker:
    """
    ByteTrack-based vehicle tracker.

    Tracks unique vehicles across frames to:
    - Prevent double-counting of the same vehicle
    - Enable per-vehicle analysis (classification, commercial detection)
    - Provide stable tracking through occlusions
    """

    def __init__(self, config: Optional[TrackingConfig] = None):
        """
        Initialize the vehicle tracker.

        Args:
            config: Tracking configuration. Uses defaults if not provided.
        """
        self.config = config or TrackingConfig()

        # Initialize ByteTrack from supervision
        self._tracker = sv.ByteTrack(
            track_activation_threshold=self.config.track_thresh,
            lost_track_buffer=self.config.track_buffer,
            minimum_matching_threshold=self.config.match_thresh,
            frame_rate=self.config.frame_rate,
        )

        # Storage for tracked vehicles
        self._tracked_vehicles: dict[int, TrackedVehicle] = {}
        self._active_track_ids: set[int] = set()
        self._previous_track_ids: set[int] = set()
        self._current_frame: int = 0

    def reset(self) -> None:
        """Reset the tracker state."""
        self._tracker.reset()
        self._tracked_vehicles.clear()
        self._active_track_ids.clear()
        self._previous_track_ids.clear()
        self._current_frame = 0

    def update(self, detection_result: DetectionResult) -> TrackingResult:
        """
        Update tracker with new detections.

        Args:
            detection_result: Detection result from the detector.

        Returns:
            TrackingResult with tracking information.
        """
        self._current_frame = detection_result.frame_number
        self._previous_track_ids = self._active_track_ids.copy()

        if not detection_result.detections:
            # No detections - all active tracks become lost
            lost_tracks = list(self._active_track_ids)
            self._active_track_ids.clear()
            return TrackingResult(
                frame_number=self._current_frame,
                active_tracks=[],
                new_tracks=[],
                lost_tracks=lost_tracks,
            )

        # Convert detections to supervision format
        xyxy = detection_result.get_bboxes_xyxy()
        confidence = detection_result.get_confidences()
        class_ids = detection_result.get_class_ids()

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_ids,
        )

        # Run ByteTrack
        tracked_detections = self._tracker.update_with_detections(sv_detections)

        # Process tracking results
        active_tracks = []
        current_track_ids = set()

        if tracked_detections.tracker_id is not None:
            for i, track_id in enumerate(tracked_detections.tracker_id):
                track_id = int(track_id)
                current_track_ids.add(track_id)

                # Get the detection data
                bbox = tuple(tracked_detections.xyxy[i])
                conf = float(tracked_detections.confidence[i])
                cls_id = int(tracked_detections.class_id[i])
                vehicle_class = VehicleClass.from_coco_id(cls_id)

                if vehicle_class is None:
                    continue

                detection = Detection(
                    bbox=bbox,
                    confidence=conf,
                    vehicle_class=vehicle_class,
                    class_id=cls_id,
                )

                active_tracks.append((track_id, detection))

                # Update or create tracked vehicle
                if track_id not in self._tracked_vehicles:
                    self._tracked_vehicles[track_id] = TrackedVehicle(
                        track_id=track_id,
                        vehicle_class=vehicle_class,
                        first_seen_frame=self._current_frame,
                        last_seen_frame=self._current_frame,
                        detections=[detection],
                    )
                else:
                    self._tracked_vehicles[track_id].add_detection(
                        detection, self._current_frame
                    )

        # Determine new and lost tracks
        new_tracks = list(current_track_ids - self._previous_track_ids)
        lost_tracks = list(self._previous_track_ids - current_track_ids)

        self._active_track_ids = current_track_ids

        return TrackingResult(
            frame_number=self._current_frame,
            active_tracks=active_tracks,
            new_tracks=new_tracks,
            lost_tracks=lost_tracks,
        )

    def get_tracked_vehicle(self, track_id: int) -> Optional[TrackedVehicle]:
        """Get a tracked vehicle by ID."""
        return self._tracked_vehicles.get(track_id)

    def get_all_tracked_vehicles(self) -> dict[int, TrackedVehicle]:
        """Get all tracked vehicles."""
        return self._tracked_vehicles.copy()

    def get_active_vehicles(self) -> list[TrackedVehicle]:
        """Get currently active tracked vehicles."""
        return [
            self._tracked_vehicles[tid]
            for tid in self._active_track_ids
            if tid in self._tracked_vehicles
        ]

    def get_unique_vehicle_count(self) -> int:
        """Get the total number of unique vehicles tracked."""
        return len(self._tracked_vehicles)

    def get_vehicle_counts_by_class(self) -> dict[VehicleClass, int]:
        """Get unique vehicle counts by class."""
        counts: dict[VehicleClass, int] = {}
        for vehicle in self._tracked_vehicles.values():
            counts[vehicle.vehicle_class] = counts.get(vehicle.vehicle_class, 0) + 1
        return counts

    def get_statistics(self) -> dict:
        """Get tracking statistics."""
        class_counts = self.get_vehicle_counts_by_class()

        return {
            "total_unique_vehicles": len(self._tracked_vehicles),
            "currently_active": len(self._active_track_ids),
            "motorcycles": class_counts.get(VehicleClass.MOTORCYCLE, 0),
            "cars": class_counts.get(VehicleClass.CAR, 0),
            "bicycles": class_counts.get(VehicleClass.BICYCLE, 0),
            "buses": class_counts.get(VehicleClass.BUS, 0),
            "trucks": class_counts.get(VehicleClass.TRUCK, 0),
            "frames_processed": self._current_frame + 1,
        }

    def mark_as_commercial(self, track_id: int) -> None:
        """Mark a tracked vehicle as commercial (ojol, angkot, etc.)."""
        if track_id in self._tracked_vehicles:
            self._tracked_vehicles[track_id].is_commercial = True

    def set_classification(self, track_id: int, classification: dict) -> None:
        """
        Set the classification result for a tracked vehicle.

        Args:
            track_id: The track ID.
            classification: Dictionary with make, model, and confidence.
        """
        if track_id in self._tracked_vehicles:
            self._tracked_vehicles[track_id].classification = classification

    def get_commercial_vehicles(self) -> list[TrackedVehicle]:
        """Get all tracked vehicles marked as commercial."""
        return [v for v in self._tracked_vehicles.values() if v.is_commercial]

    def get_non_commercial_vehicles(self) -> list[TrackedVehicle]:
        """Get all tracked vehicles not marked as commercial."""
        return [v for v in self._tracked_vehicles.values() if not v.is_commercial]

    def filter_by_min_frames(self, min_frames: int = 3) -> list[TrackedVehicle]:
        """
        Get vehicles that were tracked for at least a minimum number of frames.

        This helps filter out false positives and transient detections.

        Args:
            min_frames: Minimum number of frames a vehicle must appear in.

        Returns:
            List of tracked vehicles meeting the criteria.
        """
        return [
            v for v in self._tracked_vehicles.values()
            if v.frame_count >= min_frames
        ]

    def get_best_crops_for_classification(
        self,
        min_confidence: float = 0.5,
        min_area: float = 5000,
    ) -> dict[int, Detection]:
        """
        Get the best detection for each vehicle for classification.

        Selects detections with high confidence and large bounding boxes
        for better classification accuracy.

        Args:
            min_confidence: Minimum confidence threshold.
            min_area: Minimum bounding box area in pixels.

        Returns:
            Dictionary mapping track_id to best Detection.
        """
        best_crops = {}
        for track_id, vehicle in self._tracked_vehicles.items():
            # Find the best detection
            best = None
            best_score = 0

            for detection in vehicle.detections:
                if detection.confidence < min_confidence:
                    continue
                if detection.area < min_area:
                    continue

                # Score based on confidence and area
                score = detection.confidence * np.sqrt(detection.area)
                if score > best_score:
                    best = detection
                    best_score = score

            if best is not None:
                best_crops[track_id] = best

        return best_crops
