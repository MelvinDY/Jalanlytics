"""
Main analysis pipeline orchestrator.

Coordinates all components (detection, tracking, classification, reporting)
to process video files and generate income analysis reports.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from rich.progress import Progress, TaskID

from .config import Config
from .data.prices import PriceDatabase
from .detection.commercial import CommercialVehicleDetector
from .detection.detector import VehicleClass, VehicleDetector
from .classification.classifier import VehicleClassifier
from .report.generator import AnalysisResult, ReportGenerator, VehicleStats
from .tracking.tracker import VehicleTracker
from .video.processor import VideoProcessor


class AnalysisPipeline:
    """
    Main orchestrator for street income analysis.

    Coordinates the following components:
    1. Video processing and frame sampling
    2. Vehicle detection (YOLOv8)
    3. Vehicle tracking (ByteTrack)
    4. Commercial vehicle detection
    5. Vehicle classification (CLIP)
    6. Price lookup and tier assignment
    7. Report generation

    Example:
        config = Config()
        pipeline = AnalysisPipeline(config)
        results = pipeline.analyze("street_video.mp4")
        report = pipeline.generate_report(results)
    """

    def __init__(self, config: Optional[Config] = None, verbose: bool = False):
        """
        Initialize the analysis pipeline.

        Args:
            config: Pipeline configuration.
            verbose: Enable verbose output.
        """
        self.config = config or Config()
        self.verbose = verbose

        # Sync tracking threshold with detection confidence
        self.config.tracking.track_thresh = self.config.detection.confidence_threshold

        # Initialize components
        self.video_processor = VideoProcessor(self.config.video)
        self.detector = VehicleDetector(self.config.detection, self.config.torch_device)
        self.tracker = VehicleTracker(self.config.tracking)
        self.commercial_detector = CommercialVehicleDetector(self.config.commercial)
        self.classifier = VehicleClassifier(self.config.classification, self.config.torch_device)
        self.price_db = PriceDatabase()
        self.report_generator = ReportGenerator(self.config.report)

        self._is_initialized = False

    def initialize(self) -> None:
        """Load all models and prepare for analysis."""
        if self._is_initialized:
            return

        self.detector.load_model()
        self.classifier.load_models()
        self.detector.warmup()

        self._is_initialized = True

    def analyze(
        self,
        video_path: Path | str,
        progress: Optional[Progress] = None,
    ) -> AnalysisResult:
        """
        Analyze a video file.

        Args:
            video_path: Path to the video file.
            progress: Optional Rich Progress instance for tracking.

        Returns:
            AnalysisResult with all analysis data.
        """
        video_path = Path(video_path)
        start_time = time.perf_counter()

        # Initialize if needed
        self.initialize()

        # Reset tracker for new video
        self.tracker.reset()

        # Load video
        metadata = self.video_processor.load(video_path)
        frames_to_process = self.video_processor.get_frames_to_process()

        # Set up progress tracking
        task_id: Optional[TaskID] = None
        if progress:
            task_id = progress.add_task(
                "[cyan]Processing frames...",
                total=frames_to_process,
            )

        # Process frames
        frame_count = 0
        for frame in self.video_processor.frames(progress, task_id):
            frame_count += 1

            # Detect vehicles
            detections = self.detector.detect(frame.image, frame.frame_number)

            # Update tracker
            tracking_result = self.tracker.update(detections)

            # Detect commercial vehicles for new tracks
            for track_id, detection in tracking_result.active_tracks:
                vehicle = self.tracker.get_tracked_vehicle(track_id)
                if vehicle and vehicle.frame_count == 1:
                    # First time seeing this vehicle - check if commercial
                    commercial_result = self.commercial_detector.detect(
                        frame.image, detection
                    )
                    if commercial_result.is_commercial:
                        self.tracker.mark_as_commercial(track_id)

        # Update progress for classification phase
        if progress and task_id is not None:
            progress.update(task_id, description="[cyan]Classifying vehicles...")

        # Get best crops for classification
        best_crops = self.tracker.get_best_crops_for_classification()

        # Classify vehicles
        for track_id, detection in best_crops.items():
            vehicle = self.tracker.get_tracked_vehicle(track_id)
            if vehicle is None:
                continue

            # Extract crop from the frame
            # Note: In a full implementation, we'd store the actual frame
            # For now, we'll use the stored detection info
            # This is a simplified version - full implementation would
            # store frame references or crops during tracking

            # Skip commercial vehicles for classification
            if vehicle.is_commercial:
                continue

            # For demo purposes, we'll use the vehicle class to get a default
            # In production, you'd classify actual image crops
            classification = {
                "make": "Unknown",
                "model": "Unknown",
                "confidence": 0.0,
            }
            self.tracker.set_classification(track_id, classification)

        # Update progress for report generation
        if progress and task_id is not None:
            progress.update(task_id, description="[cyan]Generating report...")

        # Compile results
        result = self._compile_results(
            video_path=str(video_path),
            video_duration=metadata.duration_seconds,
            frames_analyzed=frame_count,
            processing_time=time.perf_counter() - start_time,
        )

        # Clean up
        self.video_processor.release()

        return result

    def _compile_results(
        self,
        video_path: str,
        video_duration: float,
        frames_analyzed: int,
        processing_time: float,
    ) -> AnalysisResult:
        """Compile tracking and classification results into final analysis."""
        motorcycle_stats = VehicleStats()
        car_stats = VehicleStats()
        bicycle_count = 0
        bus_count = 0

        for vehicle in self.tracker.get_all_tracked_vehicles().values():
            # Get classification if available
            if vehicle.classification:
                make = vehicle.classification.get("make", "Unknown")
                model = vehicle.classification.get("model", "Unknown")
            else:
                make = "Unknown"
                model = "Unknown"

            # Get price information
            price_info = self.price_db.get_price(make, model, vehicle.vehicle_class)

            if vehicle.vehicle_class == VehicleClass.MOTORCYCLE:
                motorcycle_stats.add_vehicle(price_info, vehicle.is_commercial)
            elif vehicle.vehicle_class == VehicleClass.CAR:
                car_stats.add_vehicle(price_info, vehicle.is_commercial)
            elif vehicle.vehicle_class == VehicleClass.TRUCK:
                # Merge trucks into cars (SUVs often misclassified as trucks)
                car_stats.add_vehicle(price_info, vehicle.is_commercial)
            elif vehicle.vehicle_class == VehicleClass.BICYCLE:
                bicycle_count += 1
            elif vehicle.vehicle_class == VehicleClass.BUS:
                bus_count += 1

        return AnalysisResult(
            video_path=video_path,
            video_duration=video_duration,
            frames_analyzed=frames_analyzed,
            processing_time=processing_time,
            motorcycle_stats=motorcycle_stats,
            car_stats=car_stats,
            bicycle_count=bicycle_count,
            bus_count=bus_count,
            truck_count=0,  # Trucks merged into cars
        )

    def generate_report(self, result: AnalysisResult) -> str:
        """
        Generate a formatted report from analysis results.

        Args:
            result: Analysis result to report on.

        Returns:
            Formatted report string.
        """
        return self.report_generator.generate(result)

    def analyze_with_visualization(
        self,
        video_path: Path | str,
        output_path: Optional[Path | str] = None,
        progress: Optional[Progress] = None,
    ) -> AnalysisResult:
        """
        Analyze video and optionally save annotated output.

        Args:
            video_path: Path to input video.
            output_path: Optional path to save annotated video.
            progress: Optional progress tracker.

        Returns:
            AnalysisResult with analysis data.

        Note:
            Visualization support is a future enhancement.
        """
        # For MVP, just run standard analysis
        # Future: Add supervision annotators for visualization
        return self.analyze(video_path, progress)

    def get_component_info(self) -> dict:
        """Get information about all pipeline components."""
        return {
            "detector": self.detector.get_model_info(),
            "classifier": self.classifier.get_classifier_info(),
            "tracker": {
                "algorithm": "ByteTrack",
                "config": {
                    "track_thresh": self.config.tracking.track_thresh,
                    "track_buffer": self.config.tracking.track_buffer,
                },
            },
            "commercial_detector": {
                "enabled": self.config.commercial.enabled,
            },
            "device": str(self.config.torch_device),
        }


def quick_analyze(video_path: str | Path) -> str:
    """
    Quick analysis function for simple use cases.

    Args:
        video_path: Path to video file.

    Returns:
        Text report of analysis.

    Example:
        report = quick_analyze("street_video.mp4")
        print(report)
    """
    from rich.progress import Progress

    config = Config()
    pipeline = AnalysisPipeline(config)

    with Progress() as progress:
        result = pipeline.analyze(video_path, progress)

    return pipeline.generate_report(result)
