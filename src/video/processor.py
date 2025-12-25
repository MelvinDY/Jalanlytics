"""
Video processing module for loading and sampling video frames.

Handles video file loading, frame extraction, and sampling strategies
for efficient processing of CCTV footage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from rich.progress import Progress, TaskID

from ..config import VideoConfig


@dataclass
class VideoMetadata:
    """Metadata about a video file."""

    path: Path
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str

    @property
    def duration_formatted(self) -> str:
        """Return duration in MM:SS format."""
        minutes = int(self.duration_seconds // 60)
        seconds = int(self.duration_seconds % 60)
        return f"{minutes}:{seconds:02d}"


@dataclass
class Frame:
    """Container for a video frame with metadata."""

    image: np.ndarray
    frame_number: int
    timestamp_seconds: float

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]


class VideoProcessor:
    """
    Handles video loading and frame sampling.

    Supports various video formats (mp4, avi, mov) and provides
    efficient frame sampling for processing CCTV footage.
    """

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

    def __init__(self, config: Optional[VideoConfig] = None):
        """
        Initialize the video processor.

        Args:
            config: Video processing configuration. Uses defaults if not provided.
        """
        self.config = config or VideoConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

    def load(self, video_path: Path | str) -> VideoMetadata:
        """
        Load a video file and extract metadata.

        Args:
            video_path: Path to the video file.

        Returns:
            VideoMetadata object with video information.

        Raises:
            FileNotFoundError: If the video file doesn't exist.
            ValueError: If the file format is not supported or cannot be opened.
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if video_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format: {video_path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        # Release any previously opened video
        self.release()

        # Open video capture
        self._cap = cv2.VideoCapture(str(video_path))

        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        # Extract metadata
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))

        # Decode FourCC to string
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        self._metadata = VideoMetadata(
            path=video_path,
            width=width,
            height=height,
            fps=fps if fps > 0 else 30.0,  # Default to 30 FPS if unknown
            total_frames=total_frames,
            duration_seconds=total_frames / fps if fps > 0 else 0,
            codec=codec,
        )

        return self._metadata

    @property
    def metadata(self) -> Optional[VideoMetadata]:
        """Get the metadata of the currently loaded video."""
        return self._metadata

    @property
    def is_loaded(self) -> bool:
        """Check if a video is currently loaded."""
        return self._cap is not None and self._cap.isOpened()

    def get_frames_to_process(self) -> int:
        """
        Calculate the number of frames that will be processed.

        Returns:
            Number of frames to process based on sampling interval.
        """
        if not self._metadata:
            return 0

        total = self._metadata.total_frames
        interval = self.config.frame_interval

        if self.config.max_frames:
            total = min(total, self.config.max_frames * interval)

        return (total + interval - 1) // interval

    def frames(
        self,
        progress: Optional[Progress] = None,
        task_id: Optional[TaskID] = None,
    ) -> Iterator[Frame]:
        """
        Iterate over sampled frames from the video.

        Yields frames according to the configured sampling interval.

        Args:
            progress: Optional Rich Progress instance for progress tracking.
            task_id: Optional task ID for progress updates.

        Yields:
            Frame objects containing the image and metadata.

        Raises:
            RuntimeError: If no video is loaded.
        """
        if not self.is_loaded or not self._cap or not self._metadata:
            raise RuntimeError("No video loaded. Call load() first.")

        frame_interval = self.config.frame_interval
        max_frames = self.config.max_frames
        fps = self._metadata.fps

        frame_count = 0
        processed_count = 0

        # Reset to beginning
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, image = self._cap.read()

            if not ret:
                break

            # Check if we should process this frame
            if frame_count % frame_interval == 0:
                # Resize if configured
                if self.config.resize_width or self.config.resize_height:
                    image = self._resize_frame(image)

                timestamp = frame_count / fps

                yield Frame(
                    image=image,
                    frame_number=frame_count,
                    timestamp_seconds=timestamp,
                )

                processed_count += 1

                if progress and task_id is not None:
                    progress.update(task_id, advance=1)

                # Check max frames limit
                if max_frames and processed_count >= max_frames:
                    break

            frame_count += 1

    def _resize_frame(self, image: np.ndarray) -> np.ndarray:
        """Resize a frame according to configuration."""
        if not self.config.resize_width and not self.config.resize_height:
            return image

        height, width = image.shape[:2]

        if self.config.resize_width and self.config.resize_height:
            new_width = self.config.resize_width
            new_height = self.config.resize_height
        elif self.config.resize_width:
            new_width = self.config.resize_width
            new_height = int(height * (new_width / width))
        else:
            new_height = self.config.resize_height
            new_width = int(width * (new_height / height))

        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def get_frame_at(self, frame_number: int) -> Optional[Frame]:
        """
        Get a specific frame by frame number.

        Args:
            frame_number: The frame number to retrieve (0-indexed).

        Returns:
            Frame object or None if frame cannot be retrieved.
        """
        if not self.is_loaded or not self._cap or not self._metadata:
            return None

        if frame_number < 0 or frame_number >= self._metadata.total_frames:
            return None

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, image = self._cap.read()

        if not ret:
            return None

        if self.config.resize_width or self.config.resize_height:
            image = self._resize_frame(image)

        return Frame(
            image=image,
            frame_number=frame_number,
            timestamp_seconds=frame_number / self._metadata.fps,
        )

    def extract_region(self, frame: Frame, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract a region from a frame.

        Args:
            frame: The frame to extract from.
            bbox: Bounding box as (x1, y1, x2, y2).

        Returns:
            Cropped image array.
        """
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(frame.width, int(x2))
        y2 = min(frame.height, int(y2))
        return frame.image[y1:y2, x1:x2]

    def release(self) -> None:
        """Release the video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._metadata = None

    def __enter__(self) -> "VideoProcessor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()
