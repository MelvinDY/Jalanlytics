"""
Base classifier interface for vehicle make/model classification.

Provides an abstract interface that allows different classification
approaches (CLIP, custom CNN, etc.) to be used interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..detection.detector import VehicleClass


@dataclass
class ClassificationResult:
    """Result of vehicle make/model classification."""

    predictions: list[tuple[str, float]]  # (label, confidence) pairs
    vehicle_class: VehicleClass
    processing_time_ms: float

    @property
    def top_prediction(self) -> Optional[tuple[str, float]]:
        """Get the top prediction."""
        return self.predictions[0] if self.predictions else None

    @property
    def top_label(self) -> Optional[str]:
        """Get the label of the top prediction."""
        if self.predictions:
            return self.predictions[0][0]
        return None

    @property
    def top_confidence(self) -> float:
        """Get the confidence of the top prediction."""
        if self.predictions:
            return self.predictions[0][1]
        return 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "predictions": [
                {"label": label, "confidence": conf}
                for label, conf in self.predictions
            ],
            "vehicle_class": self.vehicle_class.display_name,
            "top_label": self.top_label,
            "top_confidence": self.top_confidence,
            "processing_time_ms": self.processing_time_ms,
        }


class BaseClassifier(ABC):
    """
    Abstract base class for vehicle classifiers.

    This interface allows for pluggable classification backends:
    - CLIP-based zero-shot classification
    - Custom CNN-based classifiers
    - Ensemble methods combining multiple approaches

    Implementations should be able to classify:
    - Motorcycle makes/models (Honda, Yamaha, etc.)
    - Car makes/models (Toyota, Wuling, etc.)
    """

    @abstractmethod
    def classify(
        self,
        image: np.ndarray,
        vehicle_class: VehicleClass,
    ) -> ClassificationResult:
        """
        Classify a single vehicle image.

        Args:
            image: Cropped vehicle image as BGR numpy array.
            vehicle_class: The type of vehicle (motorcycle, car, etc.)

        Returns:
            ClassificationResult with top-k predictions.
        """
        pass

    @abstractmethod
    def classify_batch(
        self,
        images: list[np.ndarray],
        vehicle_classes: list[VehicleClass],
    ) -> list[ClassificationResult]:
        """
        Classify multiple vehicle images.

        Args:
            images: List of cropped vehicle images.
            vehicle_classes: List of vehicle types.

        Returns:
            List of ClassificationResult, one per image.
        """
        pass

    @abstractmethod
    def get_supported_labels(self, vehicle_class: VehicleClass) -> list[str]:
        """
        Get the list of labels this classifier can predict.

        Args:
            vehicle_class: The vehicle type to get labels for.

        Returns:
            List of label strings.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this classifier."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the classifier model is loaded."""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """Load the classification model."""
        pass
