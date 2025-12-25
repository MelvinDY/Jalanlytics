"""
Main vehicle classifier with ensemble support.

Provides a unified interface for vehicle classification,
supporting multiple classification backends and ensemble methods.
"""

from typing import Optional

import numpy as np
import torch

from ..config import ClassificationConfig
from ..detection.detector import VehicleClass
from .base import BaseClassifier, ClassificationResult
from .clip_classifier import CLIPClassifier


class VehicleClassifier:
    """
    Main vehicle classifier for the Jalanlytics pipeline.

    Provides:
    - Unified interface for classification
    - Support for multiple backends (CLIP, custom models)
    - Ensemble voting for improved accuracy
    - Make/model extraction utilities
    """

    def __init__(
        self,
        config: Optional[ClassificationConfig] = None,
        device: Optional[torch.device] = None,
        classifiers: Optional[list[BaseClassifier]] = None,
    ):
        """
        Initialize the vehicle classifier.

        Args:
            config: Classification configuration.
            device: PyTorch device for inference.
            classifiers: List of classifiers to use. If None, uses CLIP.
        """
        self.config = config or ClassificationConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize classifiers
        if classifiers:
            self._classifiers = classifiers
        else:
            # Default to CLIP classifier
            self._classifiers = [CLIPClassifier(config, device)]

        self._primary_classifier = self._classifiers[0]

    def load_models(self) -> None:
        """Load all classification models."""
        for classifier in self._classifiers:
            classifier.load_model()

    @property
    def is_loaded(self) -> bool:
        """Check if all models are loaded."""
        return all(c.is_loaded for c in self._classifiers)

    def classify(
        self,
        image: np.ndarray,
        vehicle_class: VehicleClass,
    ) -> ClassificationResult:
        """
        Classify a single vehicle image.

        Args:
            image: Cropped vehicle image as BGR numpy array.
            vehicle_class: The type of vehicle.

        Returns:
            ClassificationResult with predictions.
        """
        if len(self._classifiers) == 1:
            return self._primary_classifier.classify(image, vehicle_class)

        # Ensemble classification
        return self._ensemble_classify([image], [vehicle_class])[0]

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
            List of ClassificationResult.
        """
        if not images:
            return []

        if len(self._classifiers) == 1:
            return self._primary_classifier.classify_batch(images, vehicle_classes)

        return self._ensemble_classify(images, vehicle_classes)

    def _ensemble_classify(
        self,
        images: list[np.ndarray],
        vehicle_classes: list[VehicleClass],
    ) -> list[ClassificationResult]:
        """
        Perform ensemble classification using all classifiers.

        Combines predictions from multiple classifiers using weighted voting.
        """
        all_results = []
        for classifier in self._classifiers:
            results = classifier.classify_batch(images, vehicle_classes)
            all_results.append(results)

        # Combine results for each image
        combined_results = []
        for i in range(len(images)):
            # Collect all predictions for this image
            label_scores: dict[str, float] = {}
            total_time = 0.0

            for classifier_results in all_results:
                result = classifier_results[i]
                total_time += result.processing_time_ms

                for label, confidence in result.predictions:
                    if label in label_scores:
                        label_scores[label] += confidence
                    else:
                        label_scores[label] = confidence

            # Average the scores
            num_classifiers = len(self._classifiers)
            for label in label_scores:
                label_scores[label] /= num_classifiers

            # Sort by score
            sorted_predictions = sorted(
                label_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:self.config.top_k]

            combined_results.append(
                ClassificationResult(
                    predictions=list(sorted_predictions),
                    vehicle_class=vehicle_classes[i],
                    processing_time_ms=total_time / num_classifiers,
                )
            )

        return combined_results

    def add_classifier(self, classifier: BaseClassifier) -> None:
        """
        Add a classifier to the ensemble.

        Args:
            classifier: Classifier to add.
        """
        self._classifiers.append(classifier)

    def get_make_model(
        self,
        result: ClassificationResult,
    ) -> tuple[str, str, float]:
        """
        Extract make, model, and confidence from classification result.

        Args:
            result: Classification result.

        Returns:
            Tuple of (make, model, confidence).
        """
        if not result.predictions:
            return "Unknown", "Unknown", 0.0

        label, confidence = result.predictions[0]
        make, model = self._parse_label(label)
        return make, model, confidence

    def _parse_label(self, label: str) -> tuple[str, str]:
        """
        Parse a classification label into make and model.

        Args:
            label: Label like "Honda Beat motorcycle"

        Returns:
            Tuple of (make, model).
        """
        # Remove vehicle type suffixes
        clean = label
        for suffix in [" motorcycle", " car", " scooter", " SUV", " MPV", " hatchback", " sedan"]:
            clean = clean.replace(suffix, "")

        words = clean.split()
        if len(words) >= 2:
            make = words[0]
            model = " ".join(words[1:])
            return make, model
        elif words:
            return words[0], ""
        return "Unknown", ""

    def get_price_tier_hint(self, result: ClassificationResult) -> str:
        """
        Get a price tier hint based on classification.

        This provides a rough estimate before looking up actual prices.

        Args:
            result: Classification result.

        Returns:
            Price tier hint string.
        """
        if not result.predictions:
            return "unknown"

        label = result.predictions[0][0].lower()

        # Premium indicators
        if any(x in label for x in ["bmw", "mercedes", "lexus", "audi", "alphard", "land cruiser"]):
            return "luxury"
        if any(x in label for x in ["fortuner", "pajero", "palisade", "cr-v", "ioniq"]):
            return "premium"
        if any(x in label for x in ["innova", "xpander", "hr-v", "almaz", "stargazer"]):
            return "mid-range"
        if any(x in label for x in ["avanza", "ertiga", "xenia", "confero", "brio"]):
            return "economy"
        if any(x in label for x in ["ayla", "sigra", "calya", "air ev"]):
            return "budget"

        # Motorcycle tiers
        if any(x in label for x in ["cbr", "r15", "ninja", "mt ", "vespa"]):
            return "premium"
        if any(x in label for x in ["pcx", "nmax", "aerox", "adv"]):
            return "mid"
        if any(x in label for x in ["beat", "vario", "mio", "scoopy", "genio"]):
            return "budget"

        return "economy"  # Default

    def get_classifier_info(self) -> list[dict]:
        """Get information about loaded classifiers."""
        return [
            {
                "name": c.name,
                "loaded": c.is_loaded,
            }
            for c in self._classifiers
        ]
