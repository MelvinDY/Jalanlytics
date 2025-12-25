"""Vehicle classification module using CLIP and ensemble methods."""

from .base import BaseClassifier
from .clip_classifier import CLIPClassifier
from .classifier import VehicleClassifier

__all__ = ["BaseClassifier", "CLIPClassifier", "VehicleClassifier"]
