"""
CLIP-based zero-shot vehicle classifier.

Uses OpenAI's CLIP model to classify vehicle makes and models
without requiring task-specific training data.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ..config import ClassificationConfig
from ..detection.detector import VehicleClass
from .base import BaseClassifier, ClassificationResult


# Indonesian motorcycle makes and popular models
MOTORCYCLE_LABELS = [
    # Honda (market leader ~75%)
    "Honda Beat motorcycle",
    "Honda Vario motorcycle",
    "Honda PCX motorcycle",
    "Honda Scoopy motorcycle",
    "Honda ADV motorcycle",
    "Honda CBR motorcycle",
    "Honda CB motorcycle",
    "Honda Revo motorcycle",
    "Honda Genio motorcycle",
    # Yamaha (~20%)
    "Yamaha NMAX motorcycle",
    "Yamaha Aerox motorcycle",
    "Yamaha Mio motorcycle",
    "Yamaha Lexi motorcycle",
    "Yamaha Fazzio motorcycle",
    "Yamaha R15 motorcycle",
    "Yamaha MT motorcycle",
    "Yamaha XSR motorcycle",
    "Yamaha Filano motorcycle",
    # Suzuki
    "Suzuki Nex motorcycle",
    "Suzuki Address motorcycle",
    "Suzuki Satria motorcycle",
    "Suzuki GSX motorcycle",
    # Kawasaki
    "Kawasaki Ninja motorcycle",
    "Kawasaki KLX motorcycle",
    "Kawasaki W175 motorcycle",
    # Vespa
    "Vespa scooter",
    # Electric motorcycles
    "electric motorcycle",
    "Gesits electric motorcycle",
    # Generic
    "motorcycle",
    "scooter",
    "sport motorcycle",
]

# Indonesian car makes and popular models
CAR_LABELS = [
    # Toyota (market leader)
    "Toyota Avanza car",
    "Toyota Innova car",
    "Toyota Fortuner car",
    "Toyota Rush car",
    "Toyota Calya car",
    "Toyota Alphard car",
    "Toyota Yaris car",
    "Toyota Camry car",
    "Toyota Land Cruiser car",
    "Toyota Veloz car",
    "Toyota Raize car",
    # Honda
    "Honda Brio car",
    "Honda HR-V car",
    "Honda CR-V car",
    "Honda BR-V car",
    "Honda City car",
    "Honda Civic car",
    "Honda Jazz car",
    # Mitsubishi
    "Mitsubishi Xpander car",
    "Mitsubishi Pajero car",
    "Mitsubishi Triton car",
    "Mitsubishi Outlander car",
    # Daihatsu
    "Daihatsu Xenia car",
    "Daihatsu Sigra car",
    "Daihatsu Ayla car",
    "Daihatsu Terios car",
    "Daihatsu Rocky car",
    # Suzuki
    "Suzuki Ertiga car",
    "Suzuki XL7 car",
    "Suzuki Baleno car",
    "Suzuki Ignis car",
    "Suzuki Jimny car",
    # Wuling (Chinese - growing fast)
    "Wuling Air EV car",
    "Wuling Confero car",
    "Wuling Almaz car",
    "Wuling Cortez car",
    # Hyundai
    "Hyundai Creta car",
    "Hyundai Stargazer car",
    "Hyundai Ioniq car",
    "Hyundai Palisade car",
    "Hyundai Santa Fe car",
    # Kia
    "Kia Sonet car",
    "Kia Seltos car",
    "Kia Carens car",
    # DFSK (Chinese)
    "DFSK Glory car",
    "DFSK Gelora car",
    # Chery (Chinese)
    "Chery Omoda car",
    "Chery Tiggo car",
    # BYD (Chinese EV)
    "BYD Atto car",
    "BYD Seal car",
    "BYD Dolphin car",
    # MG (Chinese)
    "MG ZS car",
    "MG HS car",
    # Nissan
    "Nissan Livina car",
    "Nissan Kicks car",
    "Nissan Terra car",
    # Mazda
    "Mazda CX-5 car",
    "Mazda 2 car",
    "Mazda 3 car",
    # Premium
    "BMW car",
    "Mercedes-Benz car",
    "Lexus car",
    "Audi car",
    # Generic
    "sedan car",
    "SUV car",
    "MPV car",
    "hatchback car",
    "pickup truck",
    "van",
]


class CLIPClassifier(BaseClassifier):
    """
    CLIP-based zero-shot vehicle classifier.

    Uses CLIP's vision-language understanding to classify vehicles
    by comparing image features against text descriptions of makes/models.
    """

    def __init__(
        self,
        config: Optional[ClassificationConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the CLIP classifier.

        Args:
            config: Classification configuration.
            device: PyTorch device for inference.
        """
        self.config = config or ClassificationConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model: Optional[CLIPModel] = None
        self._processor: Optional[CLIPProcessor] = None

        # Precomputed text features for efficiency
        self._motorcycle_features: Optional[torch.Tensor] = None
        self._car_features: Optional[torch.Tensor] = None

    @property
    def name(self) -> str:
        return f"CLIP ({self.config.model_name})"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def load_model(self) -> None:
        """Load the CLIP model and precompute text features."""
        if self.is_loaded:
            return

        self._model = CLIPModel.from_pretrained(self.config.model_name)
        self._processor = CLIPProcessor.from_pretrained(self.config.model_name)

        self._model.to(self.device)
        self._model.eval()

        # Precompute text features for efficiency
        self._precompute_text_features()

    def _precompute_text_features(self) -> None:
        """Precompute text embeddings for all labels."""
        if not self._processor or not self._model:
            return

        with torch.no_grad():
            # Motorcycle features
            moto_inputs = self._processor(
                text=MOTORCYCLE_LABELS,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            moto_inputs = {k: v.to(self.device) for k, v in moto_inputs.items() if k != "pixel_values"}
            self._motorcycle_features = self._model.get_text_features(**moto_inputs)
            self._motorcycle_features = self._motorcycle_features / self._motorcycle_features.norm(
                dim=-1, keepdim=True
            )

            # Car features
            car_inputs = self._processor(
                text=CAR_LABELS,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            car_inputs = {k: v.to(self.device) for k, v in car_inputs.items() if k != "pixel_values"}
            self._car_features = self._model.get_text_features(**car_inputs)
            self._car_features = self._car_features / self._car_features.norm(dim=-1, keepdim=True)

    def classify(
        self,
        image: np.ndarray,
        vehicle_class: VehicleClass,
    ) -> ClassificationResult:
        """Classify a single vehicle image."""
        results = self.classify_batch([image], [vehicle_class])
        return results[0]

    def classify_batch(
        self,
        images: list[np.ndarray],
        vehicle_classes: list[VehicleClass],
    ) -> list[ClassificationResult]:
        """Classify multiple vehicle images."""
        if not self.is_loaded:
            self.load_model()

        if not images:
            return []

        start_time = time.perf_counter()

        # Convert images to PIL
        pil_images = []
        for img in images:
            if img is None or img.size == 0:
                # Create a small placeholder
                pil_images.append(Image.new("RGB", (224, 224)))
            else:
                # Convert BGR to RGB
                rgb = img[:, :, ::-1] if img.ndim == 3 else img
                pil_images.append(Image.fromarray(rgb))

        # Process images
        inputs = self._processor(
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        results = []
        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            for i, vehicle_class in enumerate(vehicle_classes):
                img_feat = image_features[i:i+1]

                # Select appropriate text features
                if vehicle_class == VehicleClass.MOTORCYCLE:
                    text_features = self._motorcycle_features
                    labels = MOTORCYCLE_LABELS
                else:
                    text_features = self._car_features
                    labels = CAR_LABELS

                # Calculate similarities
                similarities = (img_feat @ text_features.T).squeeze(0)
                probs = similarities.softmax(dim=0)

                # Get top-k predictions
                top_k = min(self.config.top_k, len(labels))
                top_probs, top_indices = probs.topk(top_k)

                predictions = [
                    (labels[idx.item()], prob.item())
                    for prob, idx in zip(top_probs, top_indices)
                    if prob.item() >= self.config.min_confidence
                ]

                results.append(
                    ClassificationResult(
                        predictions=predictions,
                        vehicle_class=vehicle_class,
                        processing_time_ms=(time.perf_counter() - start_time) * 1000 / len(images),
                    )
                )

        return results

    def get_supported_labels(self, vehicle_class: VehicleClass) -> list[str]:
        """Get the list of labels for a vehicle class."""
        if vehicle_class == VehicleClass.MOTORCYCLE:
            return MOTORCYCLE_LABELS.copy()
        else:
            return CAR_LABELS.copy()

    def add_custom_labels(
        self,
        labels: list[str],
        vehicle_class: VehicleClass,
    ) -> None:
        """
        Add custom labels for classification.

        Allows extending the classifier with additional makes/models.

        Args:
            labels: New labels to add.
            vehicle_class: Vehicle class to add labels to.
        """
        if not self.is_loaded:
            self.load_model()

        # Add to appropriate label list and recompute features
        if vehicle_class == VehicleClass.MOTORCYCLE:
            global MOTORCYCLE_LABELS
            MOTORCYCLE_LABELS.extend(labels)
        else:
            global CAR_LABELS
            CAR_LABELS.extend(labels)

        # Recompute text features
        self._precompute_text_features()

    def extract_make_model(self, label: str) -> tuple[str, str]:
        """
        Extract make and model from a label string.

        Args:
            label: Label like "Honda Beat motorcycle"

        Returns:
            Tuple of (make, model)
        """
        # Remove vehicle type suffix
        parts = label.replace(" motorcycle", "").replace(" car", "").replace(" scooter", "")

        # Split into make and model
        words = parts.split()
        if len(words) >= 2:
            make = words[0]
            model = " ".join(words[1:])
            return make, model
        elif words:
            return words[0], ""
        return "Unknown", ""
