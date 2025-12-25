"""
Commercial vehicle detection module.

Detects ride-hailing motorcycles (ojol), public transport (angkot),
and other commercial vehicles specific to Indonesian streets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

from ..config import CommercialConfig
from .detector import Detection, VehicleClass


class CommercialType(Enum):
    """Types of commercial vehicles in Indonesia."""

    GOJEK = auto()  # Green uniform/delivery box
    GRAB = auto()  # Green uniform/delivery box
    SHOPEE = auto()  # Orange uniform/delivery box
    MAXIM = auto()  # Red uniform
    INDRIVER = auto()  # Green/yellow
    OJOL_GENERIC = auto()  # Generic ride-hailing (has delivery box)
    ANGKOT = auto()  # Public minibus
    TAXI = auto()  # Taxi (Blue Bird, etc.)
    TRUCK_COMMERCIAL = auto()  # Commercial truck
    DELIVERY_VAN = auto()  # Delivery van

    @property
    def display_name(self) -> str:
        """Get human-readable name."""
        names = {
            CommercialType.GOJEK: "Gojek",
            CommercialType.GRAB: "Grab",
            CommercialType.SHOPEE: "Shopee Food",
            CommercialType.MAXIM: "Maxim",
            CommercialType.INDRIVER: "inDriver",
            CommercialType.OJOL_GENERIC: "Ojol (Generic)",
            CommercialType.ANGKOT: "Angkot",
            CommercialType.TAXI: "Taxi",
            CommercialType.TRUCK_COMMERCIAL: "Commercial Truck",
            CommercialType.DELIVERY_VAN: "Delivery Van",
        }
        return names.get(self, self.name)


@dataclass
class CommercialDetection:
    """Result of commercial vehicle detection."""

    is_commercial: bool
    commercial_type: Optional[CommercialType]
    confidence: float
    color_ratio: float  # Percentage of commercial colors detected
    has_delivery_box: bool
    details: dict

    @property
    def description(self) -> str:
        """Get a human-readable description."""
        if not self.is_commercial:
            return "Non-commercial vehicle"
        if self.commercial_type:
            return f"{self.commercial_type.display_name} ({self.confidence:.0%})"
        return "Commercial vehicle (unknown type)"


class CommercialVehicleDetector:
    """
    Detects commercial vehicles in Indonesian street footage.

    Uses color detection, shape analysis, and pattern matching to identify:
    - Ride-hailing motorcycles (Gojek, Grab, Shopee, etc.)
    - Public transport (angkot)
    - Commercial trucks and vans

    Commercial vehicles are tracked separately as they indicate
    commercial activity rather than residential income.
    """

    # Color ranges in HSV
    # Gojek/Grab Green
    GREEN_RANGES = [
        ((35, 80, 80), (85, 255, 255)),  # Standard green
        ((40, 100, 100), (80, 255, 255)),  # Bright green
    ]

    # Shopee Orange
    ORANGE_RANGES = [
        ((5, 100, 100), (25, 255, 255)),  # Orange
        ((0, 100, 100), (15, 255, 255)),  # Red-orange
    ]

    # Maxim Red
    RED_RANGES = [
        ((0, 100, 100), (10, 255, 255)),  # Red
        ((170, 100, 100), (180, 255, 255)),  # Red (wrapping)
    ]

    # Blue Bird Taxi Blue
    BLUE_RANGES = [
        ((100, 100, 100), (130, 255, 255)),  # Blue
    ]

    def __init__(self, config: Optional[CommercialConfig] = None):
        """
        Initialize the commercial vehicle detector.

        Args:
            config: Commercial detection configuration.
        """
        self.config = config or CommercialConfig()

    def detect(
        self,
        image: np.ndarray,
        detection: Detection,
    ) -> CommercialDetection:
        """
        Detect if a vehicle is commercial.

        Args:
            image: Full frame as BGR numpy array.
            detection: Vehicle detection to analyze.

        Returns:
            CommercialDetection with analysis results.
        """
        if not self.config.enabled:
            return CommercialDetection(
                is_commercial=False,
                commercial_type=None,
                confidence=0.0,
                color_ratio=0.0,
                has_delivery_box=False,
                details={},
            )

        # Extract vehicle region
        x1, y1, x2, y2 = detection.bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))

        if x2 <= x1 or y2 <= y1:
            return CommercialDetection(
                is_commercial=False,
                commercial_type=None,
                confidence=0.0,
                color_ratio=0.0,
                has_delivery_box=False,
                details={"error": "Invalid bounding box"},
            )

        vehicle_crop = image[y1:y2, x1:x2]

        # Analyze based on vehicle type
        if detection.vehicle_class == VehicleClass.MOTORCYCLE:
            return self._analyze_motorcycle(vehicle_crop, detection)
        elif detection.vehicle_class == VehicleClass.BUS:
            return self._analyze_bus(vehicle_crop, detection)
        elif detection.vehicle_class == VehicleClass.CAR:
            return self._analyze_car(vehicle_crop, detection)
        elif detection.vehicle_class == VehicleClass.TRUCK:
            return self._analyze_truck(vehicle_crop, detection)
        else:
            return CommercialDetection(
                is_commercial=False,
                commercial_type=None,
                confidence=0.0,
                color_ratio=0.0,
                has_delivery_box=False,
                details={},
            )

    def _analyze_motorcycle(
        self,
        crop: np.ndarray,
        detection: Detection,
    ) -> CommercialDetection:
        """Analyze a motorcycle for ojol indicators."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total_pixels = crop.shape[0] * crop.shape[1]

        details = {}

        # Check for green (Gojek/Grab)
        green_ratio = self._calculate_color_ratio(hsv, self.GREEN_RANGES)
        details["green_ratio"] = green_ratio

        # Check for orange (Shopee)
        orange_ratio = self._calculate_color_ratio(hsv, self.ORANGE_RANGES)
        details["orange_ratio"] = orange_ratio

        # Check for red (Maxim)
        red_ratio = self._calculate_color_ratio(hsv, self.RED_RANGES)
        details["red_ratio"] = red_ratio

        # Check for delivery box (rectangular shape in upper portion)
        has_box = self._detect_delivery_box(crop)
        details["has_delivery_box"] = has_box

        # Determine commercial type
        threshold = self.config.color_ratio_threshold
        commercial_type = None
        is_commercial = False
        confidence = 0.0
        color_ratio = max(green_ratio, orange_ratio, red_ratio)

        if green_ratio >= threshold:
            # Could be Gojek or Grab - hard to distinguish without logo
            commercial_type = CommercialType.GRAB if green_ratio > 0.2 else CommercialType.GOJEK
            is_commercial = True
            confidence = min(green_ratio * 3, 1.0)
        elif orange_ratio >= threshold:
            commercial_type = CommercialType.SHOPEE
            is_commercial = True
            confidence = min(orange_ratio * 3, 1.0)
        elif red_ratio >= threshold:
            commercial_type = CommercialType.MAXIM
            is_commercial = True
            confidence = min(red_ratio * 3, 1.0)
        elif has_box:
            # Has delivery box but no distinctive color
            commercial_type = CommercialType.OJOL_GENERIC
            is_commercial = True
            confidence = 0.6

        return CommercialDetection(
            is_commercial=is_commercial,
            commercial_type=commercial_type,
            confidence=confidence,
            color_ratio=color_ratio,
            has_delivery_box=has_box,
            details=details,
        )

    def _analyze_bus(
        self,
        crop: np.ndarray,
        detection: Detection,
    ) -> CommercialDetection:
        """Analyze a bus - likely angkot or public transport."""
        # Buses are typically commercial in Indonesia
        # Angkot often have distinctive colors by route

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        details = {}

        # Check for common angkot colors
        green_ratio = self._calculate_color_ratio(hsv, self.GREEN_RANGES)
        orange_ratio = self._calculate_color_ratio(hsv, self.ORANGE_RANGES)
        blue_ratio = self._calculate_color_ratio(hsv, self.BLUE_RANGES)

        details["green_ratio"] = green_ratio
        details["orange_ratio"] = orange_ratio
        details["blue_ratio"] = blue_ratio

        # Most minibuses detected as "bus" in Indonesia are angkot
        return CommercialDetection(
            is_commercial=True,
            commercial_type=CommercialType.ANGKOT,
            confidence=0.8,  # High confidence for buses
            color_ratio=max(green_ratio, orange_ratio, blue_ratio),
            has_delivery_box=False,
            details=details,
        )

    def _analyze_car(
        self,
        crop: np.ndarray,
        detection: Detection,
    ) -> CommercialDetection:
        """Analyze a car for taxi or commercial vehicle indicators."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        details = {}

        # Check for Blue Bird taxi blue
        blue_ratio = self._calculate_color_ratio(hsv, self.BLUE_RANGES)
        details["blue_ratio"] = blue_ratio

        # Check for grab green (Grab Car)
        green_ratio = self._calculate_color_ratio(hsv, self.GREEN_RANGES)
        details["green_ratio"] = green_ratio

        threshold = self.config.color_ratio_threshold

        if blue_ratio >= threshold * 2:  # Higher threshold for cars
            return CommercialDetection(
                is_commercial=True,
                commercial_type=CommercialType.TAXI,
                confidence=min(blue_ratio * 2, 0.9),
                color_ratio=blue_ratio,
                has_delivery_box=False,
                details=details,
            )
        elif green_ratio >= threshold * 2:
            return CommercialDetection(
                is_commercial=True,
                commercial_type=CommercialType.GRAB,
                confidence=min(green_ratio * 2, 0.8),
                color_ratio=green_ratio,
                has_delivery_box=False,
                details=details,
            )

        return CommercialDetection(
            is_commercial=False,
            commercial_type=None,
            confidence=0.0,
            color_ratio=max(blue_ratio, green_ratio),
            has_delivery_box=False,
            details=details,
        )

    def _analyze_truck(
        self,
        crop: np.ndarray,
        detection: Detection,
    ) -> CommercialDetection:
        """Analyze a truck - typically commercial."""
        # Trucks are almost always commercial vehicles
        return CommercialDetection(
            is_commercial=True,
            commercial_type=CommercialType.TRUCK_COMMERCIAL,
            confidence=0.9,
            color_ratio=0.0,
            has_delivery_box=False,
            details={"reason": "Trucks are typically commercial vehicles"},
        )

    def _calculate_color_ratio(
        self,
        hsv_image: np.ndarray,
        color_ranges: list[tuple[tuple, tuple]],
    ) -> float:
        """Calculate the ratio of pixels matching any of the color ranges."""
        total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
        if total_pixels == 0:
            return 0.0

        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

        for lower, upper in color_ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            color_mask = cv2.inRange(hsv_image, lower_np, upper_np)
            mask = cv2.bitwise_or(mask, color_mask)

        colored_pixels = np.count_nonzero(mask)
        return colored_pixels / total_pixels

    def _detect_delivery_box(self, crop: np.ndarray) -> bool:
        """
        Detect if a motorcycle has a delivery box.

        Uses edge detection and contour analysis to find
        rectangular structures in the upper portion of the image.
        """
        if crop.shape[0] < 50 or crop.shape[1] < 50:
            return False

        # Focus on upper half where delivery boxes usually are
        upper_half = crop[:crop.shape[0] // 2, :]

        # Convert to grayscale and find edges
        gray = cv2.cvtColor(upper_half, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Too small
                continue

            # Check for rectangular shape
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # Delivery boxes are roughly rectangular (4 corners)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0

                # Delivery boxes typically have aspect ratio 0.5 to 2.0
                if 0.5 <= aspect_ratio <= 2.0:
                    # Check if it's significant portion of the upper half
                    box_ratio = area / (upper_half.shape[0] * upper_half.shape[1])
                    if box_ratio > 0.05:  # At least 5% of upper half
                        return True

        return False

    def batch_detect(
        self,
        image: np.ndarray,
        detections: list[Detection],
    ) -> list[CommercialDetection]:
        """
        Analyze multiple detections in a single frame.

        Args:
            image: Full frame as BGR numpy array.
            detections: List of vehicle detections to analyze.

        Returns:
            List of CommercialDetection results.
        """
        return [self.detect(image, det) for det in detections]
