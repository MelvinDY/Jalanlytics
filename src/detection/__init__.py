"""Vehicle detection module using YOLOv8."""

from .detector import VehicleDetector
from .commercial import CommercialVehicleDetector

__all__ = ["VehicleDetector", "CommercialVehicleDetector"]
