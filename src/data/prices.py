"""
Vehicle price database for Indonesian market.

Provides price lookups and tier classification for vehicles
based on make and model identification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import yaml

from ..detection.detector import VehicleClass


class PriceTier(Enum):
    """Vehicle price tiers."""

    # Car tiers
    BUDGET = auto()
    ECONOMY = auto()
    MID_RANGE = auto()
    PREMIUM = auto()
    LUXURY = auto()

    # Motorcycle tiers
    MOTO_BUDGET = auto()
    MOTO_MID = auto()
    MOTO_PREMIUM = auto()

    # Non-income indicators
    COMMERCIAL = auto()
    UNKNOWN = auto()

    @property
    def display_name(self) -> str:
        """Get human-readable name."""
        names = {
            PriceTier.BUDGET: "Budget",
            PriceTier.ECONOMY: "Economy",
            PriceTier.MID_RANGE: "Mid-range",
            PriceTier.PREMIUM: "Premium",
            PriceTier.LUXURY: "Luxury",
            PriceTier.MOTO_BUDGET: "Budget",
            PriceTier.MOTO_MID: "Mid",
            PriceTier.MOTO_PREMIUM: "Premium",
            PriceTier.COMMERCIAL: "Commercial",
            PriceTier.UNKNOWN: "Unknown",
        }
        return names.get(self, self.name)


@dataclass
class VehiclePrice:
    """Price information for a vehicle."""

    make: str
    model: str
    price_idr: int
    tier: PriceTier
    category: str  # scooter, sedan, suv, mpv, etc.

    @property
    def price_formatted(self) -> str:
        """Format price in Indonesian style (Rp X.XXX.XXX)."""
        return f"Rp {self.price_idr:,.0f}".replace(",", ".")

    @property
    def price_juta(self) -> float:
        """Price in millions (juta)."""
        return self.price_idr / 1_000_000


class PriceDatabase:
    """
    Database of Indonesian vehicle prices.

    Provides price lookups and tier classification for vehicles
    detected and classified in the video analysis.
    """

    def __init__(self, data_path: Optional[Path | str] = None):
        """
        Initialize the price database.

        Args:
            data_path: Path to the YAML price database.
                      If None, uses the default bundled database.
        """
        if data_path is None:
            # Use bundled database
            data_path = Path(__file__).parent / "vehicle_prices.yaml"
        else:
            data_path = Path(data_path)

        self._data = self._load_database(data_path)
        self._build_lookup_tables()

    def _load_database(self, path: Path) -> dict:
        """Load the YAML database."""
        if not path.exists():
            return self._get_fallback_data()

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _get_fallback_data(self) -> dict:
        """Return minimal fallback data if file not found."""
        return {
            "defaults": {
                "motorcycle": 22000000,
                "car": 280000000,
                "bicycle": 0,
                "bus": 0,
                "truck": 0,
            },
            "tiers": {
                "car": {
                    "budget": {"min": 0, "max": 150000000},
                    "economy": {"min": 150000000, "max": 300000000},
                    "mid_range": {"min": 300000000, "max": 500000000},
                    "premium": {"min": 500000000, "max": 800000000},
                    "luxury": {"min": 800000000, "max": 999999999999},
                },
                "motorcycle": {
                    "budget": {"min": 0, "max": 18000000},
                    "mid": {"min": 18000000, "max": 35000000},
                    "premium": {"min": 35000000, "max": 999999999999},
                },
            },
            "motorcycles": {},
            "cars": {},
        }

    def _build_lookup_tables(self) -> None:
        """Build fast lookup tables for make/model searches."""
        self._motorcycle_lookup: dict[str, dict[str, VehiclePrice]] = {}
        self._car_lookup: dict[str, dict[str, VehiclePrice]] = {}

        # Build motorcycle lookup
        for make, models in self._data.get("motorcycles", {}).items():
            if not isinstance(models, dict):
                continue
            self._motorcycle_lookup[make.lower()] = {}
            for model_name, info in models.items():
                if not isinstance(info, dict):
                    continue
                price = info.get("price", 22000000)
                tier = self._get_motorcycle_tier(price)
                self._motorcycle_lookup[make.lower()][model_name.lower()] = VehiclePrice(
                    make=make.title(),
                    model=model_name.replace("_", " ").title(),
                    price_idr=price,
                    tier=tier,
                    category=info.get("category", "motorcycle"),
                )

        # Build car lookup
        for make, models in self._data.get("cars", {}).items():
            if not isinstance(models, dict):
                continue
            self._car_lookup[make.lower()] = {}
            for model_name, info in models.items():
                if not isinstance(info, dict):
                    continue
                price = info.get("price", 280000000)
                tier = self._get_car_tier(price)
                self._car_lookup[make.lower()][model_name.lower()] = VehiclePrice(
                    make=make.title(),
                    model=model_name.replace("_", " ").title(),
                    price_idr=price,
                    tier=tier,
                    category=info.get("category", "car"),
                )

    def _get_motorcycle_tier(self, price: int) -> PriceTier:
        """Determine motorcycle price tier."""
        tiers = self._data.get("tiers", {}).get("motorcycle", {})

        if price < tiers.get("budget", {}).get("max", 18000000):
            return PriceTier.MOTO_BUDGET
        elif price < tiers.get("mid", {}).get("max", 35000000):
            return PriceTier.MOTO_MID
        else:
            return PriceTier.MOTO_PREMIUM

    def _get_car_tier(self, price: int) -> PriceTier:
        """Determine car price tier."""
        tiers = self._data.get("tiers", {}).get("car", {})

        if price < tiers.get("budget", {}).get("max", 150000000):
            return PriceTier.BUDGET
        elif price < tiers.get("economy", {}).get("max", 300000000):
            return PriceTier.ECONOMY
        elif price < tiers.get("mid_range", {}).get("max", 500000000):
            return PriceTier.MID_RANGE
        elif price < tiers.get("premium", {}).get("max", 800000000):
            return PriceTier.PREMIUM
        else:
            return PriceTier.LUXURY

    def lookup(
        self,
        make: str,
        model: str,
        vehicle_class: VehicleClass,
    ) -> Optional[VehiclePrice]:
        """
        Look up price for a specific make/model.

        Args:
            make: Vehicle make (e.g., "Honda", "Toyota")
            model: Vehicle model (e.g., "Beat", "Avanza")
            vehicle_class: Type of vehicle

        Returns:
            VehiclePrice if found, None otherwise.
        """
        make_lower = make.lower().strip()
        model_lower = model.lower().strip().replace(" ", "_")

        if vehicle_class == VehicleClass.MOTORCYCLE:
            make_data = self._motorcycle_lookup.get(make_lower)
            if make_data:
                # Try exact match first
                if model_lower in make_data:
                    return make_data[model_lower]
                # Try partial match
                for key, price_info in make_data.items():
                    if model_lower in key or key in model_lower:
                        return price_info
        else:
            make_data = self._car_lookup.get(make_lower)
            if make_data:
                if model_lower in make_data:
                    return make_data[model_lower]
                for key, price_info in make_data.items():
                    if model_lower in key or key in model_lower:
                        return price_info

        return None

    def get_default_price(self, vehicle_class: VehicleClass) -> VehiclePrice:
        """
        Get default price for a vehicle class.

        Args:
            vehicle_class: Type of vehicle

        Returns:
            VehiclePrice with default values for the class.
        """
        defaults = self._data.get("defaults", {})

        class_map = {
            VehicleClass.MOTORCYCLE: ("motorcycle", PriceTier.MOTO_MID, 22000000),
            VehicleClass.CAR: ("car", PriceTier.ECONOMY, 280000000),
            VehicleClass.BICYCLE: ("bicycle", PriceTier.UNKNOWN, 0),
            VehicleClass.BUS: ("bus", PriceTier.COMMERCIAL, 0),
            VehicleClass.TRUCK: ("truck", PriceTier.COMMERCIAL, 0),
        }

        class_name, default_tier, fallback = class_map.get(
            vehicle_class, ("car", PriceTier.ECONOMY, 280000000)
        )
        price = defaults.get(class_name, fallback)

        return VehiclePrice(
            make="Unknown",
            model="Unknown",
            price_idr=price,
            tier=default_tier,
            category=class_name,
        )

    def get_price(
        self,
        make: str,
        model: str,
        vehicle_class: VehicleClass,
    ) -> VehiclePrice:
        """
        Get price for a vehicle, falling back to defaults if not found.

        Args:
            make: Vehicle make
            model: Vehicle model
            vehicle_class: Type of vehicle

        Returns:
            VehiclePrice (never None - returns default if not found)
        """
        result = self.lookup(make, model, vehicle_class)
        if result:
            return result

        # Try with just the make
        if vehicle_class == VehicleClass.MOTORCYCLE and make.lower() in self._motorcycle_lookup:
            models = self._motorcycle_lookup[make.lower()]
            if models:
                # Return average price for the make
                prices = [p.price_idr for p in models.values()]
                avg_price = int(sum(prices) / len(prices))
                return VehiclePrice(
                    make=make.title(),
                    model="Unknown",
                    price_idr=avg_price,
                    tier=self._get_motorcycle_tier(avg_price),
                    category="motorcycle",
                )

        if vehicle_class != VehicleClass.MOTORCYCLE and make.lower() in self._car_lookup:
            models = self._car_lookup[make.lower()]
            if models:
                prices = [p.price_idr for p in models.values()]
                avg_price = int(sum(prices) / len(prices))
                return VehiclePrice(
                    make=make.title(),
                    model="Unknown",
                    price_idr=avg_price,
                    tier=self._get_car_tier(avg_price),
                    category="car",
                )

        return self.get_default_price(vehicle_class)

    def get_all_makes(self, vehicle_class: VehicleClass) -> list[str]:
        """Get all known makes for a vehicle class."""
        if vehicle_class == VehicleClass.MOTORCYCLE:
            return list(self._motorcycle_lookup.keys())
        else:
            return list(self._car_lookup.keys())

    def get_tier_ranges(self, vehicle_class: VehicleClass) -> dict[str, tuple[int, int]]:
        """Get price ranges for each tier."""
        if vehicle_class == VehicleClass.MOTORCYCLE:
            tiers = self._data.get("tiers", {}).get("motorcycle", {})
        else:
            tiers = self._data.get("tiers", {}).get("car", {})

        return {
            name: (info.get("min", 0), info.get("max", 0))
            for name, info in tiers.items()
        }

    def format_price_idr(self, price: int) -> str:
        """Format price in Indonesian style."""
        return f"Rp {price:,.0f}".replace(",", ".")

    def format_price_juta(self, price: int) -> str:
        """Format price in millions (juta)."""
        juta = price / 1_000_000
        if juta >= 1000:
            return f"Rp {juta/1000:.1f} milyar"
        elif juta >= 1:
            return f"Rp {juta:.0f} juta"
        else:
            return f"Rp {price:,.0f}".replace(",", ".")
