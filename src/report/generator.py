"""
Report generation module for income analysis output.

Generates formatted reports with vehicle statistics, income tier analysis,
and area assessment recommendations.
"""

import json
import csv
import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..config import ReportConfig
from ..data.prices import PriceTier, VehiclePrice


@dataclass
class VehicleStats:
    """Statistics for a category of vehicles."""

    total_count: int = 0
    commercial_count: int = 0
    tier_counts: dict[str, int] = field(default_factory=dict)
    make_counts: dict[str, int] = field(default_factory=dict)
    model_counts: dict[str, int] = field(default_factory=dict)
    prices: list[int] = field(default_factory=list)

    @property
    def non_commercial_count(self) -> int:
        return self.total_count - self.commercial_count

    @property
    def median_price(self) -> int:
        if not self.prices:
            return 0
        sorted_prices = sorted(self.prices)
        n = len(sorted_prices)
        if n % 2 == 0:
            return (sorted_prices[n // 2 - 1] + sorted_prices[n // 2]) // 2
        return sorted_prices[n // 2]

    @property
    def average_price(self) -> int:
        if not self.prices:
            return 0
        return int(sum(self.prices) / len(self.prices))

    def add_vehicle(
        self,
        price: VehiclePrice,
        is_commercial: bool = False,
    ) -> None:
        """Add a vehicle to the statistics."""
        self.total_count += 1
        if is_commercial:
            self.commercial_count += 1

        # Track tier
        tier_name = price.tier.display_name
        self.tier_counts[tier_name] = self.tier_counts.get(tier_name, 0) + 1

        # Track make
        self.make_counts[price.make] = self.make_counts.get(price.make, 0) + 1

        # Track model
        full_name = f"{price.make} {price.model}"
        self.model_counts[full_name] = self.model_counts.get(full_name, 0) + 1

        # Track price (only non-zero, non-commercial)
        if price.price_idr > 0 and not is_commercial:
            self.prices.append(price.price_idr)


@dataclass
class AnalysisResult:
    """Complete analysis result from the pipeline."""

    video_path: str
    video_duration: float
    frames_analyzed: int
    processing_time: float
    motorcycle_stats: VehicleStats
    car_stats: VehicleStats
    bicycle_count: int = 0
    bus_count: int = 0
    truck_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_vehicles(self) -> int:
        return (
            self.motorcycle_stats.total_count
            + self.car_stats.total_count
            + self.bicycle_count
            + self.bus_count
            + self.truck_count
        )

    @property
    def motorcycle_to_car_ratio(self) -> float:
        if self.car_stats.total_count == 0:
            return float("inf") if self.motorcycle_stats.total_count > 0 else 0
        return self.motorcycle_stats.total_count / self.car_stats.total_count


class ReportGenerator:
    """
    Generates formatted reports from analysis results.

    Supports multiple output formats:
    - Text: Rich terminal output
    - JSON: Machine-readable format
    - CSV: Spreadsheet-compatible format
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize the report generator.

        Args:
            config: Report configuration.
        """
        self.config = config or ReportConfig()
        self._console = Console()

    def generate(self, result: AnalysisResult) -> str:
        """
        Generate a report in the configured format.

        Args:
            result: Analysis result from the pipeline.

        Returns:
            Formatted report string.
        """
        if self.config.output_format == "json":
            return self._generate_json(result)
        elif self.config.output_format == "csv":
            return self._generate_csv(result)
        else:
            return self._generate_text(result)

    def _generate_text(self, result: AnalysisResult) -> str:
        """Generate text format report."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)

        # Header
        console.print("\n" + "=" * 50)
        console.print("[bold cyan]=== Street Income Analysis Report ===[/bold cyan]")
        console.print("=" * 50 + "\n")

        # Video info
        video_name = Path(result.video_path).name
        duration_mins = int(result.video_duration // 60)
        duration_secs = int(result.video_duration % 60)

        console.print(f"[bold]Video:[/bold] {video_name}")
        console.print(f"[bold]Duration:[/bold] {duration_mins}:{duration_secs:02d}")
        console.print(f"[bold]Frames analyzed:[/bold] {result.frames_analyzed}")
        console.print(f"[bold]Processing time:[/bold] {result.processing_time:.1f}s")
        console.print()

        # Vehicle counts
        console.print("[bold underline]VEHICLE COUNTS[/bold underline]")
        console.print("-" * 30)
        console.print(f"Motorcycles detected: {result.motorcycle_stats.total_count} unique vehicles")
        if result.motorcycle_stats.commercial_count > 0:
            console.print(f"  (Commercial/Ojol: {result.motorcycle_stats.commercial_count})")
        console.print(f"Cars detected: {result.car_stats.total_count} unique vehicles")
        if result.car_stats.commercial_count > 0:
            console.print(f"  (Commercial: {result.car_stats.commercial_count})")
        if result.bicycle_count > 0:
            console.print(f"Bicycles: {result.bicycle_count}")
        if result.bus_count > 0:
            console.print(f"Buses/Angkot: {result.bus_count}")
        if result.truck_count > 0:
            console.print(f"Trucks: {result.truck_count}")
        console.print(f"[bold]Total:[/bold] {result.total_vehicles}")
        console.print()

        # Motorcycle breakdown
        if result.motorcycle_stats.total_count > 0:
            console.print("[bold underline]MOTORCYCLE BREAKDOWN[/bold underline]")
            console.print("-" * 30)

            # Top models
            sorted_models = sorted(
                result.motorcycle_stats.model_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]

            for model, count in sorted_models:
                pct = count / result.motorcycle_stats.total_count * 100
                console.print(f"{model}: {count} ({pct:.0f}%)")

            console.print()

            # Tier distribution
            console.print("Motorcycle tier distribution:")
            for tier, count in sorted(result.motorcycle_stats.tier_counts.items()):
                if count > 0:
                    pct = count / result.motorcycle_stats.total_count * 100
                    console.print(f"  {tier}: {pct:.0f}%")
            console.print()

        # Car breakdown
        if result.car_stats.total_count > 0:
            console.print("[bold underline]CAR BREAKDOWN[/bold underline]")
            console.print("-" * 30)

            sorted_models = sorted(
                result.car_stats.model_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]

            for model, count in sorted_models:
                pct = count / result.car_stats.total_count * 100
                console.print(f"{model}: {count} ({pct:.0f}%)")

            console.print()

            # Tier distribution
            console.print("Car tier distribution:")
            for tier in ["Budget", "Economy", "Mid-range", "Premium", "Luxury"]:
                count = result.car_stats.tier_counts.get(tier, 0)
                if count > 0:
                    pct = count / result.car_stats.total_count * 100
                    console.print(f"  {tier}: {pct:.0f}%")
            console.print()

        # Income analysis
        console.print("[bold underline]INCOME ANALYSIS[/bold underline]")
        console.print("-" * 30)

        if result.motorcycle_stats.median_price > 0:
            console.print(
                f"Median motorcycle value: {self._format_price(result.motorcycle_stats.median_price)}"
            )
        if result.car_stats.median_price > 0:
            console.print(
                f"Median car value: {self._format_price(result.car_stats.median_price)}"
            )

        if result.car_stats.total_count > 0:
            ratio = result.motorcycle_to_car_ratio
            console.print(f"Motorcycle-to-car ratio: {ratio:.1f}:1")
        console.print()

        # Area assessment
        console.print("[bold underline]AREA ASSESSMENT[/bold underline]")
        console.print("-" * 30)
        assessment = self._generate_assessment(result)
        console.print(f"[bold]Estimated income bracket:[/bold] {assessment['bracket']}")
        console.print(f"[bold]Characteristics:[/bold] {assessment['characteristics']}")
        console.print(f"[bold]Good location for:[/bold]")
        for item in assessment["recommendations"]:
            console.print(f"  - {item}")
        console.print()

        return output.getvalue()

    def _generate_json(self, result: AnalysisResult) -> str:
        """Generate JSON format report."""
        data = {
            "video": {
                "path": result.video_path,
                "duration_seconds": result.video_duration,
                "frames_analyzed": result.frames_analyzed,
                "processing_time_seconds": result.processing_time,
            },
            "timestamp": result.timestamp.isoformat(),
            "vehicle_counts": {
                "total": result.total_vehicles,
                "motorcycles": result.motorcycle_stats.total_count,
                "motorcycles_commercial": result.motorcycle_stats.commercial_count,
                "cars": result.car_stats.total_count,
                "cars_commercial": result.car_stats.commercial_count,
                "bicycles": result.bicycle_count,
                "buses": result.bus_count,
                "trucks": result.truck_count,
            },
            "motorcycle_analysis": {
                "tier_distribution": result.motorcycle_stats.tier_counts,
                "make_distribution": result.motorcycle_stats.make_counts,
                "top_models": dict(
                    sorted(
                        result.motorcycle_stats.model_counts.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:10]
                ),
                "median_value_idr": result.motorcycle_stats.median_price,
                "average_value_idr": result.motorcycle_stats.average_price,
            },
            "car_analysis": {
                "tier_distribution": result.car_stats.tier_counts,
                "make_distribution": result.car_stats.make_counts,
                "top_models": dict(
                    sorted(
                        result.car_stats.model_counts.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:10]
                ),
                "median_value_idr": result.car_stats.median_price,
                "average_value_idr": result.car_stats.average_price,
            },
            "income_analysis": {
                "motorcycle_to_car_ratio": (
                    result.motorcycle_to_car_ratio
                    if result.motorcycle_to_car_ratio != float("inf")
                    else None
                ),
                **self._generate_assessment(result),
            },
        }

        return json.dumps(data, indent=2)

    def _generate_csv(self, result: AnalysisResult) -> str:
        """Generate CSV format report."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(["Jalanlytics Street Income Analysis"])
        writer.writerow(["Video", Path(result.video_path).name])
        writer.writerow(["Duration (s)", result.video_duration])
        writer.writerow(["Frames Analyzed", result.frames_analyzed])
        writer.writerow([])

        # Vehicle counts
        writer.writerow(["Vehicle Type", "Count", "Commercial"])
        writer.writerow([
            "Motorcycles",
            result.motorcycle_stats.total_count,
            result.motorcycle_stats.commercial_count,
        ])
        writer.writerow([
            "Cars",
            result.car_stats.total_count,
            result.car_stats.commercial_count,
        ])
        writer.writerow(["Bicycles", result.bicycle_count, 0])
        writer.writerow(["Buses", result.bus_count, result.bus_count])
        writer.writerow(["Trucks", result.truck_count, result.truck_count])
        writer.writerow([])

        # Motorcycle models
        writer.writerow(["Motorcycle Models"])
        writer.writerow(["Model", "Count", "Percentage"])
        for model, count in sorted(
            result.motorcycle_stats.model_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            pct = count / result.motorcycle_stats.total_count * 100 if result.motorcycle_stats.total_count > 0 else 0
            writer.writerow([model, count, f"{pct:.1f}%"])
        writer.writerow([])

        # Car models
        writer.writerow(["Car Models"])
        writer.writerow(["Model", "Count", "Percentage"])
        for model, count in sorted(
            result.car_stats.model_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            pct = count / result.car_stats.total_count * 100 if result.car_stats.total_count > 0 else 0
            writer.writerow([model, count, f"{pct:.1f}%"])
        writer.writerow([])

        # Income analysis
        assessment = self._generate_assessment(result)
        writer.writerow(["Income Analysis"])
        writer.writerow(["Income Bracket", assessment["bracket"]])
        writer.writerow(["Median Motorcycle Value", result.motorcycle_stats.median_price])
        writer.writerow(["Median Car Value", result.car_stats.median_price])
        writer.writerow(["Motorcycle to Car Ratio", f"{result.motorcycle_to_car_ratio:.1f}:1"])

        return output.getvalue()

    def _format_price(self, price: int) -> str:
        """Format price in Indonesian Rupiah."""
        if price >= 1_000_000_000:
            return f"Rp {price / 1_000_000_000:.1f} milyar"
        elif price >= 1_000_000:
            return f"Rp {price / 1_000_000:.0f} juta"
        else:
            return f"Rp {price:,.0f}".replace(",", ".")

    def _generate_assessment(self, result: AnalysisResult) -> dict:
        """Generate area assessment based on analysis results."""
        # Calculate weighted income score
        moto_score = self._get_moto_income_score(result.motorcycle_stats)
        car_score = self._get_car_income_score(result.car_stats)

        # Weight car scores higher (cars are more expensive)
        if result.car_stats.non_commercial_count > 0:
            total_score = (moto_score * 0.3 + car_score * 0.7)
        else:
            total_score = moto_score

        # Determine income bracket
        if total_score >= 4:
            bracket = "HIGH-INCOME"
            characteristics = "Premium vehicle mix indicates affluent area."
        elif total_score >= 3:
            bracket = "UPPER-MIDDLE-INCOME"
            characteristics = "Good mix of mid-range and premium vehicles."
        elif total_score >= 2:
            bracket = "MIDDLE-INCOME"
            characteristics = "Typical urban traffic with economy and mid-range vehicles."
        elif total_score >= 1:
            bracket = "LOWER-MIDDLE-INCOME"
            characteristics = "Mostly budget and economy vehicles."
        else:
            bracket = "LOW-INCOME"
            characteristics = "Dominated by budget vehicles and motorcycles."

        # Additional context based on motorcycle ratio
        ratio = result.motorcycle_to_car_ratio
        if ratio > 5:
            characteristics += " Very high motorcycle density typical of residential/traditional market areas."
        elif ratio > 3:
            characteristics += " High motorcycle presence suggests mixed commercial/residential area."
        elif ratio > 1:
            characteristics += " Balanced vehicle mix indicates commercial/office district."
        else:
            characteristics += " Higher car ratio suggests upper-class residential or business district."

        # Generate recommendations
        recommendations = self._get_recommendations(bracket, ratio, result)

        return {
            "bracket": bracket,
            "score": total_score,
            "characteristics": characteristics,
            "recommendations": recommendations,
        }

    def _get_moto_income_score(self, stats: VehicleStats) -> float:
        """Calculate income score from motorcycle statistics."""
        if stats.non_commercial_count == 0:
            return 0

        tier_scores = {
            "Budget": 1,
            "Mid": 2,
            "Premium": 4,
        }

        total_score = 0
        for tier, count in stats.tier_counts.items():
            total_score += tier_scores.get(tier, 1) * count

        return total_score / stats.non_commercial_count

    def _get_car_income_score(self, stats: VehicleStats) -> float:
        """Calculate income score from car statistics."""
        if stats.non_commercial_count == 0:
            return 0

        tier_scores = {
            "Budget": 1,
            "Economy": 2,
            "Mid-range": 3,
            "Premium": 4,
            "Luxury": 5,
        }

        total_score = 0
        for tier, count in stats.tier_counts.items():
            total_score += tier_scores.get(tier, 2) * count

        return total_score / stats.non_commercial_count

    def _get_recommendations(
        self,
        bracket: str,
        ratio: float,
        result: AnalysisResult,
    ) -> list[str]:
        """Generate business recommendations based on analysis."""
        recommendations = []

        if bracket in ["HIGH-INCOME", "UPPER-MIDDLE-INCOME"]:
            recommendations.extend([
                "Premium dining and cafes",
                "Boutique retail shops",
                "High-end service businesses",
                "Professional services (clinics, salons)",
            ])
        elif bracket == "MIDDLE-INCOME":
            recommendations.extend([
                "Quick service restaurants",
                "Coffee shops",
                "Convenience stores",
                "Mid-range retail",
            ])
        else:
            recommendations.extend([
                "Warung / street food",
                "Convenience stores (Indomaret/Alfamart)",
                "Budget retail",
                "Phone accessories and repair",
            ])

        # Add motorcycle-specific recommendations
        if ratio > 3:
            recommendations.append("Motorcycle service and accessories")
            recommendations.append("Helmet and riding gear shop")

        # Commercial vehicle presence
        if result.motorcycle_stats.commercial_count > result.motorcycle_stats.total_count * 0.2:
            recommendations.append("Food for delivery (ghost kitchen)")

        return recommendations[:5]  # Limit to top 5
