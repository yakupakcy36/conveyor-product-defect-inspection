"""Configuration settings for Conveyor Product Defect Inspection Engine."""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ConveyorConfig:
    """Master configuration for camera resolution, conveyor speed, defect thresholds, and sorting."""
    image_width: int = 256
    image_height: int = 256
    conveyor_speed_m_s: float = 1.5
    min_defect_area_px: int = 20
    rejection_anomaly_threshold: float = 0.32
    target_ppm: int = 120  # Target Parts Per Minute throughput

    defect_classes: List[str] = field(default_factory=lambda: [
        "Normal",
        "Puncture / Leak",
        "Crushed Corner",
        "Label Misalignment",
        "Surface Crack",
    ])


DEFAULT_CONVEYOR_CONFIG = ConveyorConfig()
