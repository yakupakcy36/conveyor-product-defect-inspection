"""Conveyor Defect Detection, Localization, and Pneumatic Rejection Actuator Engine."""

import time
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from scipy import ndimage

from src.config import DEFAULT_CONVEYOR_CONFIG, ConveyorConfig
from src.generator import generate_conveyor_frame


@dataclass
class DefectDetectionBox:
    defect_id: int
    y_min: int
    x_min: int
    y_max: int
    x_max: int
    area_px: int
    aspect_ratio: float
    defect_class: str
    confidence: float


@dataclass
class ConveyorInspectionResult:
    unit_id: str
    pass_status: str  # PASS or REJECT
    rejection_actuator_triggered: bool
    defect_severity: str  # NONE, MINOR, CRITICAL
    primary_defect_type: str
    anomaly_score: float
    detected_defects: List[DefectDetectionBox]
    latency_ms: float


class ConveyorDefectDetector:
    """
    Industrial vision inspection engine for assembly conveyor lines.
    Performs high-speed anomaly mapping, bounding-box localization,
    multi-class defect classification, and rejection actuator signaling.
    """

    def __init__(self, config: ConveyorConfig = DEFAULT_CONVEYOR_CONFIG):
        self.config = config
        self.golden_reference = generate_conveyor_frame(
            width=config.image_width,
            height=config.image_height,
            seed=42,
        )
        # Precompute golden edge gradients
        sx0 = ndimage.sobel(self.golden_reference.astype(float), axis=1)
        sy0 = ndimage.sobel(self.golden_reference.astype(float), axis=0)
        self.golden_grad = np.hypot(sx0, sy0)

    def calibrate(self, golden_frame: np.ndarray):
        """Calibrate nominal golden reference product image."""
        self.golden_reference = golden_frame.copy()
        sx0 = ndimage.sobel(self.golden_reference.astype(float), axis=1)
        sy0 = ndimage.sobel(self.golden_reference.astype(float), axis=0)
        self.golden_grad = np.hypot(sx0, sy0)

    def inspect_frame(self, frame: np.ndarray, unit_id: str = "unit_001") -> ConveyorInspectionResult:
        """
        Inspect a single camera snapshot from the conveyor belt.
        Executes sub-10ms inspection and determines sorting pneumatic trigger.
        """
        start = time.perf_counter()

        if frame.ndim == 3:
            gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        else:
            gray = frame.astype(np.uint8)

        # 1. Pixel Residual Difference against Golden Reference
        diff_abs = np.abs(gray.astype(float) - self.golden_reference.astype(float)) / 255.0

        # 2. Differential Spatial Gradient Discontinuity
        sx1 = ndimage.sobel(gray.astype(float), axis=1)
        sy1 = ndimage.sobel(gray.astype(float), axis=0)
        curr_grad = np.hypot(sx1, sy1)
        grad_diff = np.abs(curr_grad - self.golden_grad) / 255.0

        # 3. Fused Anomaly Intensity
        fused_anomaly = 0.60 * diff_abs + 0.40 * grad_diff
        global_anomaly_score = float(np.max(fused_anomaly))

        # 4. Binary Defect Mask Segmentation
        mask = fused_anomaly >= 0.20
        cleaned_mask = ndimage.binary_opening(mask, structure=np.ones((2, 2)))

        # 5. Connected Component Analysis for Defect Localization
        labeled_mask, num_features = ndimage.label(cleaned_mask)
        bounding_boxes: List[DefectDetectionBox] = []

        primary_defect = "Normal"
        max_area = 0

        for feat_idx in range(1, num_features + 1):
            coords = np.argwhere(labeled_mask == feat_idx)
            area = len(coords)

            if area < self.config.min_defect_area_px:
                continue

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            h = max(1, y_max - y_min)
            w = max(1, x_max - x_min)
            aspect_ratio = round(float(max(h, w) / max(1, min(h, w))), 2)

            cy = (y_min + y_max) // 2
            cx = (x_min + x_max) // 2

            # Spatial & Geometric Defect Classification
            if aspect_ratio >= 1.8:
                d_class = "Surface Crack"
                conf = 0.94
            elif (cy <= 68 or cy >= 195) and (cx <= 75 or cx >= 175):
                d_class = "Crushed Corner"
                conf = 0.95
            elif (110 <= cy <= 180) and (80 <= cx <= 140) and area >= 200:
                d_class = "Label Misalignment"
                conf = 0.92
            else:
                d_class = "Puncture / Leak"
                conf = 0.90

            if area > max_area:
                max_area = area
                primary_defect = d_class

            bounding_boxes.append(DefectDetectionBox(
                defect_id=feat_idx,
                y_min=int(y_min),
                x_min=int(x_min),
                y_max=int(y_max),
                x_max=int(x_max),
                area_px=int(area),
                aspect_ratio=aspect_ratio,
                defect_class=d_class,
                confidence=conf,
            ))

        has_defect = len(bounding_boxes) > 0
        if not has_defect:
            pass_status = "PASS"
            severity = "NONE"
            primary_defect = "Normal"
            rejection_trigger = False
        else:
            pass_status = "REJECT"
            rejection_trigger = True
            severity = "CRITICAL" if (max_area > 150 or primary_defect in ["Crushed Corner", "Puncture / Leak"]) else "MINOR"

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return ConveyorInspectionResult(
            unit_id=unit_id,
            pass_status=pass_status,
            rejection_actuator_triggered=rejection_trigger,
            defect_severity=severity,
            primary_defect_type=primary_defect,
            anomaly_score=round(global_anomaly_score, 4),
            detected_defects=bounding_boxes,
            latency_ms=round(elapsed_ms, 2),
        )
