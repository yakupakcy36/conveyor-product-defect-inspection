"""Test suite for conveyor defect detector, classification, and pneumatic rejection."""

import pytest
from src.config import DEFAULT_CONVEYOR_CONFIG
from src.generator import generate_conveyor_frame, inject_product_defect
from src.detector import ConveyorDefectDetector


@pytest.fixture
def detector():
    return ConveyorDefectDetector(config=DEFAULT_CONVEYOR_CONFIG)


def test_clean_product_passes(detector):
    clean = generate_conveyor_frame(width=256, height=256, seed=42)
    res = detector.inspect_frame(clean, unit_id="TEST_PASS")
    assert res.pass_status == "PASS"
    assert res.rejection_actuator_triggered is False
    assert len(res.detected_defects) == 0
    assert res.primary_defect_type == "Normal"


def test_puncture_defect_rejected(detector):
    clean = generate_conveyor_frame(width=256, height=256, seed=42)
    defective, _ = inject_product_defect(clean, defect_type="puncture", seed=42)
    res = detector.inspect_frame(defective, unit_id="TEST_PUNCTURE")
    assert res.pass_status == "REJECT"
    assert res.rejection_actuator_triggered is True
    assert res.primary_defect_type == "Puncture / Leak"
    assert len(res.detected_defects) >= 1


def test_crushed_corner_defect_rejected(detector):
    clean = generate_conveyor_frame(width=256, height=256, seed=42)
    defective, _ = inject_product_defect(clean, defect_type="crushed_corner", seed=42)
    res = detector.inspect_frame(defective, unit_id="TEST_CRUSH")
    assert res.pass_status == "REJECT"
    assert res.rejection_actuator_triggered is True
    assert res.primary_defect_type == "Crushed Corner"


def test_label_misalignment_rejected(detector):
    clean = generate_conveyor_frame(width=256, height=256, seed=42)
    defective, _ = inject_product_defect(clean, defect_type="label_misalignment", seed=42)
    res = detector.inspect_frame(defective, unit_id="TEST_LABEL")
    assert res.pass_status == "REJECT"
    assert res.primary_defect_type == "Label Misalignment"


def test_crack_defect_rejected(detector):
    clean = generate_conveyor_frame(width=256, height=256, seed=42)
    defective, _ = inject_product_defect(clean, defect_type="crack", seed=42)
    res = detector.inspect_frame(defective, unit_id="TEST_CRACK")
    assert res.pass_status == "REJECT"
    assert res.primary_defect_type == "Surface Crack"


def test_inspection_latency_sub_30ms(detector):
    clean = generate_conveyor_frame(width=256, height=256, seed=42)
    res = detector.inspect_frame(clean)
    assert res.latency_ms < 30.0
