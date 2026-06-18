"""Test suite for conveyor frame generator and defect injection."""

import numpy as np
import pytest
from src.generator import generate_conveyor_frame, inject_product_defect


def test_conveyor_frame_properties():
    frame = generate_conveyor_frame(width=256, height=256, seed=42)
    assert frame.shape == (256, 256)
    assert frame.dtype == np.uint8
    assert 20 <= frame.min() <= frame.max() <= 255


def test_defect_injections_differ_from_base():
    base = generate_conveyor_frame(width=256, height=256, seed=42)
    for dt in ["puncture", "crushed_corner", "label_misalignment", "crack"]:
        defective, label = inject_product_defect(base, defect_type=dt, seed=42)
        assert defective.shape == base.shape
        assert not np.array_equal(base, defective)
        assert label != "Normal"
