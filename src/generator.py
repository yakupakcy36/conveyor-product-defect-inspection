"""Synthetic conveyor belt camera frame and damaged product generator."""

from typing import Tuple
import numpy as np
from scipy import ndimage


def generate_conveyor_frame(
    width: int = 256,
    height: int = 256,
    seed: int = 42,
) -> np.ndarray:
    """
    Generates a camera frame of a manufactured packaged container
    centered on an industrial moving conveyor belt.
    """
    rng = np.random.default_rng(seed)

    # 1. Conveyor belt background (dark rubber with horizontal motion grain)
    belt = rng.normal(loc=45.0, scale=4.0, size=(height, width))
    belt_grain = rng.normal(loc=0.0, scale=3.0, size=(1, width))
    belt = np.clip(belt + np.repeat(belt_grain, height, axis=0), 30.0, 60.0)

    # 2. Product container boundaries (centered product: y: 40-216, x: 58-198)
    product_mask = np.zeros((height, width), dtype=bool)
    py0, py1 = 40, 216
    px0, px1 = 58, 198
    product_mask[py0:py1, px0:px1] = True

    # Product body (metallic / composite gray)
    body = rng.normal(loc=185.0, scale=6.0, size=(height, width))

    # Product top cap / seal (darker top band: y: 40-65)
    body[py0:65, px0:px1] = rng.normal(loc=110.0, scale=4.0, size=(65 - py0, px1 - px0))

    # Product front label (bright rectangular label: y: 95-175, x: 80-176)
    body[95:175, 80:176] = rng.normal(loc=235.0, scale=3.0, size=(80, 96))

    frame = np.where(product_mask, body, belt)
    # Slight Gaussian filter to simulate factory optical lens
    frame = ndimage.gaussian_filter(frame, sigma=0.8)
    return np.clip(frame, 0.0, 255.0).astype(np.uint8)


def inject_product_defect(
    base_frame: np.ndarray,
    defect_type: str = "puncture",
    seed: int = 42,
) -> Tuple[np.ndarray, str]:
    """
    Injects a realistic manufacturing packaging defect.
    Defect types: 'normal', 'puncture', 'crushed_corner', 'label_misalignment', 'crack'
    """
    rng = np.random.default_rng(seed)
    img = base_frame.copy().astype(float)
    h, w = img.shape
    dt = defect_type.lower()

    if dt == "puncture":
        # Puncture hole / leak on the container body
        cy = rng.integers(low=110, high=190)
        cx = rng.integers(low=85, high=170)
        radius = rng.integers(low=7, high=13)
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        hole = dist <= radius
        img[hole] = np.clip(img[hole] - 140.0, 20.0, 255.0)
        label = "Puncture / Leak"

    elif dt == "crushed_corner":
        # Structural collapse at corner (e.g. top-right or bottom-right corner)
        cy, cx = 42, 195  # top-right corner of container
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        crush_mask = dist <= 24
        # Corner replaced by conveyor background illumination
        img[crush_mask] = rng.normal(loc=45.0, scale=4.0, size=img[crush_mask].shape)
        label = "Crushed Corner"

    elif dt == "label_misalignment":
        # Label shift / tear (corrupts front label zone y: 95-175, x: 80-176)
        label_mask = np.zeros((h, w), dtype=bool)
        label_mask[120:170, 80:135] = True
        img[label_mask] = np.clip(img[label_mask] - 75.0, 0.0, 255.0)
        label = "Label Misalignment"

    elif dt == "crack":
        # Jagged crack across the container body
        cy = rng.integers(low=100, high=150)
        cx = rng.integers(low=70, high=120)
        curr_y, curr_x = cy, cx
        for _ in range(55):
            curr_y += rng.choice([-1, 0, 1])
            curr_x += rng.choice([1, 2])
            if 0 <= curr_y < h and 0 <= curr_x < w:
                img[curr_y, curr_x] = max(10.0, img[curr_y, curr_x] - 130.0)
                if curr_y + 1 < h:
                    img[curr_y + 1, curr_x] = max(10.0, img[curr_y + 1, curr_x] - 70.0)
        label = "Surface Crack"

    else:
        label = "Normal"

    output = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return output, label
