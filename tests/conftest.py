import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def small_wafers():
    """Create 6 synthetic wafer maps of different sizes as uint8 arrays.

    Each wafer has values 0 (outside circle), 1 (inside circle), and a few 2s (defects).
    Sizes: 26x26, 33x31, 45x48, 64x64, 71x66, 100x90.
    """
    np.random.seed(42)
    sizes = [(26, 26), (33, 31), (45, 48), (64, 64), (71, 66), (100, 90)]
    wafers = []

    for h, w in sizes:
        # Create a circular mask: 1 inside, 0 outside
        y = np.arange(h)[:, None]
        x = np.arange(w)[None, :]
        center_y, center_x = h / 2, w / 2
        radius = min(h, w) / 3
        circle = (y - center_y) ** 2 + (x - center_x) ** 2 <= radius ** 2

        wafer = np.zeros((h, w), dtype=np.uint8)
        wafer[circle] = 1

        # Add a few defects (value 2) randomly within the circle
        num_defects = max(1, h // 20)
        circle_indices = np.argwhere(circle)
        if len(circle_indices) > 0:
            defect_locs = circle_indices[np.random.choice(len(circle_indices), min(num_defects, len(circle_indices)), replace=False)]
            for dy, dx in defect_locs:
                wafer[dy, dx] = 2

        wafers.append(wafer)

    return wafers
