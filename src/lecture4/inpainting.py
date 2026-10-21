"""Diffusion inpainting. Missing pixels are replaced by neighbour averages.

A linear ramp is harmonic, so this iteration can reconstruct it from its
boundary. The result is one completion, not a proof that the original
pixels were uniquely determined.
"""

from __future__ import annotations

import numpy as np


def diffuse(image: np.ndarray, missing: np.ndarray, iterations: int) -> np.ndarray:
    """Iterate the four-neighbour average on pixels where ``missing`` is true.

    Known pixels are reset to the damaged image's original known values
    after every iteration. The boundary condition is reflection.
    """
    if int(iterations) < 1:
        raise ValueError("iteration count must be positive.")
    known_image = np.asarray(image, dtype=np.float64)
    missing = np.asarray(missing, dtype=bool)
    current = known_image.copy()
    current[missing] = 0.0
    height, width = current.shape
    for _ in range(int(iterations)):
        padded = np.pad(current, 1, mode="edge")
        average = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        ) / 4.0
        current = np.where(missing, average, known_image)
    if current.shape != (height, width):
        raise RuntimeError("inpainting changed the image shape.")
    return current


def rectangle_mask(shape: tuple[int, int], row: int, col: int, height: int, width: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[row : row + height, col : col + width] = True
    return mask
