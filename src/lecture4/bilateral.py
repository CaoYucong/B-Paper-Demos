"""Bilateral filter: spatial Gaussian times a range Gaussian.

The implementation shifts the whole image, so it is exact for a square
window and uses reflection at the border. It is still a weighted average,
but the weights depend on the image, which is what makes it nonlinear.
"""

from __future__ import annotations

import numpy as np


def shift_reflect(image: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    height, width = image.shape
    padded = np.pad(image, ((height, height), (width, width)), mode="reflect")
    y0 = height - int(shift_y)
    x0 = width - int(shift_x)
    return padded[y0 : y0 + height, x0 : x0 + width]


def bilateral_filter(
    image: np.ndarray,
    sigma_spatial: float,
    sigma_range: float,
    radius: int,
) -> np.ndarray:
    if sigma_spatial <= 0 or sigma_range <= 0:
        raise ValueError("both sigmas must be positive.")
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be nonnegative.")
    image = np.asarray(image, dtype=np.float64)
    accumulator = np.zeros_like(image)
    weight_sum = np.zeros_like(image)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = shift_reflect(image, dy, dx)
            spatial = np.exp(-0.5 * (dy * dy + dx * dx) / float(sigma_spatial) ** 2)
            range_weight = np.exp(-0.5 * (image - shifted) ** 2 / float(sigma_range) ** 2)
            weight = spatial * range_weight
            accumulator += weight * shifted
            weight_sum += weight
    return accumulator / weight_sum
