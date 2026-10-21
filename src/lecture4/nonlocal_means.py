"""Non-local means on a small image.

For every offset inside the search window, the weight is

    exp( - mean_{patch} (f - shifted f)² / h² ).

A constant image is therefore unchanged. This is the pixel-centred form
used for teaching; it is not a fast approximate implementation.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from src.lecture4.bilateral import shift_reflect


def nonlocal_means(
    image: np.ndarray,
    patch: int,
    search: int,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the denoised image and the search-window weights at the centre pixel."""
    if patch % 2 == 0 or search % 2 == 0 or patch < 1 or search < 1:
        raise ValueError("patch and search sizes must be positive and odd.")
    if strength <= 0:
        raise ValueError("filtering strength must be positive.")
    image = np.asarray(image, dtype=np.float64)
    accumulator = np.zeros_like(image)
    weight_sum = np.zeros_like(image)
    half = search // 2
    centre_weight = np.zeros((search, search), dtype=np.float64)
    ref_y, ref_x = image.shape[0] // 2, image.shape[1] // 2
    for iy, dy in enumerate(range(-half, half + 1)):
        for ix, dx in enumerate(range(-half, half + 1)):
            shifted = shift_reflect(image, dy, dx)
            distance = uniform_filter((image - shifted) ** 2, size=int(patch), mode="reflect")
            weight = np.exp(-distance / float(strength) ** 2)
            accumulator += weight * shifted
            weight_sum += weight
            centre_weight[iy, ix] = weight[ref_y, ref_x]
    return accumulator / weight_sum, centre_weight
