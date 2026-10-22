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


def reflect_coord(index: np.ndarray, length: int) -> np.ndarray:
    """Fold an index the way ``np.pad(..., mode='reflect')`` does."""
    index = np.asarray(index)
    if length <= 1:
        return np.zeros(index.shape, dtype=int)
    period = 2 * (length - 1)
    folded = np.mod(index, period)
    reflected = np.where(folded < length, folded, period - folded)
    return reflected.astype(int)


def reflect_window(image: np.ndarray, row: int, col: int, half: int) -> np.ndarray:
    """Square neighbourhood of pixel ``(row, col)``. The centre sample is that pixel.

    ``window[half + dy, half + dx]`` is the sample at ``(row + dy, col + dx)``,
    reflected outside the image. ``shift_reflect`` subtracts its shift; the
    Gaussian is even, so the two sums are the same.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("image must be two-dimensional.")
    half = int(half)
    if half < 0:
        raise ValueError("half-width must be nonnegative.")
    row = int(row)
    col = int(col)
    height, width = image.shape
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError("pixel is outside the image.")
    offsets = np.arange(-half, half + 1)
    rows = row + offsets[:, None]
    cols = col + offsets[None, :]
    return image[reflect_coord(rows, height), reflect_coord(cols, width)]


def bilateral_weight_maps(
    image: np.ndarray,
    row: int,
    col: int,
    sigma_spatial: float,
    sigma_range: float,
    half: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unnormalised factors of the bilateral sum on a square around one pixel.

    Returns the image window, the spatial Gaussian, the range Gaussian, and
    their pointwise product. The centre of each array is ``(row, col)``.
    Dividing the product by its sum is the ``1/W`` of the filter; this
    function does not divide.
    """
    if sigma_spatial <= 0 or sigma_range <= 0:
        raise ValueError("both sigmas must be positive.")
    window = reflect_window(image, row, col, half)
    half = int(half)
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    spatial = np.exp(-0.5 * (dy * dy + dx * dx) / float(sigma_spatial) ** 2)
    centre = float(window[half, half])
    range_weight = np.exp(-0.5 * (centre - window) ** 2 / float(sigma_range) ** 2)
    return window, spatial, range_weight, spatial * range_weight


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
