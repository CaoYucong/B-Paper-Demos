"""Non-local means on a small image.

For every offset inside the search window, the weight is

    exp( - mean_{patch} (f - shifted f)² / h² ).

A constant image is therefore unchanged. This is the pixel-centred form
used for teaching; it is not a fast approximate implementation.

``weight_map`` places those weights in image coordinates for one chosen
pixel, so a similar patch lights up where it actually sits.
``repeated_marks`` is a supplementary picture for that explanation.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

from src.b14.image_processing.lecture4.bilateral import reflect_coord, reflect_window, shift_reflect

MARK_SIZE = 61
DIAGONAL_A = (30, 30)
DIAGONAL_B = (30, 46)
BAR = (24, 30)


def _require_patch(patch: int, strength: float) -> None:
    if int(patch) % 2 == 0 or int(patch) < 1:
        raise ValueError("patch size must be a positive odd integer.")
    if strength <= 0:
        raise ValueError("filtering strength must be positive.")


def _require_search(search: int) -> None:
    if int(search) % 2 == 0 or int(search) < 1:
        raise ValueError("search size must be a positive odd integer.")


def _stamp(image: np.ndarray, centre: tuple[int, int], pattern: np.ndarray) -> None:
    row, col = centre
    half = pattern.shape[0] // 2
    image[row - half : row + half + 1, col - half : col + half + 1] = pattern


def _diagonal_pattern() -> np.ndarray:
    pattern = np.full((5, 5), 0.25, dtype=np.float64)
    np.fill_diagonal(pattern, 0.85)
    return pattern


def _bar_pattern() -> np.ndarray:
    pattern = np.full((5, 5), 0.25, dtype=np.float64)
    pattern[2, :] = 0.85
    return pattern


def repeated_marks(noise: float = 0.05, seed: int = 0) -> np.ndarray:
    """Two copies of a diagonal and one horizontal bar, on a flat background.

    Supplementary teaching picture. It is not a photograph from the lecture.
    The second diagonal sits 16 columns to the right of the first, so a search
    window of 33 reaches it and a window of 15 does not. The bar is 6 rows
    above the first diagonal, so both of those windows still contain it.
    """
    if noise < 0:
        raise ValueError("noise must be non-negative.")
    image = np.full((MARK_SIZE, MARK_SIZE), 0.25, dtype=np.float64)
    _stamp(image, DIAGONAL_A, _diagonal_pattern())
    _stamp(image, DIAGONAL_B, _diagonal_pattern())
    _stamp(image, BAR, _bar_pattern())
    if noise > 0:
        rng = np.random.default_rng(int(seed))
        image = image + float(noise) * rng.standard_normal(image.shape)
    return np.clip(image, 0.0, 1.0)


def compare_patches(
    image: np.ndarray,
    row_p: int,
    col_p: int,
    row_q: int,
    col_q: int,
    patch: int,
    strength: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return the two patches, their mean squared difference, and the weight of q."""
    _require_patch(patch, strength)
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("image must be two-dimensional.")
    height, width = image.shape
    for row, col in ((row_p, col_p), (row_q, col_q)):
        if not (0 <= int(row) < height and 0 <= int(col) < width):
            raise ValueError("pixel is outside the image.")
    half = int(patch) // 2
    patch_p = reflect_window(image, int(row_p), int(col_p), half)
    patch_q = reflect_window(image, int(row_q), int(col_q), half)
    distance = float(np.mean((patch_p - patch_q) ** 2))
    weight = float(np.exp(-distance / float(strength) ** 2))
    return patch_p, patch_q, distance, weight


def weight_map(
    image: np.ndarray,
    row: int,
    col: int,
    patch: int,
    search: int,
    strength: float,
) -> np.ndarray:
    """Weights for one pixel, in image coordinates.

    ``out[q_row, q_col]`` is the weight of that pixel in the average that
    replaces ``(row, col)``. Pixels outside the search window, and outside
    the image, stay 0. The patch is the reflected window used by
    :func:`compare_patches`. This agrees with :func:`nonlocal_means` wherever
    every patch in the search window lies inside the image.
    """
    _require_patch(patch, strength)
    _require_search(search)
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("image must be two-dimensional.")
    height, width = image.shape
    row, col = int(row), int(col)
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError("pixel is outside the image.")
    half_s = int(search) // 2
    half_p = int(patch) // 2
    reference = reflect_window(image, row, col, half_p)
    rows = np.arange(row - half_s, row + half_s + 1)
    cols = np.arange(col - half_s, col + half_s + 1)
    rows = rows[(rows >= 0) & (rows < height)]
    cols = cols[(cols >= 0) & (cols < width)]
    patch_rows = np.arange(-half_p, half_p + 1)
    patch_cols = np.arange(-half_p, half_p + 1)
    sample_rows = reflect_coord(rows[:, None, None, None] + patch_rows[None, None, :, None], height)
    sample_cols = reflect_coord(cols[None, :, None, None] + patch_cols[None, None, None, :], width)
    patches = image[sample_rows, sample_cols]
    distance = np.mean((patches - reference) ** 2, axis=(2, 3))
    weights = np.exp(-distance / float(strength) ** 2)
    out = np.zeros_like(image)
    out[np.ix_(rows, cols)] = weights
    return out


def matching_patches(weights: np.ndarray, patch: int, threshold: float = 0.2) -> np.ndarray:
    """Spread every weight above ``threshold`` across that pixel's patch.

    The filter averages the centre pixel, but the reason for the weight is the
    whole patch. Painting the patch makes the match visible.
    """
    _require_patch(int(patch), strength=1.0)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("weights must be two-dimensional.")
    cores = np.where(weights >= float(threshold), weights, 0.0)
    return maximum_filter(cores, size=int(patch), mode="constant", cval=0.0)


def weighted_value(image: np.ndarray, weights: np.ndarray) -> float:
    """Normalised average of an image under a weight map."""
    image = np.asarray(image, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if image.shape != weights.shape:
        raise ValueError("weights must have the same shape as the image.")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("weights must have a positive sum.")
    return float(np.sum(weights * image) / total)


def nonlocal_means(
    image: np.ndarray,
    patch: int,
    search: int,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the denoised image and the search-window weights at the centre pixel.

    ``centre_weight[iy, ix]`` belongs to the contributor at
    ``(centre_row - dy, centre_col - dx)``, with ``dy = iy - search // 2``.
    Use :func:`weight_map` when the weights need to line up with the picture.
    """
    _require_patch(patch, strength)
    _require_search(search)
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
