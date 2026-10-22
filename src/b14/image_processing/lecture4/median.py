"""Median filter. The median of a neighbourhood is not a weighted sum."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def salt_and_pepper(image: np.ndarray, amount: float, seed: int) -> np.ndarray:
    """Replace a fraction of pixels by 0 and the same fraction's complement by 1.

    ``amount`` is the total fraction of pixels that are replaced.
    """
    image = np.array(image, dtype=np.float64, copy=True)
    amount = float(amount)
    if not 0.0 <= amount <= 1.0:
        raise ValueError("amount must lie in [0, 1].")
    rng = np.random.default_rng(int(seed))
    count = int(round(amount * image.size))
    if count == 0:
        return image
    chosen = rng.choice(image.size, count, replace=False)
    flat = image.ravel()
    half = count // 2
    flat[chosen[:half]] = 0.0
    flat[chosen[half:]] = 1.0
    return image


def median(image: np.ndarray, size: int = 3) -> np.ndarray:
    size = int(size)
    if size % 2 == 0 or size < 1:
        raise ValueError("median window must have odd size.")
    return median_filter(np.asarray(image, dtype=np.float64), size=size, mode="reflect")


def window(image: np.ndarray, row: int, col: int, size: int = 3) -> np.ndarray:
    pad = int(size) // 2
    padded = np.pad(np.asarray(image, dtype=np.float64), pad, mode="reflect")
    return padded[int(row) : int(row) + int(size), int(col) : int(col) + int(size)]


def sorted_median(values: np.ndarray) -> tuple[np.ndarray, float]:
    ordered = np.sort(np.asarray(values, dtype=np.float64).ravel())
    return ordered, float(ordered[ordered.size // 2])
