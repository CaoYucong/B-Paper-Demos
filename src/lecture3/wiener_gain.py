"""Scalar Wiener and inverse gains for one frequency."""

from __future__ import annotations

import numpy as np


def scalar_gains(magnitude: np.ndarray, gain_k: float) -> tuple[np.ndarray, np.ndarray]:
    """Return Wiener gain H*/(|H|²+K) and the naive inverse 1/H for real nonnegative H.

    Where H is 0, the inverse gain is recorded as infinity and is not used
    as a finite restoration weight.
    """
    if gain_k < 0:
        raise ValueError("K must be nonnegative.")
    h = np.asarray(magnitude, dtype=np.float64)
    wiener = h / (h * h + float(gain_k))
    inverse = np.full(h.shape, np.inf, dtype=np.float64)
    usable = h > 0
    inverse[usable] = 1.0 / h[usable]
    return wiener, inverse


def frequency_bin(shape: tuple[int, int], u: float, v: float) -> tuple[int, int]:
    """Nearest DFT bin to frequency (u, v) in cycles per pixel. Returns (row, col)."""
    height, width = shape
    row = int(np.rint(float(v) * height)) % height
    col = int(np.rint(float(u) * width)) % width
    return row, col
