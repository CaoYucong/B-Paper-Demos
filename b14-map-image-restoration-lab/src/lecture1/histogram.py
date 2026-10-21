"""Histogram equalisation. It changes intensities, not pixel positions."""

from __future__ import annotations

import numpy as np


def coin_image(n: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Darker disk with a faint emblem on a brighter, slightly uneven background."""
    size = int(n)
    yy, xx = np.mgrid[:size, :size]
    centre = (size - 1) / 2.0
    radius = np.sqrt((yy - centre) ** 2 + (xx - centre) ** 2)
    disk = radius <= 0.28 * size
    image = np.full((size, size), 0.72, dtype=np.float64)
    image += 0.05 * (xx / max(size - 1, 1))
    emblem = 0.28 + 0.05 * np.sin(2.0 * np.pi * xx / 11.0) * np.sin(2.0 * np.pi * yy / 11.0)
    image[disk] = emblem[disk]
    return image, disk


def _bin_index(values: np.ndarray, levels: int) -> np.ndarray:
    scaled = np.floor(np.clip(values, 0.0, 1.0 - 1e-12) * levels)
    return np.clip(scaled, 0, levels - 1).astype(np.int64)


def equalise_values(values: np.ndarray, levels: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map samples with s(k) = (1 / N) * sum_{j=0}^{k} n_j.

    A region that already occupies a single bin is returned unchanged.
    """
    x = np.asarray(values, dtype=np.float64)
    levels = int(levels)
    bins = _bin_index(x, levels)
    counts = np.bincount(bins.ravel(), minlength=levels).astype(np.float64)
    total = float(counts.sum())
    cdf = np.cumsum(counts) / total if total else np.zeros(levels)
    occupied = np.flatnonzero(counts)
    if occupied.size <= 1:
        mapped = x.copy()
    else:
        mapped = cdf[bins]
    return mapped.reshape(x.shape), counts, cdf


def equalise_image(image: np.ndarray, levels: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mapped, counts, cdf = equalise_values(np.asarray(image, dtype=np.float64), levels=levels)
    return mapped, counts, cdf


def equalise_regions(image: np.ndarray, mask: np.ndarray, levels: int = 256) -> np.ndarray:
    """Equalise the True and False regions with separate histograms."""
    image = np.asarray(image, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    out = np.empty_like(image)
    if np.any(mask):
        out[mask] = equalise_values(image[mask], levels=levels)[0]
    if np.any(~mask):
        out[~mask] = equalise_values(image[~mask], levels=levels)[0]
    return out
