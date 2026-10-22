"""Normalised Gaussian kernels and the separable implementation."""

from __future__ import annotations

import numpy as np

from src.b14.image_processing.lecture1.spatial_filters import correlate2d, correlate_separable


def gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    size = int(size)
    if size % 2 == 0 or size < 1:
        raise ValueError("kernel size must be a positive odd integer.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    radius = np.arange(size, dtype=np.float64) - (size // 2)
    kernel = np.exp(-0.5 * (radius / float(sigma)) ** 2)
    kernel /= kernel.sum()
    return kernel


def gaussian_kernel_2d(size: int, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel_1d(size, sigma)
    return np.outer(kernel, kernel)


def gaussian_filter(image: np.ndarray, size: int, sigma: float, boundary: str = "reflect") -> tuple[np.ndarray, np.ndarray]:
    kernel = gaussian_kernel_2d(size, sigma)
    return correlate2d(image, kernel, mode="same", boundary=boundary), kernel


def gaussian_filter_separable(
    image: np.ndarray, size: int, sigma: float, boundary: str = "reflect"
) -> tuple[np.ndarray, np.ndarray]:
    kernel = gaussian_kernel_1d(size, sigma)
    filtered = correlate_separable(image, kernel, kernel, boundary=boundary)
    return filtered, kernel


def multiplications_per_pixel(size: int) -> tuple[int, int]:
    """Direct k×k correlation versus two length-k correlations."""
    size = int(size)
    return size * size, 2 * size
