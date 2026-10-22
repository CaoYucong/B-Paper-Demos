"""Frequency-domain masks. Cutoff is in cycles per pixel."""

from __future__ import annotations

import numpy as np

from src.b14.image_processing.lecture2.spatial_frequency import dft


def frequency_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    v = np.fft.fftfreq(height)[:, None]
    u = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(u * u + v * v)
    return u, v, radius


def frequency_mask(
    shape: tuple[int, int],
    kind: str,
    cutoff: float,
    width: float = 0.05,
    angle: float = 0.0,
) -> np.ndarray:
    """Real multiplicative mask aligned with ``numpy.fft.fft2`` (DC at index 0)."""
    _u, v, radius = frequency_grid(shape)
    u = _u
    cutoff = max(float(cutoff), 1e-8)
    width = max(float(width), 1e-8)
    if kind == "ideal_low":
        mask = (radius <= cutoff).astype(np.float64)
    elif kind == "gaussian_low":
        mask = np.exp(-0.5 * (radius / cutoff) ** 2)
    elif kind == "ideal_high":
        mask = (radius >= cutoff).astype(np.float64)
    elif kind == "gaussian_high":
        mask = 1.0 - np.exp(-0.5 * (radius / cutoff) ** 2)
    elif kind == "band":
        mask = np.exp(-0.5 * ((radius - cutoff) / width) ** 2)
    elif kind == "directional":
        orientation = np.arctan2(v, u)
        delta = np.angle(np.exp(1j * (orientation - float(angle))))
        opposite = np.angle(np.exp(1j * (orientation - float(angle) + np.pi)))
        mask = np.exp(-0.5 * (np.minimum(np.abs(delta), np.abs(opposite)) / width) ** 2)
        mask[0, 0] = 0.0
    else:
        raise ValueError(kind)
    return mask


def apply_frequency_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(dft(image) * mask).real


def hybrid_image(image_low: np.ndarray, image_high: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Low-pass of one image plus high-pass of another. Both filters are Gaussian."""
    low = apply_frequency_mask(image_low, frequency_mask(image_low.shape, "gaussian_low", cutoff))
    high = apply_frequency_mask(image_high, frequency_mask(image_high.shape, "gaussian_high", cutoff))
    return low, high, low + high
