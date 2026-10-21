"""Downsampling, aliasing, and a two-grating Moiré pattern."""

from __future__ import annotations

import numpy as np

from src.lecture1.gaussian import gaussian_filter


def stripe_image(n: int, frequency: float) -> np.ndarray:
    """Vertical cosine bars. ``frequency`` is cycles per pixel."""
    x = np.arange(int(n), dtype=np.float64)
    row = 0.5 + 0.5 * np.cos(2.0 * np.pi * float(frequency) * x)
    return np.broadcast_to(row, (int(n), int(n))).copy()


def checkerboard(n: int, period: int) -> np.ndarray:
    period = max(int(period), 1)
    y = np.arange(int(n))[:, None]
    x = np.arange(int(n))[None, :]
    return ((y // period + x // period) % 2).astype(np.float64)


def downsample(image: np.ndarray, factor: int, prefilter_sigma: float | None = None) -> np.ndarray:
    """Point-sample every ``factor`` pixels, optionally after a Gaussian prefilter."""
    image = np.asarray(image, dtype=np.float64)
    factor = int(factor)
    if prefilter_sigma is not None and prefilter_sigma > 0:
        size = int(2 * np.ceil(3 * prefilter_sigma) + 1)
        if size % 2 == 0:
            size += 1
        image, _kernel = gaussian_filter(image, size=size, sigma=float(prefilter_sigma), boundary="reflect")
    return image[::factor, ::factor].copy()


def moire(n: int, frequency: float, angle_degrees: float) -> np.ndarray:
    """Sum of a horizontal grating and a rotated copy. A small angle produces beats."""
    size = int(n)
    y = np.arange(size, dtype=np.float64)[:, None]
    x = np.arange(size, dtype=np.float64)[None, :]
    theta = np.deg2rad(float(angle_degrees))
    first = np.cos(2.0 * np.pi * float(frequency) * x)
    rotated = np.cos(2.0 * np.pi * float(frequency) * (x * np.cos(theta) + y * np.sin(theta)))
    return 0.5 + 0.25 * (first + rotated)


def dominant_frequency(image: np.ndarray) -> tuple[float, float]:
    """Largest non-DC magnitude bin, as cycles per pixel (u, v)."""
    spectrum = np.fft.fft2(np.asarray(image, dtype=np.float64))
    spectrum[0, 0] = 0.0
    index = int(np.argmax(np.abs(spectrum)))
    y, x = divmod(index, spectrum.shape[1])
    v = float(np.fft.fftfreq(spectrum.shape[0])[y])
    u = float(np.fft.fftfreq(spectrum.shape[1])[x])
    return u, v
