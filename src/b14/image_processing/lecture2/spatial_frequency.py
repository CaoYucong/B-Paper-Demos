"""Sinusoidal images and the display convention for their spectra."""

from __future__ import annotations

import numpy as np


def sinusoid(
    shape: tuple[int, int],
    u: float,
    v: float,
    phase: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """f[y, x] = A cos(2π (u x + v y) + φ).

    ``u`` and ``v`` are cycles per pixel. ``x`` increases to the right and
    ``y`` increases downward, in row/column index order.
    """
    height, width = shape
    y = np.arange(height, dtype=np.float64)[:, None]
    x = np.arange(width, dtype=np.float64)[None, :]
    return float(amplitude) * np.cos(2.0 * np.pi * (float(u) * x + float(v) * y) + float(phase))


def dft(image: np.ndarray) -> np.ndarray:
    """Unnormalised 2D DFT, the same convention as ``numpy.fft.fft2``."""
    return np.fft.fft2(np.asarray(image, dtype=np.float64))


def shifted_magnitude_phase(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the raw DFT, fftshifted magnitude, and fftshifted phase."""
    spectrum = dft(image)
    return spectrum, np.fft.fftshift(np.abs(spectrum)), np.fft.fftshift(np.angle(spectrum))


def log_magnitude(magnitude: np.ndarray) -> np.ndarray:
    """Display compression log10(1 + |F|). This is not the DFT itself."""
    return np.log10(1.0 + np.asarray(magnitude, dtype=np.float64))
