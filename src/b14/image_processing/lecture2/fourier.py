"""Reconstruction from selected DFT coefficients, and magnitude/phase swaps."""

from __future__ import annotations

import numpy as np

from src.b14.image_processing.lecture2.spatial_frequency import dft


def _partner(y: int, x: int, height: int, width: int) -> tuple[int, int]:
    return (-y) % height, (-x) % width


def keep_conjugate_pairs(spectrum: np.ndarray, pairs: int) -> np.ndarray:
    """Keep the largest-magnitude coefficients together with their conjugates.

    A real image has F[k] = conj(F[-k]). Dropping one of the pair makes the
    inverse DFT complex. ``pairs`` counts unique unordered pairs; the DC
    term is its own partner and counts as one pair.
    """
    spectrum = np.asarray(spectrum)
    height, width = spectrum.shape
    magnitude = np.abs(spectrum)
    order = np.argsort(magnitude.ravel())[::-1]
    keep = np.zeros(spectrum.shape, dtype=bool)
    chosen = 0
    for index in order:
        if chosen >= int(pairs):
            break
        y, x = divmod(int(index), width)
        if keep[y, x]:
            continue
        keep[y, x] = True
        py, px = _partner(y, x, height, width)
        keep[py, px] = True
        chosen += 1
    return np.where(keep, spectrum, 0.0)


def reconstruct_pairs(image: np.ndarray, pairs: int) -> tuple[np.ndarray, np.ndarray]:
    spectrum = dft(image)
    kept = keep_conjugate_pairs(spectrum, pairs)
    reconstructed = np.fft.ifft2(kept).real
    return reconstructed, np.fft.fftshift(np.abs(kept))


def combine_magnitude_phase(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Inverse DFT of |F| exp(i φ). Zero-magnitude samples have phase 0 if omitted upstream."""
    spectrum = np.asarray(magnitude, dtype=np.float64) * np.exp(1j * np.asarray(phase, dtype=np.float64))
    return np.fft.ifft2(spectrum).real


def magnitude_only(image: np.ndarray) -> np.ndarray:
    spectrum = dft(image)
    return combine_magnitude_phase(np.abs(spectrum), np.zeros(spectrum.shape))


def phase_only(image: np.ndarray) -> np.ndarray:
    """Unit magnitude, with phase 0 wherever the DFT magnitude is numerically zero."""
    spectrum = dft(image)
    phase = np.angle(spectrum)
    phase = np.where(np.abs(spectrum) < 1e-12, 0.0, phase)
    return combine_magnitude_phase(np.ones(spectrum.shape), phase)


def swap_magnitude_phase(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    spectrum_a = dft(image_a)
    spectrum_b = dft(image_b)
    phase_b = np.where(np.abs(spectrum_b) < 1e-12, 0.0, np.angle(spectrum_b))
    return combine_magnitude_phase(np.abs(spectrum_a), phase_b)


def reconstruction_error(image: np.ndarray) -> float:
    """Max absolute error of the unmodified inverse DFT. It should be numerical noise."""
    return float(np.max(np.abs(np.fft.ifft2(dft(image)).real - np.asarray(image, dtype=np.float64))))


def linearity_error(image_a: np.ndarray, image_b: np.ndarray, scale_a: float, scale_b: float) -> float:
    left = dft(scale_a * image_a + scale_b * image_b)
    right = scale_a * dft(image_a) + scale_b * dft(image_b)
    return float(np.max(np.abs(left - right)))


def translation_phase_error(image: np.ndarray, shift_y: int, shift_x: int) -> float:
    """||F{shifted} - F{image} exp(-2πi (v Δy + u Δx))||_max."""
    image = np.asarray(image, dtype=np.float64)
    shifted = np.roll(np.roll(image, int(shift_y), axis=0), int(shift_x), axis=1)
    height, width = image.shape
    v = np.fft.fftfreq(height)[:, None]
    u = np.fft.fftfreq(width)[None, :]
    predicted = dft(image) * np.exp(-2j * np.pi * (v * int(shift_y) + u * int(shift_x)))
    return float(np.max(np.abs(dft(shifted) - predicted)))


def conjugate_symmetry_error(image: np.ndarray) -> float:
    """Max |F[k] - conj(F[-k])| for a real image."""
    spectrum = dft(image)
    height, width = spectrum.shape
    rows = (-np.arange(height)) % height
    cols = (-np.arange(width)) % width
    partner = np.conj(spectrum[np.ix_(rows, cols)])
    return float(np.max(np.abs(spectrum - partner)))
