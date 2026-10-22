"""Directional motion blur on the same periodic convolution operator as Lecture 3."""

from __future__ import annotations

import numpy as np

from src.b14.image_processing.lecture3.operators import apply_otf, psf_to_otf


def directional_motion_psf(shape: tuple[int, int], length: int, angle_degrees: float) -> np.ndarray:
    """Unit-sum PSF along a line of ``length`` pixels starting at the image centre.

    Angle 0 points to the right and matches ``operators.motion_blur_psf``.
    The kernel is not symmetric, so A and Aᵀ differ.
    """
    length = int(length)
    if length < 1:
        raise ValueError("motion length must be at least 1.")
    height, width = int(shape[0]), int(shape[1])
    psf = np.zeros((height, width), dtype=np.float64)
    centre_y, centre_x = height // 2, width // 2
    theta = np.deg2rad(float(angle_degrees))
    for step in range(length):
        row = int(np.rint(centre_y + step * np.sin(theta))) % height
        col = int(np.rint(centre_x + step * np.cos(theta))) % width
        psf[row, col] += 1.0
    psf /= psf.sum()
    return psf


def apply_psf(image: np.ndarray, psf: np.ndarray, noise_std: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return blurred image, noise, and observation. Nothing is clipped."""
    otf = psf_to_otf(psf)
    blurred = apply_otf(image, otf)
    rng = np.random.default_rng(int(seed))
    noise = float(noise_std) * rng.standard_normal(np.shape(image))
    return blurred, noise, blurred + noise
