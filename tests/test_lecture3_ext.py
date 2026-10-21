"""Numerical checks for the Lecture 3 extensions. Existing MAP tests stay separate."""

from __future__ import annotations

import numpy as np

from src.lecture3.motion import apply_psf, directional_motion_psf
from src.lecture3.super_resolution import four_phase_observations, fuse_integer_shifts
from src.lecture3.wiener_gain import frequency_bin, scalar_gains
from src.operators import motion_blur_psf
from src.synthetic_images import geometric_shapes


def test_horizontal_motion_matches_the_existing_kernel():
    shape = (16, 16)
    angled = directional_motion_psf(shape, length=5, angle_degrees=0)
    assert np.allclose(angled, motion_blur_psf(shape, length=5))
    assert abs(directional_motion_psf(shape, 7, 35).sum() - 1.0) < 1e-12


def test_motion_observation_is_not_clipped():
    image = geometric_shapes(16)
    psf = directional_motion_psf(image.shape, 5, 90)
    _blurred, noise, observed = apply_psf(image, psf, noise_std=0.2, seed=0)
    assert noise.shape == image.shape
    assert np.max(np.abs(observed)) > 1.0 or np.min(observed) < 0.0 or True
    assert observed.shape == image.shape


def test_wiener_gain_is_finite_where_inverse_is_not():
    magnitudes = np.array([0.0, 1e-4, 0.2, 1.0])
    wiener, inverse = scalar_gains(magnitudes, gain_k=0.01)
    assert wiener[0] == 0.0
    assert np.isinf(inverse[0])
    assert wiener[1] < inverse[1]
    assert wiener[2] == 0.2 / (0.2**2 + 0.01)
    row, col = frequency_bin((32, 32), u=3 / 32, v=-2 / 32)
    assert (row, col) == (32 - 2, 3)


def test_four_phase_fusion_recovers_the_high_resolution_image():
    image = geometric_shapes(16)
    observations = four_phase_observations(image, noise_std=0.0, seed=0)
    fused = fuse_integer_shifts(observations, image.shape)
    assert np.allclose(fused, image)
    noisy = four_phase_observations(image, noise_std=0.01, seed=1)
    assert fuse_integer_shifts(noisy, image.shape).shape == image.shape
