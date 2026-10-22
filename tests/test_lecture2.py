"""Numerical checks for Lecture 2."""

from __future__ import annotations

import numpy as np

from src.b14.image_processing.lecture2.aliasing import dominant_frequency, downsample, stripe_image
from src.b14.image_processing.lecture2.convolution_theorem import circular_convolve_same, linear_convolve_direct, linear_convolve_fft
from src.b14.image_processing.lecture2.fourier import (
    conjugate_symmetry_error,
    linearity_error,
    magnitude_only,
    reconstruct_pairs,
    reconstruction_error,
    swap_magnitude_phase,
    translation_phase_error,
)
from src.b14.image_processing.lecture2.frequency_filters import apply_frequency_mask, frequency_mask
from src.b14.image_processing.lecture2.spatial_frequency import dft, sinusoid
from src.b14.image_processing.synthetic_images import geometric_shapes


def test_sinusoid_lands_on_one_frequency_pair():
    image = sinusoid((32, 32), u=3 / 32, v=0.0, phase=0.0)
    spectrum = dft(image)
    row = np.abs(spectrum[0])
    assert int(np.argmax(row)) in (3, 32 - 3)
    assert np.allclose(row[3], row[32 - 3])


def test_inverse_and_phase_swap_round_trip():
    image = geometric_shapes(32)
    assert reconstruction_error(image) < 1e-10
    reconstructed, _kept = reconstruct_pairs(image, pairs=image.size)
    assert np.allclose(reconstructed, image, atol=1e-8)
    identity = swap_magnitude_phase(image, image)
    assert np.allclose(identity, image, atol=1e-8)
    assert np.mean((magnitude_only(image) - image) ** 2) > 1e-4


def test_frequency_mask_and_properties():
    image = geometric_shapes(32)
    low = frequency_mask(image.shape, "ideal_low", cutoff=0.1)
    assert low[0, 0] == 1.0
    high = frequency_mask(image.shape, "ideal_high", cutoff=0.1)
    assert high[0, 0] == 0.0
    filtered = apply_frequency_mask(image, low)
    assert filtered.shape == image.shape
    other = np.flip(image, axis=1)
    assert linearity_error(image, other, 0.3, -0.7) < 1e-8
    assert translation_phase_error(image, 4, -3) < 1e-8
    assert conjugate_symmetry_error(image) < 1e-8


def test_linear_convolution_matches_fft_and_differs_from_circular():
    rng = np.random.default_rng(0)
    image = rng.normal(size=(8, 8))
    kernel = rng.normal(size=(3, 3))
    direct = linear_convolve_direct(image, kernel)
    spectral = linear_convolve_fft(image, kernel)
    assert direct.shape == (10, 10)
    assert np.allclose(direct, spectral, atol=1e-10)
    circular = circular_convolve_same(image, kernel)
    assert circular.shape == image.shape
    assert not np.allclose(circular, direct[:8, :8])


def test_downsampling_aliases_a_high_frequency():
    image = stripe_image(64, frequency=0.3125)
    sampled = downsample(image, factor=4)
    u, _v = dominant_frequency(sampled)
    assert abs(abs(u) - 0.25) < 1e-8
    prefiltered = downsample(image, factor=4, prefilter_sigma=2.0)
    assert np.std(prefiltered) < np.std(sampled)
