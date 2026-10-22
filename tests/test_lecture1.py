"""Numerical checks for Lecture 1."""

from __future__ import annotations

import numpy as np

from src.b14.image_processing.lecture1.fundamentals import bayer_rggb, demosaic_bilinear, quantise, synthetic_rgb, temporal_average
from src.b14.image_processing.lecture1.gaussian import gaussian_filter, gaussian_filter_separable, gaussian_kernel_2d, multiplications_per_pixel
from src.b14.image_processing.lecture1.histogram import bin_counts, coin_image, equalise_image, equalise_regions
from src.b14.image_processing.lecture1.matched_filter import add_noise, matched_response, place_template, template_image
from src.b14.image_processing.lecture1.point_operations import apply_point
from src.b14.image_processing.lecture1.sobel import rotated_step, sobel, sobel_separable
from src.b14.image_processing.lecture1.spatial_filters import KERNELS, convolve2d, correlate2d, neighbourhood, valid_output_shape


def test_quantise_endpoints_and_one_bit():
    _codes, reconstructed = quantise(np.array([0.0, 0.2, 0.8, 1.0]), bits=1)
    assert np.allclose(reconstructed, [0.0, 0.0, 1.0, 1.0])


def test_point_operation_ignores_neighbours():
    image = np.array([[0.2, 0.8], [0.4, 0.6]])
    changed = image.copy()
    changed[0, 0] = 0.9
    output = apply_point(image, "brightness", brightness=0.1)
    output_changed = apply_point(changed, "brightness", brightness=0.1)
    assert output[0, 1] == output_changed[0, 1]
    assert output[0, 0] == image[0, 0] + 0.1


def test_histogram_equalisation_keeps_positions_and_constant_regions():
    image, disk = coin_image(32)
    mapped, _counts, cdf = equalise_image(image)
    assert cdf[-1] == 1.0
    assert mapped.shape == image.shape
    constant = np.full((8, 8), 0.3)
    unchanged, _counts, _cdf = equalise_image(constant)
    assert np.allclose(unchanged, constant)
    before = bin_counts(image, 64)
    after = bin_counts(mapped, 64)
    assert before.sum() == after.sum() == image.size
    assert not np.array_equal(before, after)
    regional = equalise_regions(image, disk)
    assert regional.shape == image.shape
    assert np.std(regional[disk]) > np.std(image[disk])
    disk_counts = bin_counts(regional[disk], 64)
    background_counts = bin_counts(regional[~disk], 64)
    assert disk_counts.sum() + background_counts.sum() == image.size


def test_correlation_flips_for_convolution():
    image = np.zeros((5, 5))
    image[2, 3] = 1.0
    kernel = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    assert correlate2d(image, kernel)[2, 2] == 1.0
    assert convolve2d(image, kernel)[2, 2] == 0.0
    image[2, 1] = 1.0
    image[2, 3] = 0.0
    assert convolve2d(image, kernel)[2, 2] == 1.0


def test_valid_shape_and_padding_changes_the_edge():
    assert valid_output_shape((7, 7), (3, 3)) == (5, 5)
    image = np.zeros((5, 5))
    image[0, :] = 1.0
    kernel = KERNELS["box"]
    zero = correlate2d(image, kernel, boundary="zero")
    replicate = correlate2d(image, kernel, boundary="replicate")
    assert zero[0, 2] < replicate[0, 2]


def test_separable_gaussian_matches_direct_kernel():
    rng = np.random.default_rng(0)
    image = rng.random((16, 16))
    direct, kernel = gaussian_filter(image, size=5, sigma=1.2)
    separable, factor = gaussian_filter_separable(image, size=5, sigma=1.2)
    assert np.allclose(kernel, np.outer(factor, factor))
    assert abs(kernel.sum() - 1.0) < 1e-12
    assert np.allclose(direct, separable, atol=1e-12)
    assert multiplications_per_pixel(5) == (25, 10)
    assert np.allclose(gaussian_kernel_2d(5, 1.2), kernel)


def test_sobel_orientation_and_separability():
    vertical_edge = rotated_step(31, 0.0)
    gx, gy, _magnitude, orientation = sobel(vertical_edge)
    edge = slice(14, 17)
    assert np.mean(np.abs(gx[:, edge])) > 5.0 * np.mean(np.abs(gy[:, edge]))
    assert abs(np.median(orientation[:, 16])) < 0.2
    separable_x, separable_y = sobel_separable(vertical_edge)
    assert np.allclose(gx, separable_x)
    assert np.allclose(gy, separable_y)


def test_matched_filter_peaks_on_the_template():
    template = template_image("vertical", 9)
    clean = place_template((32, 32), template, 12, 20)
    noisy = add_noise(clean, noise_std=0.05, seed=1)
    response = matched_response(noisy, template)
    assert np.unravel_index(int(np.argmax(response)), response.shape) == (12, 20)


def test_temporal_average_reduces_variance():
    image = np.full((8, 8), 0.4)
    stack, mean = temporal_average(image, frames=16, noise_std=0.2, seed=2)
    assert stack.shape == (16, 8, 8)
    assert np.var(mean) < np.var(stack)


def test_bayer_keeps_measured_samples():
    rgb = synthetic_rgb(8)
    mosaic = bayer_rggb(rgb)
    restored = demosaic_bilinear(mosaic)
    assert np.allclose(restored[0::2, 0::2, 0], rgb[0::2, 0::2, 0])
    assert np.allclose(restored[1::2, 1::2, 2], rgb[1::2, 1::2, 2])
    window = neighbourhood(rgb[:, :, 0], 1, 1, (3, 3), "reflect")
    assert window.shape == (3, 3)
