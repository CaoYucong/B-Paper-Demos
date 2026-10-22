"""Numerical checks for Lecture 4."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import uniform_filter

from src.b14.image_processing.lecture1.gaussian import gaussian_filter
from src.b14.image_processing.lecture4.bilateral import bilateral_filter, bilateral_weight_maps, reflect_window, shift_reflect
from src.b14.image_processing.lecture4.compression import rle_decode, rle_encode, storage_bits, storage_report
from src.b14.image_processing.lecture4.dct import basis_montage, dct2, dct_matrix, idct2, keep_frequencies
from src.b14.image_processing.lecture4.huffman import average_length, build_tree, codes_from_tree, decode, encode, frequencies
from src.b14.image_processing.lecture4.inpainting import diffuse, rectangle_mask
from src.b14.image_processing.lecture4.jpeg_demo import jpeg_like, zigzag, zigzag_indices
from src.b14.image_processing.lecture4.median import median, salt_and_pepper, sorted_median, window
from src.b14.image_processing.lecture4.window_pick import clamp_centre, clamp_inside, clamp_origin, window_png
from src.b14.image_processing.lecture4.nonlocal_means import (
    BAR,
    DIAGONAL_A,
    DIAGONAL_B,
    compare_patches,
    matching_patches,
    nonlocal_means,
    repeated_marks,
    weight_map,
    weighted_value,
)
from src.b14.image_processing.metrics import mse


def test_storage_units_and_rle_round_trip():
    assert storage_bits(100, 100, 8, 1, 1) == 80_000
    report = storage_report(100, 100, 8, 1, 1, compression_ratio=4, transfer_bits_per_second=8_000)
    assert report["uncompressed_bytes"] == 10_000
    assert report["compressed_bytes"] == 2_500
    assert report["transfer_seconds"] == 2.5
    sequence = [1, 1, 1, 0, 0]
    assert rle_decode(rle_encode(sequence)) == sequence


def test_huffman_is_lossless_and_prefix_free():
    text = "ABRACADABRA"
    counts = frequencies(text)
    tree = build_tree(counts)
    codes = codes_from_tree(tree)
    assert decode(encode(text, codes), tree) == text
    assert average_length(counts, codes) < 8
    single = build_tree({"Q": 3})
    single_codes = codes_from_tree(single)
    assert decode(encode("QQQ", single_codes), single) == "QQQ"


def test_dct_round_trip_and_jpeg_sparsity():
    matrix = dct_matrix(8)
    assert np.allclose(matrix @ matrix.T, np.eye(8), atol=1e-10)
    rng = np.random.default_rng(0)
    block = rng.random((8, 8))
    assert np.allclose(idct2(dct2(block, matrix), matrix), block, atol=1e-10)
    limited = keep_frequencies(dct2(block, matrix), max_sum=2)
    assert limited[-1, -1] == 0.0
    assert basis_montage(8).shape == (64, 64)
    image = rng.random((32, 32))
    gentle = jpeg_like(image, strength=0.05)
    strong = jpeg_like(image, strength=2.0)
    assert strong["sparsity"] > gentle["sparsity"]
    indices = zigzag_indices(8)
    assert len(indices) == 64 and len(set(indices)) == 64
    assert zigzag(np.arange(64).reshape(8, 8)).shape == (64,)


def test_inside_window_cannot_hang_off_the_image():
    assert clamp_inside(0, 0, 32, 40, 11) == (5, 5)
    assert clamp_inside(100, 100, 32, 40, 11) == (26, 34)
    with pytest.raises(ValueError):
        clamp_inside(3, 3, 32, 40, 8)


def test_dct_block_stays_inside_the_image():
    assert clamp_origin(-3, 4, 32, 40, 8) == (0, 4)
    assert clamp_origin(30, 100, 32, 40, 8) == (24, 32)
    with pytest.raises(ValueError):
        clamp_origin(0, 0, 7, 8, 8)


def test_median_window_centre_is_the_chosen_pixel():
    image = np.arange(16, dtype=np.float64).reshape(4, 4)
    values = window(image, 0, 0, 3)
    assert values.shape == (3, 3)
    assert values[1, 1] == image[0, 0]
    assert clamp_centre(-4, 9, 4, 4) == (0, 3)
    raw = base64.b64decode(window_png(np.array([[0.0, 1.0], [0.25, 0.5]])))
    decoded = np.asarray(Image.open(io.BytesIO(raw)))
    assert decoded.shape == (2, 2)
    assert decoded[0, 1] == 255
    assert decoded[1, 0] == 64


def test_median_beats_gaussian_on_salt_and_pepper():
    rng = np.random.default_rng(0)
    clean = np.tile(np.linspace(0.2, 0.8, 32), (32, 1))
    noisy = salt_and_pepper(clean, amount=0.08, seed=3)
    med = median(noisy, size=3)
    gauss, _kernel = gaussian_filter(noisy, size=5, sigma=1.0)
    assert mse(med, clean) < mse(gauss, clean)
    values = window(noisy, 4, 4, 3)
    ordered, centre = sorted_median(values)
    assert ordered[4] == centre
    assert med.shape == clean.shape
    assert rng.random() <= 1.0


def test_weight_maps_match_the_filter_and_split_a_step():
    rng = np.random.default_rng(1)
    image = rng.random((9, 10))
    height, width = image.shape
    padded = np.pad(image, ((height, height), (width, width)), mode="reflect")
    for row, col in ((2, 3), (0, 0), (8, 9)):
        window = reflect_window(image, row, col, half=2)
        assert window[2, 2] == image[row, col]
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                assert window[2 + dy, 2 + dx] == padded[height + row + dy, width + col + dx]

    step = np.zeros((21, 21))
    step[:, 11:] = 1.0
    sampled, spatial, range_weight, product = bilateral_weight_maps(step, 10, 14, 2.0, 0.2, half=6)
    assert sampled.shape == (13, 13)
    assert sampled[6, 6] == 1.0
    assert spatial[6, 6] == 1.0
    assert np.allclose(spatial, spatial[::-1, :])
    assert np.allclose(spatial, spatial[:, ::-1])
    assert abs(spatial[6, 7] - np.exp(-0.5 / 4.0)) < 1e-12
    assert range_weight[6, 6] == 1.0
    assert range_weight[6, 0] < 0.01
    assert range_weight[6, 4] > 0.5
    assert np.allclose(product, spatial * range_weight)
    assert product.max() == 1.0

    flat = np.full((9, 9), 0.3)
    _, _, flat_range, _ = bilateral_weight_maps(flat, 1, 1, 1.5, 0.2, half=2)
    assert np.allclose(flat_range, 1.0)

    wide = np.zeros((15, 15))
    wide[:, 8:] = 1.0
    _, spatial_wide, range_wide, product_wide = bilateral_weight_maps(wide, 7, 10, 2.0, 5.0, half=4)
    assert range_wide.min() > 0.95
    assert np.allclose(product_wide, spatial_wide * range_wide)

    noisy = step + 0.05 * rng.standard_normal(step.shape)
    radius = 2
    sigma_s, sigma_r = 1.7, 0.25
    filtered = bilateral_filter(noisy, sigma_s, sigma_r, radius)
    values, _, _, weights = bilateral_weight_maps(noisy, 4, 16, sigma_s, sigma_r, half=6)
    block = weights[4:9, 4:9]
    pixels = values[4:9, 4:9]
    average = float(np.sum(block * pixels) / np.sum(block))
    assert abs(average - float(filtered[4, 16])) < 1e-9


def test_bilateral_preserves_a_step_better_than_a_gaussian():
    clean = np.zeros((32, 32))
    clean[:, 16:] = 1.0
    rng = np.random.default_rng(4)
    noisy = clean + 0.05 * rng.standard_normal(clean.shape)
    gauss, _kernel = gaussian_filter(noisy, size=7, sigma=2.0)
    bilateral = bilateral_filter(noisy, sigma_spatial=2.0, sigma_range=0.12, radius=3)
    gauss_jump = abs(float(gauss[:, 16].mean() - gauss[:, 15].mean()))
    bilateral_jump = abs(float(bilateral[:, 16].mean() - bilateral[:, 15].mean()))
    assert bilateral_jump > gauss_jump
    constant = np.full((8, 8), 0.4)
    assert np.allclose(bilateral_filter(constant, 1.0, 0.2, 1), constant)


def _reference_weight_map(image: np.ndarray, row: int, col: int, patch: int, search: int, strength: float) -> np.ndarray:
    """The filter's own shift, read back at one pixel and placed on the contributor."""
    image = np.asarray(image, dtype=np.float64)
    out = np.zeros_like(image)
    half = search // 2
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            shifted = shift_reflect(image, dy, dx)
            distance = uniform_filter((image - shifted) ** 2, size=patch, mode="reflect")
            weight = float(np.exp(-distance[row, col] / strength**2))
            q_row, q_col = row - dy, col - dx
            if 0 <= q_row < image.shape[0] and 0 <= q_col < image.shape[1]:
                out[q_row, q_col] = weight
    return out


def test_nonlocal_means_and_inpainting():
    constant = np.full((12, 12), 0.3)
    denoised, weights = nonlocal_means(constant, patch=3, search=5, strength=0.1)
    assert np.allclose(denoised, constant)
    assert weights.shape == (5, 5)
    mapped = weight_map(constant, 6, 6, patch=3, search=5, strength=0.1)
    assert np.allclose(mapped[4:9, 4:9], 1.0)
    assert np.count_nonzero(mapped) == 25
    size = 24
    ramp = np.tile(np.linspace(0.0, 1.0, size), (size, 1))
    missing = rectangle_mask(ramp.shape, 8, 8, 8, 8)
    damaged = ramp.copy()
    damaged[missing] = 0.0
    restored = diffuse(damaged, missing, iterations=800)
    assert mse(restored[missing], ramp[missing]) < 1e-3


def test_weight_map_lines_up_with_the_filter_and_with_a_single_pixel():
    image = np.zeros((9, 9))
    image[4, 4] = 0.5
    image[4, 7] = 0.5
    image[6, 4] = 1.0
    weights = weight_map(image, 4, 4, patch=1, search=7, strength=0.5)
    assert abs(weights[4, 4] - 1.0) < 1e-12
    assert abs(weights[4, 7] - 1.0) < 1e-12
    assert abs(weights[6, 4] - np.exp(-1.0)) < 1e-12
    assert weights[4, 8] == 0.0
    patch_p, patch_q, distance, weight = compare_patches(image, 4, 4, 6, 4, patch=1, strength=0.5)
    assert patch_p.shape == (1, 1)
    assert abs(distance - 0.25) < 1e-12
    assert abs(weight - weights[6, 4]) < 1e-12

    rng = np.random.default_rng(0)
    sample = rng.random((20, 20))
    restored, _centre = nonlocal_means(sample, patch=5, search=7, strength=0.2)
    mapped = weight_map(sample, 10, 10, patch=5, search=7, strength=0.2)
    reference = _reference_weight_map(sample, 10, 10, patch=5, search=7, strength=0.2)
    assert np.allclose(mapped, reference)
    assert abs(weighted_value(sample, mapped) - float(restored[10, 10])) < 1e-9


def test_repeated_marks_light_up_the_matching_diagonal():
    clean = repeated_marks(noise=0.0)
    weights = weight_map(clean, *DIAGONAL_A, patch=5, search=33, strength=0.12)
    assert abs(weights[DIAGONAL_A] - 1.0) < 1e-12
    assert abs(weights[DIAGONAL_B] - 1.0) < 1e-9
    assert weights[BAR] < 1e-3
    narrow = weight_map(clean, *DIAGONAL_A, patch=5, search=15, strength=0.12)
    assert narrow[DIAGONAL_B] == 0.0
    assert narrow[BAR] > 0.0
    assert narrow[BAR] < 1e-3

    noisy = repeated_marks(noise=0.04, seed=0)
    noisy_weights = weight_map(noisy, *DIAGONAL_A, patch=5, search=33, strength=0.08)
    assert noisy_weights[DIAGONAL_B] > 0.5
    assert noisy_weights[BAR] < 0.05
    glow = matching_patches(noisy_weights, patch=5)
    assert glow[28:33, 44:49].min() > 0.5
    assert glow[22:27, 28:33].max() == 0.0
    _patch_p, _patch_q, _distance, compared = compare_patches(noisy, *DIAGONAL_A, *DIAGONAL_B, patch=5, strength=0.08)
    assert abs(compared - float(noisy_weights[DIAGONAL_B])) < 1e-12

    restored, _centre = nonlocal_means(noisy, patch=5, search=33, strength=0.08)
    gauss, _kernel = gaussian_filter(noisy, size=7, sigma=1.5)
    stroke = clean > 0.5
    assert mse(restored[stroke], clean[stroke]) < mse(gauss[stroke], clean[stroke])
