"""Numerical checks for Lecture 4."""

from __future__ import annotations

import numpy as np

from src.lecture1.gaussian import gaussian_filter
from src.lecture4.bilateral import bilateral_filter
from src.lecture4.compression import rle_decode, rle_encode, storage_bits, storage_report
from src.lecture4.dct import basis_montage, dct2, dct_matrix, idct2, keep_frequencies
from src.lecture4.huffman import average_length, build_tree, codes_from_tree, decode, encode, frequencies
from src.lecture4.inpainting import diffuse, rectangle_mask
from src.lecture4.jpeg_demo import jpeg_like, zigzag, zigzag_indices
from src.lecture4.median import median, salt_and_pepper, sorted_median, window
from src.lecture4.nonlocal_means import nonlocal_means
from src.metrics import mse


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


def test_nonlocal_means_and_inpainting():
    constant = np.full((12, 12), 0.3)
    denoised, weights = nonlocal_means(constant, patch=3, search=5, strength=0.1)
    assert np.allclose(denoised, constant)
    assert weights.shape == (5, 5)
    size = 24
    ramp = np.tile(np.linspace(0.0, 1.0, size), (size, 1))
    missing = rectangle_mask(ramp.shape, 8, 8, 8, 8)
    damaged = ramp.copy()
    damaged[missing] = 0.0
    restored = diffuse(damaged, missing, iterations=800)
    assert mse(restored[missing], ramp[missing]) < 1e-3
