"""Orthonormal DCT-II used by the JPEG-like demonstration."""

from __future__ import annotations

import numpy as np


def dct_matrix(n: int = 8) -> np.ndarray:
    """Rows are orthonormal DCT-II basis vectors."""
    n = int(n)
    basis = np.zeros((n, n), dtype=np.float64)
    samples = np.arange(n, dtype=np.float64)
    for k in range(n):
        alpha = np.sqrt(1.0 / n) if k == 0 else np.sqrt(2.0 / n)
        basis[k] = alpha * np.cos(np.pi * k * (samples + 0.5) / n)
    return basis


def dct2(block: np.ndarray, matrix: np.ndarray | None = None) -> np.ndarray:
    matrix = dct_matrix(block.shape[0]) if matrix is None else matrix
    return matrix @ np.asarray(block, dtype=np.float64) @ matrix.T


def idct2(coefficients: np.ndarray, matrix: np.ndarray | None = None) -> np.ndarray:
    matrix = dct_matrix(coefficients.shape[0]) if matrix is None else matrix
    return matrix.T @ np.asarray(coefficients, dtype=np.float64) @ matrix


def basis_image(k: int, ell: int, n: int = 8) -> np.ndarray:
    matrix = dct_matrix(n)
    return np.outer(matrix[int(k)], matrix[int(ell)])


def basis_montage(n: int = 8) -> np.ndarray:
    """All n² basis images, one tile each, scaled independently for display."""
    canvas = np.zeros((n * n, n * n), dtype=np.float64)
    for k in range(n):
        for ell in range(n):
            tile = basis_image(k, ell, n)
            peak = np.max(np.abs(tile))
            if peak > 0:
                tile = tile / peak
            canvas[k * n : (k + 1) * n, ell * n : (ell + 1) * n] = tile
    return canvas


def keep_frequencies(coefficients: np.ndarray, max_sum: int) -> np.ndarray:
    """Zero coefficients whose row index plus column index exceeds ``max_sum``."""
    kept = np.array(coefficients, dtype=np.float64, copy=True)
    yy, xx = np.mgrid[: kept.shape[0], : kept.shape[1]]
    kept[yy + xx > int(max_sum)] = 0.0
    return kept
