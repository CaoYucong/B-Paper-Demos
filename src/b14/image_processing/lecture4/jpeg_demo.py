"""Educational JPEG-like pipeline. It does not write a JPEG file.

The luminance table is the usual example table from the JPEG standard,
included as a teaching quantiser. It has not been checked against a
particular lecture slide. Entropy coding is demonstrated separately.
"""

from __future__ import annotations

import numpy as np

from src.b14.image_processing.lecture4.dct import dct_matrix

# JPEG luminance quantisation table, Annex K of ITU-T T.81.
LUMINANCE_Q = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float64,
)


def quantisation_table(strength: float) -> np.ndarray:
    if strength <= 0:
        raise ValueError("quantisation strength must be positive.")
    return np.maximum(LUMINANCE_Q * float(strength), 1.0)


def _blocks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = image.shape[0] - image.shape[0] % 8
    width = image.shape[1] - image.shape[1] % 8
    cropped = np.asarray(image[:height, :width], dtype=np.float64)
    blocks = cropped.reshape(height // 8, 8, width // 8, 8).swapaxes(1, 2)
    return cropped, blocks


def _unblock(blocks: np.ndarray) -> np.ndarray:
    swapped = np.swapaxes(blocks, 1, 2)
    height = swapped.shape[0] * 8
    width = swapped.shape[2] * 8
    return swapped.reshape(height, width)


def jpeg_like(image: np.ndarray, strength: float) -> dict[str, np.ndarray | float]:
    """Level-shift, DCT, quantise, reconstruct. Edges are cropped to a multiple of 8.

    Intensities are in [0, 1], so the level shift is 0.5 rather than 128.
    The returned reconstruction is not clipped.
    """
    _cropped, blocks = _blocks(image)
    matrix = dct_matrix(8)
    coefficients = np.einsum("ik,yxkl,jl->yxij", matrix, blocks - 0.5, matrix)
    table = quantisation_table(strength)
    indices = np.rint(coefficients / table)
    reconstructed_coeff = indices * table
    restored_blocks = np.einsum("ki,yxkl,lj->yxij", matrix, reconstructed_coeff, matrix) + 0.5
    restored = _unblock(restored_blocks)
    return {
        "cropped": _cropped,
        "coefficients": _unblock(coefficients),
        "indices": _unblock(indices),
        "reconstructed": restored,
        "sparsity": float(np.mean(indices == 0.0)),
        "table": table,
    }


def zigzag_indices(n: int = 8) -> list[tuple[int, int]]:
    indices: list[tuple[int, int]] = []
    for diagonal in range(2 * n - 1):
        if diagonal % 2 == 0:
            row = min(diagonal, n - 1)
            col = diagonal - row
            while row >= 0 and col < n:
                indices.append((row, col))
                row -= 1
                col += 1
        else:
            col = min(diagonal, n - 1)
            row = diagonal - col
            while col >= 0 and row < n:
                indices.append((row, col))
                col -= 1
                row += 1
    return indices


def zigzag(block: np.ndarray) -> np.ndarray:
    return np.array([block[row, col] for row, col in zigzag_indices(block.shape[0])], dtype=np.float64)
