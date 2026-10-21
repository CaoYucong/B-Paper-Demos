"""Correlation with a chosen template. Additional to the extracted slide titles."""

from __future__ import annotations

import numpy as np

from src.lecture1.spatial_filters import correlate2d


def template_image(kind: str, size: int = 9) -> np.ndarray:
    size = int(size)
    if size % 2 == 0:
        raise ValueError("template size must be odd.")
    image = np.zeros((size, size), dtype=np.float64)
    centre = size // 2
    if kind == "horizontal":
        image[centre - 1 : centre + 2, :] = 1.0
    elif kind == "vertical":
        image[:, centre - 1 : centre + 2] = 1.0
    elif kind == "disk":
        yy, xx = np.mgrid[:size, :size]
        image[(yy - centre) ** 2 + (xx - centre) ** 2 <= (size * 0.28) ** 2] = 1.0
    elif kind == "diagonal":
        for i in range(size):
            for j in range(-1, 2):
                col = i + j
                if 0 <= col < size:
                    image[i, col] = 1.0
    else:
        raise ValueError(kind)
    image -= image.mean()
    return image


def place_template(shape: tuple[int, int], template: np.ndarray, row: int, col: int) -> np.ndarray:
    image = np.zeros(shape, dtype=np.float64)
    half_y = template.shape[0] // 2
    half_x = template.shape[1] // 2
    image[row - half_y : row + half_y + 1, col - half_x : col + half_x + 1] = template
    return image


def matched_response(image: np.ndarray, template: np.ndarray, boundary: str = "reflect") -> np.ndarray:
    """Correlation response. The peak is a detection statistic, not a probability."""
    return correlate2d(image, template, mode="same", boundary=boundary)


def add_noise(image: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return np.asarray(image, dtype=np.float64) + float(noise_std) * rng.standard_normal(image.shape)
