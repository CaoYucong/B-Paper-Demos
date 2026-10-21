"""Sobel gradient by correlation, and its separable factors."""

from __future__ import annotations

import numpy as np

from src.lecture1.spatial_filters import KERNELS, correlate2d, correlate_separable


SOBEL_X = KERNELS["sobel_x"]
SOBEL_Y = KERNELS["sobel_y"]
# outer([1, 2, 1], [-1, 0, 1]) = Sobel X
SOBEL_X_VERTICAL = np.array([1.0, 2.0, 1.0])
SOBEL_X_HORIZONTAL = np.array([-1.0, 0.0, 1.0])
# outer([-1, 0, 1], [1, 2, 1]) = Sobel Y
SOBEL_Y_VERTICAL = np.array([-1.0, 0.0, 1.0])
SOBEL_Y_HORIZONTAL = np.array([1.0, 2.0, 1.0])


def sobel(image: np.ndarray, boundary: str = "reflect") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Gx, Gy, magnitude, and gradient orientation in radians.

    Orientation is ``atan2(Gy, Gx)``, the direction of the intensity
    gradient. The edge itself is perpendicular to that direction.
    """
    gx = correlate2d(image, SOBEL_X, boundary=boundary)
    gy = correlate2d(image, SOBEL_Y, boundary=boundary)
    magnitude = np.hypot(gx, gy)
    orientation = np.arctan2(gy, gx)
    return gx, gy, magnitude, orientation


def sobel_separable(image: np.ndarray, boundary: str = "reflect") -> tuple[np.ndarray, np.ndarray]:
    gx = correlate_separable(image, SOBEL_X_VERTICAL, SOBEL_X_HORIZONTAL, boundary=boundary)
    gy = correlate_separable(image, SOBEL_Y_VERTICAL, SOBEL_Y_HORIZONTAL, boundary=boundary)
    return gx, gy


def rotated_step(n: int, angle_degrees: float) -> np.ndarray:
    """Unit step whose normal points at ``angle_degrees`` from the +x axis."""
    size = int(n)
    yy, xx = np.mgrid[:size, :size]
    yy = yy - (size - 1) / 2.0
    xx = xx - (size - 1) / 2.0
    theta = np.deg2rad(float(angle_degrees))
    projection = xx * np.cos(theta) + yy * np.sin(theta)
    return (projection >= 0.0).astype(np.float64)


def threshold_magnitude(magnitude: np.ndarray, level: float) -> np.ndarray:
    return (np.asarray(magnitude, dtype=np.float64) >= float(level)).astype(np.float64)
