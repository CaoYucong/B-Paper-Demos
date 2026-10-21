"""Point operations g(x, y) = T(f(x, y)). Neighbours are not used."""

from __future__ import annotations

import numpy as np


def apply_point(
    image: np.ndarray,
    kind: str,
    brightness: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    threshold: float = 0.5,
) -> np.ndarray:
    """Apply one intensity mapping. The result is not clipped to [0, 1].

    Gamma is defined here for nonnegative intensities. Negative inputs are
    left unchanged and should be reported by the caller via ``negative_count``.
    """
    x = np.asarray(image, dtype=np.float64)
    if kind == "brightness":
        return x + float(brightness)
    if kind == "contrast":
        return (x - 0.5) * float(contrast) + 0.5
    if kind == "invert":
        return 1.0 - x
    if kind == "gamma":
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        safe = np.where(x < 0.0, 0.0, x)
        mapped = np.power(safe, float(gamma))
        return np.where(x < 0.0, x, mapped)
    if kind == "threshold":
        return (x >= float(threshold)).astype(np.float64)
    raise ValueError(f"unknown point operation: {kind}")


def negative_count(image: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(image) < 0.0))


def mapping_curve(
    kind: str,
    brightness: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    threshold: float = 0.5,
    samples: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Input and output of T on [0, 1]."""
    x = np.linspace(0.0, 1.0, int(samples))
    y = apply_point(x, kind, brightness=brightness, contrast=contrast, gamma=gamma, threshold=threshold)
    return x, y


def histogram(image: np.ndarray, bins: int = 64, value_range: tuple[float, float] = (0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(np.asarray(image, dtype=np.float64), bins=int(bins), range=value_range)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, counts.astype(np.float64)
