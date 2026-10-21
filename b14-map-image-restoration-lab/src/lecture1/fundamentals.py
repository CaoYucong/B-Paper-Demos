"""Sampling, quantisation, Bayer mosaicing, and temporal averaging."""

from __future__ import annotations

import numpy as np


def quantise(image: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Mid-riser quantisation of a [0, 1] image.

    Returns integer codes in ``0 .. 2^bits - 1`` and the reconstructed
    intensities ``code / (2^bits - 1)``. Values outside [0, 1] are counted
    into the end bins; the reconstruction itself is not a display clip of
    an intermediate filter.
    """
    levels = 2 ** int(bits)
    if levels < 2:
        raise ValueError("bit depth must be at least 1.")
    x = np.asarray(image, dtype=np.float64)
    scaled = np.clip(x, 0.0, 1.0) * (levels - 1)
    codes = np.rint(scaled).astype(np.int64)
    reconstructed = codes.astype(np.float64) / (levels - 1)
    return codes, reconstructed


def downsample(image: np.ndarray, factor: int, method: str = "point") -> np.ndarray:
    """Keep every ``factor``-th sample, or the mean of each factor×factor block.

    Point sampling keeps image[::factor, ::factor]. Area sampling averages
    complete blocks and drops a trailing partial block.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError("sampling factor must be at least 1.")
    image = np.asarray(image, dtype=np.float64)
    if method == "point":
        return image[::factor, ::factor].copy()
    if method != "area":
        raise ValueError("method must be 'point' or 'area'.")
    height = image.shape[0] - image.shape[0] % factor
    width = image.shape[1] - image.shape[1] % factor
    cropped = image[:height, :width]
    blocks = cropped.reshape(height // factor, factor, width // factor, factor)
    return blocks.mean(axis=(1, 3))


def upsample_nearest(image: np.ndarray, factor: int, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Repeat each sample into a factor×factor block. Crop to ``shape`` if given."""
    factor = int(factor)
    zoomed = np.repeat(np.repeat(np.asarray(image, dtype=np.float64), factor, axis=0), factor, axis=1)
    if shape is None:
        return zoomed
    return zoomed[: shape[0], : shape[1]]


def bright_border(n: int, width: int = 2) -> np.ndarray:
    """Dark field with a bright frame, used to expose padding artefacts."""
    image = np.full((int(n), int(n)), 0.08, dtype=np.float64)
    image[:width, :] = 1.0
    image[-width:, :] = 1.0
    image[:, :width] = 1.0
    image[:, -width:] = 1.0
    return image


def synthetic_rgb(n: int) -> np.ndarray:
    """Three-channel test image. Channels are independent spatial patterns."""
    size = int(n)
    y = np.linspace(0.0, 1.0, size, dtype=np.float64)[:, None]
    x = np.linspace(0.0, 1.0, size, dtype=np.float64)[None, :]
    red = np.broadcast_to(x, (size, size)).copy()
    blue = np.broadcast_to(y, (size, size)).copy()
    yy, xx = np.mgrid[:size, :size]
    green = ((yy - size / 2.0) ** 2 + (xx - size / 2.0) ** 2 <= (0.28 * size) ** 2).astype(np.float64)
    return np.stack([red, green, blue], axis=-1)


def bayer_rggb(rgb: np.ndarray) -> np.ndarray:
    """RGGB Bayer mosaic. One measured channel per pixel."""
    rgb = np.asarray(rgb, dtype=np.float64)
    mosaic = np.zeros(rgb.shape[:2], dtype=np.float64)
    mosaic[0::2, 0::2] = rgb[0::2, 0::2, 0]
    mosaic[0::2, 1::2] = rgb[0::2, 1::2, 1]
    mosaic[1::2, 0::2] = rgb[1::2, 0::2, 1]
    mosaic[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return mosaic


def demosaic_bilinear(mosaic: np.ndarray) -> np.ndarray:
    """Fill each Bayer channel by averaging the available 4-neighbours.

    This is a teaching interpolation, not a camera pipeline. Known samples
    are left unchanged.
    """
    mosaic = np.asarray(mosaic, dtype=np.float64)
    height, width = mosaic.shape
    red = np.zeros_like(mosaic)
    green = np.zeros_like(mosaic)
    blue = np.zeros_like(mosaic)
    red[0::2, 0::2] = mosaic[0::2, 0::2]
    green[0::2, 1::2] = mosaic[0::2, 1::2]
    green[1::2, 0::2] = mosaic[1::2, 0::2]
    blue[1::2, 1::2] = mosaic[1::2, 1::2]
    known = {
        0: np.zeros_like(mosaic, dtype=bool),
        1: np.zeros_like(mosaic, dtype=bool),
        2: np.zeros_like(mosaic, dtype=bool),
    }
    known[0][0::2, 0::2] = True
    known[1][0::2, 1::2] = True
    known[1][1::2, 0::2] = True
    known[2][1::2, 1::2] = True
    for index, channel in enumerate((red, green, blue)):
        mask = known[index]
        values = np.pad(channel, 1, mode="constant")
        weights = np.pad(mask.astype(np.float64), 1, mode="constant")
        neighbour_sum = np.zeros_like(channel)
        neighbour_count = np.zeros_like(channel)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbour_sum += values[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
                neighbour_count += weights[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
        filled = np.zeros_like(channel)
        np.divide(neighbour_sum, neighbour_count, out=filled, where=neighbour_count > 0)
        channel[~mask] = filled[~mask]
    return np.stack((red, green, blue), axis=-1)


def temporal_average(
    image: np.ndarray, frames: int, noise_std: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Independent noisy frames and their mean. The noise is not clipped."""
    frames = int(frames)
    if frames < 1:
        raise ValueError("frame count must be positive.")
    image = np.asarray(image, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    stack = image + float(noise_std) * rng.standard_normal((frames,) + image.shape)
    return stack, stack.mean(axis=0)
