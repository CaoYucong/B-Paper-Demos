"""Spatial correlation, convolution, and boundary conditions.

Correlation uses the kernel as displayed: kernel[0, 0] multiplies the
top-left sample of the neighbourhood. Convolution flips the kernel by
180 degrees first, then uses the same sum. Lecture 3's circular
convolution is the periodic case of this convolution, implemented with
the FFT in ``src.operators``.
"""

from __future__ import annotations

import numpy as np

_PAD = {
    "zero": "constant",
    "replicate": "edge",
    "reflect": "reflect",
    "periodic": "wrap",
}


def pad_image(image: np.ndarray, pad_y: int, pad_x: int, mode: str) -> np.ndarray:
    if mode not in _PAD:
        raise ValueError(f"unknown boundary mode: {mode}")
    return np.pad(np.asarray(image, dtype=np.float64), ((pad_y, pad_y), (pad_x, pad_x)), mode=_PAD[mode])


def valid_output_shape(image_shape: tuple[int, int], kernel_shape: tuple[int, int]) -> tuple[int, int]:
    """Output size when the kernel must lie entirely inside the image."""
    out_y = int(image_shape[0]) - int(kernel_shape[0]) + 1
    out_x = int(image_shape[1]) - int(kernel_shape[1]) + 1
    if out_y < 1 or out_x < 1:
        raise ValueError("kernel is larger than the image, so there is no valid output.")
    return out_y, out_x


def _require_odd(kernel: np.ndarray) -> None:
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("this laboratory uses odd kernels so the centre sample is defined.")


def correlate2d(image: np.ndarray, kernel: np.ndarray, mode: str = "same", boundary: str = "reflect") -> np.ndarray:
    """Correlation. ``same`` returns the input shape; ``valid`` does not pad."""
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    _require_odd(kernel)
    kh, kw = kernel.shape
    if mode == "valid":
        padded = image
    elif mode == "same":
        padded = pad_image(image, kh // 2, kw // 2, boundary)
    else:
        raise ValueError("mode must be 'same' or 'valid'.")
    out_h = padded.shape[0] - kh + 1
    out_w = padded.shape[1] - kw + 1
    out = np.zeros((out_h, out_w), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            weight = kernel[i, j]
            if weight == 0.0:
                continue
            out += weight * padded[i : i + out_h, j : j + out_w]
    return out


def convolve2d(image: np.ndarray, kernel: np.ndarray, mode: str = "same", boundary: str = "reflect") -> np.ndarray:
    """Convolution: correlation against the kernel rotated by 180 degrees."""
    return correlate2d(image, np.flip(np.asarray(kernel, dtype=np.float64)), mode=mode, boundary=boundary)


def correlate_separable(
    image: np.ndarray,
    vertical: np.ndarray,
    horizontal: np.ndarray,
    boundary: str = "reflect",
) -> np.ndarray:
    """Same-size correlation with kernel[i, j] = vertical[i] * horizontal[j].

    The image is padded once. A second padding of the intermediate result
    would not match the two-dimensional kernel at the boundary.
    """
    image = np.asarray(image, dtype=np.float64)
    vertical = np.asarray(vertical, dtype=np.float64).reshape(-1)
    horizontal = np.asarray(horizontal, dtype=np.float64).reshape(-1)
    if vertical.size % 2 == 0 or horizontal.size % 2 == 0:
        raise ValueError("separable factors must have odd length.")
    hy, hx = vertical.size // 2, horizontal.size // 2
    padded = pad_image(image, hy, hx, boundary)
    temp = np.zeros((padded.shape[0], image.shape[1]), dtype=np.float64)
    for j, weight in enumerate(horizontal):
        temp += weight * padded[:, j : j + image.shape[1]]
    out = np.zeros_like(image)
    for i, weight in enumerate(vertical):
        out += weight * temp[i : i + image.shape[0], :]
    return out


def kernel_products(window: np.ndarray, kernel: np.ndarray, flipped: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    """Element-wise products and their sum for one centred neighbourhood."""
    kernel = np.asarray(kernel, dtype=np.float64)
    used = np.flip(kernel) if flipped else kernel
    products = np.asarray(window, dtype=np.float64) * used
    return used, products, float(products.sum())


def neighbourhood(image: np.ndarray, row: int, col: int, kernel_shape: tuple[int, int], boundary: str) -> np.ndarray:
    kh, kw = kernel_shape
    padded = pad_image(image, kh // 2, kw // 2, boundary)
    return padded[int(row) : int(row) + kh, int(col) : int(col) + kw]


KERNELS: dict[str, np.ndarray] = {
    "identity": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    "box": np.ones((3, 3)) / 9.0,
    "horizontal": np.array([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]]),
    "vertical": np.array([[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]]),
    "sobel_x": np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]),
    "sobel_y": np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]),
    "sharpen": np.array([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]]),
}
