"""Linear convolution by direct summation and by zero-padded FFTs.

Circular convolution is a different operation. It is what an unpadded
FFT product computes, and it is the operator used in Lecture 3.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def linear_convolve_direct(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Full linear convolution. Output shape is (H+Kh-1, W+Kw-1)."""
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    out_h = image.shape[0] + kernel.shape[0] - 1
    out_w = image.shape[1] + kernel.shape[1] - 1
    padded = np.zeros((out_h, out_w), dtype=np.float64)
    padded[: image.shape[0], : image.shape[1]] = image
    out = np.zeros_like(padded)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            weight = kernel[i, j]
            if weight == 0.0:
                continue
            out[i : i + image.shape[0], j : j + image.shape[1]] += weight * image
    return out


def linear_convolve_fft(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """FFT product after implicit zero padding to the linear-convolution size."""
    return fftconvolve(np.asarray(image, dtype=np.float64), np.asarray(kernel, dtype=np.float64), mode="full")


def circular_convolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Circular convolution with the kernel's top-left treated as lag zero.

    No zero padding is added, so the kernel wraps. This is not linear convolution.
    """
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.shape[0] > image.shape[0] or kernel.shape[1] > image.shape[1]:
        raise ValueError("circular demonstration expects the kernel to fit in the image.")
    kernel_full = np.zeros_like(image)
    kernel_full[: kernel.shape[0], : kernel.shape[1]] = kernel
    return np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(kernel_full)).real
