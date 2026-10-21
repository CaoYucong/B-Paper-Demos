"""误差指标。峰值按归一化图像的 1 计算。"""

from __future__ import annotations

import numpy as np


def mse(estimate: np.ndarray, reference: np.ndarray) -> float:
    difference = np.asarray(estimate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return float(np.mean(difference * difference))


def psnr(estimate: np.ndarray, reference: np.ndarray, peak: float = 1.0) -> float:
    error = mse(estimate, reference)
    if error <= 1e-18:
        return float("inf")
    return float(10.0 * np.log10((float(peak) ** 2) / error))


def format_db(value: float) -> str:
    if not np.isfinite(value):
        return "∞"
    return f"{value:.2f} dB"
