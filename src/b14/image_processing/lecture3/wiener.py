"""频域逆滤波和维纳滤波。

二者都使用与前向算子相同的离散传递函数 H，并使用共轭 H*。
周期边界和同一个 H 是它们能与 MAP 对照的前提。

    F_inverse = G / H，   仅在 |H| 不小于阈值时
    F_Wiener  = H* G / (|H|^2 + K)

常数 K 的维纳滤波对应白高斯先验，也就是惩罚 ||f||^2。
梯度 MAP 的分母第二项是 λ(|D_x|^2 + |D_y|^2)，随频率变化。
所以两者相关，但一般不相等。
"""

from __future__ import annotations

import numpy as np


def inverse_filter(observed: np.ndarray, otf: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    """朴素逆滤波。返回 (图像, 被阈值抑制的频率个数)。

    |H| 很小时，G/H 会把噪声和舍入误差放大到不可用。
    阈值是硬截断：这些频率直接置零，而不是像维纳滤波那样软衰减。
    """
    if threshold < 0:
        raise ValueError("稳定性阈值不能为负。")
    spectrum = np.fft.fft2(observed)
    magnitude = np.abs(otf)
    mask = magnitude >= float(threshold)
    restored = np.zeros_like(spectrum)
    restored[mask] = spectrum[mask] / otf[mask]
    suppressed = int(np.count_nonzero(~mask))
    return np.fft.ifft2(restored).real, suppressed


def wiener_filter(observed: np.ndarray, otf: np.ndarray, gain: float) -> np.ndarray:
    """常数 K 的维纳滤波。gain 就是公式中的 K，必须非负。"""
    if gain < 0:
        raise ValueError("维纳常数 K 不能为负。")
    spectrum = np.fft.fft2(observed)
    denominator = np.abs(otf) ** 2 + float(gain)
    restored = (np.conj(otf) / denominator) * spectrum
    return np.fft.ifft2(restored).real
