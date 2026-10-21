"""标量高斯逆问题的先验、似然、后验和 MAP 闭式解。

模型是 g = a f + n，n ~ N(0, σ_n²)，f ~ N(μ_f, σ_f²)。
这个先验直接约束 f 本身，对应图像问题里的零阶 Tikhonov 项，
不是梯度惩罚。贝叶斯结构和图像 MAP 相同。
"""

from __future__ import annotations

import numpy as np


def scalar_map(observation: float, coefficient: float, noise_std: float, prior_mean: float, prior_std: float) -> float:
    """高斯标量 MAP 的精确公式。

    f_MAP = (a g / σ_n² + μ / σ_f²) / (a² / σ_n² + 1 / σ_f²)
    """
    noise_var = float(noise_std) ** 2
    prior_var = float(prior_std) ** 2
    if noise_var <= 0 or prior_var <= 0:
        raise ValueError("标准差必须为正。")
    numerator = coefficient * observation / noise_var + prior_mean / prior_var
    denominator = (coefficient**2) / noise_var + 1.0 / prior_var
    return float(numerator / denominator)


def scalar_posterior_variance(coefficient: float, noise_std: float, prior_std: float) -> float:
    """后验精度是似然精度与先验精度之和，方差是精度的倒数。"""
    noise_var = float(noise_std) ** 2
    prior_var = float(prior_std) ** 2
    precision = (coefficient**2) / noise_var + 1.0 / prior_var
    return float(1.0 / precision)


def gaussian_pdf(samples: np.ndarray, mean: float, std: float) -> np.ndarray:
    variance = float(std) ** 2
    scale = std * np.sqrt(2.0 * np.pi)
    return np.exp(-0.5 * (samples - mean) ** 2 / variance) / scale


def prior_density(samples: np.ndarray, prior_mean: float, prior_std: float) -> np.ndarray:
    return gaussian_pdf(samples, prior_mean, prior_std)


def likelihood_density(samples: np.ndarray, observation: float, coefficient: float, noise_std: float) -> np.ndarray:
    """把 p(g | f) 看成 f 的函数。均值在 f = g/a（a 不为 0 时）。"""
    variance = float(noise_std) ** 2
    scale = noise_std * np.sqrt(2.0 * np.pi)
    residual = observation - coefficient * samples
    return np.exp(-0.5 * residual**2 / variance) / scale


def posterior_density_unnormalised(
    samples: np.ndarray,
    observation: float,
    coefficient: float,
    noise_std: float,
    prior_mean: float,
    prior_std: float,
) -> np.ndarray:
    return likelihood_density(samples, observation, coefficient, noise_std) * prior_density(
        samples, prior_mean, prior_std
    )


def _trapz(values: np.ndarray, samples: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, samples))
    return float(np.trapz(values, samples))


def normalised_posterior(
    samples: np.ndarray,
    observation: float,
    coefficient: float,
    noise_std: float,
    prior_mean: float,
    prior_std: float,
) -> np.ndarray:
    """用数值积分把 prior × likelihood 归一化，用来检验解析后验。"""
    raw = posterior_density_unnormalised(samples, observation, coefficient, noise_std, prior_mean, prior_std)
    area = _trapz(raw, samples)
    if area <= 0 or not np.isfinite(area):
        raise ValueError("后验无法归一化。")
    return raw / area


def grid_map(samples: np.ndarray, posterior: np.ndarray) -> float:
    return float(samples[int(np.argmax(posterior))])


def suggested_axis(
    observation: float,
    coefficient: float,
    noise_std: float,
    prior_mean: float,
    prior_std: float,
) -> np.ndarray:
    """覆盖先验、似然峰和 MAP，两侧各留数个标准差。"""
    f_map = scalar_map(observation, coefficient, noise_std, prior_mean, prior_std)
    points = [prior_mean, f_map]
    if abs(coefficient) > 1e-8:
        points.append(observation / coefficient)
    span = max(4.0 * prior_std, 4.0 * noise_std / max(abs(coefficient), 0.25), 1.0)
    low = min(points) - span
    high = max(points) + span
    return np.linspace(low, high, 1600)
