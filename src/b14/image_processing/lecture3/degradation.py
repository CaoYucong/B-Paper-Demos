"""前向退化 g = A f + n。

A 是周期高斯卷积，n 是逐像素独立的高斯噪声。
这里故意不调用 scipy.ndimage.gaussian_filter：那个函数默认使用反射边界，
和本实验的圆卷积不是同一个算子，传函也不能混用。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.b14.image_processing.lecture3.operators import apply_otf, gaussian_psf, psf_to_otf


@dataclass
class Observation:
    """一次完整的前向实验。original 只用于展示和计算误差。"""

    label: str
    original: np.ndarray
    blurred: np.ndarray
    noise: np.ndarray
    observed: np.ndarray
    psf: np.ndarray
    otf: np.ndarray
    sigma: float
    noise_std: float
    seed: int


def degrade(
    original: np.ndarray,
    sigma: float,
    noise_std: float,
    seed: int,
    label: str = "image",
) -> Observation:
    """生成模糊图像 Af 和观测 g = Af + n。

    噪声使用 numpy 的 Generator，固定 seed 时结果可重复。
    观测 g 不裁剪到 [0, 1]，否则模型就不再是加性高斯噪声。
    """
    if noise_std < 0:
        raise ValueError("噪声标准差不能为负。")
    image = np.array(original, dtype=np.float64, copy=True)
    if image.ndim != 2:
        raise ValueError("只接受二维灰度图。")
    if not np.isfinite(image).all():
        raise ValueError("图像含有非有限数值。")

    psf = gaussian_psf(image.shape, sigma)
    otf = psf_to_otf(psf)
    blurred = apply_otf(image, otf)
    if noise_std == 0:
        noise = np.zeros_like(image)
    else:
        generator = np.random.default_rng(int(seed))
        noise = generator.normal(0.0, float(noise_std), size=image.shape)
    observed = blurred + noise
    return Observation(
        label=label,
        original=image,
        blurred=blurred,
        noise=noise,
        observed=observed,
        psf=psf,
        otf=otf,
        sigma=float(sigma),
        noise_std=float(noise_std),
        seed=int(seed),
    )
