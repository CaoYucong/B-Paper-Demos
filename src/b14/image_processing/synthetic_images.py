"""内置灰度图和上传图像读取。所有图像都是 float64，范围 [0, 1]。"""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
from PIL import Image


def geometric_shapes(n: int) -> np.ndarray:
    """黑白几何图形：矩形、圆环、圆盘和细十字，用来观察边缘如何被抹开。"""
    size = int(n)
    image = np.zeros((size, size), dtype=np.float64)
    yy, xx = np.mgrid[0:size, 0:size]
    image[int(0.12 * size) : int(0.42 * size), int(0.08 * size) : int(0.36 * size)] = 1.0

    cy, cx, radius = 0.67 * size, 0.72 * size, 0.13 * size
    image[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2] = 1.0

    cy2, cx2 = 0.33 * size, 0.72 * size
    outer, inner = 0.15 * size, 0.10 * size
    dist2 = (yy - cy2) ** 2 + (xx - cx2) ** 2
    image[(dist2 <= outer**2) & (dist2 >= inner**2)] = 1.0

    thickness = max(1, size // 64)
    image[int(0.62 * size) : int(0.62 * size) + thickness, int(0.10 * size) : int(0.42 * size)] = 1.0
    image[int(0.50 * size) : int(0.84 * size), int(0.24 * size) : int(0.24 * size) + thickness] = 1.0
    return image


def gradient_with_edges(n: int) -> np.ndarray:
    """左半边是平滑渐变，中线是阶跃，另外有两条水平锐边。"""
    size = int(n)
    y = np.linspace(0.0, 1.0, size, dtype=np.float64)[:, None]
    x = np.linspace(0.0, 1.0, size, dtype=np.float64)[None, :]
    image = 0.18 + 0.55 * np.broadcast_to(x, (size, size)) + 0.12 * np.broadcast_to(y, (size, size))
    image = image.copy()
    image[:, size // 2 :] = 0.12 + 0.75 * y
    band = max(1, size // 48)
    image[size // 3 : size // 3 + band, :] = 0.0
    image[2 * size // 3 : 2 * size // 3 + band, :] = 1.0
    return np.clip(image, 0.0, 1.0)


def repeated_stripes(n: int) -> np.ndarray:
    """上半部分是竖条纹，下半部分是对角条纹，方便观察不同方向的频率。"""
    size = int(n)
    x = np.arange(size, dtype=np.float64)
    y = np.arange(size, dtype=np.float64)[:, None]
    vertical = 0.5 + 0.4 * np.sign(np.sin(2.0 * np.pi * x / 6.0))
    image = np.broadcast_to(vertical, (size, size)).astype(np.float64, copy=True)
    diagonal = 0.5 + 0.4 * np.sign(np.sin(2.0 * np.pi * (x + y) / 5.0))
    image[size // 2 :, :] = diagonal[size // 2 :, :]
    return image


def smooth_and_texture(n: int) -> np.ndarray:
    """低频起伏、一块常数圆盘，以及右上角的细棋盘纹理。"""
    size = int(n)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    image = 0.55 + 0.20 * np.sin(2.0 * np.pi * x / size) * np.cos(2.0 * np.pi * y / (1.35 * size))
    image += 0.07 * np.sin(2.0 * np.pi * (x + 2.0 * y) / (2.4 * size))
    cy, cx, radius = 0.38 * size, 0.32 * size, 0.15 * size
    image[(y - cy) ** 2 + (x - cx) ** 2 <= radius**2] = 0.84
    texture = ((np.floor(x / 2.0) + np.floor(y / 2.0)) % 2.0) * 0.82 + 0.08
    mask = (x > 0.58 * size) & (x < 0.90 * size) & (y > 0.16 * size) & (y < 0.70 * size)
    image[mask] = texture[mask]
    image[: max(1, size // 8), : max(1, size // 8)] = 0.08
    return np.clip(image, 0.0, 1.0)


IMAGE_BUILDERS: dict[str, Callable[[int], np.ndarray]] = {
    "几何图形": geometric_shapes,
    "锐边渐变": gradient_with_edges,
    "重复条纹": repeated_stripes,
    "平滑与纹理": smooth_and_texture,
}


def make_image(name: str, size: int) -> np.ndarray:
    try:
        builder = IMAGE_BUILDERS[name]
    except KeyError as exc:
        known = ", ".join(IMAGE_BUILDERS)
        raise ValueError(f"未知图像 {name!r}。可选：{known}。") from exc
    image = builder(int(size))
    if image.shape != (size, size):
        raise RuntimeError("合成图像尺寸不正确。")
    return image


def load_grayscale_image(data: bytes, size: int) -> np.ndarray:
    """把 PNG 或 JPEG 转成 size×size 的灰度图，强度除以 255。"""
    with Image.open(io.BytesIO(data)) as image:
        grayscale = image.convert("L")
        resized = grayscale.resize((int(size), int(size)), Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float64) / 255.0
    return np.clip(array, 0.0, 1.0)
