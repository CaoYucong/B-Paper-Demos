"""周期边界下的线性图像算子。

矩阵 A 在概念上是 N×N 的稠密矩阵（N 为像素数）。128×128 的图像会得到
大约 2.7 亿个元素，因此这里从不显式构造 A，只实现

    向量 v  ->  A v
    向量 r  ->  A^T r

高斯核是实偶函数，所以它的圆卷积算子是自伴随的：A^T = A。
代码仍按共轭转置实现伴随，这样不对称的运动模糊也能通过同一个检验。
"""

from __future__ import annotations

import numpy as np

# 三点滑动平均。教学用小矩阵使用它；二维复原使用高斯核，二者不要混为一谈。
THREE_POINT_WEIGHT = 1.0 / 3.0


def gaussian_psf(shape: tuple[int, int], sigma: float) -> np.ndarray:
    """返回以图像中心为峰值、总和为 1 的周期离散高斯 PSF。

    周期距离保证核在边界处仍是偶对称的。偶对称实核的 DFT 为实数，
    从而前向圆卷积与其伴随一致。
    """
    if sigma <= 0:
        raise ValueError("高斯标准差 sigma 必须为正。")
    height, width = (int(shape[0]), int(shape[1]))
    if height < 2 or width < 2:
        raise ValueError("图像至少为 2×2。")
    cy, cx = height // 2, width // 2
    yy, xx = np.ogrid[:height, :width]
    dy = np.minimum(np.abs(yy - cy), height - np.abs(yy - cy)).astype(np.float64)
    dx = np.minimum(np.abs(xx - cx), width - np.abs(xx - cx)).astype(np.float64)
    kernel = np.exp(-(dx * dx + dy * dy) / (2.0 * float(sigma) ** 2))
    total = kernel.sum()
    if total <= 0:
        raise ValueError("高斯核求和失败。")
    kernel /= total
    return kernel


def motion_blur_psf(shape: tuple[int, int], length: int = 9) -> np.ndarray:
    """单向水平运动模糊。PSF 从图像中心向右延伸，因此不是偶对称。

    这个核只用于说明 A 与 A^T 可以不同。正式的复原实验使用高斯核。
    """
    if length < 1:
        raise ValueError("运动长度至少为 1。")
    height, width = (int(shape[0]), int(shape[1]))
    psf = np.zeros((height, width), dtype=np.float64)
    cy, cx = height // 2, width // 2
    for offset in range(int(length)):
        psf[cy, (cx + offset) % width] += 1.0
    psf /= psf.sum()
    return psf


def psf_to_otf(psf_centered: np.ndarray) -> np.ndarray:
    """由中心对齐的 PSF 构造传递函数 H。

    numpy.fft.ifftshift 把中心像素（零滞后）移到索引 (0, 0)，
    随后 FFT 才是圆卷积定理所要求的核。不要改成 fftshift，否则冲激响应会平移。
    """
    origin_kernel = np.fft.ifftshift(np.asarray(psf_centered, dtype=np.float64))
    return np.fft.fft2(origin_kernel)


def apply_otf(image: np.ndarray, otf: np.ndarray) -> np.ndarray:
    """前向算子 A(f) = F^{-1}(H F(f))，周期卷积。"""
    spectrum = np.fft.fft2(image) * otf
    return np.fft.ifft2(spectrum).real


def apply_otf_adjoint(image: np.ndarray, otf: np.ndarray) -> np.ndarray:
    """伴随算子 A^T(r) = F^{-1}(H^* F(r))。

    圆卷积的伴随是与翻转核的卷积。翻转核的 DFT 正好是 H 的共轭。
    """
    spectrum = np.fft.fft2(image) * np.conj(otf)
    return np.fft.ifft2(spectrum).real


def forward_diff_x(image: np.ndarray) -> np.ndarray:
    """水平前向差分，周期边界：(D_x f)[i, j] = f[i, j+1] - f[i, j]。"""
    return np.roll(image, -1, axis=1) - image


def forward_diff_y(image: np.ndarray) -> np.ndarray:
    """垂直前向差分，周期边界：(D_y f)[i, j] = f[i+1, j] - f[i, j]。"""
    return np.roll(image, -1, axis=0) - image


def adjoint_diff_x(field: np.ndarray) -> np.ndarray:
    """D_x 的精确伴随：(D_x^T v)[i, j] = v[i, j-1] - v[i, j]。"""
    return np.roll(field, 1, axis=1) - field


def adjoint_diff_y(field: np.ndarray) -> np.ndarray:
    """D_y 的精确伴随：(D_y^T v)[i, j] = v[i-1, j] - v[i, j]。"""
    return np.roll(field, 1, axis=0) - field


def apply_gradient_penalty(image: np.ndarray) -> np.ndarray:
    """正则项的梯度因子 D^T D f = D_x^T D_x f + D_y^T D_y f。"""
    return adjoint_diff_x(forward_diff_x(image)) + adjoint_diff_y(forward_diff_y(image))


def gradient_squared_norm(image: np.ndarray) -> float:
    """||Df||^2 = ||D_x f||^2 + ||D_y f||^2，不取平方根。"""
    dx = forward_diff_x(image)
    dy = forward_diff_y(image)
    return float(np.dot(dx.ravel(), dx.ravel()) + np.dot(dy.ravel(), dy.ravel()))


def frequency_gradient_power(shape: tuple[int, int]) -> np.ndarray:
    """周期前向差分的功率 |D_x(ω)|^2 + |D_y(ω)|^2。

    numpy 的 FFT 约定下，roll(f, -1) 的乘数是 exp(+2π i f)。
    因此 D_x 的频率响应是 exp(2π i f_x) - 1。
    其模平方为 4 sin^2(π f)，在奈奎斯特频率处等于 4。符号取反不改变模平方。
    """
    height, width = (int(shape[0]), int(shape[1]))
    fy = np.fft.fftfreq(height)
    fx = np.fft.fftfreq(width)
    dy = np.abs(np.exp(2j * np.pi * fy) - 1.0) ** 2
    dx = np.abs(np.exp(2j * np.pi * fx) - 1.0) ** 2
    return dy[:, None] + dx[None, :]


def crop_centered(image: np.ndarray, half: int) -> np.ndarray:
    """裁出以中心像素为中心的窗口，用于显示 PSF。"""
    height, width = image.shape
    cy, cx = height // 2, width // 2
    half = int(max(0, half))
    top = max(0, cy - half)
    bottom = min(height, cy + half + 1)
    left = max(0, cx - half)
    right = min(width, cx + half + 1)
    return image[top:bottom, left:right]


def impulse_at_center(shape: tuple[int, int]) -> np.ndarray:
    """中心为 1 的冲激图像。A(δ) 应当正好等于中心对齐的 PSF。"""
    image = np.zeros(shape, dtype=np.float64)
    image[shape[0] // 2, shape[1] // 2] = 1.0
    return image


def inner_product(left: np.ndarray, right: np.ndarray) -> float:
    """实向量内积 <u, v> = sum u v。"""
    return float(np.dot(np.ravel(left), np.ravel(right)))


def averaging_matrix(n: int, weight: float = THREE_POINT_WEIGHT) -> np.ndarray:
    """周期边界上的对称三点平均矩阵，尺寸 n×n，n 在 5 到 8 之间最适合课堂演示。

    第 i 行只在列 i-1、i、i+1 上非零（下标按 n 取模）。
    这是显式的小矩阵。二维高斯模糊不是这个矩阵。
    """
    if n < 3:
        raise ValueError("三点平均至少需要 3 个像素。")
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        matrix[i, (i - 1) % n] += weight
        matrix[i, i] += weight
        matrix[i, (i + 1) % n] += weight
    return matrix


def circular_three_point_average(vector: np.ndarray) -> np.ndarray:
    """与 averaging_matrix 相同的三点圆卷积，用来对照矩阵乘法。"""
    values = np.asarray(vector, dtype=np.float64)
    return (np.roll(values, 1) + values + np.roll(values, -1)) * THREE_POINT_WEIGHT


def otf_log_magnitude(otf: np.ndarray) -> np.ndarray:
    """把 |H| 的对数移到零频居中，只用于显示，不参与滤波。"""
    magnitude = np.maximum(np.abs(otf), 1e-12)
    return np.fft.fftshift(np.log10(magnitude))
