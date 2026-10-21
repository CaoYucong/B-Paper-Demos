"""Stable experiment names shared by the lecture pages and the practice links."""

from __future__ import annotations

import re

L1_EXPERIMENTS: tuple[tuple[str, str, str], ...] = (
    ("fundamentals", "Sampling and quantisation", "采样与量化"),
    ("bayer", "Bayer sampling", "Bayer 采样"),
    ("point", "Point operations", "点运算"),
    ("histogram", "Histogram equalisation", "直方图均衡"),
    ("temporal", "Temporal averaging", "时间平均"),
    ("convolution", "Convolution playground", "卷积实验台"),
    ("padding", "Padding and boundaries", "填充与边界"),
    ("sobel", "Sobel edges", "Sobel 边缘"),
    ("gaussian", "Gaussian filtering", "高斯滤波"),
    ("matched", "Matched filtering", "匹配滤波"),
)

L2_EXPERIMENTS: tuple[tuple[str, str, str], ...] = (
    ("spatial", "Spatial frequency", "空间频率"),
    ("decomposition", "Fourier decomposition", "傅里叶分解"),
    ("properties", "Fourier properties", "傅里叶性质"),
    ("phase", "Magnitude and phase", "幅度与相位"),
    ("filters", "Frequency-domain filtering", "频域滤波"),
    ("theorem", "Convolution theorem", "卷积定理"),
    ("aliasing", "Sampling and aliasing", "采样与混叠"),
    ("hybrid", "Hybrid images", "混合图像"),
)

L3_EXPERIMENTS: tuple[tuple[str, str, str], ...] = (
    ("forward", "Degradation model", "退化模型"),
    ("operator", "Operator A", "算子 A"),
    ("inverse", "Inverse filtering", "逆滤波"),
    ("wiener", "Wiener filtering", "维纳滤波"),
    ("motion", "Directional motion blur", "有方向的运动模糊"),
    ("bayes", "Bayesian MAP, scalar", "标量贝叶斯 MAP"),
    ("map", "MAP gradient descent", "MAP 梯度下降"),
    ("compare", "MAP, Wiener, and inverse", "MAP、维纳与逆滤波"),
    ("superres", "Super-resolution, supplementary", "超分辨率，补充"),
)

L4_EXPERIMENTS: tuple[tuple[str, str, str], ...] = (
    ("storage", "Storage calculator", "存储量计算"),
    ("rle", "Run-length encoding", "游程编码"),
    ("huffman", "Huffman coding", "霍夫曼编码"),
    ("dct", "Discrete cosine transform", "离散余弦变换"),
    ("jpeg", "JPEG-like pipeline", "类 JPEG 流程"),
    ("median", "Median filtering", "中值滤波"),
    ("bilateral", "Bilateral filtering", "双边滤波"),
    ("nlm", "Non-local means", "非局部均值"),
    ("inpaint", "Inpainting", "修补"),
)


def section_anchor(item: tuple[str, str, str]) -> str:
    """Hash used in the address bar. ``Bayer sampling`` becomes ``Bayersampling``."""
    return re.sub(r"[^A-Za-z0-9]", "", item[1]) or item[0]


def option(item: tuple[str, str, str]) -> str:
    return f"{item[1]} / {item[2]}"


def option_list(items: tuple[tuple[str, str, str], ...]) -> list[str]:
    return [option(item) for item in items]


def experiment_id(items: tuple[tuple[str, str, str], ...], selected: str) -> str:
    for item in items:
        if option(item) == selected:
            return item[0]
    return items[0][0]


SA_L1_EXPERIMENTS: tuple[tuple[str, str, str], ...] = (
    ("types", "Sampling and quantisation", "采样与量化"),
    ("superposition", "Superposition", "叠加"),
    ("convolution", "Convolution", "卷积"),
    ("sinusoid", "Sinusoid through an LTI system", "正弦通过 LTI 系统"),
    ("ideal", "Ideal low-pass filter", "理想低通"),
    ("butterworth", "Butterworth low-pass", "巴特沃斯低通"),
)

SA_L2_EXPERIMENTS: tuple[tuple[str, str, str], ...] = (
    ("aliasing", "Aliasing", "混叠"),
    ("hold", "Zero-order hold", "零阶保持"),
    ("parabolic", "Five-point parabolic filter", "五点抛物线滤波"),
    ("impulse", "FIR and IIR", "FIR 与 IIR"),
    ("stability", "Stability of a real pole", "实极点的稳定性"),
    ("distance", "Pole distance", "极点距离"),
)

SA_L3_EXPERIMENTS: tuple[tuple[str, str, str], ...] = (
    ("hanning", "Three-point Hanning filter", "三点汉宁滤波"),
    ("phase", "Linear phase", "线性相位"),
    ("window", "Windowed FIR low-pass", "加窗 FIR 低通"),
    ("bilinear", "Bilinear warping", "双线性频率弯折"),
    ("iir", "Digital Butterworth example", "数字巴特沃斯例子"),
)

SA_L4_EXPERIMENTS: tuple[tuple[str, str, str], ...] = (
    ("dft", "Four-point DFT example", "四点 DFT 例子"),
    ("padding", "Zero padding", "补零"),
    ("leakage", "Spectral leakage", "频谱泄漏"),
    ("ar", "Autoregressive fit", "自回归拟合"),
    ("psd", "Periodogram and AR spectrum", "周期图与 AR 谱"),
)


PRACTICE_LINKS: dict[str, str] = {
    "gaussian-cost": "l1-gaussian",
    "valid-size": "l1-padding",
    "histogram-cdf": "l1-histogram",
    "sobel-response": "l1-sobel",
    "linear-length": "l2-theorem",
    "median-window": "l4-median",
    "storage-bytes": "l4-storage",
    "wiener-k": "l3-wiener",
}
