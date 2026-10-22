"""Experiment names for the Signal Analysis lecture page."""

from __future__ import annotations

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
