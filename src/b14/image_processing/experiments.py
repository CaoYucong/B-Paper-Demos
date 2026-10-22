"""Experiment names for the Image Processing lecture page and its practice links."""

from __future__ import annotations

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
    ("panorama", "Panorama super-resolution", "全景超分辨率"),
    ("samples", "Why one frame is not enough", "为什么一帧不够"),
    ("benefits", "Resolution and noise", "分辨率与噪声"),
    ("measure", "The matrix M", "矩阵 M"),
    ("stacked", "Stacked equations", "叠放的方程"),
    ("mapsr", "MAP for several frames", "多帧的 MAP"),
    ("huber", "Huber loss", "Huber 损失"),
    ("compare", "MAP, Wiener, and inverse", "MAP、维纳与逆滤波"),
    ("examples", "Examples from the slides", "讲义中的例子"),
    ("blind", "Blind deblurring, not examinable", "盲去模糊，非考试"),
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
