"""Original practice problems.

The example-sheet PDF did not yield extractable question text, so these
are not the official B14 example-sheet questions. Each one is labelled
as an original problem in the interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.b14.image_processing.lecture1.histogram import equalise_values
from src.b14.image_processing.lecture1.spatial_filters import correlate2d, valid_output_shape
from src.b14.image_processing.lecture4.compression import storage_bits
from src.b14.image_processing.lecture4.median import sorted_median


@dataclass(frozen=True)
class NumericProblem:
    id: str
    topic_en: str
    topic_zh: str
    lecture: str
    prompt_en: str
    prompt_zh: str
    answer: float
    tolerance: float
    solution_en: str
    solution_zh: str


@dataclass(frozen=True)
class QuizQuestion:
    id: str
    lecture: str
    prompt_en: str
    prompt_zh: str
    choices_en: tuple[str, ...]
    choices_zh: tuple[str, ...]
    correct: int
    explanation_en: str
    explanation_zh: str


_PATCH = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]
)
_SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_RESPONSE = float(correlate2d(_PATCH, _SOBEL_X, mode="valid")[0, 0])

_CDF_FIRST = float(equalise_values(np.array([0.0, 0.4, 0.8, 0.9]), levels=3)[2][0])


PROBLEMS: tuple[NumericProblem, ...] = (
    NumericProblem(
        id="gaussian-cost",
        topic_en="Gaussian filter implementation",
        topic_zh="高斯滤波的实现",
        lecture="l1",
        prompt_en="A separable 5×1 Gaussian is applied horizontally and then vertically. How many multiplications does one output pixel use? Count one multiplication per tap, and ignore the border.",
        prompt_zh="把一个可分离的 5×1 高斯先横向再纵向作用。忽略边界时，每个输出像素做多少次乘法？每个抽头算一次。",
        answer=10,
        tolerance=0,
        solution_en="Each direction uses 5 taps, so the two passes use 5 + 5 = 10 multiplications. The direct 5×5 kernel would use 25.",
        solution_zh="每个方向 5 个抽头，两遍一共 5 + 5 = 10 次乘法。直接的 5×5 核要 25 次。",
    ),
    NumericProblem(
        id="valid-size",
        topic_en="Filtering bounds and padding",
        topic_zh="滤波范围与填充",
        lecture="l1",
        prompt_en="An 8×8 image is correlated with a 3×3 kernel and no padding is added. How many rows does the valid output have?",
        prompt_zh="8×8 图像与 3×3 核做相关，不加填充。有效输出有多少行？",
        answer=float(valid_output_shape((8, 8), (3, 3))[0]),
        tolerance=0,
        solution_en="Valid output length is n − m + 1 = 8 − 3 + 1 = 6.",
        solution_zh="有效输出长度是 n − m + 1 = 8 − 3 + 1 = 6。",
    ),
    NumericProblem(
        id="histogram-cdf",
        topic_en="Histogram equalisation",
        topic_zh="直方图均衡",
        lecture="l1",
        prompt_en="Four samples fall into 3 bins with counts [1, 1, 2]. Using s(k) = (sum of counts up to k) / N, what is s for the lowest bin?",
        prompt_zh="四个样本落入 3 个箱子，计数是 [1, 1, 2]。按 s(k) =（从第 0 箱累加到 k 的计数）/ N，最低箱的 s 是多少？",
        answer=_CDF_FIRST,
        tolerance=1e-9,
        solution_en="N = 4 and the first count is 1, so s(0) = 1/4 = 0.25.",
        solution_zh="N = 4，第一箱计数是 1，所以 s(0) = 1/4 = 0.25。",
    ),
    NumericProblem(
        id="sobel-response",
        topic_en="Sobel filters",
        topic_zh="Sobel 滤波",
        lecture="l1",
        prompt_en="The 3×3 patch has rows [0, 0, 1], [0, 0, 1], [0, 0, 1]. Correlate it with Sobel Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], with no padding. What is the single output value?",
        prompt_zh="3×3 小块的三行是 [0, 0, 1]、[0, 0, 1]、[0, 0, 1]。用 Sobel Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] 做相关，不填充。唯一的输出是多少？",
        answer=_SOBEL_RESPONSE,
        tolerance=1e-9,
        solution_en="Only the last column is 1. The last column of Gx sums to 1 + 2 + 1 = 4.",
        solution_zh="只有最后一列是 1。Gx 最后一列相加是 1 + 2 + 1 = 4。",
    ),
    NumericProblem(
        id="linear-length",
        topic_en="Convolution theorem",
        topic_zh="卷积定理",
        lecture="l2",
        prompt_en="A length-5 signal is linearly convolved with a length-3 kernel. How long is the full linear result?",
        prompt_zh="长度为 5 的信号与长度为 3 的核做线性卷积。完整线性结果有多长？",
        answer=7,
        tolerance=0,
        solution_en="Full linear convolution has length n + k − 1 = 5 + 3 − 1 = 7. A circular convolution of length 5 is a different operation.",
        solution_zh="完整线性卷积长度是 n + k − 1 = 7。长度为 5 的循环卷积是另一种运算。",
    ),
    NumericProblem(
        id="median-window",
        topic_en="Median filtering",
        topic_zh="中值滤波",
        lecture="l4",
        prompt_en="Nine neighbourhood values are 1, 9, 2, 8, 0, 7, 3, 6, 4. What is their median?",
        prompt_zh="邻域里的九个数是 1, 9, 2, 8, 0, 7, 3, 6, 4。中值是多少？",
        answer=sorted_median(np.array([1, 9, 2, 8, 0, 7, 3, 6, 4]))[1],
        tolerance=0,
        solution_en="Sorted order is 0, 1, 2, 3, 4, 6, 7, 8, 9. The middle entry is 4.",
        solution_zh="排序后是 0, 1, 2, 3, 4, 6, 7, 8, 9。中间一项是 4。",
    ),
    NumericProblem(
        id="storage-bytes",
        topic_en="Storage calculation",
        topic_zh="存储量计算",
        lecture="l4",
        prompt_en="A 640×480 grayscale frame uses 8 bits per pixel and one channel. How many bytes is one uncompressed frame?",
        prompt_zh="一帧 640×480 的灰度图像，每像素 8 位、一个通道。未压缩的一帧是多少字节？",
        answer=storage_bits(640, 480, 8, 1, 1) / 8.0,
        tolerance=0,
        solution_en="640 × 480 × 8 bits = 2,457,600 bits. Divide by 8 to get 307,200 bytes.",
        solution_zh="640 × 480 × 8 = 2,457,600 位。除以 8 得到 307,200 字节。",
    ),
    NumericProblem(
        id="wiener-k",
        topic_en="Wiener gain",
        topic_zh="维纳增益",
        lecture="l3",
        prompt_en="A real transfer value is H = 0.2 and the Wiener constant is K = 0.04. For this nonnegative H the gain is H / (H² + K). What is that gain?",
        prompt_zh="实传递值 H = 0.2，维纳常数 K = 0.04。非负 H 的增益是 H / (H² + K)。这个增益是多少？",
        answer=0.2 / (0.2**2 + 0.04),
        tolerance=1e-9,
        solution_en="H² + K = 0.04 + 0.04 = 0.08, so the gain is 0.2 / 0.08 = 2.5. The naive inverse would be 1/0.2 = 5.",
        solution_zh="H² + K = 0.08，增益是 0.2 / 0.08 = 2.5。朴素逆滤波则是 1/0.2 = 5。",
    ),
)


QUIZ: tuple[QuizQuestion, ...] = (
    QuizQuestion(
        "q-point",
        "l1",
        "A point operation T(f) uses which pixels?",
        "点运算 T(f) 使用哪些像素？",
        ("Only the pixel being mapped", "A 3×3 neighbourhood", "The whole Fourier spectrum"),
        ("只使用正在映射的那个像素", "3×3 邻域", "整幅傅里叶频谱"),
        0,
        "g(x, y) = T(f(x, y)) does not read a neighbour.",
        "g(x, y) = T(f(x, y)) 不读取邻域。",
    ),
    QuizQuestion(
        "q-he",
        "l1",
        "Histogram equalisation changes which of these?",
        "直方图均衡改变的是哪一项？",
        ("The intensity distribution", "The spatial order of the pixels", "The image dimensions"),
        ("强度的分布", "像素的空间排列", "图像尺寸"),
        0,
        "Each intensity is replaced by a function of its rank. Positions stay put.",
        "每个强度被换成它的秩的函数。位置不动。",
    ),
    QuizQuestion(
        "q-conv",
        "l1",
        "How does convolution differ from correlation in this lab?",
        "在本实验里，卷积和相关差在哪里？",
        ("Convolution flips the kernel by 180° first", "They are always identical", "Convolution uses the median"),
        ("卷积先把核旋转 180°", "二者永远相同", "卷积使用中值"),
        0,
        "For an asymmetric kernel the two outputs differ. Sobel masks are written as correlation kernels.",
        "核不对称时，两个输出不同。Sobel 模板按相关来写。",
    ),
    QuizQuestion(
        "q-phase",
        "l2",
        "Swapping magnitude and phase usually shows that structure follows which spectrum?",
        "交换幅度和相位时，结构通常跟谁走？",
        ("The phase", "The magnitude", "The DC coefficient only"),
        ("相位", "幅度", "只跟直流系数"),
        0,
        "A magnitude-only reconstruction with zero phase loses the alignment of the sinusoids.",
        "相位置零的纯幅度重建会丢掉各个正弦分量的对齐方式。",
    ),
    QuizQuestion(
        "q-alias",
        "l2",
        "Downsampling without a prefilter can create aliasing when the image contains frequencies above what?",
        "不做预滤波就降采样，图像里高于什么频率的成分会造成混叠？",
        ("Half the new sampling rate", "The DC term", "The kernel size"),
        ("新采样率的一半", "直流项", "核的大小"),
        0,
        "After keeping every k-th pixel, the new Nyquist rate is 0.5/k cycles per original pixel.",
        "每隔 k 个像素取一次之后，新的奈奎斯特频率是每原像素 0.5/k 周。",
    ),
    QuizQuestion(
        "q-inverse",
        "l3",
        "Naive inverse filtering divides by H. What happens where |H| is very small?",
        "朴素逆滤波要除以 H。|H| 很小时会发生什么？",
        ("Noise and roundoff are amplified", "The noise is removed exactly", "The blur kernel becomes a median"),
        ("噪声和舍入误差被放大", "噪声被精确消除", "模糊核变成中值"),
        0,
        "The inverse gain is 1/|H|. Wiener's |H|² + K keeps that gain finite.",
        "逆增益是 1/|H|。维纳滤波的 |H|² + K 使增益保持有限。",
    ),
    QuizQuestion(
        "q-map",
        "l3",
        "Which object is hidden from gradient descent, conjugate gradients, and the Fourier MAP solver?",
        "梯度下降、共轭梯度和傅里叶 MAP 求解器都看不到哪一个量？",
        ("The original image, except when scoring error afterwards", "The observation g", "The operator A"),
        ("原图；它只在事后计算误差时使用", "观测 g", "算子 A"),
        0,
        "The original image builds g and the reported MSE. It is not an input to the solvers.",
        "原图用来生成 g 和报告 MSE。它不是求解器的输入。",
    ),
    QuizQuestion(
        "q-wiener-map",
        "l3",
        "A constant-K Wiener filter matches quadratic MAP under which prior?",
        "常数 K 的维纳滤波在哪种先验下与二次 MAP 相同？",
        ("A white Gaussian penalty on f itself, not on its gradient", "The Sobel edge prior", "No prior at all"),
        ("惩罚 f 本身的白高斯先验，而不是惩罚梯度", "Sobel 边缘先验", "完全没有先验"),
        0,
        "The gradient MAP denominator is |H|² + λ(|Dx|² + |Dy|²), which varies with frequency.",
        "梯度 MAP 的分母是 |H|² + λ(|Dx|² + |Dy|²)，随频率变化。",
    ),
    QuizQuestion(
        "q-median",
        "l4",
        "Why is a median filter nonlinear?",
        "为什么中值滤波是非线性的？",
        ("The median of a sum is not the sum of the medians", "It is a Gaussian convolution", "It only changes the DC term"),
        ("和的中值不等于中值的和", "它是高斯卷积", "它只改变直流项"),
        0,
        "Sorting and selecting the middle sample cannot be written as a fixed kernel.",
        "排序后取中间样本，不能写成一个固定的核。",
    ),
    QuizQuestion(
        "q-jpeg",
        "l4",
        "What does this lab's JPEG-like page actually produce?",
        "本实验的类 JPEG 页面实际产生的是什么？",
        ("A reconstructed array after DCT and quantisation, not a JPEG file", "A standards-compliant .jpg file", "A Huffman-coded video stream"),
        ("DCT 和量化之后的重建数组，不是 JPEG 文件", "符合标准的 .jpg 文件", "霍夫曼编码的视频流"),
        0,
        "Entropy coding is shown separately. The quantisation table is a teaching table.",
        "熵编码单独演示。量化表是教学用表。",
    ),
)


def grade_numeric(problem_id: str, submitted: float) -> bool:
    problem = next(item for item in PROBLEMS if item.id == problem_id)
    return abs(float(submitted) - problem.answer) <= problem.tolerance + 1e-12
