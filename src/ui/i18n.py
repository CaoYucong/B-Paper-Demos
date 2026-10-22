"""Interface language: 中文, English, or Bilingual.

Widget values stay in the language they were stored in. Only text that is
shown to the reader is translated, so changing language does not reset the
selected lecture or experiment.
"""

from __future__ import annotations

import html
import re

import streamlit as st

LANGUAGES = ("zh", "en", "both")
LANGUAGE_LABELS = {"zh": "中文", "en": "English", "both": "Bilingual"}
_CJK = re.compile(r"[\u4e00-\u9fff]")
_INSTALLED = False

# Exact widget labels and figure titles from the original Lecture 3 pages.
GLOSSARY: dict[str, tuple[str, str]] = {
    "几何图形": ("Geometric shapes", "几何图形"),
    "锐边渐变": ("Sharp edge and ramp", "锐边渐变"),
    "重复条纹": ("Repeated stripes", "重复条纹"),
    "平滑与纹理": ("Smooth and texture", "平滑与纹理"),
    "观测图像 g": ("Observed image g", "观测图像 g"),
    "全零": ("All zeros", "全零"),
    "显示噪声 n": ("Show noise n", "显示噪声 n"),
    "显示高斯 PSF 核": ("Show the Gaussian PSF", "显示高斯 PSF 核"),
    "原始图像  f": ("Original image f", "原始图像  f"),
    "模糊图像  Af": ("Blurred image Af", "模糊图像  Af"),
    "观测图像  g = Af + n": ("Observed image g = Af + n", "观测图像  g = Af + n"),
    "噪声  n = g − Af": ("Noise n = g − Af", "噪声  n = g − Af"),
    "高斯 PSF（中心窗口）": ("Gaussian PSF, centre window", "高斯 PSF（中心窗口）"),
    "实验 1：看懂算子 A": ("Experiment 1: what the operator A does", "实验 1：看懂算子 A"),
    "一维玩具：把 A 写成真正的矩阵": ("One-dimensional toy: write A as a real matrix", "一维玩具：把 A 写成真正的矩阵"),
    "像素个数": ("Number of pixels", "像素个数"),
    "拖动下面的滑块改变原始像素 f。矩阵乘法会立刻更新。": (
        "Move the sliders to change the pixels in f. The matrix product updates immediately.",
        "拖动下面的滑块改变原始像素 f。矩阵乘法会立刻更新。",
    ),
    "高亮 A 的第 i 行": ("Highlight row i of A", "高亮 A 的第 i 行"),
    "相对误差": ("Relative error", "相对误差"),
    "二维图像：A 是圆卷积，不是那张三点矩阵": (
        "Two-dimensional image: A is circular convolution, not that three-point matrix",
        "二维图像：A 是圆卷积，不是那张三点矩阵",
    ),
    "高斯模糊": ("Gaussian blur", "高斯模糊"),
    "额外对比一个不对称的运动模糊（它不是本实验的复原模型）": (
        "Also compare an asymmetric motion blur. It is not the restoration model used here.",
        "额外对比一个不对称的运动模糊（它不是本实验的复原模型）",
    ),
    "运动模糊": ("Motion blur", "运动模糊"),
    "为什么梯度里出现的是 Aᵀ，而不是 A": (
        "Why the gradient contains Aᵀ rather than A",
        "为什么梯度里出现的是 Aᵀ，而不是 A",
    ),
    "原始像素向量  f": ("Original pixel vector f", "原始像素向量  f"),
    "矩阵乘法  Af（此玩具不加噪声）": ("Matrix product Af. This toy adds no noise.", "矩阵乘法  Af（此玩具不加噪声）"),
    "A(f)：高斯卷积": ("A(f): Gaussian convolution", "A(f)：高斯卷积"),
    "检验向量 y": ("Probe vector y", "检验向量 y"),
    "运动模糊 PSF = A(δ)": ("Motion-blur PSF = A(δ)", "运动模糊 PSF = A(δ)"),
    "log₁₀|H|，零频在中心。这是 A 的频率响应，不是 PSF": (
        "log₁₀|H|, DC at the centre. This is the frequency response of A, not the PSF.",
        "log₁₀|H|，零频在中心。这是 A 的频率响应，不是 PSF",
    ),
    "学习率比例  s = η / (1/L)": ("Step fraction s = η / (1/L)", "学习率比例  s = η / (1/L)"),
    "迭代次数": ("Iterations", "迭代次数"),
    "初始化": ("Initialisation", "初始化"),
    "每步把像素裁剪到 [0, 1]。这是投影梯度下降，不是普通梯度下降": (
        "Clip pixels to [0, 1] after every step. This is projected gradient descent, not plain gradient descent.",
        "每步把像素裁剪到 [0, 1]。这是投影梯度下降，不是普通梯度下降",
    ),
    "谱半径 ρ": ("Spectral radius ρ", "谱半径 ρ"),
    "Lipschitz 常数 L": ("Lipschitz constant L", "Lipschitz 常数 L"),
    "实际学习率 η": ("Step size η", "实际学习率 η"),
    "等价小步长 α = η/N": ("Equivalent small step α = η/N", "等价小步长 α = η/N"),
    "运行优化": ("Run optimisation", "运行优化"),
    "查看已保存的中间迭代（不会重新优化）": (
        "View a saved iteration. This does not run the optimiser again.",
        "查看已保存的中间迭代（不会重新优化）",
    ),
    "当前迭代": ("Iteration", "当前迭代"),
    "总损失 J": ("Total cost J", "总损失 J"),
    "数据保真": ("Data fidelity", "数据保真"),
    "正则项": ("Regulariser", "正则项"),
    "梯度范数 ||∇J||": ("Gradient norm ||∇J||", "梯度范数 ||∇J||"),
    "MSE（相对原图，仅评估）": ("MSE against the original, evaluation only", "MSE（相对原图，仅评估）"),
    "PSNR（仅评估）": ("PSNR, evaluation only", "PSNR（仅评估）"),
    "数据保真 + 正则项 = 总损失。MSE 和 PSNR 使用了原图，优化器没有使用。": (
        "Data fidelity plus the regulariser equals the total cost. MSE and PSNR use the original; the optimiser does not.",
        "数据保真 + 正则项 = 总损失。MSE 和 PSNR 使用了原图，优化器没有使用。",
    ),
    "L 是 ∇J 的 Lipschitz 常数。s = 1 表示 η = 1/L；s < 2 才有无约束收敛保证。": (
        "L is a Lipschitz constant of ∇J. s = 1 means η = 1/L. s < 2 is the unconstrained convergence guarantee.",
        "L 是 ∇J 的 Lipschitz 常数。s = 1 表示 η = 1/L；s < 2 才有无约束收敛保证。",
    ),
    "s 接近稳定边界。若损失曲线上升，请减小 s。": (
        "s is near the stability boundary. Reduce s if the cost rises.",
        "s 接近稳定边界。若损失曲线上升，请减小 s。",
    ),
    "λ 非常小，最小值接近逆滤波，噪声会被放大。": (
        "λ is very small, so the minimiser is close to inverse filtering and noise is amplified.",
        "λ 非常小，最小值接近逆滤波，噪声会被放大。",
    ),
    "λ 很大，解会被拉向常数图像，边缘发糊。这是过正则。": (
        "λ is large, so the solution is pulled toward a constant image and edges blur. This is over-regularisation.",
        "λ 很大，解会被拉向常数图像，边缘发糊。这是过正则。",
    ),
    "裁剪打开后，迭代一般不会停在无约束傅里叶解上。下面的损失仍是无约束的 J，所以不一定单调下降。": (
        "With clipping, the iteration need not stop at the unconstrained Fourier solution. The cost below is still the unconstrained J, so it need not decrease at every step.",
        "裁剪打开后，迭代一般不会停在无约束傅里叶解上。下面的损失仍是无约束的 J，所以不一定单调下降。",
    ),
    "正在计算梯度下降。128×128、几百步通常只要一两秒。": (
        "Running gradient descent. A 128×128 image and a few hundred steps usually take a second or two.",
        "正在计算梯度下降。128×128、几百步通常只要一两秒。",
    ),
    "设置好 λ、学习率和迭代次数后，点击「运行优化」。中间结果会存下来，拖动滑块不会重新计算。": (
        "Set λ, the step fraction, and the iteration count, then press Run optimisation. Snapshots are stored, so moving the slider does not recompute.",
        "设置好 λ、学习率和迭代次数后，点击「运行优化」。中间结果会存下来，拖动滑块不会重新计算。",
    ),
    "左侧参数或本页的优化设置已经改变。下面仍是上一次运行的结果，请重新点击按钮。": (
        "A parameter changed after the last run. The pictures below are from that run. Press the button again.",
        "左侧参数或本页的优化设置已经改变。下面仍是上一次运行的结果，请重新点击按钮。",
    ),
    "迭代出现非有限数值，已停在最后一个有限的图像。请把学习率比例降到 2 以下。": (
        "A non-finite value appeared. The last finite image was kept. Set the step fraction below 2.",
        "迭代出现非有限数值，已停在最后一个有限的图像。请把学习率比例降到 2 以下。",
    ),
    "损失比起点更高。这一步长对当前的 L 不稳定。": (
        "The cost is higher than at the start. This step is unstable for the current L.",
        "损失比起点更高。这一步长对当前的 L 不稳定。",
    ),
    "投影之后，无约束目标 J 没有下降。可以关掉裁剪，对照普通梯度下降。": (
        "After projection, the unconstrained cost J did not fall. Turn clipping off to compare with plain gradient descent.",
        "投影之后，无约束目标 J 没有下降。可以关掉裁剪，对照普通梯度下降。",
    ),
    "原图 f（只用于评估）": ("Original f, evaluation only", "原图 f（只用于评估）"),
    "观测 g": ("Observation g", "观测 g"),
    "当前 MAP 估计": ("Current MAP estimate", "当前 MAP 估计"),
    "把估计再模糊  A(f̂)": ("Re-blurred estimate A(f̂)", "把估计再模糊  A(f̂)"),
    "残差  g − A(f̂)": ("Residual g − A(f̂)", "残差  g − A(f̂)"),
    "同一条损失曲线的 Matplotlib 静态图": (
        "The same cost curve as a Matplotlib figure",
        "同一条损失曲线的 Matplotlib 静态图",
    ),
    "实验 2 和实验 3": ("Experiments 2 and 3", "实验 2 和实验 3"),
    "方法": ("Method", "方法"),
    "时间 (ms)": ("Time (ms)", "时间 (ms)"),
    "目标 J": ("Cost J", "目标 J"),
    "说明": ("Note", "说明"),
    "log₁₀(K)，维纳滤波的常数": ("log₁₀(K), the Wiener constant", "log₁₀(K)，维纳滤波的常数"),
    "log₁₀(逆滤波阈值)": ("log₁₀(inverse-filter threshold)", "log₁₀(逆滤波阈值)"),
    "|H| 低于阈值的频率被置零。这是硬截断，不是维纳滤波的软衰减。": (
        "Frequencies with |H| below the threshold are set to zero. This is a hard cutoff, not the soft decay of a Wiener filter.",
        "|H| 低于阈值的频率被置零。这是硬截断，不是维纳滤波的软衰减。",
    ),
    "不复原（观测 g）": ("No restoration (observed g)", "不复原（观测 g）"),
    "把 g 当作估计。": ("Use g as the estimate.", "把 g 当作估计。"),
    "朴素逆滤波（inverse filter）": ("Plain inverse filter", "朴素逆滤波（inverse filter）"),
    "维纳滤波（Wiener）": ("Wiener filter", "维纳滤波（Wiener）"),
    "常数 K，对应白先验 λ||f||²，不是梯度先验。": (
        "Constant K. This matches a white prior λ||f||², not the gradient prior.",
        "常数 K，对应白先验 λ||f||²，不是梯度先验。",
    ),
    "MAP 傅里叶闭式解（Fourier MAP）": ("Fourier MAP closed form", "MAP 傅里叶闭式解（Fourier MAP）"),
    "当前 λ 下二次目标的精确最小值。": (
        "Exact minimiser of the quadratic cost at the current λ.",
        "当前 λ 下二次目标的精确最小值。",
    ),
    "不复原：观测 g": ("No restoration: observed g", "不复原：观测 g"),
    "朴素逆滤波": ("Plain inverse filter", "朴素逆滤波"),
    "维纳滤波": ("Wiener filter", "维纳滤波"),
    "MAP 傅里叶闭式解": ("Fourier MAP solution", "MAP 傅里叶闭式解"),
    "λ 如何改变精确 MAP 解": ("How λ changes the exact MAP solution", "λ 如何改变精确 MAP 解"),
    "迭代法是否走到同一个二次解": (
        "Whether the iterative methods reach the same quadratic solution",
        "迭代法是否走到同一个二次解",
    ),
    "对比用的梯度下降迭代次数": ("Gradient-descent iterations for the comparison", "对比用的梯度下降迭代次数"),
    "这里的梯度下降固定使用稳定步长 s = 1，从 g 出发，并且不裁剪像素。学习率实验请回到第 3 页。": (
        "Gradient descent here uses the stable fraction s = 1, starts from g, and does not clip. The step-size experiment is on the MAP page.",
        "这里的梯度下降固定使用稳定步长 s = 1，从 g 出发，并且不裁剪像素。学习率实验请回到第 3 页。",
    ),
    "运行梯度下降与共轭梯度": ("Run gradient descent and conjugate gradients", "运行梯度下降与共轭梯度"),
    "正在运行梯度下降和矩阵自由共轭梯度…": (
        "Running gradient descent and matrix-free conjugate gradients…",
        "正在运行梯度下降和矩阵自由共轭梯度…",
    ),
    "参数已变化。下面的迭代结果属于上一次按钮运行。": (
        "A parameter changed. The iterative result below belongs to the previous button press.",
        "参数已变化。下面的迭代结果属于上一次按钮运行。",
    ),
    "MAP 梯度下降（gradient descent）": ("MAP gradient descent", "MAP 梯度下降（gradient descent）"),
    "MAP 共轭梯度（conjugate gradient）": ("MAP conjugate gradients", "MAP 共轭梯度（conjugate gradient）"),
    "MAP 梯度下降": ("MAP gradient descent", "MAP 梯度下降"),
    "MAP 共轭梯度": ("MAP conjugate gradients", "MAP 共轭梯度"),
    "共轭梯度 − 傅里叶解": ("Conjugate gradients minus the Fourier solution", "共轭梯度 − 傅里叶解"),
    "实验 2：在这张 PSNR 曲线上选 λ": ("Experiment 2: choose λ on this PSNR curve", "实验 2：在这张 PSNR 曲线上选 λ"),
    "退化系数 a": ("Degradation coefficient a", "退化系数 a"),
    "噪声标准差 σₙ": ("Noise standard deviation σₙ", "噪声标准差 σₙ"),
    "先验均值 μ_f": ("Prior mean μ_f", "先验均值 μ_f"),
    "先验标准差 σ_f": ("Prior standard deviation σ_f", "先验标准差 σ_f"),
    "解析 MAP": ("Analytic MAP", "解析 MAP"),
    "网格搜索最大值": ("Grid-search maximum", "网格搜索最大值"),
    "|解析 − 网格|": ("|analytic − grid|", "|解析 − 网格|"),
    "后验标准差": ("Posterior standard deviation", "后验标准差"),
    "由观测反推的 g/a": ("g/a inverted from the observation", "由观测反推的 g/a"),
    "从标量公式走到图像上的优化问题": (
        "From the scalar formula to the optimisation problem on an image",
        "从标量公式走到图像上的优化问题",
    ),
    "三条曲线的纵轴各自缩放。橙色虚线是 MAP。似然画的是 p(g|f) 作为 f 的函数，不是作为 g 的函数。": (
        "Each curve has its own vertical scale. The orange dashed line is the MAP. The likelihood is p(g|f) as a function of f, not of g.",
        "三条曲线的纵轴各自缩放。橙色虚线是 MAP。似然画的是 p(g|f) 作为 f 的函数，不是作为 g 的函数。",
    ),
    "a = 0 时，观测里没有 f 的信息，似然对 f 是常数，MAP 等于先验均值。": (
        "When a = 0 the observation carries no information about f, the likelihood is constant in f, and the MAP equals the prior mean.",
        "a = 0 时，观测里没有 f 的信息，似然对 f 是常数，MAP 等于先验均值。",
    ),
    "总损失 J(f)": ("Total cost J(f)", "总损失 J(f)"),
    "数据保真 (1/N)||Af−g||²": ("Data fidelity (1/N)||Af−g||²", "数据保真 (1/N)||Af−g||²"),
    "正则项 λ(1/N)||Df||²": ("Regulariser λ(1/N)||Df||²", "正则项 λ(1/N)||Df||²"),
    "损失随迭代的变化": ("Cost against iteration", "损失随迭代的变化"),
    "迭代 k": ("Iteration k", "迭代 k"),
    "损失": ("Cost", "损失"),
    "梯度范数": ("Gradient norm", "梯度范数"),
    "MAP 目标函数": ("MAP objective", "MAP 目标函数"),
    "显式模糊矩阵 A": ("Explicit blur matrix A", "显式模糊矩阵 A"),
    "输入像素 j": ("Input pixel j", "输入像素 j"),
    "输出像素 i": ("Output pixel i", "输出像素 i"),
    "权重": ("Weight", "权重"),
    "强度": ("Intensity", "强度"),
    "像素下标": ("Pixel index", "像素下标"),
    "虚线是 MAP 估计": ("The dashed line is the MAP estimate", "虚线是 MAP 估计"),
    "未知量 f": ("Unknown f", "未知量 f"),
    "先验 prior  p(f)": ("Prior p(f)", "先验 prior  p(f)"),
    "似然 likelihood  p(g | f)": ("Likelihood p(g | f)", "似然 likelihood  p(g | f)"),
    "归一化后验 posterior  p(f | g)": ("Normalised posterior p(f | g)", "归一化后验 posterior  p(f | g)"),
    "PSNR（相对原图）": ("PSNR against the original", "PSNR（相对原图）"),
    "解的数据保真项": ("Data fidelity of the solution", "解的数据保真项"),
    "改变 λ 时，傅里叶 MAP 解的表现": (
        "Fourier MAP solution as λ changes",
        "改变 λ 时，傅里叶 MAP 解的表现",
    ),
    "行 %{y}<br>列 %{x}<br>值 %{z:.4f}<extra></extra>": (
        "row %{y}<br>col %{x}<br>value %{z:.4f}<extra></extra>",
        "行 %{y}<br>列 %{x}<br>值 %{z:.4f}<extra></extra>",
    ),
    "输出 i=%{y}<br>输入 j=%{x}<br>A=%{z:.3f}<extra></extra>": (
        "output i=%{y}<br>input j=%{x}<br>A=%{z:.3f}<extra></extra>",
        "输出 i=%{y}<br>输入 j=%{x}<br>A=%{z:.3f}<extra></extra>",
    ),
    "像素 %{x}<br>值 %{y:.4f}<extra></extra>": (
        "pixel %{x}<br>value %{y:.4f}<extra></extra>",
        "像素 %{x}<br>值 %{y:.4f}<extra></extra>",
    ),
    "色标按残差的 99% 分位对称设置。": (
        "The colour scale is symmetric about the 99th percentile of the absolute residual.",
        "色标按残差的 99% 分位对称设置。",
    ),
    "色标覆盖核的最小值到最大值。": (
        "The colour scale covers the minimum and maximum of the kernel.",
        "色标覆盖核的最小值到最大值。",
    ),
    "显示范围固定为 [0, 1]。": ("The display range is fixed at [0, 1].", "显示范围固定为 [0, 1]。"),
}


def language_choice(raw: str | None) -> str:
    """A stored language, or bilingual when the value is missing or unknown."""
    return raw if raw in LANGUAGES else "both"


def language() -> str:
    try:
        value = st.session_state.get("ui_language", "both")
    except Exception:
        return "both"
    return language_choice(value if isinstance(value, str) else None)


def html_pair(en: str, zh: str) -> str:
    """Both languages in the page. CSS shows the one the reader picked."""
    return (
        f'<span class="b14-en">{html.escape(en)}</span>'
        f'<span class="b14-sep"> / </span>'
        f'<span class="b14-zh">{html.escape(zh)}</span>'
    )


def t(en: str, zh: str) -> str:
    return html_pair(en, zh)


def plain(en: str, zh: str) -> str:
    """Buttons, options, figures, and progress text cannot hide one language."""
    return f"{en} / {zh}"


def split_pair(text: str) -> tuple[str, str] | None:
    if " / " not in text or "<" in text or _CJK.search(text) is None:
        return None
    english, chinese = text.split(" / ", 1)
    if _CJK.search(chinese) is None or _CJK.search(english):
        return None
    return english, chinese


_SPAN = re.compile(r'<span class="b14-(en|zh|sep)">(.*?)</span>', re.S)


def plain_from_spans(text: str) -> str:
    matches = list(_SPAN.finditer(text))
    if not matches:
        return text
    english = [html.unescape(match.group(2)) for match in matches if match.group(1) == "en"]
    chinese = [html.unescape(match.group(2)) for match in matches if match.group(1) == "zh"]
    if len(english) == 1 and len(chinese) == 1 and _SPAN.sub("", text).strip() == "":
        return f"{english[0]} / {chinese[0]}"
    return _SPAN.sub(
        lambda match: "" if match.group(1) == "sep" else html.unescape(match.group(2)),
        text,
    )


def L(text: str) -> str:
    """Split one ``English / 中文`` string. Strings without that shape are unchanged."""
    if "b14-en" in text:
        return plain_from_spans(text)
    pair = split_pair(text)
    if pair is None:
        return text
    return plain(*pair)


def shown_label(text: str) -> str:
    if "b14-en" in text:
        return plain_from_spans(text)
    pair = GLOSSARY.get(text)
    if pair is not None:
        return plain(*pair)
    return L(text)


def shown_body(text: str) -> str:
    """Keep both languages in the page. A short single-language caption is tagged, not dropped."""
    # Tags, anchors, and text we already marked must stay intact. A later
    # language choice hides a marked caption; it must not hide a section anchor.
    if not text or "<" in text:
        return text
    translated = shown_label(text)
    if translated != text:
        return translated
    if "$$" in text:
        return text
    cjk = len(_CJK.findall(text))
    latin = sum(character.isascii() and character.isalpha() for character in text)
    if cjk >= 4 and latin < 15 and len(text) < 500:
        return f'<span class="b14-only b14-zh"></span>\n\n{text}'
    if cjk == 0 and latin >= 24 and len(text) < 500:
        return f'<span class="b14-only b14-en"></span>\n\n{text}'
    return text


def say(en: str, zh: str) -> None:
    """Markdown stays markdown. The marker lets CSS hide one language."""
    st.markdown(f'<span class="b14-only b14-en"></span>\n\n{en}', unsafe_allow_html=True)
    st.markdown(f'<span class="b14-only b14-zh"></span>\n\n{zh}', unsafe_allow_html=True)


def colour_notes(kind: str, low: float, high: float) -> tuple[str, str]:
    if kind == "residual":
        return (
            "The colour scale is symmetric about the 99th percentile of the absolute residual.",
            "色标按残差的 99% 分位对称设置。",
        )
    if kind == "kernel":
        return (
            "The colour scale covers the minimum and maximum of the kernel.",
            "色标覆盖核的最小值到最大值。",
        )
    if low >= -0.05 and high <= 1.05:
        return ("The display range is fixed at [0, 1].", "显示范围固定为 [0, 1]。")
    return (
        f"Values leave [0, 1] ({low:.2f} to {high:.2f}); the figure is stretched to the 1%–99% percentiles.",
        f"数值超出 [0, 1]（{low:.2f} 到 {high:.2f}），图按 1%–99% 分位拉伸。",
    )


def forget_translation_keys(*keys: str) -> None:
    """Drop language-specific copies so the next selectbox follows the canonical value."""
    for key in keys:
        for suffix in LANGUAGES:
            st.session_state.pop(f"{key}@@{suffix}", None)


_BOOT = """
<script>
(function () {
  function choice(value) {
    return value === "zh" || value === "en" || value === "both" ? value : "both";
  }
  function saved() {
    try { return choice(localStorage.getItem("b14-language")); }
    catch (error) { return "both"; }
  }
  function apply(lang) {
    lang = choice(lang);
    document.documentElement.setAttribute("data-lang", lang);
    try {
      if (window.top !== window) window.top.document.documentElement.setAttribute("data-lang", lang);
    } catch (error) {}
  }
  apply(saved());
  if (window.__b14LangPick) return;
  window.__b14LangPick = true;
  document.addEventListener("click", function (event) {
    var button = event.target && event.target.closest ? event.target.closest("[data-set-lang]") : null;
    if (!button) return;
    event.preventDefault();
    var lang = choice(button.getAttribute("data-set-lang"));
    try { localStorage.setItem("b14-language", lang); } catch (error) {}
    apply(lang);
  }, true);
})();
</script>
"""

_HEADING = {"title": 1, "header": 2, "subheader": 3}
_HTML_LABEL = {"slider", "select_slider", "number_input", "file_uploader", "selectbox", "radio", "text_input", "metric"}
_ALERT = {"info", "warning", "success", "error"}
_ORIGINAL_MARKDOWN = None


def mount_language() -> None:
    """Read the saved language before the page text is shown. Clicks do not rerun the script."""
    st.html(_BOOT, unsafe_allow_javascript=True)


def language_switch() -> None:
    st.markdown(
        '<p class="b14-field">'
        + t("Language", "语言")
        + '</p><div class="b14-lang" role="radiogroup" aria-label="Language">'
        + '<button type="button" data-set-lang="zh">中文</button>'
        + '<button type="button" data-set-lang="en">English</button>'
        + '<button type="button" data-set-lang="both">Bilingual</button>'
        + "</div>",
        unsafe_allow_html=True,
    )


def _lift_markup(text: str) -> str:
    def heading(match: re.Match[str]) -> str:
        return f"<h{len(match.group(1))}>{match.group(2)}</h{len(match.group(1))}>"

    text = re.sub(r"^(#{1,6})[ \t]+(.+)$", heading, text, count=1, flags=re.M)
    text = re.sub(
        r"\*\*((?:<span class=\"b14-(?:en|zh|sep)\">.*?</span>)+)\*\*",
        r"<strong>\1</strong>",
        text,
        flags=re.S,
    )

    def link(match: re.Match[str]) -> str:
        href = html.escape(match.group(3), quote=True)
        return f'{match.group(1)}<a href="{href}">{match.group(2)}</a>'

    return re.sub(
        r"((?:&nbsp;|\s)*)\[((?:<span class=\"b14-(?:en|zh|sep)\">.*?</span>)+)\]\(([^)\s]+)\)",
        link,
        text,
        flags=re.S,
    )


def present_text(text: str) -> tuple[str, bool]:
    """Turn a bilingual string into HTML when CSS should be able to hide one half."""
    if "<style" in text:
        return text, True
    if "b14-en" in text or "b14-zh" in text or "b14-only" in text:
        return _lift_markup(text), True
    heading = re.fullmatch(r"(#{1,6})[ \t]+([\s\S]+)", text)
    if heading:
        inner, unsafe = present_text(heading.group(2).strip())
        if unsafe:
            level = len(heading.group(1))
            return f"<h{level}>{inner}</h{level}>", True
    bold = re.fullmatch(r"\*\*([\s\S]+)\*\*", text)
    if bold:
        inner, unsafe = present_text(bold.group(1))
        if unsafe:
            return f"<strong>{inner}</strong>", True
    link = re.fullmatch(r"((?:&nbsp;|\s)*)\[([^\]]+)\]\(([^)\s]+)\)", text)
    if link:
        inner, unsafe = present_text(link.group(2))
        if unsafe:
            href = html.escape(link.group(3), quote=True)
            return f'{link.group(1)}<a href="{href}">{inner}</a>', True
    if "\n\n" in text:
        parts = text.split("\n\n")
        rendered = [present_text(part) for part in parts]
        if any(flag for _, flag in rendered):
            chunks = [
                f"<p>{html_text if flag else html.escape(part)}</p>"
                for (html_text, flag), part in zip(rendered, parts)
            ]
            return "".join(chunks), True
    pair = split_pair(text)
    if pair is not None:
        return html_pair(*pair), True
    return text, False


def _emit(body: str) -> None:
    _ORIGINAL_MARKDOWN(body, unsafe_allow_html=True)


def _label_of(args: tuple, kwargs: dict) -> str | None:
    if args and isinstance(args[0], str):
        return args[0]
    label = kwargs.get("label")
    return label if isinstance(label, str) else None


def _put_label(args: tuple, kwargs: dict, label: str) -> tuple:
    if args and isinstance(args[0], str):
        return (label,) + args[1:], kwargs
    kwargs["label"] = label
    return args, kwargs


def _translate_format(format_func):
    def wrapped(option, user_format=format_func):
        rendered = option if user_format is None else user_format(option)
        if isinstance(rendered, str):
            return shown_label(rendered)
        return str(option) if user_format is None else rendered

    return wrapped


def _wrap_label(name: str) -> None:
    original = getattr(st, name)

    def wrapped(*args, **kwargs):
        label = _label_of(args, kwargs)
        if isinstance(kwargs.get("help"), str):
            kwargs["help"] = shown_label(kwargs["help"])
        if name in {"selectbox", "radio", "select_slider"}:
            kwargs["format_func"] = _translate_format(kwargs.get("format_func"))
        user_key = kwargs.get("key") if name == "selectbox" else None
        if isinstance(user_key, str):
            lang_key = f"{user_key}@@{language()}"
            if lang_key not in st.session_state and user_key in st.session_state:
                st.session_state[lang_key] = st.session_state[user_key]
            kwargs["key"] = lang_key
        if name == "metric" and len(args) >= 2 and isinstance(args[1], str) and "b14-en" in args[1]:
            args = (args[0], shown_label(args[1]), *args[2:])
        if isinstance(label, str) and name in _HEADING:
            html_text, unsafe = present_text(label)
            if not unsafe:
                html_text = html.escape(html_text)
            _emit(f"<h{_HEADING[name]}>{html_text}</h{_HEADING[name]}>")
            return None
        if isinstance(label, str) and name in _HTML_LABEL:
            html_text, unsafe = present_text(label)
            if not unsafe:
                html_text = html.escape(html_text)
            _emit(f'<p class="b14-field">{html_text}</p>')
            kwargs["label_visibility"] = "collapsed"
            args, kwargs = _put_label(args, kwargs, shown_label(label) or " ")
        elif isinstance(label, str):
            args, kwargs = _put_label(args, kwargs, shown_label(label))
        result = original(*args, **kwargs)
        if isinstance(user_key, str):
            st.session_state[user_key] = result
        return result

    setattr(st, name, wrapped)


def _wrap_body(name: str) -> None:
    original = getattr(st, name)

    def wrapped(*args, **kwargs):
        if not args or not isinstance(args[0], str):
            return original(*args, **kwargs)
        text = shown_body(args[0])
        html_text, unsafe = present_text(text)
        if name in _ALERT:
            body = html_text if unsafe else html.escape(text)
            _emit(f'<div class="b14-note b14-note-{name}">{body}</div>')
            return None
        if name == "text" and unsafe:
            _emit(html_text)
            return None
        if unsafe:
            kwargs["unsafe_allow_html"] = True
            args = (html_text,) + args[1:]
        else:
            args = (text,) + args[1:]
        return original(*args, **kwargs)

    setattr(st, name, wrapped)


def install() -> None:
    """Translate labels at display time. Call once, after page config and before widgets."""
    global _INSTALLED, _ORIGINAL_MARKDOWN
    if _INSTALLED:
        return
    _INSTALLED = True
    _ORIGINAL_MARKDOWN = st.markdown
    for name in ("markdown", "caption", "write", "info", "warning", "success", "error", "text"):
        _wrap_body(name)
    for name in (
        "button",
        "download_button",
        "slider",
        "select_slider",
        "checkbox",
        "metric",
        "number_input",
        "file_uploader",
        "selectbox",
        "radio",
        "text_input",
        "expander",
        "header",
        "subheader",
        "title",
    ):
        _wrap_label(name)
