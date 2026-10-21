"""Original numerical checks for Signal Analysis.

These are not an official example sheet. Each answer is computed from the
same functions as the lecture page.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.signal.analog import butterworth_order, butterworth_polynomial_gain
from src.signal.design import (
    EXAMPLE_FIR,
    EXAMPLE_RECURSION,
    HANNING_TAPS,
    example_order_estimate,
    example_prewarp,
    iir_frequency_response,
)
from src.signal.digital import fir_frequency_response, parabolic_closed_form
from src.signal.spectral import DFT_EXAMPLE_SPECTRUM


@dataclass(frozen=True)
class NumericProblem:
    id: str
    topic_en: str
    topic_zh: str
    prompt_en: str
    prompt_zh: str
    answer: float
    tolerance: float
    solution_en: str
    solution_zh: str
    anchor: str


@dataclass(frozen=True)
class QuizQuestion:
    id: str
    prompt_en: str
    prompt_zh: str
    choices_en: tuple[str, ...]
    choices_zh: tuple[str, ...]
    correct: int


def _answers() -> dict[str, float]:
    order = butterworth_order(100.0, 2.0)
    gain = butterworth_polynomial_gain(2j, 7)
    attenuation = 1.0 / abs(gain)
    parabolic_dc = float(np.real(parabolic_closed_form(np.array([0.0]))[0]))
    hanning_nyquist = abs(complex(fir_frequency_response(HANNING_TAPS, np.array([math.pi]))[0]))
    analog_pass, analog_stop = example_prewarp()
    digital_order = example_order_estimate(analog_pass, analog_stop)
    dc_gain = abs(complex(iir_frequency_response(EXAMPLE_FIR, EXAMPLE_RECURSION, np.array([0.0]))[0]))
    cutoff_gain = abs(
        complex(iir_frequency_response(EXAMPLE_FIR, EXAMPLE_RECURSION, np.array([0.4 * math.pi]))[0])
    )
    return {
        "butterworth-order": math.ceil(order - 1e-9),
        "butterworth-check": attenuation,
        "parabolic-dc": parabolic_dc,
        "hanning-nyquist": hanning_nyquist,
        "iir-order": math.ceil(digital_order - 1e-9),
        "iir-dc": dc_gain,
        "iir-cutoff": cutoff_gain,
        "dft-dc": float(np.real(DFT_EXAMPLE_SPECTRUM[0]) / 4.0),
    }


ANSWERS = _answers()


PROBLEMS: tuple[NumericProblem, ...] = (
    NumericProblem(
        id="butterworth-order",
        topic_en="Lecture 1 · Butterworth order",
        topic_zh="第 1 讲 · 巴特沃斯阶数",
        prompt_en="A low-pass Butterworth filter needs 40 dB attenuation at 1 kHz and 3 dB at 500 Hz. The notes use n = log(A²−1) / (2 log(ωa/ωc)) with A = 100, then round up. What integer order meets the specification?",
        prompt_zh="低通巴特沃斯要求 1 kHz 处衰减 40 dB，500 Hz 处为 3 dB。笔记用 n = log(A²−1) / (2 log(ωa/ωc))，A = 100，再向上取整。满足指标的整数阶数是多少？",
        answer=ANSWERS["butterworth-order"],
        tolerance=0.01,
        solution_en="ωa/ωc = 2 and A = 100, so n ≈ 6.64. The next integer is 7.",
        solution_zh="ωa/ωc = 2，A = 100，所以 n ≈ 6.64。下一个整数是 7。",
        anchor="Butterworthlowpass",
    ),
    NumericProblem(
        id="butterworth-check",
        topic_en="Lecture 1 · Attenuation check",
        topic_zh="第 1 讲 · 衰减核对",
        prompt_en="Using the n = 7 row of the coefficient table, what is the attenuation A = 1/|G| at ωa/ωc = 2? Give A, not decibels.",
        prompt_zh="用系数表中 n = 7 那一行，在 ωa/ωc = 2 处衰减 A = 1/|G| 是多少？填 A，不要填分贝。",
        answer=ANSWERS["butterworth-check"],
        tolerance=0.5,
        solution_en="Substitute s/ωc = 2j into the degree-7 table. The notes record A = 128; the same polynomial gives that value.",
        solution_zh="把 s/ωc = 2j 代入 7 阶系数表。笔记写下 A = 128；同一多项式给出这个值。",
        anchor="Butterworthlowpass",
    ),
    NumericProblem(
        id="parabolic-dc",
        topic_en="Lecture 2 · Parabolic gain at DC",
        topic_zh="第 2 讲 · 抛物线滤波器的直流增益",
        prompt_en="For the causal five-point parabolic filter, what is |G(e^{jωT})| at ωT = 0?",
        prompt_zh="因果的五点抛物线滤波器在 ωT = 0 处的 |G(e^{jωT})| 是多少？",
        answer=ANSWERS["parabolic-dc"],
        tolerance=1e-6,
        solution_en="|G| = |17 + 24 cos ωT − 6 cos 2ωT| / 35. At ωT = 0 this is 35/35 = 1.",
        solution_zh="|G| = |17 + 24 cos ωT − 6 cos 2ωT| / 35。ωT = 0 时是 35/35 = 1。",
        anchor="Fivepointparabolicfilter",
    ),
    NumericProblem(
        id="hanning-nyquist",
        topic_en="Lecture 3 · Hanning zero",
        topic_zh="第 3 讲 · 汉宁零点",
        prompt_en="The three-point Hanning filter is y[k] = (1/4)(x[k] + 2x[k−1] + x[k−2]). What is |G| at z = −1?",
        prompt_zh="三点汉宁滤波器是 y[k] = (1/4)(x[k] + 2x[k−1] + x[k−2])。z = −1 处的 |G| 是多少？",
        answer=ANSWERS["hanning-nyquist"],
        tolerance=1e-9,
        solution_en="G(z) = (1/4)(1 + z^{−1})², so there is a double zero at z = −1 and the gain there is 0.",
        solution_zh="G(z) = (1/4)(1 + z^{−1})²，所以 z = −1 是二重零点，增益为 0。",
        anchor="ThreepointHanningfilter",
    ),
    NumericProblem(
        id="iir-order",
        topic_en="Lecture 3 · Digital Butterworth order",
        topic_zh="第 3 讲 · 数字巴特沃斯阶数",
        prompt_en="After pre-warping the 2 kHz and 4.25 kHz edges at fs = 10 kHz, the notes estimate n ≈ log10(A) / log10(ωa2/ωa1) for 30 dB. What integer order do they adopt?",
        prompt_zh="在 fs = 10 kHz 下把 2 kHz 和 4.25 kHz 预弯折之后，笔记对 30 dB 用 n ≈ log10(A) / log10(ωa2/ωa1)。他们采用的整数阶数是多少？",
        answer=ANSWERS["iir-order"],
        tolerance=0.01,
        solution_en="The estimate is about 1.98, so n = 2.",
        solution_zh="估计值约为 1.98，所以 n = 2。",
        anchor="DigitalButterworthexample",
    ),
    NumericProblem(
        id="iir-dc",
        topic_en="Lecture 3 · DC gain",
        topic_zh="第 3 讲 · 直流增益",
        prompt_en="For the printed difference equation of that example, what is |G(1)|?",
        prompt_zh="对那个例子印出的差分方程，|G(1)| 是多少？",
        answer=ANSWERS["iir-dc"],
        tolerance=1e-4,
        solution_en="G(z) = 0.20657 (z² + 2z + 1) / (z² − 0.36953 z + 0.19582). At z = 1 the gain is 1.",
        solution_zh="G(z) = 0.20657 (z² + 2z + 1) / (z² − 0.36953 z + 0.19582)。z = 1 时增益为 1。",
        anchor="DigitalButterworthexample",
    ),
    NumericProblem(
        id="iir-cutoff",
        topic_en="Lecture 3 · Cutoff gain",
        topic_zh="第 3 讲 · 截止增益",
        prompt_en="At the 3 dB digital frequency ωd = 0.4π, what is |G| for that same example? The notes record 0.707.",
        prompt_zh="在 3 dB 数字频率 ωd = 0.4π 处，同一个例子的 |G| 是多少？笔记记下 0.707。",
        answer=ANSWERS["iir-cutoff"],
        tolerance=0.002,
        solution_en="Evaluating the printed rational function at e^{j 0.4π} gives |G| = 0.707.",
        solution_zh="把印出的有理函数代到 e^{j 0.4π}，得到 |G| = 0.707。",
        anchor="DigitalButterworthexample",
    ),
    NumericProblem(
        id="dft-dc",
        topic_en="Lecture 4 · DC from F[0]",
        topic_zh="第 4 讲 · 由 F[0] 得到的直流",
        prompt_en="The four-point example has F[0] = 20. What DC value does the inverse transform assign, F[0]/N?",
        prompt_zh="四点例子中 F[0] = 20。逆变换给出的直流值 F[0]/N 是多少？",
        answer=ANSWERS["dft-dc"],
        tolerance=1e-9,
        solution_en="N = 4, so the DC term is 20/4 = 5, which is the constant term in f(t).",
        solution_zh="N = 4，所以直流项是 20/4 = 5，也就是 f(t) 里的常数项。",
        anchor="FourpointDFTexample",
    ),
)


QUIZ: tuple[QuizQuestion, ...] = (
    QuizQuestion(
        id="causal",
        prompt_en="A continuous system is causal exactly when its impulse response satisfies which condition?",
        prompt_zh="连续系统为因果，当且仅当冲激响应满足哪个条件？",
        choices_en=("g(t) = 0 for t < 0", "g(t) = 0 for t > 0", "g(t) is a sinc", "g(t) is constant"),
        choices_zh=("t < 0 时 g(t) = 0", "t > 0 时 g(t) = 0", "g(t) 是 sinc", "g(t) 是常数"),
        correct=0,
    ),
    QuizQuestion(
        id="stable-analogue",
        prompt_en="For a rational transfer function that is both causal and stable, where are the poles?",
        prompt_zh="既因果又稳定的有理传递函数，极点在哪里？",
        choices_en=("Open left half of the s-plane", "Open right half of the s-plane", "On the imaginary axis", "At infinity only"),
        choices_zh=("s 平面的开左半平面", "s 平面的开右半平面", "在虚轴上", "只在无穷远"),
        correct=0,
    ),
    QuizQuestion(
        id="nyquist",
        prompt_en="The notes call fs/2 the frequency above which a continuous sinusoid cannot be distinguished from a lower one after sampling. What do they name it?",
        prompt_zh="笔记把 fs/2 称为采样后无法与更低频率正弦区分开的那个频率。他们怎么称呼它？",
        choices_en=("Nyquist or folding frequency", "Cutoff of a Butterworth filter", "A DFT bin", "A window sidelobe"),
        choices_zh=("奈奎斯特频率或折叠频率", "巴特沃斯截止频率", "一个 DFT 频点", "窗函数的旁瓣"),
        correct=0,
    ),
    QuizQuestion(
        id="fir-poles",
        prompt_en="Where are the poles of a causal FIR pulse transfer function?",
        prompt_zh="因果 FIR 的脉冲传递函数，极点在哪里？",
        choices_en=("Only at the origin", "Only on the unit circle", "Anywhere inside the unit circle", "FIR filters have no z-transform"),
        choices_zh=("只在原点", "只在单位圆上", "单位圆内任意位置", "FIR 没有 z 变换"),
        correct=0,
    ),
    QuizQuestion(
        id="linear-phase",
        prompt_en="Which impulse responses does the lecture say have linear phase?",
        prompt_zh="讲义说哪一类冲激响应具有线性相位？",
        choices_en=(
            "Symmetric or antisymmetric FIR coefficients",
            "Every IIR filter",
            "Only Butterworth filters",
            "Only filters with a pole at z = 1",
        ),
        choices_zh=("对称或反对称的 FIR 系数", "每一个 IIR", "只有巴特沃斯", "只有 z = 1 处有极点的滤波器"),
        correct=0,
    ),
    QuizQuestion(
        id="gibbs",
        prompt_en="Truncating the ideal low-pass impulse response with a rectangle produces ripples. What do the notes call this?",
        prompt_zh="用矩形窗截断理想低通的冲激响应会产生波纹。笔记怎么称呼它？",
        choices_en=("Gibbs phenomenon", "Aliasing", "Spectral leakage", "Pre-warping"),
        choices_zh=("吉布斯现象", "混叠", "频谱泄漏", "预弯折"),
        correct=0,
    ),
    QuizQuestion(
        id="bilinear",
        prompt_en="What does the bilinear map do with the imaginary axis of the s-plane?",
        prompt_zh="双线性变换把 s 平面的虚轴映到哪里？",
        choices_en=("Onto the unit circle", "Onto the real axis of the z-plane", "Onto a horizontal strip", "Onto the origin only"),
        choices_zh=("映到单位圆上", "映到 z 平面的实轴", "映到一条水平带", "只映到原点"),
        correct=0,
    ),
    QuizQuestion(
        id="dft-period",
        prompt_en="The DFT treats a finite sequence as one period of which extension?",
        prompt_zh="DFT 把有限序列当作哪一种延拓的一个周期？",
        choices_en=("A periodic extension of the same length", "A zero-padded infinite sequence", "A continuous Fourier transform", "An AR model"),
        choices_zh=("同样长度的周期延拓", "补零后的无穷序列", "连续傅里叶变换", "一个 AR 模型"),
        correct=0,
    ),
)


def grade_numeric(problem_id: str, submitted: float) -> bool:
    problem = next(item for item in PROBLEMS if item.id == problem_id)
    return abs(float(submitted) - problem.answer) <= problem.tolerance
