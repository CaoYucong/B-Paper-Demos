"""One scrolling page for B14 Signal Analysis, in the order of the notes."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.b14.signal_analysis.lecture1.analog import (
    BUTTERWORTH_TABLE,
    butterworth_order,
    butterworth_polynomial_gain,
    butterworth_squared_magnitude,
    discrete_convolution,
    exponential_frequency_response,
    first_difference,
    ideal_lowpass_kernel,
    sampled_sinusoid,
    stable_butterworth_poles,
    uniform_quantise,
)
from src.b14.signal_analysis.lecture3.design import (
    EXAMPLE_CUTOFF_HZ,
    EXAMPLE_FIR,
    EXAMPLE_RECURSION,
    EXAMPLE_SAMPLE_RATE_HZ,
    EXAMPLE_STOP_HZ,
    HANNING_TAPS,
    apply_iir,
    bilinear_analog_frequency,
    endpoint_window,
    example_order_estimate,
    example_prewarp,
    iir_frequency_response,
    windowed_fir,
)
from src.b14.signal_analysis.lecture2.digital import (
    PARABOLIC_TAPS,
    alias_pair,
    apply_fir,
    causal_stable_alpha,
    fir_frequency_response,
    parabolic_closed_form,
    recursive_lowpass_response,
)
from src.b14.signal_analysis.practice.problems import PROBLEMS, QUIZ, grade_numeric
from src.b14.signal_analysis.lecture4.spectral import (
    DFT_EXAMPLE_SAMPLES,
    REPORTED_AR4,
    ar_psd,
    conjugate_pair_term,
    dft,
    idft,
    least_squares_ar,
    periodogram_bins,
    synthesise_ar,
    yule_walker_ar,
)
from src.ui.components import experiment_heading, lecture_anchor
from src.b14.signal_analysis.experiments import (
    SA_L1_EXPERIMENTS,
    SA_L2_EXPERIMENTS,
    SA_L3_EXPERIMENTS,
    SA_L4_EXPERIMENTS,
)
from src.ui.i18n import language_switch, plain, say, shown_label, t
from src.ui.navigation import _contents, _leave, _scroll_to_hash
from src.ui.routes import Route, module_of, paper_of, topic_route
from src.ui.common import show_fig


def _plot(x, series, title: str, xlabel: str, ylabel: str) -> None:
    show_fig(line_ready(x, series, title, xlabel, ylabel))


def line_ready(x, series, title: str, xlabel: str, ylabel: str):
    from src.ui.components import line_figure

    return line_figure(np.asarray(x, dtype=np.float64), [(name, np.asarray(values, dtype=np.float64)) for name, values in series], title, xlabel, ylabel)


def _plane(x, y, title: str, circle: bool) -> None:
    figure = go.Figure()
    if circle:
        theta = np.linspace(0.0, 2.0 * math.pi, 256)
        figure.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode="lines", name=shown_label("Unit circle / 单位圆"), line=dict(dash="dot")))
    figure.add_trace(go.Scatter(x=x, y=y, mode="markers", name=shown_label("Poles / 极点"), marker=dict(size=10)))
    figure.update_layout(
        title=dict(text=shown_label(title), font=dict(size=15)),
        height=360,
        margin=dict(l=12, r=12, t=46, b=12),
        xaxis_title=shown_label("Real / 实部"),
        yaxis_title=shown_label("Imaginary / 虚部"),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    show_fig(figure)


def render_signal_home() -> None:
    st.markdown("<div id='home'></div>", unsafe_allow_html=True)
    st.title(t("B14 Signal Analysis", "B14 信号分析"))
    st.caption(t("Michaelmas · Steve Roberts", "Michaelmas · Steve Roberts"))
    say(
        "The four lectures are one page, in the order of the notes. "
        "Lecture 1 is continuous-time LTI systems and analogue low-pass design. "
        "Lecture 2 samples those systems and introduces the z-transform. "
        "Lecture 3 designs FIR filters with windows and IIR filters with the bilinear transformation. "
        "Lecture 4 estimates a spectrum with the DFT and with an autoregressive model.",
        "四讲按笔记顺序排在这一页上。"
        "第 1 讲是连续时间 LTI 系统和模拟低通设计。"
        "第 2 讲对这些系统采样，并引入 z 变换。"
        "第 3 讲用窗函数设计 FIR，用双线性变换设计 IIR。"
        "第 4 讲用 DFT 和自回归模型估计频谱。",
    )
    say(
        "Chebyshev, elliptic, Kaiser, impulse invariance, Welch, and the Levinson–Durbin recursion are named in the notes and are not computed on this page. "
        "The practice questions at the bottom are original checks of the printed formulas. They are not an official example sheet.",
        "切比雪夫、椭圆、凯泽、冲激响应不变、Welch，以及 Levinson–Durbin 递推都在笔记里点过名，这一页不算它们。"
        "页末的练习是按印出的公式写的原创核对，不是官方习题纸。",
    )


def render_signal_lecture1() -> None:
    lecture_anchor("l1")
    st.title(t("Lecture 1 — Introduction and analogue filters", "第 1 讲 — 引言与模拟滤波器"))
    say(
        "A linear system obeys superposition, and it does not create new frequencies. "
        "A time-invariant system delays its output by the same amount as its input. "
        "A system is causal when g(t) = 0 for t < 0, and BIBO stable when g(t) is absolutely integrable.",
        "线性系统满足叠加，并且不产生新的频率。"
        "时不变系统把输出延迟得和输入一样多。"
        "t < 0 时 g(t) = 0，系统才是因果的；g(t) 绝对可积，系统才是 BIBO 稳定的。",
    )

    experiment_heading("l1", SA_L1_EXPERIMENTS[0])
    say(
        "The notes write a continuous sinusoid xa(t) = A cos(2π fa t + φ) and its samples xd[n] = A cos(2π fd n + φ), with fd = fa/fs. "
        "Frequencies separated by an integer number of cycles per sample are the same sequence. "
        "The uniform quantiser drawn under the samples is an illustration of the amplitude axis on the signal-types slide. The slide does not specify this rule.",
        "笔记写连续正弦 xa(t) = A cos(2π fa t + φ)，以及样本 xd[n] = A cos(2π fd n + φ)，其中 fd = fa/fs。"
        "相差整数个周期每样本的频率，得到的序列相同。"
        "样本下面的均匀量化只是信号类型那一页上幅度轴的示意。那一页没有给出这个规则。",
    )
    st.latex(r"x_d[n] = A\cos(2\pi f_d n + \phi), \quad f_d = f_a/f_s")
    amplitude = st.slider("Amplitude A / 幅度 A", 0.2, 2.0, 1.0, key="sig_amp")
    frequency = st.slider("Frequency fa (Hz) / 频率 fa（Hz）", 0.5, 40.0, 5.0, key="sig_fa")
    rate = st.slider("Sampling rate fs (Hz) / 采样率 fs（Hz）", 8.0, 80.0, 20.0, key="sig_fs")
    levels = st.slider("Quantiser levels / 量化级数", 2, 16, 4, key="sig_levels")
    dense_t = np.linspace(0.0, 1.0, 400)
    dense = amplitude * np.cos(2.0 * math.pi * frequency * dense_t)
    times, samples = sampled_sinusoid(amplitude, frequency, 0.0, rate, int(rate) + 1)
    quantised = uniform_quantise(samples, levels, amplitude)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=dense_t, y=dense, mode="lines", name=shown_label("Continuous / 连续")))
    figure.add_trace(go.Scatter(x=times, y=samples, mode="lines+markers", name=shown_label("Samples / 样本")))
    figure.update_layout(
        title=dict(text=shown_label("Continuous sinusoid / 连续正弦"), font=dict(size=15)),
        height=340,
        margin=dict(l=12, r=12, t=46, b=12),
        xaxis_title=shown_label("t (s) / 时间（秒）"),
        yaxis_title=shown_label("Amplitude / 幅度"),
        legend=dict(orientation="h"),
    )
    show_fig(figure)
    _plot(times, [("Samples / 样本", samples), ("Quantised / 量化", quantised)], "Samples and a uniform quantiser / 样本与均匀量化", "t (s) / 时间（秒）", "Amplitude / 幅度")
    st.caption(t(
        f"fd = {frequency / rate:.3f} cycles per sample. The Nyquist frequency is fs/2 = {rate / 2:.3g} Hz.",
        f"fd = {frequency / rate:.3f} 周期每样本。奈奎斯特频率是 fs/2 = {rate / 2:.3g} Hz。",
    ))

    experiment_heading("l1", SA_L1_EXPERIMENTS[1])
    say(
        "If x1 and x2 produce y1 and y2, then a x1 + b x2 produces a y1 + b y2. "
        "The system here is the sampled form of the lecture example y(t) = x(t) − x(t − 1), using one sample as the delay.",
        "若 x1、x2 分别得到 y1、y2，则 a x1 + b x2 得到 a y1 + b y2。"
        "这里的系统是讲义例子 y(t) = x(t) − x(t − 1) 的采样形式，延迟取一个样本。",
    )
    weight_a = st.slider("a", -2.0, 2.0, 1.0, key="sig_sup_a")
    weight_b = st.slider("b", -2.0, 2.0, -0.5, key="sig_sup_b")
    index = np.arange(64)
    first = np.sin(2.0 * math.pi * index / 16.0)
    second = np.cos(2.0 * math.pi * index / 9.0)
    combined_in = weight_a * first + weight_b * second
    direct = first_difference(combined_in)
    separate = weight_a * first_difference(first) + weight_b * first_difference(second)
    error = float(np.max(np.abs(direct - separate)))
    _plot(index, [("a y1 + b y2 / 分别计算", separate), ("y(a x1 + b x2) / 先相加", direct)], "Superposition / 叠加", "n / 样本", "Output / 输出")
    st.caption(t(f"The largest absolute difference is {error:.2e}.", f"最大绝对差是 {error:.2e}。"))

    experiment_heading("l1", SA_L1_EXPERIMENTS[2])
    say(
        "For a discrete LTI system the notes replace the convolution integral by y[n] = sum_k x[k] g[n − k]. "
        "The two short sequences below are the input and the impulse response.",
        "对离散 LTI 系统，笔记把卷积积分写成 y[n] = sum_k x[k] g[n − k]。"
        "下面两段短序列是输入和冲激响应。",
    )
    st.latex(r"y[n] = \sum_k x[k]\, g[n-k]")
    left, right = st.columns(2)
    signal = []
    impulse = []
    with left:
        for i, default in enumerate((1.0, 2.0, 1.0, 0.0)):
            signal.append(st.number_input(f"x[{i}]", value=default, key=f"sig_conv_x_{i}"))
    with right:
        for i, default in enumerate((0.25, 0.5, 0.25, 0.0)):
            impulse.append(st.number_input(f"g[{i}]", value=default, key=f"sig_conv_g_{i}"))
    output = discrete_convolution(np.array(signal), np.array(impulse))
    _plot(np.arange(output.size), [("y[n] / 输出", output)], "Discrete convolution / 离散卷积", "n / 样本", "y[n]")

    experiment_heading("l1", SA_L1_EXPERIMENTS[3])
    say(
        "The notes show that x(t) = e^{st} is an eigenfunction of an LTI system, with eigenvalue G(s). "
        "On the imaginary axis, a sinusoid comes out at the same frequency with amplitude |G(jω)| and phase arg G(jω). "
        "For the causal exponential g(t) = e^{−α t} u(t) with α > 0, they give G(s) = 1/(s + α).",
        "笔记说明 x(t) = e^{st} 是 LTI 系统的特征函数，特征值是 G(s)。"
        "在虚轴上，正弦以同一频率出去，幅度是 |G(jω)|，相位是 arg G(jω)。"
        "对 α > 0 的因果指数 g(t) = e^{−α t} u(t)，他们给出 G(s) = 1/(s + α)。",
    )
    st.latex(r"G(j\omega)=\frac{1}{\alpha+j\omega}")
    alpha = st.slider("α", 0.2, 4.0, 1.0, key="sig_alpha")
    omega = st.slider("ω (rad/s) / ω（弧度每秒）", 0.2, 8.0, 2.0, key="sig_omega")
    response = exponential_frequency_response(alpha, omega)
    time = np.linspace(0.0, 8.0, 400)
    steady = amplitude_from(response, omega, time)
    _plot(time, [("Steady output / 稳态输出", steady)], "Same frequency, new amplitude and phase / 频率不变，幅度和相位改变", "t (s) / 时间（秒）", "y(t)")
    st.caption(t(
        f"|G| = {abs(response):.4f}, phase = {math.degrees(math.atan2(response.imag, response.real)):.2f} degrees.",
        f"|G| = {abs(response):.4f}，相位 = {math.degrees(math.atan2(response.imag, response.real)):.2f} 度。",
    ))

    experiment_heading("l1", SA_L1_EXPERIMENTS[4])
    say(
        "An ideal low-pass of height 1 and cutoff ωc has impulse response g(t) = (ωc/π) sin(ωc t)/(ωc t). "
        "The response starts before the impulse, so the filter is not realisable. "
        "A delay makes most of the response occur later and adds a phase −ωT. The delayed filter is still not exactly realisable.",
        "高度为 1、截止频率为 ωc 的理想低通，冲激响应是 g(t) = (ωc/π) sin(ωc t)/(ωc t)。"
        "响应在冲激之前就开始了，所以这个滤波器不可实现。"
        "加一个延迟能让大部分响应出现得更晚，并带上相位 −ωT。延迟之后仍然不是严格可实现的。",
    )
    st.latex(r"g(t)=\frac{\omega_c}{\pi}\frac{\sin\omega_c t}{\omega_c t}")
    cutoff = st.slider("ωc (rad/s) / ωc（弧度每秒）", 0.5, 6.0, 2.0, key="sig_wc")
    delay = st.slider("Delay T (s) / 延迟 T（秒）", 0.0, 4.0, 1.5, key="sig_delay")
    span = 4.0 * math.pi / cutoff
    grid = np.linspace(-span, span, 500)
    _plot(
        grid,
        [("g(t) / 冲激响应", ideal_lowpass_kernel(grid, cutoff)), ("Delayed / 延迟后", ideal_lowpass_kernel(grid - delay, cutoff))],
        "Ideal low-pass kernel / 理想低通的核",
        "t (s) / 时间（秒）",
        "g(t)",
    )

    experiment_heading("l1", SA_L1_EXPERIMENTS[5])
    say(
        "The Butterworth choice is |G(jω)|² = 1 / (1 + (ω/ωc)^{2n}). "
        "At ω = ωc the power gain is 1/2, so the magnitude is 1/√2. "
        "The curves for every order meet at that cutoff. For large ω the notes give a slope of about −20n dB/decade. "
        "Poles lie equally spaced on the circle of radius ωc. The stable ones, in the open left half-plane, are assigned to G(s).",
        "巴特沃斯取 |G(jω)|² = 1 / (1 + (ω/ωc)^{2n})。"
        "在 ω = ωc 处功率增益是 1/2，所以幅度是 1/√2。"
        "各阶曲线在这个截止频率相交。ω 很大时，笔记给出大约 −20n dB/十倍频程的斜率。"
        "极点等间隔地落在半径 ωc 的圆上。开左半平面里的稳定极点分给 G(s)。",
    )
    st.latex(r"|G(j\omega)|^2=\frac{1}{1+(\omega/\omega_c)^{2n}}")
    order = int(st.slider("Order n / 阶数 n", 1, 8, 3, key="sig_bw_n"))
    frequencies = np.linspace(0.0, 4.0, 400)
    magnitude = np.sqrt(butterworth_squared_magnitude(frequencies, 1.0, order))
    _plot(frequencies, [("|G|", magnitude)], "Butterworth magnitude, ωc = 1 / 巴特沃斯幅度，ωc = 1", "ω/ωc", "|G|")
    poles = stable_butterworth_poles(order, 1.0)
    _plane(np.real(poles), np.imag(poles), "Stable Butterworth poles / 稳定的巴特沃斯极点", circle=True)
    st.caption(t(
        f"The denominator coefficients in the lecture table for n = {order} are {', '.join(f'{c:.4g}' for c in BUTTERWORTH_TABLE[order])}.",
        f"讲义系数表中 n = {order} 的分母系数是 {', '.join(f'{c:.4g}' for c in BUTTERWORTH_TABLE[order])}。",
    ))
    st.markdown(f"**{t('The worked specification', '笔记里的设计例子')}**")
    say(
        "The notes ask for at least 40 dB at 1 kHz and 3 dB at 500 Hz. They take A = 100 and n = log(A² − 1) / (2 log(ωa/ωc)), then round up.",
        "笔记要求 1 kHz 处至少 40 dB，500 Hz 处 3 dB。他们取 A = 100，n = log(A² − 1) / (2 log(ωa/ωc))，再向上取整。",
    )
    atten_db = st.slider("Attenuation (dB) / 衰减（dB）", 10.0, 80.0, 40.0, key="sig_bw_db")
    ratio = st.slider("ωa/ωc", 1.1, 4.0, 2.0, key="sig_bw_ratio")
    exact = butterworth_order(10.0 ** (atten_db / 20.0), ratio)
    chosen = int(math.ceil(exact - 1e-9))
    st.caption(t(f"n = {exact:.3f}, so the integer order is {chosen}.", f"n = {exact:.3f}，所以整数阶数是 {chosen}。"))
    if chosen <= 8:
        check = 1.0 / abs(butterworth_polynomial_gain(1j * ratio, chosen))
        st.caption(t(
            f"On the coefficient table, A = 1/|G| at this ratio is {check:.2f}.",
            f"用系数表，这个频率比上的 A = 1/|G| 是 {check:.2f}。",
        ))
    say(
        "The notes say a sharper transition is possible if ripples are allowed: Chebyshev Type I in the pass-band, Chebyshev Type II in the stop-band, and elliptic in both. Those designs are not calculated here.",
        "笔记说如果允许波纹，过渡可以更陡：切比雪夫 I 型的波纹在通带，II 型在阻带，椭圆滤波器两边都有。这些设计这里不算。",
    )


def amplitude_from(response: complex, omega: float, time: np.ndarray) -> np.ndarray:
    phase = math.atan2(float(np.imag(response)), float(np.real(response)))
    return abs(response) * np.cos(omega * time + phase)


def render_signal_lecture2() -> None:
    lecture_anchor("l2")
    st.title(t("Lecture 2 — Digital filters", "第 2 讲 — 数字滤波器"))
    say(
        "Sampling at interval T replaces xa(t) by the sequence x[k] = xa(kT). "
        "Digital frequencies outside [−1/2, 1/2] cycles per sample are aliases. "
        "The highest continuous frequency that remains distinct is fs/2, the Nyquist or folding frequency.",
        "以间隔 T 采样，是用序列 x[k] = xa(kT) 代替 xa(t)。"
        "[−1/2, 1/2] 周期每样本之外的数字频率是混叠。"
        "仍能区分的最高连续频率是 fs/2，也就是奈奎斯特频率或折叠频率。",
    )

    experiment_heading("l2", SA_L2_EXPERIMENTS[0])
    say(
        "The notes compare cos(π t / (2T)), one cycle every four samples, with cos(5π t / (2T)), five cycles every four samples. "
        "The samples coincide. Sampling also repeats the spectrum at intervals of fs. "
        "An analogue low-pass before the converter, the anti-aliasing filter, is there to suppress frequencies above fs/2.",
        "笔记比较 cos(π t / (2T))（每四个样本一个周期）和 cos(5π t / (2T))（每四个样本五个周期）。"
        "样本重合。采样还会把频谱以 fs 为间隔重复。"
        "转换器之前的模拟低通，就是抗混叠滤波器，用来压掉 fs/2 以上的频率。",
    )
    count = st.slider("Number of samples / 样本数", 8, 32, 12, key="sig_alias_n")
    k = np.arange(count)
    low, high = alias_pair(k)
    fine = np.linspace(0.0, count - 1, 400)
    _plot(
        fine,
        [("fd = 1/4 / 低频率", np.cos(math.pi * fine / 2.0)), ("fd = 5/4 / 高频率", np.cos(5.0 * math.pi * fine / 2.0))],
        "Two continuous cosines / 两条连续余弦",
        "t/T / 以采样间隔计的时间",
        "Amplitude / 幅度",
    )
    _plot(k, [("Shared samples / 相同的样本", low)], "Samples of either cosine / 两条余弦的样本", "k / 样本", "x[k]")
    st.caption(t(
        f"The largest difference between the two sampled sequences is {float(np.max(np.abs(low - high))):.2e}.",
        f"两条采样序列的最大差是 {float(np.max(np.abs(low - high))):.2e}。",
    ))

    experiment_heading("l2", SA_L2_EXPERIMENTS[1])
    say(
        "After the digital filter, a D/A converter holds each sample until the next one. The notes call this a step, or zero-order, interpolation. "
        "A following analogue low-pass is the recovery filter. "
        "The hold is convolution with a box, so the spectrum is multiplied by a sinc: high-frequency copies appear, and low frequencies are attenuated.",
        "数字滤波之后，D/A 把每个样本保持到下一个样本。笔记称之为阶梯插值，或零阶插值。"
        "后面的模拟低通是恢复滤波器。"
        "保持相当于与一个方波卷积，所以频谱乘上一个 sinc：出现高频副本，低频也被衰减。",
    )
    hold_n = st.slider("Samples to hold / 被保持的样本数", 6, 24, 10, key="sig_hold_n")
    held_samples = np.sin(2.0 * math.pi * np.arange(hold_n) / 7.0)
    repeats = 8
    staircase_t = np.arange(hold_n * repeats) / repeats
    staircase = np.repeat(held_samples, repeats)
    _plot(staircase_t, [("Zero-order hold / 零阶保持", staircase)], "Held samples / 保持后的样本", "t/T / 以采样间隔计的时间", "Amplitude / 幅度")

    experiment_heading("l2", SA_L2_EXPERIMENTS[2])
    say(
        "A least-squares parabola through x[−2] … x[2] has centre value s0 = (1/35)(−3 x[−2] + 12 x[−1] + 17 x[0] + 12 x[1] − 3 x[2]). "
        "Waiting until all five samples have arrived makes the filter causal and delays every frequency by 2T. "
        "The closed form is |G| = |17 + 24 cos ωT − 6 cos 2ωT| / 35, equal to 1 at ωT = 0 and about 0.707 near ωT = 0.48π.",
        "过 x[−2] … x[2] 的最小二乘抛物线，中心值是 s0 = (1/35)(−3 x[−2] + 12 x[−1] + 17 x[0] + 12 x[1] − 3 x[2])。"
        "等五个样本都到了再输出，滤波器就成为因果的，并且每个频率都延迟 2T。"
        "闭式是 |G| = |17 + 24 cos ωT − 6 cos 2ωT| / 35。ωT = 0 时它等于 1，在 ωT ≈ 0.48π 附近约为 0.707。",
    )
    st.latex(r"|G(e^{j\omega T})|=\frac{1}{35}\left|17+24\cos\omega T-6\cos 2\omega T\right|")
    tone = st.slider("Tone (cycles/sample) / 正弦频率（周期每样本）", 0.01, 0.45, 0.08, key="sig_par_f")
    noise_std = st.slider("Noise standard deviation / 噪声标准差", 0.0, 0.8, 0.25, key="sig_par_noise")
    length = 180
    generator = np.random.default_rng(int(st.session_state.get("sig_seed", 1)))
    clean = np.sin(2.0 * math.pi * tone * np.arange(length))
    noisy = clean + noise_std * generator.normal(size=length)
    filtered = apply_fir(noisy, PARABOLIC_TAPS)[4: length + 4]
    _plot(np.arange(length), [("Noisy / 含噪声", noisy), ("Parabola / 抛物线滤波", filtered)], "Five-point smoother / 五点平滑", "k / 样本", "Amplitude / 幅度")
    grid = np.linspace(0.0, math.pi, 400)
    _plot(grid, [("|G|", np.abs(parabolic_closed_form(grid)))], "Magnitude of the causal parabolic filter / 因果抛物线滤波器的幅度", "ωT (rad/sample) / ωT（弧度每样本）", "|G|")

    experiment_heading("l2", SA_L2_EXPERIMENTS[3])
    say(
        "A non-recursive filter y[k] = sum_i a_i x[k − i] has a finite impulse response: the coefficients are that response. "
        "A recursive filter also feeds back earlier outputs and generally has an infinite impulse response. "
        "An FIR can still be written with recursion; the notes give y[k] = y[k−1] + x[k] − 2x[k−1] + x[k−2] as a recursive filter whose impulse response is finite.",
        "非递归滤波器 y[k] = sum_i a_i x[k − i] 的冲激响应有限：那些系数就是冲激响应。"
        "递归滤波器还把更早的输出反馈回来，冲激响应一般是无限的。"
        "FIR 仍然可以写成递归形式。笔记举出 y[k] = y[k−1] + x[k] − 2x[k−1] + x[k−2]，它是递归的，但冲激响应有限。",
    )
    alpha = st.slider("α for y[k] = α y[k−1] + x[k−1] / 递归系数 α", 0.05, 0.95, 0.8, key="sig_iir_alpha")
    impulse = np.zeros(24)
    impulse[0] = 1.0
    fir_out = apply_fir(impulse, PARABOLIC_TAPS)[:24]
    iir_out = apply_iir(impulse, np.array([0.0, 1.0]), np.array([alpha]))[:24]
    _plot(np.arange(24), [("Parabolic FIR / 抛物线 FIR", fir_out), ("Recursive / 递归", iir_out)], "Impulse responses / 冲激响应", "k / 样本", "g[k]")

    experiment_heading("l2", SA_L2_EXPERIMENTS[4])
    say(
        "For a causal rational pulse transfer function, every pole must lie inside the unit circle. "
        "The notes take y[k] = x[k−1] + α y[k−1] with α > 0, find a pole at z = α, and conclude that stability needs 0 < α < 1. "
        "The step-function series F(z) = 1/(1 − z^{−1}) is stated only for |z^{−1}| < 1.",
        "因果的有理脉冲传递函数，每个极点都必须在单位圆内。"
        "笔记取 α > 0 的 y[k] = x[k−1] + α y[k−1]，得到极点 z = α，并得出稳定条件 0 < α < 1。"
        "阶跃的级数 F(z) = 1/(1 − z^{−1}) 只在 |z^{−1}| < 1 时成立。",
    )
    st.latex(r"G(z)=\frac{1}{z-\alpha}")
    pole = st.slider("α / 极点 α", 0.05, 1.2, 0.8, key="sig_pole")
    if causal_stable_alpha(pole):
        st.caption(t("0 < α < 1, so this causal filter is stable.", "0 < α < 1，所以这个因果滤波器是稳定的。"))
    else:
        st.warning(t(
            "α is not inside (0, 1). The notes do not treat the unit-circle values as the frequency response of a stable causal filter.",
            "α 不在 (0, 1) 内。笔记不把单位圆上的取值当作稳定因果滤波器的频率响应。",
        ))
    _plane([pole], [0.0], "Pole of 1/(z − α) / 1/(z − α) 的极点", circle=True)

    experiment_heading("l2", SA_L2_EXPERIMENTS[5])
    say(
        "The frequency response is the pulse transfer function on the unit circle, z = e^{jωT}. "
        "For G(z) = 1/(z − 0.8) the magnitude at a frequency is the reciprocal of the distance from the pole to that point on the unit circle. "
        "A pole near z = 1 therefore passes low frequencies.",
        "频率响应是单位圆上的脉冲传递函数，z = e^{jωT}。"
        "对 G(z) = 1/(z − 0.8)，某一频率上的幅度是极点到单位圆上对应点的距离的倒数。"
        "所以靠近 z = 1 的极点会放过低频。",
    )
    st.latex(r"|G(e^{j\omega T})|=\frac{1}{|e^{j\omega T}-0.8|}")
    grid = np.linspace(0.0, math.pi, 400)
    distance = np.abs(recursive_lowpass_response(0.8, grid))
    _plot(grid, [("|G|", distance)], "Low-pass from a pole at 0.8 / 极点在 0.8 的低通", "ωT (rad/sample) / ωT（弧度每样本）", "|G|")


def render_signal_lecture3() -> None:
    lecture_anchor("l3")
    st.title(t("Lecture 3 — Design of digital filters", "第 3 讲 — 数字滤波器设计"))
    say(
        "Design specifies a magnitude, approximates it with a causal discrete system, then realises that system in finite precision. "
        "Poles of a stable causal filter stay inside the unit circle. A zero on the unit circle sets the gain to zero at that frequency.",
        "设计先给定幅度，再用一个因果的离散系统去逼近，最后用有限精度实现。"
        "稳定因果滤波器的极点留在单位圆内。单位圆上的零点把该频率的增益设为 0。",
    )

    experiment_heading("l3", SA_L3_EXPERIMENTS[0])
    say(
        "The Hanning filter in the notes is y[k] = (1/4)(x[k] + 2 x[k−1] + x[k−2]). "
        "Its pulse transfer function (1/4)(1 + z^{−1})² has a double zero at z = −1, so it attenuates the Nyquist frequency.",
        "笔记里的汉宁滤波器是 y[k] = (1/4)(x[k] + 2 x[k−1] + x[k−2])。"
        "它的脉冲传递函数 (1/4)(1 + z^{−1})² 在 z = −1 有二重零点，所以它衰减奈奎斯特频率。",
    )
    st.latex(r"G(z)=\frac14(1+z^{-1})^2")
    grid = np.linspace(0.0, math.pi, 400)
    _plot(grid, [("|G|", np.abs(fir_frequency_response(HANNING_TAPS, grid)))], "Three-point Hanning magnitude / 三点汉宁的幅度", "ωT (rad/sample) / ωT（弧度每样本）", "|G|")
    _plane([-1.0], [0.0], "Double zero at z = −1 / z = −1 处的二重零点", circle=True)

    experiment_heading("l3", SA_L3_EXPERIMENTS[1])
    say(
        "If the phase is φ = −ω k0 T, a sinusoid is delayed by k0 samples and scaled, and the waveform of a sum of sinusoids is kept. "
        "Symmetric or antisymmetric FIR coefficients have this linear phase. IIR filters do not have it exactly. "
        "A jump of π in a plotted phase is a change of sign of a real amplitude factor. The causal parabolic filter has phase −2ωT apart from those jumps.",
        "若相位 φ = −ω k0 T，正弦被延迟 k0 个样本并乘上一个比例，若干正弦之和的波形得以保持。"
        "对称或反对称的 FIR 系数具有这种线性相位。IIR 不能严格做到。"
        "画出的相位若跳变 π，是实幅度因子变了号。因果抛物线滤波器的相位是 −2ωT，另加这些跳变。",
    )
    n = np.arange(220)
    tones = (
        0.6 * np.sin(2.0 * math.pi * 0.04 * n)
        + 0.3 * np.sin(2.0 * math.pi * 0.09 * n + 0.4)
        + 0.2 * np.sin(2.0 * math.pi * 0.16 * n)
    )
    linear = apply_fir(tones, PARABOLIC_TAPS)[4: n.size + 4]
    warped = apply_iir(tones, np.array([0.0, 1.0]), np.array([0.85]))
    _plot(
        n,
        [("Input / 输入", tones), ("Parabolic FIR / 抛物线 FIR", linear), ("Pole at 0.85 / 极点 0.85", warped)],
        "Three tones through a linear-phase FIR and a one-pole IIR / 三个正弦经过线性相位 FIR 和单极点 IIR",
        "k / 样本",
        "Amplitude / 幅度",
    )

    experiment_heading("l3", SA_L3_EXPERIMENTS[2])
    say(
        "The ideal cutoff ωc = π/(4T) is π/4 radians per sample. Sampling the sinc gives an infinite, non-causal coefficient sequence. "
        "The notes first truncate at |k| = 10 and shift by 10, which produces 21 coefficients and the ripples of the Gibbs phenomenon. "
        "They say the overshoot in their figure is about 9%. A Hamming (α = 0.54) or Hanning (α = 0.5) window, w[k] = α + (1 − α) cos(kπ/N), reduces the ripple and makes the cutoff less sharp. "
        "Kaiser is named and is not given a formula here.",
        "理想截止 ωc = π/(4T) 就是每样本 π/4 弧度。对 sinc 采样得到无穷长、非因果的系数。"
        "笔记先在 |k| = 10 处截断，再平移 10，得到 21 个系数，以及吉布斯现象的波纹。"
        "他们说自己那张图里的过冲大约是 9%。汉明窗（α = 0.54）或汉宁窗（α = 0.5），w[k] = α + (1 − α) cos(kπ/N)，能减小波纹，同时让截止变缓。"
        "凯泽窗只被点名，这里没有给出它的公式。",
    )
    st.latex(r"w[k]=\alpha+(1-\alpha)\cos\frac{k\pi}{N},\quad -N\le k\le N")
    half = int(st.slider("Truncation N / 截断 N", 4, 20, 10, key="sig_fir_n"))
    window = st.selectbox(
        "Window / 窗",
        ("rectangular", "hanning", "hamming"),
        format_func=lambda name: {"rectangular": "Rectangular / 矩形", "hanning": "Hanning α=0.5 / 汉宁", "hamming": "Hamming α=0.54 / 汉明"}[name],
        key="sig_fir_window",
    )
    alpha = {"rectangular": None, "hanning": 0.5, "hamming": 0.54}[window]
    taps = windowed_fir(half, math.pi / 4.0, alpha)
    grid = np.linspace(0.0, math.pi, 600)
    _plot(grid, [("|G|", np.abs(fir_frequency_response(taps, grid)))], "Truncated ideal low-pass / 截断后的理想低通", "ω (rad/sample) / ω（弧度每样本）", "|G|")
    st.caption(t(
        f"There are {taps.size} coefficients. The DC gain of this truncation is {float(np.sum(taps)):.4f}.",
        f"共有 {taps.size} 个系数。这一截断的直流增益是 {float(np.sum(taps)):.4f}。",
    ))

    experiment_heading("l3", SA_L3_EXPERIMENTS[3])
    say(
        "The bilinear map is s = (2/T) (1 − z^{−1}) / (1 + z^{−1}). It sends the whole imaginary axis to the unit circle and the open left half-plane to the inside of that circle. "
        "Frequencies are related by ωa = (2/T) tan(ωd/2). For T = 2 the notes' figure reduces this to ωa = tan(ωd/2), against the straight sampling line ωa = ωd/T. "
        "Cutoffs are pre-warped before the analogue design. Impulse invariance is named as an alternative that aliases; it is not carried out here.",
        "双线性变换是 s = (2/T) (1 − z^{−1}) / (1 + z^{−1})。它把整条虚轴映到单位圆，把开左半平面映到圆内。"
        "频率关系是 ωa = (2/T) tan(ωd/2)。T = 2 时，笔记的图把它写成 ωa = tan(ωd/2)，并和采样直线 ωa = ωd/T 对照。"
        "模拟设计之前要先把截止频率预弯折。冲激响应不变被当作会混叠的另一种方法点了名，这里不做。",
    )
    st.latex(r"\omega_a=\frac{2}{T}\tan\frac{\omega_d}{2}")
    period = st.slider("Sampling period T / 采样周期 T", 0.5, 4.0, 2.0, key="sig_bilinear_t")
    digital = np.linspace(0.0, 0.95 * math.pi, 300)
    warped = bilinear_analog_frequency(digital, period)
    straight = digital / period
    _plot(digital, [("Bilinear / 双线性", warped), ("ωd/T / 采样直线", straight)], "Frequency warping / 频率弯折", "ωd (rad/sample) / ωd（弧度每样本）", "ωa / 模拟角频率")

    experiment_heading("l3", SA_L3_EXPERIMENTS[4])
    say(
        "The worked example is a digital Butterworth low-pass: 3 dB at 2 kHz, at least 30 dB at 4.25 kHz, fs = 10 kHz. "
        "Pre-warping produces analogue edges near 14.53 kHz-rad/s and 83.3 krad/s. The order estimate is about 1.98, so n = 2. "
        "The printed difference equation is used below without re-deriving its decimals.",
        "笔记里的例子是数字巴特沃斯低通：2 kHz 处 3 dB，4.25 kHz 处至少 30 dB，fs = 10 kHz。"
        "预弯折后的模拟边界大约是 14.53 krad/s 和 83.3 krad/s。阶数估计约为 1.98，所以 n = 2。"
        "下面直接使用印出的差分方程，不再重推那些小数。",
    )
    st.latex(r"y[k]=0.20657\,x[k]+0.41314\,x[k-1]+0.20657\,x[k-2]+0.36953\,y[k-1]-0.19582\,y[k-2]")
    analog_pass, analog_stop = example_prewarp()
    estimate = example_order_estimate(analog_pass, analog_stop)
    st.caption(t(
        f"Pre-warped edges: {analog_pass:.0f} and {analog_stop:.0f} rad/s. The order estimate is {estimate:.3f}.",
        f"预弯折后的边界：{analog_pass:.0f} 和 {analog_stop:.0f} rad/s。阶数估计是 {estimate:.3f}。",
    ))
    hz = np.linspace(0.0, 0.49 * EXAMPLE_SAMPLE_RATE_HZ, 500)
    digital = 2.0 * math.pi * hz / EXAMPLE_SAMPLE_RATE_HZ
    gain_db = 20.0 * np.log10(np.maximum(np.abs(iir_frequency_response(EXAMPLE_FIR, EXAMPLE_RECURSION, digital)), 1e-8))
    _plot(hz, [("20 log10 |G| (dB)", gain_db)], "Example magnitude / 例子的幅度", "Frequency (Hz) / 频率（Hz）", "dB / 分贝")
    st.caption(t(
        f"Vertical reading points from the specification are {EXAMPLE_CUTOFF_HZ:.0f} Hz and {EXAMPLE_STOP_HZ:.0f} Hz. "
        "The printed numerator is proportional to (z+1)², so the gain is zero at fs/2 and the curve stops short of that point.",
        f"指标里用来读图的两个频率是 {EXAMPLE_CUTOFF_HZ:.0f} Hz 和 {EXAMPLE_STOP_HZ:.0f} Hz。"
        "印出的分子与 (z+1)² 成比例，所以 fs/2 处增益为零，曲线在那之前停下。",
    ))
    k = np.arange(250)
    mixed = np.sin(2.0 * math.pi * 500.0 * k / EXAMPLE_SAMPLE_RATE_HZ) + 0.7 * np.sin(
        2.0 * math.pi * 4000.0 * k / EXAMPLE_SAMPLE_RATE_HZ
    )
    filtered = apply_iir(mixed, EXAMPLE_FIR, EXAMPLE_RECURSION)
    _plot(k, [("Input / 输入", mixed), ("Filtered / 滤波后", filtered)], "500 Hz and 4 kHz through the printed equation / 500 Hz 与 4 kHz 经过印出的方程", "k / 样本", "Amplitude / 幅度")


def render_signal_lecture4() -> None:
    lecture_anchor("l4")
    st.title(t("Lecture 4 — Spectral estimation", "第 4 讲 — 谱估计"))
    say(
        "The periodogram is (1/N) |sum_n x[n] e^{−jω n T}|². The DFT samples that transform. "
        "An autoregressive model instead writes the sequence as a recursive filter driven by white noise, and the spectrum follows from the filter.",
        "周期图是 (1/N) |sum_n x[n] e^{−jω n T}|²。DFT 对这个变换采样。"
        "自回归模型则把序列写成白噪声驱动的递归滤波器，频谱由这个滤波器得到。",
    )

    experiment_heading("l4", SA_L4_EXPERIMENTS[0])
    say(
        "The notes sample f(t) = 5 + 2 cos(2π t − 90°) + 3 cos(4π t) at fs = 4 Hz and keep four samples: 8, 4, 8, 0. "
        "The DFT, with the 1/N factor on the inverse, is F = (20, −j4, 12, j4). "
        "Then F[0]/N = 5 is the DC term, (2/N)|F[1]| = 2 is the amplitude of the 1 Hz cosine, and F[2]/N = 3 is the 2 Hz term. "
        "For a real sequence, F[N − n] = F[n]*.",
        "笔记把 f(t) = 5 + 2 cos(2π t − 90°) + 3 cos(4π t) 以 fs = 4 Hz 采样，并保留四个样本：8、4、8、0。"
        "1/N 放在逆变换上时，DFT 是 F = (20, −j4, 12, j4)。"
        "于是 F[0]/N = 5 是直流，(2/N)|F[1]| = 2 是 1 Hz 余弦的幅度，F[2]/N = 3 是 2 Hz 项。"
        "实序列满足 F[N − n] = F[n]*。",
    )
    st.latex(r"F[n]=\sum_{k=0}^{N-1} f[k] e^{-j 2\pi n k/N}")
    spectrum = dft(DFT_EXAMPLE_SAMPLES)
    _plot(np.arange(4), [("|F[n]|", np.abs(spectrum))], "DFT magnitude of the four samples / 四个样本的 DFT 幅度", "n / 频点", "|F[n]|")
    k = np.arange(4)
    dc = np.full(4, float(np.real(spectrum[0]) / 4.0))
    one_hz = conjugate_pair_term(spectrum, 1, k)
    two_hz = np.real(spectrum[2]) / 4.0 * np.cos(math.pi * k)
    _plot(
        k,
        [("Samples / 样本", DFT_EXAMPLE_SAMPLES), ("DC + 1 Hz + 2 Hz / 三项之和", dc + one_hz + two_hz)],
        "Inverse contributions / 逆变换的各项",
        "k / 样本",
        "f[k]",
    )
    say(
        "The notes place four transforms side by side: the Fourier series and the Fourier transform for continuous time, and the discrete-time Fourier transform and the DFT for sequences. "
        "The DFT is what you get by treating the finite record as one period and sampling the DTFT at the harmonic frequencies 2π n/N.",
        "笔记把四种变换放在一起：连续时间的傅里叶级数和傅里叶变换，以及序列的离散时间傅里叶变换和 DFT。"
        "DFT 的做法是把有限记录当成一个周期，再在谐波频率 2π n/N 上对 DTFT 采样。",
    )

    experiment_heading("l4", SA_L4_EXPERIMENTS[1])
    say(
        "Sampling the z-transform at L > N equally spaced points on the unit circle reconstructs the original N samples followed by zeros. "
        "That is zero-padding. It changes the display of the spectrum and does not add information. "
        "If L < N, one period of the reconstruction is a sum of shifted copies, which the notes call time-domain aliasing, and the original sequence is not recovered.",
        "在单位圆上取 L > N 个等间隔样本，重建出原来的 N 个样本，后面接零。"
        "这就是补零。它改变频谱的显示，不增加信息。"
        "若 L < N，重建的一个周期是若干移位副本之和。笔记称之为时域混叠，原序列不能恢复。",
    )
    record = 16
    base = np.sin(2.0 * math.pi * 2.0 * np.arange(record) / record) + 0.5 * np.sin(2.0 * math.pi * 3.0 * np.arange(record) / record)
    length = int(st.slider("DFT length L / DFT 长度 L", 8, 64, 16, key="sig_pad_l"))
    if length >= record:
        spectrum = dft(np.concatenate([base, np.zeros(length - record)]))
        recovered = np.real(idft(spectrum))[:record]
    else:
        k = np.arange(record)
        n = np.arange(length).reshape(-1, 1)
        spectrum = np.exp(-2j * math.pi * n * k / length) @ base
        recovered = np.real(idft(spectrum))
    frequencies = np.arange(length) / length
    _plot(frequencies, [("(1/L)|F|²", np.abs(spectrum) ** 2 / length)], "Samples of the spectrum / 频谱样本", "cycles/sample / 周期每样本", "Periodogram bin / 周期图频点")
    if length >= record:
        st.caption(t(
            f"The largest error in the recovered {record} samples is {float(np.max(np.abs(recovered - base))):.2e}.",
            f"恢复出的 {record} 个样本，最大误差是 {float(np.max(np.abs(recovered - base))):.2e}。",
        ))
    else:
        st.caption(t(
            f"L < N, so these {length} reconstructed samples are not the start of the original record.",
            f"L < N，所以这 {length} 个重建样本不是原记录的开头。",
        ))

    experiment_heading("l4", SA_L4_EXPERIMENTS[2])
    say(
        "A DFT over a non-integer number of cycles treats the record as one period of a waveform with jumps at the joins. "
        "Energy then appears in every bin. This is spectral leakage. "
        "The notes reduce it with the Hamming or Hanning windows from the FIR lecture, which taper the ends of the record.",
        "若 DFT 的长度里不是整数个周期，它把这段记录当成一个在衔接处有跳变的周期波形。"
        "能量于是出现在每个频点上。这就是频谱泄漏。"
        "笔记用 FIR 那一讲的汉明窗或汉宁窗来减小它。这些窗把记录的两端收下去。",
    )
    cycles = st.slider("Cycles in the record / 记录内的周期数", 2.0, 8.0, 4.5, 0.1, key="sig_leak_cycles")
    window = st.selectbox(
        "Leakage window / 泄漏用的窗",
        ("rectangular", "hanning", "hamming"),
        format_func=lambda name: {"rectangular": "Rectangular / 矩形", "hanning": "Hanning α=0.5 / 汉宁", "hamming": "Hamming α=0.54 / 汉明"}[name],
        key="sig_leak_window",
    )
    length = 64
    tone = np.cos(2.0 * math.pi * cycles * np.arange(length) / length)
    if window != "rectangular":
        tone = tone * endpoint_window(length, 0.5 if window == "hanning" else 0.54)
    magnitude = np.abs(dft(tone))[: length // 2]
    _plot(np.arange(magnitude.size), [("|F[n]|", magnitude)], "DFT of one cosine / 一条余弦的 DFT", "n / 频点", "|F[n]|")

    experiment_heading("l4", SA_L4_EXPERIMENTS[3])
    say(
        "An AR(p) model is x_t = sum_{k=1}^{p} a_k x_{t−k} + e_t, with e_t white and zero mean. "
        "In matrix form, x = M a + e, and the least-squares estimate is â = (Mᵀ M)⁻¹ Mᵀ x. "
        "The Yule–Walker form is r = R a. The notes name the Levinson–Durbin recursion for the Toeplitz matrix; this page solves the linear system directly. "
        "The four default coefficients are the ones reported for their AR(4) figure. This page synthesises a series from them. It does not reproduce the samples in that figure.",
        "AR(p) 模型是 x_t = sum_{k=1}^{p} a_k x_{t−k} + e_t，e_t 是零均值白噪声。"
        "写成矩阵是 x = M a + e，最小二乘估计是 â = (Mᵀ M)⁻¹ Mᵀ x。"
        "Yule–Walker 形式是 r = R a。笔记为那个 Toeplitz 矩阵点了 Levinson–Durbin 递推的名；这一页直接解线性方程组。"
        "四个默认系数是他们 AR(4) 图里报告的系数。这一页用它们合成一条序列，并不复现那张图里的样本。",
    )
    coeffs = []
    columns = st.columns(4)
    for index, default in enumerate(REPORTED_AR4):
        with columns[index]:
            coeffs.append(st.number_input(f"a{index + 1}", value=float(default), key=f"sig_ar_a{index}", format="%.4f"))
    noise_std = st.slider("Driving-noise standard deviation / 驱动噪声的标准差", 0.01, 0.6, 0.28, key="sig_ar_std")
    count = int(st.slider("Length N / 长度 N", 80, 400, 200, key="sig_ar_n"))
    generator = np.random.default_rng(int(st.session_state.get("sig_seed", 1)))
    series = synthesise_ar(np.array(coeffs), noise_std * generator.normal(size=count))
    if not np.isfinite(series).all():
        st.warning(t(
            "This coefficient choice makes the recurrence overflow. The trace below is not a usable AR series.",
            "这组系数让递推溢出了。下面的曲线不是一条可用的 AR 序列。",
        ))
        series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
    estimated, residual, variance = least_squares_ar(series, 4)
    try:
        walker = yule_walker_ar(series, 4)
        walker_text = ", ".join(f"{c:.3f}" for c in walker)
    except np.linalg.LinAlgError:
        walker_text = plain("not available for this series", "这条序列上无解")
    prediction = np.full(count, np.nan)
    prediction[4:] = series[4:] - residual
    _plot(
        np.arange(count),
        [("Series / 序列", series), ("Least-squares prediction / 最小二乘预测", prediction)],
        "One-step AR prediction / AR 一步预测",
        "t / 时间",
        "x",
    )
    signal_var = float(np.var(series))
    st.caption(t(
        f"Least squares: {', '.join(f'{c:.3f}' for c in estimated)}. Yule–Walker: {walker_text}. "
        f"Residual mean square {variance:.4f}. Series variance {signal_var:.4f}.",
        f"最小二乘：{', '.join(f'{c:.3f}' for c in estimated)}。Yule–Walker：{', '.join(f'{c:.3f}' for c in walker)}。"
        f"残差均方 {variance:.4f}。序列方差 {signal_var:.4f}。",
    ))

    experiment_heading("l4", SA_L4_EXPERIMENTS[4])
    say(
        "For the AR model the notes give Pxx(ω) = σe² / |1 − sum_k a_k e^{−j k ω T}|². "
        "An AR(p) spectrum can show p/2 peaks, and it can be drawn at any frequency, not only at the DFT bins. "
        "Their comparison figure uses Welch's method. Welch is not implemented here. The left-hand curve is the periodogram (1/N)|F[n]|².",
        "对 AR 模型，笔记给出 Pxx(ω) = σe² / |1 − sum_k a_k e^{−j k ω T}|²。"
        "AR(p) 的谱最多看出 p/2 个峰，而且可以在任意频率上画，不限于 DFT 频点。"
        "他们的对照图用的是 Welch 方法。这里没有实现 Welch。左边的曲线是周期图 (1/N)|F[n]|²。",
    )
    st.latex(r"P_{xx}(\omega)=\frac{\sigma_e^2}{\left|1-\sum_{k=1}^{p} a_k e^{-jk\omega T}\right|^2}")
    first = st.slider("First tone (cycles/sample) / 第一个正弦（周期每样本）", 0.02, 0.4, 0.10, key="sig_psd_f1")
    second = st.slider("Second tone (cycles/sample) / 第二个正弦（周期每样本）", 0.02, 0.4, 0.14, key="sig_psd_f2")
    order = int(st.slider("AR order p / AR 阶数 p", 2, 12, 4, key="sig_psd_p"))
    psd_n = 256
    generator = np.random.default_rng(int(st.session_state.get("sig_seed", 1)) + 1)
    k = np.arange(psd_n)
    tones = np.sin(2.0 * math.pi * first * k) + np.sin(2.0 * math.pi * second * k)
    observed = tones + 0.4 * generator.normal(size=psd_n)
    bins = periodogram_bins(observed)
    coefficients, _, noise_var = least_squares_ar(observed - np.mean(observed), order)
    omega = np.linspace(0.0, math.pi, 400)
    smooth = ar_psd(coefficients, noise_var, omega)
    _plot(np.arange(psd_n // 2) / psd_n, [("Periodogram / 周期图", bins[: psd_n // 2])], "Periodogram / 周期图", "cycles/sample / 周期每样本", "Pxx")
    _plot(omega / (2.0 * math.pi), [("AR spectrum / AR 谱", smooth)], "Autoregressive spectrum / 自回归谱", "cycles/sample / 周期每样本", "Pxx")


def render_signal_practice() -> None:
    st.markdown("<div id='practice'></div>", unsafe_allow_html=True)
    st.title(t("Practice", "练习"))
    st.warning(t(
        "Original checks of formulas printed in the notes. They are not an official example sheet.",
        "这些是按笔记里印出的公式写的原创核对，不是官方习题纸。",
    ))
    for problem in PROBLEMS:
        with st.container(border=True):
            st.markdown(f"**{t(problem.topic_en, problem.topic_zh)}**")
            say(problem.prompt_en, problem.prompt_zh)
            submitted = st.number_input("Your answer / 你的答案", value=0.0, format="%.6f", key=f"sig-answer-{problem.id}")
            if st.button("Check / 检查", key=f"sig-check-{problem.id}"):
                st.session_state[f"sig-mark-{problem.id}"] = grade_numeric(problem.id, submitted)
            mark = st.session_state.get(f"sig-mark-{problem.id}")
            if mark is True:
                st.success(t("Within tolerance.", "在容差以内。"))
            elif mark is False:
                st.error(t("Outside tolerance.", "超出容差。"))
            with st.expander(t("Worked solution", "题解")):
                say(problem.solution_en, problem.solution_zh)
                st.write(t(f"Answer: {problem.answer:.6g}", f"答案：{problem.answer:.6g}"))
            st.markdown(f"[{t('Go to this experiment', '滚到这个实验')}](#{problem.anchor})")

    st.header(t("Quiz", "小测"))
    for question in QUIZ:
        labels = [plain(en, zh) for en, zh in zip(question.choices_en, question.choices_zh)]
        st.radio(t(question.prompt_en, question.prompt_zh), labels, key=f"sig-quiz-{question.id}")
    if st.button(t("Score the quiz", "给小测评分"), key="sig-quiz-score"):
        correct = 0
        details = []
        for question in QUIZ:
            labels = [plain(en, zh) for en, zh in zip(question.choices_en, question.choices_zh)]
            selected = st.session_state[f"sig-quiz-{question.id}"]
            is_correct = selected == labels[question.correct]
            correct += int(is_correct)
            details.append((question, is_correct))
        st.session_state["sig_quiz_details"] = details
        st.session_state["sig_quiz_score"] = (correct, len(QUIZ))
    if "sig_quiz_score" in st.session_state:
        correct, total = st.session_state["sig_quiz_score"]
        st.write(t(f"Score: {correct} / {total}", f"得分：{correct} / {total}"))


def signal_groups():
    return (
        ("home", "Overview", "总览", ()),
        ("l1", "Lecture 1", "第 1 讲", SA_L1_EXPERIMENTS),
        ("l2", "Lecture 2", "第 2 讲", SA_L2_EXPERIMENTS),
        ("l3", "Lecture 3", "第 3 讲", SA_L3_EXPERIMENTS),
        ("l4", "Lecture 4", "第 4 讲", SA_L4_EXPERIMENTS),
        ("practice", "Practice", "练习", ()),
    )


def _reset_signal() -> None:
    for key in list(st.session_state.keys()):
        if key in {"sig_seed", "sig_reset"} or key.startswith("sig_seed@@") or key.startswith("sig_reset@@"):
            continue
        if key.startswith("sig_") or key.startswith("sig-"):
            del st.session_state[key]
    st.rerun()

def _sidebar_signal(route: Route) -> None:
    paper = paper_of(route)
    module = module_of(route)
    with st.sidebar:
        language_switch()
        st.header(t("B14 · Signal Analysis", "B14 · 信号分析"))
        st.caption(t("B14 Signal Analysis", "B14 信号分析"))
        _leave(f"← {module.title_en} / {module.title_zh}", topic_route(paper, module), "back-signal")
        _contents(signal_groups())
        st.divider()
        st.number_input("Random seed / 随机种子", min_value=0, max_value=100000, step=1, key="sig_seed")
        if st.button("Reset signal values / 恢复信号实验参数", key="sig_reset"):
            _reset_signal()

def render_signal_lab(route: Route) -> None:
    _scroll_to_hash()
    _sidebar_signal(route)
    render_signal_home()
    render_signal_lecture1()
    render_signal_lecture2()
    render_signal_lecture3()
    render_signal_lecture4()
    render_signal_practice()
