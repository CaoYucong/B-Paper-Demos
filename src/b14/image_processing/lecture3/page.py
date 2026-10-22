"""Lecture 3. The original MAP, Wiener, and operator demonstrations are called unchanged."""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.b14.image_processing.lecture3.degradation import degrade
from src.b14.image_processing.lecture3.motion import apply_psf, directional_motion_psf
from src.b14.image_processing.lecture3.multiframe_page import (
    render_benefits,
    render_blind,
    render_examples,
    render_huber,
    render_map_sr,
    render_measure,
    render_samples,
    render_stacked,
)
from src.b14.image_processing.lecture3.panorama import load_example_frames, load_placement_steps, load_registered_panorama
from src.b14.image_processing.lecture3.super_resolution import four_phase_observations, fuse_integer_shifts
from src.b14.image_processing.lecture3.wiener_gain import frequency_bin, scalar_gains
from src.b14.image_processing.metrics import format_db, mse, psnr
from src.b14.image_processing.lecture3.operators import psf_to_otf
from src.ui.components import concept, controls_heading, experiment_heading, lecture_anchor, line_figure, model, show_array, show_row, steps, takeaways, try_this, visual_heading
from src.b14.image_processing.experiments import L3_EXPERIMENTS
from src.ui.i18n import shown_label, t
from src.b14.image_processing.lecture3.bayes_page import render_bayes
from src.b14.image_processing.lecture3.compare_page import render_compare
from src.b14.image_processing.lecture3.forward_page import render_forward
from src.b14.image_processing.lecture3.map_page import render_map
from src.b14.image_processing.lecture3.operator_page import render_operator
from src.b14.image_processing.lecture3.wiener import inverse_filter, wiener_filter


def render_lecture3(image: np.ndarray, label: str) -> None:
    lecture_anchor("l3")
    st.title(t("Lecture 3 — Image Restoration", "第 3 讲 — 图像复原"))
    st.caption(shown_label(label))
    say_note = (
        "Blur, noise, and λ below are shared by the Gaussian experiments on this lecture. "
        "Motion blur and super-resolution have their own noise controls. "
        "Gradient descent and conjugate gradients still run only when their buttons are pressed."
    )
    st.markdown(say_note)
    st.caption("下面的模糊、噪声和 λ 供本讲的高斯实验共用。运动模糊和超分辨率有自己的噪声。梯度下降和共轭梯度仍然只在按下按钮后运行。")
    observation, lam = _observation(image, label)
    for item in L3_EXPERIMENTS:
        experiment_heading("l3", item)
        if item[0] == "forward":
            render_forward(observation, lam)
        elif item[0] == "operator":
            render_operator(observation, lam)
        elif item[0] == "inverse":
            _inverse(observation)
        elif item[0] == "wiener":
            _wiener(observation)
        elif item[0] == "motion":
            _motion(image, label)
        elif item[0] == "bayes":
            render_bayes()
        elif item[0] == "map":
            render_map(observation, lam)
        elif item[0] == "panorama":
            _panorama()
        elif item[0] == "samples":
            render_samples()
        elif item[0] == "benefits":
            render_benefits()
        elif item[0] == "measure":
            render_measure()
        elif item[0] == "stacked":
            render_stacked()
        elif item[0] == "mapsr":
            render_map_sr()
        elif item[0] == "huber":
            render_huber()
        elif item[0] == "compare":
            render_compare(observation, lam)
        elif item[0] == "examples":
            render_examples()
        elif item[0] == "blind":
            render_blind()
        elif item[0] == "superres":
            _superres(image)


def _observation(image: np.ndarray, label: str):
    controls_heading()
    left, middle, right = st.columns(3)
    with left:
        sigma = float(st.slider("Gaussian blur σ / 高斯模糊", 0.5, 5.0, step=0.1, key="sigma"))
    with middle:
        noise = float(st.slider("Noise σ / 噪声标准差", 0.0, 0.2, step=0.005, key="noise_std"))
    with right:
        st.slider("log₁₀(λ)", -5.0, 1.0, step=0.1, key="log10_lambda")
    lam = float(10.0 ** float(st.session_state["log10_lambda"]))
    st.caption(f"λ = {lam:.4g}. The original image is not passed to a solver. / 原图不会传给求解器。")
    observation = degrade(image, sigma=sigma, noise_std=noise, seed=int(st.session_state["seed"]), label=label)
    return observation, lam


def _inverse(observation) -> None:
    concept(
        "Dividing G by H undoes a convolution where H is not small. Where the blur removed a frequency, the same division amplifies noise.",
        "在 H 不太小的地方，G 除以 H 可以撤销卷积。模糊已经抹掉某个频率时，同一次除法会放大噪声。",
    )
    model(r"\hat F = G/H \quad \mathrm{when}\ |H|\ge \tau,\quad \mathrm{else}\ 0")
    threshold = float(10.0 ** st.slider("log₁₀(threshold τ) / 阈值", -6.0, -1.0, -3.0, 0.1, key="inverse_page_threshold"))
    restored, suppressed = inverse_filter(observation.observed, observation.otf, threshold)
    magnitude = np.abs(observation.otf)
    inverse_gain = np.zeros_like(magnitude)
    usable = magnitude >= threshold
    inverse_gain[usable] = 1.0 / magnitude[usable]
    visual_heading()
    show_row(
        [
            ("Original, evaluation only / 原图，只用于评价", observation.original, "image"),
            ("Observed g / 观测", observation.observed, "image"),
            ("Inverse filtered / 逆滤波", restored, "image"),
        ]
    )
    show_row(
        [
            ("|H|, shifted / 模糊传递函数", np.fft.fftshift(np.log10(1.0 + magnitude)), "otf"),
            ("Inverse gain, shifted; zeros are the safeguard / 逆增益", np.fft.fftshift(np.log10(1.0 + inverse_gain)), "otf"),
            ("Residual g prediction is not shown; this is f error / 与原图的差", restored - observation.original, "residual"),
        ]
    )
    st.write(
        f"Suppressed bins {suppressed} of {magnitude.size}. "
        f"MSE {mse(restored, observation.original):.4g}. PSNR {format_db(psnr(restored, observation.original))}."
    )
    st.caption(f"被抑制的频率箱有 {suppressed} 个，共 {magnitude.size} 个。均方误差和 PSNR 用原图计算，求解时没有使用原图。")
    st.caption("The gain picture uses log10(1 + gain) so that large values remain visible. Bins below τ are drawn as zero because the safeguard sets them to zero. / 增益图使用 log10(1+增益)。低于 τ 的频率箱被保护措施置零，所以画成零。")
    steps(
        [
            ("Blur multiplies each DFT coefficient by H.", "模糊把每个 DFT 系数乘上 H。"),
            ("Inversion divides by that same H.", "逆滤波再除以同一个 H。"),
            ("A tiny H makes the quotient huge. The threshold replaces those quotients by zero. That is a hard cutoff, not a Wiener roll-off.", "很小的 H 会让商变得极大。阈值把这些商换成零。这是硬切断，不是维纳滤波的平滑衰减。"),
        ]
    )
    takeaways([("Lower the noise to zero and the threshold can be tiny. Raise the noise and the same threshold is not enough.", "把噪声降到零时，阈值可以很小。升高噪声后，同一个阈值就不够了。")])
    try_this("Set the noise to 0 and the threshold to 1e-6. Then raise the noise to 0.1 without changing τ. / 先把噪声设为 0、阈值设为 1e-6。再把噪声升到 0.1，不要改 τ。", "The residual is the error against the original, which the filter did not see. / 残差是相对原图的误差，滤波器并没有看到原图。")


def _wiener(observation) -> None:
    concept(
        "The Wiener weight H* / (|H|² + K) agrees with 1/H when |H| is large, and it falls toward zero when |H| is small. Constant K is the ratio of noise power to signal power in the white-prior derivation.",
        "维纳权重 H* / (|H|² + K) 在 |H| 大时接近 1/H，在 |H| 小时趋向零。在白先验的推导里，常数 K 是噪声功率与信号功率之比。",
    )
    model(r"W=\frac{H^*}{|H|^2+K}")
    log_k = float(st.slider("log₁₀(K)", -6.0, 0.0, -2.0, 0.1, key="wiener_page_k"))
    gain_k = float(10.0 ** log_k)
    restored = wiener_filter(observation.observed, observation.otf, gain_k)
    magnitude = np.abs(observation.otf)
    weight = np.conj(observation.otf) / (magnitude**2 + gain_k)
    visual_heading()
    show_row(
        [
            ("|H| / 传递函数幅度", np.fft.fftshift(magnitude), "otf"),
            ("Phase of H / H 的相位", np.fft.fftshift(np.angle(observation.otf)), "residual"),
            ("|W| / 维纳增益", np.fft.fftshift(np.abs(weight)), "otf"),
        ]
    )
    show_row(
        [
            ("Observed / 观测", observation.observed, "image"),
            ("Wiener / 维纳", restored, "image"),
            ("Error vs original / 相对原图的误差", restored - observation.original, "residual"),
        ]
    )
    axis = np.linspace(0.0, 1.0, 300)
    wiener_curve, inverse_curve = scalar_gains(axis, gain_k)
    inverse_curve = np.where(np.isfinite(inverse_curve), inverse_curve, np.nan)
    from src.ui.common import show_fig

    show_fig(
        line_figure(
            axis,
            [("Wiener |W|", wiener_curve), ("Inverse 1/|H|", inverse_curve)],
            "Single-frequency gain / 单频率增益",
            "|H|",
            "gain / 增益",
        )
    )
    u = float(st.slider("Inspect u / 查看的水平频率", 0.0, 0.5, 0.1, 0.01))
    v = float(st.slider("Inspect v / 查看的垂直频率", -0.5, 0.5, 0.0, 0.01))
    row, col = frequency_bin(observation.otf.shape, u, v)
    h_value = observation.otf[row, col]
    st.write(
        f"Bin ({row}, {col}): H = {h_value.real:.4g}{h_value.imag:+.4g}j, "
        f"|W| = {abs(weight[row, col]):.4g}. MSE {mse(restored, observation.original):.4g}. "
        f"PSNR {format_db(psnr(restored, observation.original))}."
    )
    st.caption("H* is the complex conjugate. For this even Gaussian it is just H, because H is real. A motion blur in the next experiment makes H complex. / H* 是复共轭。这里的偶高斯使 H 为实数，所以 H* 就是 H。下一个实验的运动模糊会让 H 变成复数。")
    steps(
        [
            ("K = 0 reproduces inverse filtering wherever H is nonzero.", "在 H 不为零的地方，K = 0 就是逆滤波。"),
            ("A larger K trusts the prior more and leaves more blur in the result.", "更大的 K 更信任先验，结果里会留下更多模糊。"),
            ("Constant-K Wiener matches quadratic MAP with a penalty on f itself. The gradient MAP penalty λ||Df||² is frequency dependent, so the two filters are not the same.", "常数 K 的维纳滤波对应惩罚 f 本身的二次 MAP。梯度 MAP 的惩罚 λ||Df||² 随频率变化，所以两种滤波器并不相同。"),
        ]
    )
    takeaways([("The curve shows why a small H is safe: the Wiener gain stays below the inverse gain.", "这条曲线说明小的 H 为什么是安全的：维纳增益低于逆增益。")])
    try_this("Set K very small and compare the picture with the inverse-filtering experiment. / 把 K 设得很小，再和逆滤波实验的图像比较。", "Move the inspected frequency outward and watch |W| fall. / 把查看的频率向外移，观察 |W| 下降。")


def _motion(image: np.ndarray, label: str) -> None:
    concept(
        "Motion blur is still a convolution, so it still has a transfer function. The kernel is no longer even, and H is complex.",
        "运动模糊仍然是卷积，所以仍然有传递函数。这个核不再是偶函数，H 是复数。",
    )
    model(r"g = h_{\theta}*f + n")
    controls_heading()
    length = int(st.slider("Motion length, pixels / 运动长度", 1, 21, 9, 1))
    angle = float(st.slider("Angle, degrees / 角度", 0.0, 180.0, 20.0, 1.0))
    noise = float(st.slider("Noise σ / 噪声", 0.0, 0.2, 0.02, 0.005, key="motion-noise"))
    log_k = float(st.slider("log₁₀(K)", -6.0, 0.0, -2.0, 0.1, key="motion-k"))
    psf = directional_motion_psf(image.shape, length, angle)
    blurred, _noise, observed = apply_psf(image, psf, noise, int(st.session_state["seed"]))
    otf = psf_to_otf(psf)
    restored = wiener_filter(observed, otf, 10.0 ** log_k)
    visual_heading()
    show_row(
        [
            ("PSF / 点扩散函数", psf, "kernel"),
            ("|H| / 传递函数幅度", np.fft.fftshift(np.abs(otf)), "otf"),
            ("Phase of H / 相位", np.fft.fftshift(np.angle(otf)), "residual"),
        ]
    )
    show_row(
        [
            ("Observed / 观测", observed, "image"),
            ("Wiener / 维纳", restored, "image"),
            ("Blurred, before noise / 加噪声前的模糊", blurred, "image"),
        ]
    )
    st.write(f"PSNR against the original {format_db(psnr(restored, image))}. Label {label}.")
    st.caption("PSNR 相对原图计算。角度为 0 时，这个核与原来的水平运动模糊相同。")
    steps(
        [
            ("The line of pixels is normalised so that the PSF sums to one.", "这条像素线被归一化，使 PSF 的和为 1。"),
            ("The same FFT convention as the Gaussian blur turns the PSF into H.", "与高斯模糊相同的 FFT 约定把 PSF 变成 H。"),
            ("Wiener filtering uses H*, which is no longer equal to H.", "维纳滤波使用 H*，此时 H* 不再等于 H。"),
        ]
    )
    takeaways([("Changing the angle changes which frequencies the blur destroys.", "改变角度会改变模糊毁掉的是哪些频率。")])
    try_this("Set the angle to 0° and then 90°. The |H| stripes rotate. / 把角度设为 0° 再设为 90°。|H| 的条纹会旋转。", "This page does not run gradient descent. The MAP solver can use the same H because the adjoint code already accepts a complex OTF. / 这一页不运行梯度下降。MAP 求解器可以使用同一个 H，因为伴随代码已经接受复传递函数。")


def _panorama() -> None:
    concept(
        "The lecture's panorama example uses four photographs of one scene, each taken from a different place. Registration warps them onto one plane. The samples are then not on a common grid, and that offset can be used to beat the Nyquist limit. A single horizontal shift does not do this: the walls stay double-exposed. The wide picture is the registered result. The blank corners were covered by no frame. This page does not recompute the non-examinable sketch solution.",
        "讲义的全景例子是同一场景的四张照片，拍摄位置各不相同。配准把它们变到同一个平面上。配准之后样本不在同一网格上，这个错位可以用来突破奈奎斯特极限。只做一次水平平移做不到这一点，墙会重影。下面的宽图是配准后的结果。空白的角没有任何一幅覆盖。这一页不重算那份非考试的草图解。",
    )
    frames = load_example_frames()
    registered = load_registered_panorama()
    st.caption(
        t(
            "This section does not use the workspace image.",
            "这一节不使用工作区图像。",
        )
    )
    visual_heading()
    show_row([(f"Frame {index + 1} / 第 {index + 1} 幅", frame, "image") for index, frame in enumerate(frames)])
    st.caption(
        t(
            "Each photograph is warped onto the registered plane and then added. A new edge is that frame's border after the warp. Where two frames cover the same place, this illustration averages them.",
            "每张照片先变到配准后的平面上，再加进去。新出现的边是这幅图变形后的边界。两幅盖住同一处时，这里取平均。",
        )
    )
    placed = load_placement_steps()
    show_array(placed[0], "1. Warp frame 1 onto the plane / 1. 把第 1 幅变到平面上", "image", wide=True)
    for index in range(1, 4):
        show_array(
            placed[index],
            f"{index + 1}. Add frame {index + 1} / {index + 1}. 加上第 {index + 1} 幅",
            "image",
            wide=True,
        )
    show_array(registered, "After registration / 配准之后", "image", wide=True)
    steps(
        [
            ("Each frame is one photograph of the courtyard, from its own position.", "每一幅都是从各自位置拍下的庭院。"),
            ("Shifting the frames by one horizontal offset leaves the chapel and the walls on top of each other.", "用同一个水平位移去挪，礼拜堂和墙会叠在一起。"),
            ("Registration warps frame 1 onto the plane, then adds frame 2, frame 3, and frame 4. The same wall continues across the join.", "配准先把第 1 幅变到平面上，再依次加上第 2、第 3、第 4 幅。同一面墙从接缝处延续过去。"),
            ("The slanted borders are the frame edges after that warp. The white corners are plane that no frame covers.", "斜边是变形之后的画面边缘。白角是没有任何一幅覆盖到的平面。"),
        ]
    )
    takeaways(
        [
            ("The panorama is the four views after they have been warped onto one plane.", "全景是四幅视图变到同一平面之后的结果。"),
            ("The registered samples do not fill one upright rectangle, so their grids are not the same.", "配准后的样本填不满一个正矩形，说明它们的网格并不相同。"),
        ]
    )
    try_this(
        "Find the chapel in two of the frames, then find it once in the registered panorama.",
        "在两幅图里找出礼拜堂，再在配准后的全景里看它只出现一次。",
    )


def _superres(image: np.ndarray) -> None:
    concept(
        "Supplementary. The lecture introduces several shifted observations and marks the full sketch solution as not examinable. This page only interleaves four known pixel phases:",
        "补充内容。讲义介绍了多幅位移观测，并把完整草图解标为非考试内容。这一页只把四个已知的像素相位交错拼回去：",
    )
    model(r"g_{a,b}[p,q]=f[2p+a,2q+b]+n_{a,b}[p,q]")
    controls_heading()
    noise = float(st.slider("Noise σ on each low-resolution image / 每幅低分辨率图的噪声", 0.0, 0.1, 0.0, 0.005))
    if image.shape[0] % 2 or image.shape[1] % 2:
        st.error("The workspace image must have even side lengths.")
        return
    observations = four_phase_observations(image, noise, int(st.session_state["seed"]))
    fused = fuse_integer_shifts(observations, image.shape)
    visual_heading()
    show_row(
        [
            ("Phase (0, 0)", observations[(0, 0)], "image"),
            ("Phase (0, 1)", observations[(0, 1)], "image"),
            ("Phase (1, 0)", observations[(1, 0)], "image"),
        ]
    )
    show_row(
        [
            ("Phase (1, 1)", observations[(1, 1)], "image"),
            ("Fused / 拼合", fused, "image"),
            ("Original, evaluation only / 原图，只用于评价", image, "image"),
        ]
    )
    st.write(f"MSE {mse(fused, image):.4g}. With zero noise the four phases tile the original exactly.")
    st.caption(f"均方误差 {mse(fused, image):.4g}。噪声为零时，四个相位恰好铺满原图。")
    steps(
        [
            ("Each observation keeps every other row and every other column, at one of the four offsets.", "每个观测按四种偏移之一，隔行隔列地保留像素。"),
            ("The samples do not overlap, so placing each one back is the least-squares answer for independent Gaussian noise.", "这些样本互不重叠，所以在独立高斯噪声下，把它们放回去就是最小二乘解。"),
            ("A real camera shift is not this integer lattice. That harder problem is the non-examinable sketch.", "真实的相机位移不是这个整数网格。更难的问题才是讲义里的非考试草图。"),
        ]
    )
    takeaways([("Several incomplete observations can contain complementary samples of the same high-resolution grid.", "多幅不完整的观测可以含有同一高分辨率网格上互补的样本。")])
    try_this(
        "Raise the noise from zero. The fused error equals the noise because no denoising is applied. The sections above are a small periodic model of the sketch. This page only interleaves four phases.",
        "把噪声从零升高。拼合误差等于噪声，因为这里没有做去噪。上面几节用一个很小的周期模型说明草图解。这一页只交错四个相位。",
    )
