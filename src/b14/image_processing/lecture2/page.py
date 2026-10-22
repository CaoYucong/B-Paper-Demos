"""Lecture 2 experiments. Spectra are displayed fftshifted; calculations are not."""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.b14.image_processing.lecture2.aliasing import checkerboard, dominant_frequency, downsample, moire, stripe_image
from src.b14.image_processing.lecture2.convolution_theorem import circular_convolve_same, linear_convolve_direct, linear_convolve_fft
from src.b14.image_processing.lecture2.fourier import (
    conjugate_symmetry_error,
    linearity_error,
    magnitude_only,
    phase_only,
    reconstruct_pairs,
    swap_magnitude_phase,
    translation_phase_error,
)
from src.b14.image_processing.lecture2.frequency_filters import apply_frequency_mask, frequency_mask, hybrid_image
from src.b14.image_processing.lecture2.spatial_frequency import log_magnitude, shifted_magnitude_phase, sinusoid
from src.b14.image_processing.metrics import mse
from src.b14.image_processing.synthetic_images import geometric_shapes, repeated_stripes
from src.ui.components import concept, controls_heading, experiment_heading, lecture_anchor, model, save_workspace, show_array, show_row, steps, takeaways, try_this, visual_heading
from src.b14.image_processing.experiments import L2_EXPERIMENTS
from src.ui.i18n import shown_label, t


def render_lecture2(image: np.ndarray, label: str) -> None:
    lecture_anchor("l2")
    st.title(t("Lecture 2 — Fourier Analysis", "第 2 讲 — 傅里叶分析"))
    st.caption(shown_label(label))
    st.caption(
        "DFT is numpy.fft.fft2, unnormalised. Display uses fftshift so that DC is in the centre. "
        "/ 计算使用未归一化的 numpy.fft.fft2。显示时用 fftshift，把直流放在中心。"
    )
    runners = {
        "spatial": _spatial,
        "decomposition": _decomposition,
        "properties": _properties,
        "phase": _phase,
        "filters": _filters,
        "theorem": _theorem,
        "aliasing": _aliasing,
        "hybrid": _hybrid,
    }
    for item in L2_EXPERIMENTS:
        experiment_heading("l2", item)
        runners[item[0]](image)


def _spectrum_row(image: np.ndarray, title: str) -> None:
    _spectrum, magnitude, phase = shifted_magnitude_phase(image)
    show_row(
        [
            (title, image, "image"),
            ("log10(1+|F|), shifted / 对数幅度", log_magnitude(magnitude), "otf"),
            ("Phase, shifted / 相位", phase, "residual"),
        ]
    )


def _spatial(image: np.ndarray) -> None:
    del image
    concept(
        "The integers (u, v), measured in cycles per pixel, are the orientation and spacing of a sinusoid. A horizontal wave has v = 0.",
        "以每周每像素计的 (u, v) 决定正弦的方向和间距。水平波的 v = 0。",
    )
    model(r"f[y,x]=A\cos(2\pi(ux+vy)+\phi)")
    controls_heading()
    size = int(st.session_state.get("image_size", 128))
    u = float(st.slider("u, cycles per pixel / 水平频率", 0.0, 0.5, 3 / size, 1 / size))
    v = float(st.slider("v, cycles per pixel / 垂直频率", -0.5, 0.5, 0.0, 1 / size))
    phase = float(st.slider("Phase φ, radians / 相位", -np.pi, np.pi, 0.0, 0.05))
    amplitude = float(st.slider("Amplitude A / 振幅", 0.0, 1.0, 0.5, 0.05))
    picture = 0.5 + sinusoid((size, size), u, v, phase, amplitude)
    visual_heading()
    _spectrum_row(picture, "Sinusoid, shown around 0.5 / 正弦，显示时以 0.5 为中心")
    st.caption("Adding 0.5 is display offset so the cosine fits a grey image. The DFT of a pure cosine is a pair of spikes; the offset adds DC. / 加上 0.5 只是为了把余弦放进灰度图。纯余弦的 DFT 是一对尖峰，这个偏移会额外产生直流。")
    steps(
        [
            ("Increasing u packs the bars more tightly from left to right.", "增大 u 会让左右方向的条纹更密。"),
            ("A nonzero v tilts the bars. The direction of the bars is perpendicular to the frequency vector (u, v).", "v 不为零时条纹会倾斜。条纹方向垂直于频率向量 (u, v)。"),
            ("Phase slides the pattern. It does not change the magnitude spikes.", "相位平移图案，不改变幅度尖峰。"),
        ]
    )
    takeaways([("One DFT bin is one of these patterns, plus its conjugate partner for a real image.", "一个 DFT 频率箱就是这样一种图案；实图像还要带上它的共轭伙伴。")])
    try_this("Set u = v and watch the bars run diagonally. Then set both to 0 and see only DC. / 令 u = v，观察对角条纹。再把两者都设为 0，只剩下直流。", "The slider step is one bin of the current image size. / 滑块步进是当前图像尺寸上的一个频率箱。")


def _decomposition(image: np.ndarray) -> None:
    concept(
        "The inverse DFT adds the retained complex coefficients. Keeping a conjugate pair preserves a real image.",
        "逆 DFT 把保留下来的复系数加起来。成对保留共轭系数，重建才是实图像。",
    )
    model(r"\hat f = \mathcal{F}^{-1}\{F\cdot M\}")
    controls_heading()
    pairs = int(st.slider("Conjugate pairs kept / 保留的共轭对个数", 1, min(64, image.size), 8))
    reconstructed, kept = reconstruct_pairs(image, pairs)
    visual_heading()
    show_row(
        [
            ("Original / 原图", image, "image"),
            ("Kept magnitude, shifted / 保留的幅度", log_magnitude(kept), "otf"),
            ("Reconstruction / 重建", reconstructed, "image"),
        ]
    )
    st.write(f"MSE {mse(reconstructed, image):.4g}.")
    st.caption(f"均方误差 {mse(reconstructed, image):.4g}。")
    steps(
        [
            ("Order coefficients by magnitude.", "按幅度给系数排序。"),
            ("Keep a coefficient and the coefficient at the negated frequency.", "保留一个系数，以及负频率上的那个系数。"),
            ("Set the rest to zero and invert. The error is the energy that was discarded.", "其余置零后做逆变换。误差就是被丢掉的能量。"),
        ]
    )
    takeaways([("A few low-frequency pairs recover the broad shapes. Edges need the pairs that were discarded.", "少数低频对能恢复大的形状。边缘需要那些被丢掉的频率对。")])
    try_this("Move from 1 pair to the maximum and watch the MSE fall. / 从 1 对加到最大，观察均方误差下降。", "The count is pairs, not individual complex numbers. / 这里数的是共轭对，不是单个复数。")
    save_workspace(reconstructed, "fourier-partial", "ws-dft")


def _phase(image: np.ndarray) -> None:
    concept(
        "Magnitude says how strong each sinusoid is. Phase says where it sits. Zeroing the phase lines the sinusoids up at the origin and usually destroys the picture.",
        "幅度表示每个正弦有多强。相位表示它放在哪里。把相位置零会让这些正弦在原点对齐，通常会毁掉图像。",
    )
    model(r"F=|F|e^{i\phi}")
    controls_heading()
    other_name = st.selectbox("Second image / 第二幅图", ["几何图形", "重复条纹"])
    size = image.shape[0]
    other = geometric_shapes(size) if other_name == "几何图形" else repeated_stripes(size)
    visual_heading()
    mag_only = magnitude_only(image)
    only_phase = phase_only(image)
    swap_ab = swap_magnitude_phase(image, other)
    swap_ba = swap_magnitude_phase(other, image)
    show_row(
        [
            ("Image A / 图像 A", image, "image"),
            ("Magnitude only, zero phase / 仅幅度，相位为零", mag_only, "kernel"),
            ("Phase only, unit magnitude / 仅相位，幅度为 1", only_phase, "kernel"),
        ]
    )
    show_row(
        [
            ("Image B / 图像 B", other, "image"),
            ("|A| with phase of B / A 的幅度配 B 的相位", swap_ab, "kernel"),
            ("|B| with phase of A / B 的幅度配 A 的相位", swap_ba, "kernel"),
        ]
    )
    st.caption(
        "Phase-only and swapped images are not scaled to [0, 1]. Where a DFT coefficient is numerically zero, its phase is set to 0 before the inverse transform. "
        "仅相位和交换后的图像没有被缩放到 [0, 1]。DFT 系数在数值上为零时，逆变换前把相位设为 0。"
    )
    steps(
        [
            ("Magnitude only replaces every phase by zero.", "仅幅度把每个相位换成零。"),
            ("Phase only replaces every nonzero magnitude by one.", "仅相位把每个非零幅度换成一。"),
            ("A swap keeps one image's magnitudes and the other's phases.", "交换保留一幅图的幅度和另一幅图的相位。"),
        ]
    )
    takeaways([("The recognisable structure follows the phase donor, not the magnitude donor.", "认得出的结构跟着提供相位的那幅图，而不是提供幅度的那幅图。")])
    try_this("Compare |A| with phase of B against image B. / 把「A 的幅度配 B 的相位」和图像 B 比较。", "The grey scale may be stretched. Read the caption before judging contrast. / 灰度可能被拉伸。先看说明再判断对比度。")


def _filters(image: np.ndarray) -> None:
    concept(
        "Multiplying the DFT by a mask is circular convolution with the mask's inverse DFT. An ideal cutoff has a sharp edge, so its spatial kernel rings.",
        "把 DFT 乘上一个掩模，就是和该掩模的逆 DFT 做循环卷积。理想截止有陡峭的边界，所以它的空间核会振铃。",
    )
    model(r"G = F \odot H")
    controls_heading()
    kind = st.selectbox(
        "Mask / 掩模",
        ["ideal_low", "gaussian_low", "ideal_high", "gaussian_high", "band", "directional"],
        format_func=lambda name: {
            "ideal_low": "Ideal low-pass / 理想低通",
            "gaussian_low": "Gaussian low-pass / 高斯低通",
            "ideal_high": "Ideal high-pass / 理想高通",
            "gaussian_high": "Gaussian high-pass / 高斯高通",
            "band": "Band-pass / 带通",
            "directional": "Directional / 方向",
        }[name],
    )
    cutoff = float(st.slider("Cutoff, cycles per pixel / 截止频率", 0.02, 0.45, 0.12, 0.01))
    width = float(st.slider("Width / 宽度", 0.01, 0.2, 0.04, 0.01))
    angle = float(st.slider("Direction, radians / 方向", -np.pi, np.pi, 0.0, 0.05))
    mask = frequency_mask(image.shape, kind, cutoff, width=width, angle=angle)
    filtered = apply_frequency_mask(image, mask)
    visual_heading()
    _spectrum, magnitude, _phase = shifted_magnitude_phase(image)
    show_row(
        [
            ("Original / 原图", image, "image"),
            ("Spectrum / 频谱", log_magnitude(magnitude), "otf"),
            ("Mask, shifted / 掩模", np.fft.fftshift(mask), "kernel"),
        ]
    )
    filtered_magnitude = log_magnitude(np.fft.fftshift(np.abs(np.fft.fft2(filtered))))
    show_row(
        [
            ("Filtered spectrum / 滤波后的频谱", filtered_magnitude, "otf"),
            ("Reconstruction / 重建", filtered, "image"),
            ("Difference / 差值", filtered - image, "residual"),
        ]
    )
    steps(
        [
            ("A low-pass keeps a disk around DC and removes the fine bars.", "低通保留直流周围的圆盘，去掉细条纹。"),
            ("A high-pass does the opposite and also removes the mean when the disk includes DC.", "高通相反，而且当圆盘包含直流时会去掉均值。"),
            ("The ideal disk has a discontinuous edge. That edge is why the reconstruction rings. The Gaussian mask does not have that edge.", "理想圆盘的边界不连续。重建振铃来自这条边界。高斯掩模没有这条边界。"),
        ]
    )
    takeaways([("This product is circular convolution, not linear convolution. The padding question is the next experiment.", "这个乘积是循环卷积，不是线性卷积。填充问题在下一个实验。")])
    try_this("Compare ideal low-pass and Gaussian low-pass at the same cutoff on the geometric image. / 在几何图形上，用相同截止频率比较理想低通和高斯低通。", "A directional mask removes DC on purpose. The reconstruction then has mean zero. / 方向掩模故意去掉直流，重建的均值就为零。")
    save_workspace(filtered, f"freq-{kind}", "ws-freq")


def _theorem(image: np.ndarray) -> None:
    concept(
        "The DFT turns circular convolution into a product. Linear convolution needs enough zero padding that the kernel never wraps around.",
        "DFT 把循环卷积变成乘积。线性卷积需要足够的零填充，使核不会绕回到另一边。",
    )
    model(r"\mathcal{F}\{f*h\}=\mathcal{F}\{f\}\odot\mathcal{F}\{h\}")
    controls_heading()
    size = int(st.slider("Kernel width / 核的宽度", 3, 9, 5, 2))
    yy, xx = np.mgrid[:size, :size]
    kernel = np.exp(-0.5 * ((yy - size // 2) ** 2 + (xx - size // 2) ** 2) / 1.2**2)
    kernel = kernel / kernel.sum()
    visual_heading()
    linear_direct = linear_convolve_direct(image, kernel)
    linear_fft = linear_convolve_fft(image, kernel)
    circular = circular_convolve_same(image, kernel)
    show_row(
        [
            ("Kernel / 核", kernel, "kernel"),
            ("Linear, centre crop / 线性卷积中心", linear_fft[size // 2 : size // 2 + image.shape[0], size // 2 : size // 2 + image.shape[1]], "image"),
            ("Circular, no padding / 不填充的循环卷积", circular, "image"),
        ]
    )
    st.write(
        f"Max |direct linear − FFT linear| = {np.max(np.abs(linear_direct - linear_fft)):.3g}. "
        f"Linear shape {linear_fft.shape}. Circular shape {circular.shape}."
    )
    st.caption(
        f"直接线性卷积与 FFT 线性卷积的最大差是 {np.max(np.abs(linear_direct - linear_fft)):.3g}。"
        f"线性结果尺寸 {linear_fft.shape}。循环结果尺寸 {circular.shape}。"
    )
    steps(
        [
            ("Full linear convolution of an n×n image and an m×m kernel has shape (n+m−1)×(n+m−1).", "n×n 图像与 m×m 核的完整线性卷积尺寸是 (n+m−1)×(n+m−1)。"),
            ("scipy.signal.fftconvolve pads to that size, multiplies DFTs, and inverts.", "scipy.signal.fftconvolve 填充到这个尺寸，相乘 DFT，再逆变换。"),
            ("The circular result uses no padding, so samples from the opposite border enter the sum.", "循环结果不填充，对边的样本会进入求和。"),
        ]
    )
    takeaways([("FFT filtering is the fast method once the padded kernel is large. It is not automatically the same operator as a sliding window with reflect padding.", "填充后的核很大时，FFT 滤波更快。它不会自动等于反射填充的滑动窗口。")])
    try_this("Increase the kernel width and look at the border of the circular result. / 增大核的宽度，看循环结果的边界。", "The reported difference should be numerical noise, not a visible image error. / 报告的差值应是数值噪声，而不是看得见的图像误差。")


def _properties(image: np.ndarray) -> None:
    concept(
        "These identities are exact for this DFT, up to floating-point error. They are not approximations.",
        "对这些 DFT，下面的恒等式在浮点误差内是精确的，不是近似。",
    )
    model(r"\mathcal{F}\{c_1 f_1+c_2 f_2\}=c_1 F_1+c_2 F_2")
    controls_heading()
    shift_y = int(st.slider("Shift down, pixels / 向下平移", -20, 20, 6))
    shift_x = int(st.slider("Shift right, pixels / 向右平移", -20, 20, -4))
    other = np.roll(image, 7, axis=0)
    visual_heading()
    shifted = np.roll(np.roll(image, shift_y, axis=0), shift_x, axis=1)
    show_row([("Original / 原图", image, "image"), ("Shifted / 平移后", shifted, "image")])
    st.write(
        f"Linearity error {linearity_error(image, other, 0.4, -0.2):.3g}. "
        f"Translation phase error {translation_phase_error(image, shift_y, shift_x):.3g}. "
        f"Conjugate-symmetry error {conjugate_symmetry_error(image):.3g}."
    )
    st.caption("线性误差、平移相位误差、共轭对称误差都显示在上面。它们应接近舍入误差。")
    from src.b14.image_processing.lecture3.operators import gaussian_psf, psf_to_otf

    psf = gaussian_psf(image.shape, 2.0)
    otf = psf_to_otf(psf)
    show_row(
        [
            ("Periodic Gaussian / 周期高斯", psf, "kernel"),
            ("Real part of its OTF / 传递函数的实部", np.fft.fftshift(otf.real), "otf"),
            ("Imaginary part / 虚部", np.fft.fftshift(otf.imag), "residual"),
        ]
    )
    st.caption(
        f"An even real kernel has a real DFT. Max |Im H| = {np.max(np.abs(otf.imag)):.3g}. "
        "On a discrete periodic grid this is the exact transform of that kernel, not a continuous Gaussian identity. "
        f"偶实核的 DFT 是实数。最大虚部是 {np.max(np.abs(otf.imag)):.3g}。在离散周期网格上，这是该核的精确变换，不是连续高斯的恒等式。"
    )
    steps(
        [
            ("Linearity is the DFT of a sum versus the sum of the DFTs.", "线性是和的 DFT 与 DFT 的和。"),
            ("A spatial shift multiplies F by a complex exponential. The magnitude is unchanged.", "空间平移给 F 乘一个复指数。幅度不变。"),
            ("A real image satisfies F[k] = conj(F[-k]).", "实图像满足 F[k] = conj(F[-k])。"),
        ]
    )
    takeaways([("The imaginary part of the Gaussian OTF is numerical noise because the kernel was built to be even.", "高斯传递函数的虚部是数值噪声，因为这个核被做成了偶函数。")])
    try_this("Change the shift and check that the phase error stays tiny while the picture moves. / 改变平移，确认图像在动而相位误差仍然很小。", "Scaling is visible in the sinusoid experiment: doubling (u, v) packs the same pattern more tightly. / 缩放可以在正弦实验里看到：把 (u, v) 加倍，同样的图案会更密。")


def _aliasing(image: np.ndarray) -> None:
    del image
    concept(
        "A discrete image already has a Nyquist frequency of 0.5 cycles per pixel. Keeping every k-th sample lowers that limit to 0.5/k. Content above the new limit folds back.",
        "离散图像的奈奎斯特频率已经是每像素 0.5 周。每隔 k 个样本取一次，会把这个上限降到 0.5/k。高于新上限的成分会折回来。",
    )
    model(r"f_{\mathrm{Nyquist}}=1/(2\Delta)")
    controls_heading()
    size = int(st.session_state.get("image_size", 128))
    pattern = st.radio("Pattern / 图案", ["stripes", "checkerboard"], horizontal=True, format_func=lambda name: "Stripes / 条纹" if name == "stripes" else "Checkerboard / 棋盘")
    frequency = float(st.slider("Stripe frequency / 条纹频率", 0.05, 0.45, 0.31, 0.01))
    factor = int(st.select_slider("Downsample factor / 降采样倍数", options=[2, 4, 8], value=4))
    sigma = float(st.slider("Prefilter σ / 预滤波 σ", 0.0, 4.0, 1.5, 0.1))
    angle = float(st.slider("Moiré angle, degrees / 摩尔条纹夹角", 0.0, 20.0, 4.0, 0.5))
    source = stripe_image(size, frequency) if pattern == "stripes" else checkerboard(size, period=max(1, int(round(0.5 / max(frequency, 0.05)))))
    raw = downsample(source, factor)
    safe = downsample(source, factor, prefilter_sigma=sigma if sigma > 0 else None)
    visual_heading()
    show_row(
        [
            ("Original / 原图", source, "image"),
            ("Downsampled, no prefilter / 直接降采样", raw, "image"),
            ("Downsampled after Gaussian / 先高斯再降采样", safe, "image"),
        ]
    )
    _spectrum_row(raw, "Sampled image / 采样后的图像")
    if pattern == "stripes":
        u, v = dominant_frequency(raw)
        st.write(f"Strongest non-DC bin of the downsampled stripes: u = {u:.4f}, v = {v:.4f} cycles per new pixel.")
        st.caption(f"降采样条纹最强的非直流频率箱：u = {u:.4f}，v = {v:.4f}，单位是每个新像素的周数。")
    beats = moire(size, frequency, angle)
    show_array(beats, "Two-grating Moiré / 两组光栅的摩尔条纹", "image")
    steps(
        [
            ("Point sampling keeps the high-frequency oscillation and lets it masquerade as a lower frequency.", "点抽样保留高频振荡，并让它伪装成更低的频率。"),
            ("A Gaussian before sampling attenuates those frequencies. It also blurs genuine detail.", "采样前的高斯会衰减这些频率，同时也会模糊真正的细节。"),
            ("The Moiré picture is the sum of a grating and a slightly rotated copy. The eye sees the beat, not only the fine lines.", "摩尔图是一组光栅和轻微旋转后的副本之和。眼睛看到的是拍频，不只是细线。"),
        ]
    )
    takeaways([("Anti-aliasing is a filter applied before the samples are thrown away, not a repair applied afterwards.", "抗混叠是在丢掉样本之前做的滤波，不是事后修补。")])
    try_this("Use stripes at frequency 0.31 and factor 4. The new peak is not at 0.31 cycle per new pixel. / 用频率 0.31 的条纹和倍数 4。新的峰值不在每个新像素 0.31 周。", "Increase the Moiré angle from 0. The beat gets finer. / 把摩尔夹角从 0 增大，拍频会变细。")


def _hybrid(image: np.ndarray) -> None:
    concept(
        "The lecture uses hybrid images as a filtering application: low frequencies from one picture and high frequencies from another.",
        "这一讲把混合图像当作滤波应用：一幅图提供低频，另一幅图提供高频。",
    )
    model(r"h = H_{\mathrm{low}}F_A + H_{\mathrm{high}}F_B")
    controls_heading()
    size = image.shape[0]
    second = repeated_stripes(size) if not np.allclose(image, repeated_stripes(size)) else geometric_shapes(size)
    cutoff = float(st.slider("Gaussian cutoff / 高斯截止", 0.02, 0.3, 0.08, 0.01))
    low, high, mixed = hybrid_image(image, second, cutoff)
    visual_heading()
    show_row(
        [
            ("Low-pass of the workspace / 工作图像的低通", low, "image"),
            ("High-pass of the second image / 第二幅图的高通", high, "kernel"),
            ("Sum / 和", mixed, "image"),
        ]
    )
    steps(
        [
            ("Both masks are Gaussian, so this sum does not ring the way an ideal cutoff does.", "两个掩模都是高斯，所以这个和不会像理想截止那样振铃。"),
            ("The high-pass image has its mean removed. The sum's mean comes from the low-pass image.", "高通图像去掉了均值。和的均值来自低通图像。"),
        ]
    )
    takeaways([("Looking from far away emphasises the low-pass image. Looking closely reveals the high-pass image. That is the demonstration, not a measurement of vision.", "远看强调低通图像，近看露出高通图像。这是演示，不是对视觉的测量。")])
    try_this("Lower the cutoff until the workspace image becomes a soft blob and the stripes dominate up close. / 降低截止频率，直到工作图像变成模糊的一团，而条纹在近处占主导。", "Save the sum as the workspace if you want to compress it in Lecture 4. / 如果想在第 4 讲压缩这个和，把它存入工作区。")
    save_workspace(mixed, "hybrid", "ws-hybrid")
