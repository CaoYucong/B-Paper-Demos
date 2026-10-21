"""标签页 1：前向模型 g = Af + n。"""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.degradation import Observation
from src.operators import crop_centered
from src.ui.i18n import say, t
from src.ui_common import banner, render_images


def render_forward(observation: Observation, lam: float) -> None:
    banner(
        observation.label,
        observation.original.shape,
        observation.sigma,
        observation.noise_std,
        observation.seed,
        lam,
    )
    say(
        r"""
Restoration starts from a model of how the image was damaged:

$$
g = Af + n
$$

$f$ is the unknown grayscale image, $A$ is Gaussian blur, $g$ is the image we actually see, and $n$ is additive Gaussian noise. The blur and noise controls apply to every restoration experiment, so they compare one observation.
""",
        """
图像复原先要写清图像是怎样变坏的。本实验使用

$$
g = Af + n
$$

其中 $f$ 是未知的原始灰度图，$A$ 是高斯模糊，$g$ 是我们真正看到的图，$n$ 是加性高斯噪声。
模糊和噪声会同时作用于后面每一个复原实验，因此各页比较的是同一幅观测。
""",
    )
    items = [
        ("原始图像  f", observation.original, "image"),
        ("模糊图像  Af", observation.blurred, "image"),
        ("观测图像  g = Af + n", observation.observed, "image"),
    ]
    if st.session_state.get("show_noise", False):
        items.append(("噪声  n = g − Af", observation.noise, "residual"))
    render_images(items, columns_per_row=3)

    left, right = st.columns(2)
    with left:
        st.checkbox("显示噪声 n", key="show_noise")
    with right:
        st.checkbox("显示高斯 PSF 核", key="show_psf")

    if st.session_state["show_psf"]:
        half = int(np.clip(np.ceil(4.0 * observation.sigma), 6, observation.psf.shape[0] // 2))
        window = crop_centered(observation.psf, half)
        render_images([("高斯 PSF（中心窗口）", window, "kernel")], columns_per_row=1)
        st.caption(
            t(
                f"Mass inside the window is {float(window.sum()):.4f}. Mass of the full kernel is {float(observation.psf.sum()):.6f}. "
                "The PSF is the point spread function: the image of one bright point at the centre after A.",
                f"窗口内的质量和为 {float(window.sum()):.4f}，完整核的质量和为 {float(observation.psf.sum()):.6f}。"
                " PSF 是点扩散函数（point spread function）：一个位于中心的亮点经过 A 之后变成的样子。",
            )
        )

    say(
        f"""
**Blur standard deviation σ = {observation.sigma:.2f} changes the operator A.**
The discrete Gaussian is $k_{{ij}} \\propto \\exp(-(i^2+j^2)/(2\\sigma^2))$, then normalised to sum to 1.
A larger σ mixes more distant neighbours into each output pixel, so $Af$ is blurrier.
σ does not change the noise. The kernel sums to {float(observation.psf.sum()):.6f}, and the DC gain is $H(0) =$ {float(np.real(observation.otf[0, 0])):.6f}, so a uniform grey level is unchanged by the blur.

**Noise standard deviation σₙ = {observation.noise_std:.3f} changes n, not A.**
Each pixel is independent $\\mathcal N(0,\\sigma_n^2)$. Intensities lie near $[0,1]$, so 0.03 is mild and 0.1 already hides detail. When σₙ = 0, $g = Af$.
$g$ is not clipped back to $[0,1]$, because that would no longer be the additive Gaussian model.
""",
        f"""
**模糊标准差 σ = {observation.sigma:.2f} 改变的是算子 A。**
离散高斯核 $k_{{ij}} \\propto \\exp(-(i^2+j^2)/(2\\sigma^2))$，再归一化使总和为 1。
σ 越大，核越宽，A 的每一行就会把更远的邻居混进同一个输出像素，所以 $Af$ 更糊。
σ 不改变噪声的分布。当前核总和为 {float(observation.psf.sum()):.6f}，
直流增益 $H(0) =$ {float(np.real(observation.otf[0, 0])):.6f}，因此一片均匀灰度不会被模糊改变亮度。

**噪声标准差 σₙ = {observation.noise_std:.3f} 改变的是 n，不是 A。**
每个像素的噪声独立服从 $\\mathcal N(0,\\sigma_n^2)$。强度已经归一化到大约 $[0,1]$，
所以 0.03 是轻度噪声，0.1 以上会明显盖住细节。σₙ = 0 时 $g = Af$。
程序不会把 $g$ 裁剪回 $[0,1]$，否则模型就不再是题目中的加性高斯噪声。
""",
    )
    with st.expander("实验 1：看懂算子 A"):
        say(
            """
1. Choose Geometric shapes, set the noise to 0, and move the blur σ through 0.5, 2.0, and 4.0.
2. Show the PSF. As σ grows, the kernel widens and rectangles and thin lines lose their edges first.
3. Fix σ at 2.0 and raise the noise from 0 to 0.1. $Af$ stays the same; only $g$ gets dirtier.
4. Switch to Repeated stripes. High-frequency stripes are smoothed by the same A more than a smooth ramp is.

The next page takes A apart: it is a matrix in principle and a convolution in the calculation.
""",
            """
1. 选择「几何图形」，把噪声设为 0，只拖动模糊 σ：0.5、2.0、4.0。
2. 打开 PSF。σ 变大时，核变宽，矩形和细线先丢失锐利边缘。
3. 把 σ 固定在 2.0，再把噪声从 0 调到 0.1。$Af$ 不变，只有 $g$ 变脏。
4. 换「重复条纹」。高频条纹比平滑渐变更容易被同一个 A 抹平。

下一页会把这个 A 拆开：它在概念上是矩阵，在计算上是卷积。
""",
        )
