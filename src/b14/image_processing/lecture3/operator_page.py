"""标签页 2：矩阵 A、圆卷积，以及伴随为什么出现在梯度里。"""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.b14.image_processing.lecture3.degradation import Observation
from src.b14.image_processing.lecture3.operators import (
    apply_otf,
    apply_otf_adjoint,
    averaging_matrix,
    inner_product,
    motion_blur_psf,
    otf_log_magnitude,
    psf_to_otf,
)
from src.ui.i18n import say, shown_label
from src.ui.common import banner, render_images, show_fig
from src.ui.visualization import image_png, matrix_figure, vector_bars


def _default_pixels(n: int) -> np.ndarray:
    values = np.zeros(n, dtype=np.float64)
    values[n // 2 :] = 1.0
    if n >= 6:
        values[-1] = 0.35
    return values


def _sync_toy_pixels(n: int) -> None:
    ready = st.session_state.get("_toy_n_applied") == n and all(
        f"toy_v_{i}" in st.session_state for i in range(n)
    )
    if ready:
        return
    values = _default_pixels(n)
    for i in range(8):
        st.session_state[f"toy_v_{i}"] = float(values[i]) if i < n else 0.0
    st.session_state["_toy_n_applied"] = n


def _render_toy() -> None:
    st.subheader("一维玩具：把 A 写成真正的矩阵")
    st.markdown(
        """
二维图像的 A 大到不能画出来。这里用 5–8 个像素做一个**不同的**算子：
周期边界上的对称三点平均，权重是 $1/3,1/3,1/3$。
它不是后面复原所用的高斯模糊，只用来看清「矩阵的一行 = 输出像素如何混合邻居」。
"""
    )
    n_pixels = st.slider("像素个数", min_value=5, max_value=8, step=1, key="toy_n")
    _sync_toy_pixels(int(n_pixels))
    st.caption("拖动下面的滑块改变原始像素 f。矩阵乘法会立刻更新。")
    value_columns = st.columns(int(n_pixels))
    values = np.array(
        [
            value_columns[i].slider(f"f[{i}]", min_value=0.0, max_value=1.0, step=0.01, key=f"toy_v_{i}")
            for i in range(int(n_pixels))
        ],
        dtype=np.float64,
    )
    matrix = averaging_matrix(int(n_pixels))
    blurred = matrix @ values
    if int(st.session_state.get("toy_row", 0)) > int(n_pixels) - 1:
        st.session_state["toy_row"] = int(n_pixels) - 1
    highlight = st.slider("高亮 A 的第 i 行", min_value=0, max_value=int(n_pixels) - 1, key="toy_row")
    show_fig(matrix_figure(matrix, int(highlight)))
    left, right = st.columns(2)
    with left:
        show_fig(vector_bars(values, "原始像素向量  f", "#0B3A5B"))
    with right:
        show_fig(vector_bars(blurred, "矩阵乘法  Af（此玩具不加噪声）", "#C4492C"))

    terms = [f"{matrix[highlight, j]:.3f}\\times {values[j]:.3f}" for j in range(int(n_pixels)) if matrix[highlight, j] != 0]
    st.latex(rf"(Af)_{{{int(highlight)}}} = " + " + ".join(terms) + rf" = {float(blurred[highlight]):.4f}")
    st.markdown(
        f"""
第 {int(highlight)} 行不是单位脉冲，所以输出像素是左右邻居的平均，而不是 f 本身。
周期边界让第 0 行用到最后一个像素，因此矩阵右上角是 $1/3$，不是 0。
完整模型里还要再加上噪声才得到 $g$。这个玩具停在 $Af$，是为了把矩阵乘法单独看清楚。
"""
    )


def _adjoint_report(name: str, image: np.ndarray, otf: np.ndarray, probe: np.ndarray) -> None:
    forward = apply_otf(image, otf)
    adjoint_probe = apply_otf_adjoint(probe, otf)
    lhs = inner_product(forward, probe)
    rhs = inner_product(image, adjoint_probe)
    relative = abs(lhs - rhs) / max(1.0, abs(lhs))
    self_gap = float(np.max(np.abs(forward - apply_otf_adjoint(image, otf))))
    metric_cols = st.columns(4)
    metric_cols[0].metric(f"{name}  ⟨A(f), y⟩", f"{lhs:.6e}")
    metric_cols[1].metric("⟨f, Aᵀ(y)⟩", f"{rhs:.6e}")
    metric_cols[2].metric("相对误差", f"{relative:.2e}")
    metric_cols[3].metric("max |A(f) − Aᵀ(f)|", f"{self_gap:.2e}")
    if relative < 1e-8:
        st.success(f"{name}：伴随检验通过。⟨A(f), y⟩ 与 ⟨f, Aᵀ(y)⟩ 的差别只在浮点误差里。")
    else:
        st.error(f"{name}：伴随检验未通过。前向算子和 Aᵀ 的实现不一致。")


def render_operator(observation: Observation, lam: float) -> None:
    banner(
        observation.label,
        observation.original.shape,
        observation.sigma,
        observation.noise_std,
        observation.seed,
        lam,
    )
    st.caption(
        "Operator A is circular convolution. The small matrix below is a one-dimensional teaching model, not the two-dimensional restoration operator. "
        "/ 算子 A 是循环卷积。下面的小矩阵是一维教学模型，不是二维复原算子。"
    )
    _render_toy()
    st.divider()
    st.subheader("二维图像：A 是圆卷积，不是那张三点矩阵")
    say(
        r"""
For an image of $N$ pixels, A is an $N\times N$ matrix in principle. A $128\times 128$ image has about $2.7\times 10^8$ entries, which should neither be drawn nor stored. The normalised Gaussian makes A a **circular convolution**:

$$
(Af) = k * f
$$

`ifftshift` then an FFT of that same discrete kernel gives the transfer function $H$. The forward operation is $A(f) = \mathcal F^{-1}(H\,\mathcal F(f))$.
Do not treat SciPy's default reflect-boundary Gaussian filter as this A: the boundary condition is different, and the transfer function would not match.
This kernel is even, so $H$ is real and $A^T = A$. The code still implements the adjoint with the conjugate $H^*$, so the asymmetric motion blur below passes the same check.
""",
        r"""
对 $N$ 个像素的图像，概念上的 A 是 $N\times N$ 矩阵。$128\times 128$ 就有大约 $2.7\times 10^8$ 个元素，
既画不出来，也不该存下来。归一化高斯核对应的 A 是**周期卷积**：

$$
(Af) = k * f
$$

同一个离散核经过 `ifftshift` 再做 FFT，得到传递函数 $H$。
前向运算是 $A(f) = \mathcal F^{-1}(H\,\mathcal F(f))$。
不要把 SciPy 默认的反射边界高斯滤波当成这个 A：边界条件不同，传函也不能混用。
本实验的核是偶对称的，因此 $H$ 为实数，并且 $A^T = A$。代码仍按共轭 $H^*$ 实现伴随，
这样下面的不对称运动模糊也能用同一个检验。
""",
    )
    probe = np.random.default_rng(12345).normal(size=observation.original.shape)
    _adjoint_report("高斯模糊", observation.original, observation.otf, probe)
    render_images(
        [
            ("A(f)：高斯卷积", observation.blurred, "image"),
            ("检验向量 y", probe, "residual"),
            ("Aᵀ(y)", apply_otf_adjoint(probe, observation.otf), "residual"),
        ],
        columns_per_row=3,
    )
    otf_display = otf_log_magnitude(observation.otf)
    st.image(
        image_png(
            otf_display,
            shown_label("log₁₀|H|，零频在中心。这是 A 的频率响应，不是 PSF"),
            zmin=float(np.min(otf_display)),
            zmax=float(np.max(otf_display)),
            colorscale="Magma",
        ),
        use_container_width=True,
    )
    st.caption(
        f"高斯核的 max |Im H| = {float(np.max(np.abs(observation.otf.imag))):.2e}。"
        " 它接近 0，说明这个对称核的前向算子和伴随算子相同。"
    )

    st.checkbox("额外对比一个不对称的运动模糊（它不是本实验的复原模型）", key="show_motion")
    if st.session_state["show_motion"]:
        motion = motion_blur_psf(observation.original.shape, length=9)
        motion_otf = psf_to_otf(motion)
        st.markdown(
            "运动模糊的 PSF 只向一侧延伸，所以 $A \\neq A^T$。伴随依然成立，因为伴随是核翻转之后的卷积，频域里就是 $H^*$。"
        )
        _adjoint_report("运动模糊", observation.original, motion_otf, probe)
        render_images(
            [
                ("运动模糊 PSF = A(δ)", motion, "kernel"),
                ("A(f)", apply_otf(observation.original, motion_otf), "image"),
                ("Aᵀ(f)", apply_otf_adjoint(observation.original, motion_otf), "image"),
            ],
            columns_per_row=3,
        )

    with st.expander("为什么梯度里出现的是 Aᵀ，而不是 A"):
        st.markdown(
            """
数据项是平均平方误差

$$
J_{\\mathrm{data}}(f) = \\frac{1}{N}\\lVert Af - g\\rVert^2.
$$

沿方向 $v$ 的方向导数可以用内积写出来。令 $r = Af - g$，则

$$
\\mathrm d J_{\\mathrm{data}} = \\frac{2}{N}\\langle r, A v\\rangle = \\frac{2}{N}\\langle A^{T} r, v\\rangle.
$$

最后一个等号就是伴随的定义：$\\langle Ax, y\\rangle = \\langle x, A^{T} y\\rangle$。
因此数据项的梯度是 $(2/N)\\,A^{T}(Af-g)$，不是 $A(Af-g)$。
正则项同样由 $D$ 的伴随给出 $(2/N)\\,\\lambda D^{T}Df$。
高斯核恰好自伴随，所以数值上 $A^{T}$ 和 $A$ 一样；推导不能省掉 $A^{T}$，否则换一个核就会错。
"""
        )
