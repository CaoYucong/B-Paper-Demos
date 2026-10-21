"""标签页 4：逆滤波、维纳滤波和两条 MAP 直接求解。"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from src.comparison import lambda_sweep
from src.degradation import Observation
from src.map_estimation import compute_gradient, objective, run_gradient_descent, solve_map_cg, solve_map_fourier
from src.metrics import format_db, mse, psnr
from src.ui.i18n import say
from src.ui_common import banner, digest_array, render_images, show_fig
from src.visualization import sweep_figure
from src.wiener import inverse_filter, wiener_filter


def _fast_signature(observation: Observation, lam: float) -> tuple:
    return (
        digest_array(observation.observed),
        round(float(lam), 12),
        round(float(st.session_state["wiener_log10_k"]), 4),
        round(float(st.session_state["inverse_log10_threshold"]), 4),
        int(st.session_state["compare_iters"]),
    )


def _record(name: str, image: np.ndarray, seconds: float, observation: Observation, lam: float, note: str) -> dict:
    return {
        "方法": name,
        "时间 (ms)": f"{seconds * 1000:.2f}",
        "MSE": f"{mse(image, observation.original):.4e}",
        "PSNR": format_db(psnr(image, observation.original)),
        "目标 J": f"{objective(image, observation.observed, observation.otf, lam):.4e}",
        "说明": note,
    }


def render_compare(observation: Observation, lam: float) -> None:
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
Five methods share one observation $g$ and one circular Gaussian operator $A$. MSE and PSNR use the original image; none of the solvers sees it. Closed forms update immediately. Gradient descent and conjugate gradients run when the button is pressed, so moving a slider does not recompute them.

$$
F_{\mathrm{inverse}} = G/H
\qquad
F_{\mathrm{Wiener}} = \frac{H^*}{|H|^2+K}G
$$

$$
\hat F = \frac{H^* G}{|H|^2+\lambda(|D_x|^2+|D_y|^2)}
$$

The last expression is the exact quadratic-gradient MAP solution under periodic boundaries. $H$ comes from the same discrete kernel as the forward convolution, and the filter uses the conjugate $H^*$.
""",
        r"""
五种处理使用同一幅 $g$ 和同一个周期高斯算子 $A$。
MSE、PSNR 用原图计算，求解时谁也不看原图。
闭式方法会立刻更新；梯度下降和共轭梯度按下按钮后运行，避免每次拖动滑块都重算。

$$
F_{\mathrm{inverse}} = G/H
\qquad
F_{\mathrm{Wiener}} = \frac{H^*}{|H|^2+K}G
$$

$$
\hat F = \frac{H^* G}{|H|^2+\lambda(|D_x|^2+|D_y|^2)}
$$

最后一式是周期边界下二次梯度 MAP 的精确解。$H$ 来自与前向卷积相同的离散核，并使用共轭 $H^*$。
""",
    )
    left, right = st.columns(2)
    with left:
        st.slider("log₁₀(K)，维纳滤波的常数", min_value=-6.0, max_value=0.0, step=0.1, key="wiener_log10_k")
    with right:
        st.slider(
            "log₁₀(逆滤波阈值)",
            min_value=-6.0,
            max_value=-1.0,
            step=0.1,
            key="inverse_log10_threshold",
            help="|H| 低于阈值的频率被置零。这是硬截断，不是维纳滤波的软衰减。",
        )
    wiener_k = 10.0 ** float(st.session_state["wiener_log10_k"])
    threshold = 10.0 ** float(st.session_state["inverse_log10_threshold"])
    st.caption(f"当前 K = {wiener_k:.3e}，逆滤波阈值 = {threshold:.3e}，MAP 的 λ = {lam:.4g}。")

    started = time.perf_counter()
    inverse, suppressed = inverse_filter(observation.observed, observation.otf, threshold)
    inverse_seconds = time.perf_counter() - started
    started = time.perf_counter()
    wiener = wiener_filter(observation.observed, observation.otf, wiener_k)
    wiener_seconds = time.perf_counter() - started
    started = time.perf_counter()
    fourier = solve_map_fourier(observation.observed, observation.otf, lam)
    fourier_seconds = time.perf_counter() - started
    gradient = compute_gradient(fourier, observation.observed, observation.otf, lam)
    gradient_norm = float(np.linalg.norm(gradient))

    rows = [
        _record("不复原（观测 g）", observation.observed, 0.0, observation, lam, "把 g 当作估计。"),
        _record(
            "朴素逆滤波（inverse filter）",
            inverse,
            inverse_seconds,
            observation,
            lam,
            f"抑制了 {suppressed} 个 |H| 过小的频率。",
        ),
        _record(
            "维纳滤波（Wiener）",
            wiener,
            wiener_seconds,
            observation,
            lam,
            "常数 K，对应白先验 λ||f||²，不是梯度先验。",
        ),
        _record(
            "MAP 傅里叶闭式解（Fourier MAP）",
            fourier,
            fourier_seconds,
            observation,
            lam,
            "当前 λ 下二次目标的精确最小值。",
        ),
    ]
    render_images(
        [
            ("原图 f（只用于评估）", observation.original, "image"),
            ("不复原：观测 g", observation.observed, "image"),
            ("朴素逆滤波", inverse, "image"),
            ("维纳滤波", wiener, "image"),
            ("MAP 傅里叶闭式解", fourier, "image"),
        ],
        columns_per_row=3,
    )
    if gradient_norm < 1e-6:
        st.success(f"傅里叶解处的梯度范数 ||∇J|| = {gradient_norm:.3e}。无约束梯度在这个最小值上约为零。")
    else:
        st.error(f"傅里叶解处的梯度范数是 {gradient_norm:.3e}，闭式解和梯度不一致。")
    st.caption(
        f"逆滤波把 {suppressed} 个频率置零。|H| 很小时，G/H 会把噪声和舍入误差放大，"
        "所以必须有阈值。被置零的频率相当于承认「这些细节已经无法从 g 里可靠地收回」。"
    )

    st.subheader("λ 如何改变精确 MAP 解")
    sweep = lambda_sweep(observation, np.logspace(-5, 1, 21))
    show_fig(sweep_figure(sweep["lambda"], sweep["psnr"], sweep["data"], lam))
    st.caption(
        "虚线是左侧当前的 λ。PSNR 使用原图，只用于观察；每张图的求解只使用 g。"
        "曲线上的数据保真项可以互相比较。不同 λ 的总损失 J 不能比大小，因为 λ 改变了目标本身。"
        "左端通常欠正则，右端过正则，中间某个 λ 的 PSNR 最高。这个最好的 λ 依赖噪声和图像，不能事先当成唯一正确答案。"
    )

    st.subheader("迭代法是否走到同一个二次解")
    st.slider("对比用的梯度下降迭代次数", min_value=10, max_value=1000, step=10, key="compare_iters")
    st.caption("这里的梯度下降固定使用稳定步长 s = 1，从 g 出发，并且不裁剪像素。学习率实验请回到第 3 页。")
    if st.button("运行梯度下降与共轭梯度", type="primary"):
        with st.spinner("正在运行梯度下降和矩阵自由共轭梯度…"):
            started = time.perf_counter()
            gd = run_gradient_descent(
                observation.observed,
                observation.otf,
                lam,
                n_iterations=int(st.session_state["compare_iters"]),
                step_fraction=1.0,
                initialisation="observed",
                project=False,
                max_snapshots=2,
            )
            gd_seconds = time.perf_counter() - started
            started = time.perf_counter()
            cg_image, info = solve_map_cg(observation.observed, observation.otf, lam)
            cg_seconds = time.perf_counter() - started
        st.session_state["compare_iterative"] = {
            "signature": _fast_signature(observation, lam),
            "gd_image": gd.final,
            "gd_seconds": gd_seconds,
            "gd_note": f"普通梯度下降 {int(gd.iteration[-1])} 步。" + (" 已发散。" if gd.diverged else ""),
            "cg_image": cg_image,
            "cg_seconds": cg_seconds,
            "cg_info": info,
        }

    saved = st.session_state.get("compare_iterative")
    if saved is None:
        st.info("傅里叶解已经是这个二次目标的精确 MAP。点击按钮后，可以检查有限步梯度下降和共轭梯度离它有多远。")
    else:
        if saved["signature"] != _fast_signature(observation, lam):
            st.warning("参数已变化。下面的迭代结果属于上一次按钮运行。")
        gap = float(np.max(np.abs(saved["cg_image"] - fourier)))
        cg_note = "与当前傅里叶解的最大绝对差为 " + f"{gap:.3e}。"
        if saved["cg_info"] == 0:
            cg_note += " 共轭梯度已达到容差。"
        else:
            cg_note += f" 共轭梯度 info = {saved['cg_info']}。"
        rows.append(_record("MAP 梯度下降（gradient descent）", saved["gd_image"], saved["gd_seconds"], observation, lam, saved["gd_note"]))
        rows.append(_record("MAP 共轭梯度（conjugate gradient）", saved["cg_image"], saved["cg_seconds"], observation, lam, cg_note))
        render_images(
            [
                ("MAP 梯度下降", saved["gd_image"], "image"),
                ("MAP 共轭梯度", saved["cg_image"], "image"),
                ("共轭梯度 − 傅里叶解", saved["cg_image"] - fourier, "residual"),
            ]
        )
        if saved["signature"] == _fast_signature(observation, lam) and gap < 1e-4:
            st.success("共轭梯度和傅里叶闭式解一致。二者解的是同一条法方程 (AᵀA + λ DᵀD) f = Aᵀg。")

    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.markdown(
        """
**MAP 和维纳滤波什么时候像，为什么一般又不同。**

高斯后验的均值就是 MAP，所以「维纳滤波 = 后验均值」和「MAP」并不冲突。
它们的公式不同，是因为先验不同。

- 常数 $K$ 的维纳滤波对应白高斯先验，惩罚的是 $\\lVert f\\rVert^2$。频域里每个频率加同一个常数。
- 本实验的 MAP 惩罚的是 $\\lVert Df\\rVert^2$，也就是鼓励相邻像素接近。它的「K」是 $\\lambda(|D_x|^2+|D_y|^2)$，低频几乎不罚，高频罚得很重。
- 只有把维纳公式中的常数 $K$ 换成这个随频率变化的项，两者才变成同一个滤波器。
- 负对数后验里的 $1/2$ 会消掉。本实验的 $\\lambda$ 对应 $\\sigma_n^2/\\sigma_f^2$。再除以像素数 $N$ 不改变最小值，只改变梯度尺度。

傅里叶闭式解成立，是因为周期卷积和周期差分都是循环矩阵，能被同一个傅里叶基对角化。
边界若改成反射或零填充，就没有这道闭式解，那时矩阵自由共轭梯度仍然适用。
"""
    )
    with st.expander("实验 2：在这张 PSNR 曲线上选 λ"):
        st.markdown(
            """
固定模糊和噪声，看曲线而不是只看一张图。
1. 把 σ 固定为 2，噪声固定为 0.03。
2. 拖动左侧 log₁₀(λ)。虚线移动时，图像、PSNR 和数据保真一起变化。
3. λ = 10⁻⁵ 时看噪声是否还在；λ = 10 时看边缘是否被抹平。
4. 把逆滤波阈值从 10⁻⁶ 调到 10⁻²，看硬截断怎样换取稳定。
5. 把维纳 K 调到接近某个常数，再和梯度 MAP 比较。两者可以都好看，但目标函数 J 一般只有傅里叶 MAP 最小。
"""
        )
