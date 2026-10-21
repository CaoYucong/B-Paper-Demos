"""标签页 3：用梯度下降最小化 MAP 目标。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.degradation import Observation
from src.map_estimation import run_gradient_descent, snapshot_image, step_size_from_fraction
from src.metrics import format_db, mse, psnr
from src.operators import apply_otf
from src.settings import INIT_TO_CODE
from src.ui.i18n import say, t
from src.ui_common import banner, digest_array, render_images, show_fig
from src.visualization import display_limits, grad_norm_figure, image_png, loss_figure, loss_matplotlib

_CACHED_FRAMES = 16


def _signature(observation: Observation, lam: float) -> tuple:
    return (
        digest_array(observation.observed),
        digest_array(np.ascontiguousarray(observation.otf)),
        round(float(lam), 12),
        round(float(st.session_state["lr_fraction"]), 6),
        int(st.session_state["n_iters"]),
        INIT_TO_CODE[st.session_state["init_mode"]],
        bool(st.session_state["use_projection"]),
    )


def render_map(observation: Observation, lam: float) -> None:
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
Maximum a posteriori (MAP) estimation, with Gaussian noise and a Gaussian smoothness prior, is the minimum of this quadratic:

$$
J(f)=\frac{1}{N}\lVert Af-g\rVert^2+\lambda\frac{1}{N}\lVert Df\rVert^2
$$

$$
\nabla J(f)=\frac{2}{N}\left[A^{T}(Af-g)+\lambda D^{T}Df\right]
$$

$$
f_{k+1}=f_k-\eta\nabla J(f_k)
$$

$N$ is the number of pixels. The original $f$ does **not** enter the optimisation; it is used only afterwards for MSE and PSNR. $D$ is the periodic forward difference and $D^T$ is its exact adjoint. λ is shared with the comparison page. Because the cost is divided by $N$, the gradient carries $2/N$, so the numerical value of $\eta$ is comparatively large.
""",
        r"""
最大后验（MAP, Maximum A Posteriori）在高斯噪声和高斯平滑先验下，变成下面这个二次函数的最小值。
$N$ 是像素个数。原图 $f$ **不**进入优化，只在事后计算 MSE 和 PSNR。

$$
J(f)=\frac{1}{N}\lVert Af-g\rVert^2+\lambda\frac{1}{N}\lVert Df\rVert^2
$$

$$
\nabla J(f)=\frac{2}{N}\left[A^{T}(Af-g)+\lambda D^{T}Df\right]
$$

$$
f_{k+1}=f_k-\eta\nabla J(f_k)
$$

$D$ 是周期前向差分，$D^T$ 是它的精确伴随。λ 与对比页使用同一个值。
损失除以 $N$ 之后，梯度里带有 $2/N$，所以 $\eta$ 的数值会比较大。
""",
    )
    fraction = float(st.session_state["lr_fraction"])
    step = step_size_from_fraction(observation.otf, lam, fraction)
    control_left, control_right = st.columns(2)
    with control_left:
        st.slider(
            "学习率比例  s = η / (1/L)",
            min_value=0.05,
            max_value=2.5,
            step=0.05,
            key="lr_fraction",
            help="L 是 ∇J 的 Lipschitz 常数。s = 1 表示 η = 1/L；s < 2 才有无约束收敛保证。",
        )
        st.slider("迭代次数", min_value=10, max_value=1000, step=10, key="n_iters")
    with control_right:
        st.selectbox("初始化", list(INIT_TO_CODE), key="init_mode")
        st.checkbox(
            "每步把像素裁剪到 [0, 1]。这是投影梯度下降，不是普通梯度下降",
            key="use_projection",
        )

    metric_cols = st.columns(4)
    metric_cols[0].metric("谱半径 ρ", f"{step.spectral_radius:.4f}")
    metric_cols[1].metric("Lipschitz 常数 L", f"{step.lipschitz:.4e}")
    metric_cols[2].metric("实际学习率 η", f"{step.eta:.4g}")
    metric_cols[3].metric("等价小步长 α = η/N", f"{step.alpha:.4g}")
    st.caption(
        "周期算子在傅里叶基下对角化，所以 ρ = max(|H|² + λ(|Dₓ|²+|Dᵧ|²)) 是精确的，不是估计。"
        f" 稳定条件是 0 < η < 2/L = {step.stable_limit:.4g}，也就是 s < 2。"
        " 另一种完全等价的更新是 f ← f − 2α [Aᵀ(Af−g) + λ DᵀDf]，最小值不变。"
        " 本页按带 2/N 的梯度实现，没有把平均损失和未平均的梯度混用。"
    )
    if fraction >= 2.0:
        st.error(
            "s ≥ 2，已经超过「对所有频率都收敛」的充分条件。直流分量曲率最大，而高斯模糊不改变平均值，"
            "所以 s 只是略大于 2 时损失可能仍下降。把 s 调到 3 以上，通常能看到损失上升或数值发散。推荐值是 s = 1。"
        )
    elif fraction > 1.5:
        st.warning("s 接近稳定边界。若损失曲线上升，请减小 s。")
    if lam < 1e-4:
        st.info("λ 非常小，最小值接近逆滤波，噪声会被放大。")
    elif lam > 2:
        st.info("λ 很大，解会被拉向常数图像，边缘发糊。这是过正则。")
    if st.session_state["use_projection"]:
        st.warning("裁剪打开后，迭代一般不会停在无约束傅里叶解上。下面的损失仍是无约束的 J，所以不一定单调下降。")

    if st.button("运行优化", type="primary"):
        n_steps = int(st.session_state["n_iters"])
        progress = st.progress(0.0, text=t(f"Iteration 0 / {n_steps}", f"第 0 / {n_steps} 步"))
        frame = st.empty()

        def _show_progress(step: int, total: int, image: np.ndarray, saved: bool) -> None:
            progress.progress(step / total if total else 1.0, text=t(f"Iteration {step} / {total}", f"第 {step} / {total} 步"))
            if saved:
                low, high, _note = display_limits(image, "image")
                frame.image(
                    image_png(image, t(f"Iteration {step}", f"第 {step} 步"), zmin=low, zmax=high),
                    width="stretch",
                )

        result = run_gradient_descent(
            observation.observed,
            observation.otf,
            lam,
            n_iterations=n_steps,
            step_fraction=float(st.session_state["lr_fraction"]),
            initialisation=INIT_TO_CODE[st.session_state["init_mode"]],
            project=bool(st.session_state["use_projection"]),
            max_snapshots=_CACHED_FRAMES,
            on_iterate=_show_progress,
        )
        progress.progress(1.0, text=t(f"Finished {int(result.iteration[-1])} iterations", f"完成 {int(result.iteration[-1])} 步"))
        st.session_state["gd_result"] = {"signature": _signature(observation, lam), "result": result}

    saved = st.session_state.get("gd_result")
    if saved is None:
        st.info(t(
            "Set λ, the step size, and the iteration count, then press Run. The bar advances with the iteration, and 16 frames are kept.",
            "设置好 λ、学习率和迭代次数后，点击「运行优化」。进度条会跟着迭代走，并留下 16 张图像。",
        ))
        return

    result = saved["result"]
    if saved["signature"] != _signature(observation, lam):
        st.warning("左侧参数或本页的优化设置已经改变。下面仍是上一次运行的结果，请重新点击按钮。")

    if result.diverged:
        st.error("迭代出现非有限数值，已停在最后一个有限的图像。请把学习率比例降到 2 以下。")
    elif (not result.projected) and result.total_loss[-1] > result.total_loss[0] + 1e-8:
        st.error("损失比起点更高。这一步长对当前的 L 不稳定。")
    elif result.projected and result.total_loss[-1] > result.total_loss[0] + 1e-8:
        st.warning("投影之后，无约束目标 J 没有下降。可以关掉裁剪，对照普通梯度下降。")
    else:
        st.success(
            f"完成 {int(result.iteration[-1])} 步。J 从 {result.total_loss[0]:.4e} 降到 {result.total_loss[-1]:.4e}。"
            " 初始化只决定起点：用 g 通常更快，用全零则说明算法并不需要先看到答案。"
        )

    frames = result.snapshot_iterations
    kept = len(frames)
    st.progress(1.0, text=t(f"{kept} cached frames, through iteration {int(frames[-1])}", f"缓存 {kept} 张，直到第 {int(frames[-1])} 步"))
    st.caption(t(
        f"{kept} frames are kept, spread across the run. The estimate below is the last one.",
        f"全程保留 {kept} 张，均匀取自这次运行。下面的估计是最后一张。",
    ))
    per_row = 4
    for start in range(0, kept, per_row):
        columns = st.columns(per_row)
        for column, index in zip(columns, range(start, min(start + per_row, kept))):
            step = int(frames[index])
            picture = snapshot_image(result, index)
            low, high, _note = display_limits(picture, "image")
            with column:
                st.image(
                    image_png(picture, t(f"Iteration {step}", f"第 {step} 步"), zmin=low, zmax=high),
                    width="stretch",
                )
    iteration = int(result.iteration[-1])
    position = len(result.iteration) - 1
    estimate = result.final
    reblurred = apply_otf(estimate, observation.otf)
    residual = observation.observed - reblurred

    stat_cols = st.columns(4)
    stat_cols[0].metric("当前迭代", iteration)
    stat_cols[1].metric("总损失 J", f"{result.total_loss[position]:.4e}")
    stat_cols[2].metric("数据保真", f"{result.data_loss[position]:.4e}")
    stat_cols[3].metric("正则项", f"{result.regularisation_loss[position]:.4e}")
    eval_cols = st.columns(3)
    eval_cols[0].metric("梯度范数 ||∇J||", f"{result.grad_norm[position]:.3e}")
    eval_cols[1].metric("MSE（相对原图，仅评估）", f"{mse(estimate, observation.original):.4e}")
    eval_cols[2].metric("PSNR（仅评估）", format_db(psnr(estimate, observation.original)))
    st.caption("数据保真 + 正则项 = 总损失。MSE 和 PSNR 使用了原图，优化器没有使用。")

    render_images(
        [
            ("原图 f（只用于评估）", observation.original, "image"),
            ("观测 g", observation.observed, "image"),
            ("当前 MAP 估计", estimate, "image"),
        ]
    )
    render_images(
        [
            ("把估计再模糊  A(f̂)", reblurred, "image"),
            ("残差  g − A(f̂)", residual, "residual"),
        ],
        columns_per_row=2,
    )
    show_fig(loss_figure(result.iteration, result.total_loss, result.data_loss, result.regularisation_loss))
    show_fig(grad_norm_figure(result.iteration, result.grad_norm))
    with st.expander("同一条损失曲线的 Matplotlib 静态图"):
        figure = loss_matplotlib(result.iteration, result.total_loss, result.data_loss, result.regularisation_loss)
        st.pyplot(figure)
        plt.close(figure)

    with st.expander("实验 2 和实验 3"):
        st.markdown(
            """
**实验 2：看懂 λ。** 固定 σ = 2、噪声 0.03、种子 0，学习率比例保持 1，迭代 300 或更多。
把左侧 log₁₀(λ) 分别设为 −5、−1.3 和 1（也就是 λ ≈ 10⁻⁵、0.05 和 10）。
λ 太小会留下噪声，λ 太大会把边缘抹掉。第 4 页的 PSNR–λ 曲线可以用精确解把这个现象一次看完。

**实验 3：看懂优化。** 固定 λ，不要改 A 和 g。
- 把 s 设为 0.2，迭代 80：损失下降，但走得慢。
- 把 s 设为 1，迭代 300：这是按 1/L 选取的稳定步长。
- 把 s 设为 4，迭代 40：明显超过稳定上限。损失会急剧上升，或被判定为发散。s 只略大于 2 时不一定立刻爆炸，因为最陡的直流方向上误差通常已经几乎为零。
用进度条回看这 16 张缓存图。估计是逐渐变清楚的，而不是某一步突然知道了原图。
"""
        )
