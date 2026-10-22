"""标签页 5：标量高斯逆问题里的 Bayes 定理。"""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.b14.image_processing.lecture3.bayes import (
    grid_map,
    likelihood_density,
    normalised_posterior,
    prior_density,
    scalar_map,
    scalar_posterior_variance,
    suggested_axis,
)
from src.ui.i18n import say
from src.ui.i18n import say
from src.ui.common import show_fig
from src.ui.visualization import posterior_figure


def render_bayes() -> None:
    say(
        r"""
The cost in the image experiment is an instance of Bayes' theorem

$$
p(f\mid g)\propto p(g\mid f)\,p(f).
$$

The MAP is the $f$ that maximises the posterior. A scalar problem is small enough that the densities can be drawn:

$$
g = af + n,\qquad n\sim\mathcal N(0,\sigma_n^2),\qquad f\sim\mathcal N(\mu_f,\sigma_f^2).
$$

This prior acts on $f$ itself. It corresponds to the zeroth-order penalty $\lVert f\rVert^2$ in an image, **not** to the gradient penalty. The Bayesian split is the same: the likelihood comes from the noise, and the prior comes from the structure we are willing to assume.
""",
        r"""
图像实验里的目标函数不是凭空写出来的。Bayes 定理说

$$
p(f\mid g)\propto p(g\mid f)\,p(f).
$$

MAP 就是后验密度最大的那个 $f$。先看一个能把密度真正画出来的标量问题：

$$
g = af + n,\qquad n\sim\mathcal N(0,\sigma_n^2),\qquad f\sim\mathcal N(\mu_f,\sigma_f^2).
$$

这里的先验直接作用在 $f$ 上，对应图像里的零阶惩罚 $\lVert f\rVert^2$，**不是**梯度惩罚。
贝叶斯结构是一样的：似然来自噪声，先验来自我们愿意相信的结构。
""",
    )
    left, right = st.columns(2)
    with left:
        st.slider("观测 g", min_value=-1.0, max_value=3.0, step=0.05, key="bayes_g")
        st.slider("退化系数 a", min_value=0.0, max_value=2.0, step=0.05, key="bayes_a")
        st.slider("噪声标准差 σₙ", min_value=0.05, max_value=2.0, step=0.05, key="bayes_sigma_n")
    with right:
        st.slider("先验均值 μ_f", min_value=-1.0, max_value=2.0, step=0.05, key="bayes_mu")
        st.slider("先验标准差 σ_f", min_value=0.05, max_value=2.0, step=0.05, key="bayes_sigma_f")

    observation = float(st.session_state["bayes_g"])
    coefficient = float(st.session_state["bayes_a"])
    noise_std = float(st.session_state["bayes_sigma_n"])
    prior_mean = float(st.session_state["bayes_mu"])
    prior_std = float(st.session_state["bayes_sigma_f"])
    estimate = scalar_map(observation, coefficient, noise_std, prior_mean, prior_std)
    variance = scalar_posterior_variance(coefficient, noise_std, prior_std)
    samples = suggested_axis(observation, coefficient, noise_std, prior_mean, prior_std)
    prior = prior_density(samples, prior_mean, prior_std)
    likelihood = likelihood_density(samples, observation, coefficient, noise_std)
    posterior = normalised_posterior(samples, observation, coefficient, noise_std, prior_mean, prior_std)
    numerical = grid_map(samples, posterior)
    spacing = float(samples[1] - samples[0])

    noise_var = noise_std**2
    prior_var = prior_std**2
    st.latex(
        rf"""
\hat f_{{\mathrm{{MAP}}}}
= \frac{{a g/\sigma_n^2 + \mu_f/\sigma_f^2}}{{a^2/\sigma_n^2 + 1/\sigma_f^2}}
= \frac{{{coefficient:.3f}\times {observation:.3f}/{noise_var:.4f} + {prior_mean:.3f}/{prior_var:.4f}}}
{{{coefficient ** 2:.4f}/{noise_var:.4f} + 1/{prior_var:.4f}}}
= {estimate:.4f}
"""
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("解析 MAP", f"{estimate:.4f}")
    metric_cols[1].metric("网格搜索最大值", f"{numerical:.4f}")
    metric_cols[2].metric("|解析 − 网格|", f"{abs(estimate - numerical):.4f}")
    metric_cols[3].metric("后验标准差", f"{np.sqrt(variance):.4f}")
    if abs(estimate - numerical) <= 0.5 * spacing + 1e-9:
        st.success(f"网格间距是 {spacing:.4f}。最大值与闭式解的差别不超过半个间距，公式和归一化后验一致。")
    else:
        st.warning("网格最大值和闭式解差得比半个间距还多。请检查归一化或搜索范围。")

    if abs(coefficient) < 1e-8:
        st.info("a = 0 时，观测里没有 f 的信息，似然对 f 是常数，MAP 等于先验均值。")
    else:
        implied = observation / coefficient
        distance_data = abs(estimate - implied)
        distance_prior = abs(estimate - prior_mean)
        extra = st.columns(2)
        extra[0].metric("由观测反推的 g/a", f"{implied:.4f}")
        extra[1].metric("先验均值 μ_f", f"{prior_mean:.4f}")
        if distance_data + 1e-9 < distance_prior:
            st.info(
                "当前 MAP 更接近 g/a。噪声越小，或先验标准差越大，数据就越有发言权，估计会朝观测移动。"
            )
        else:
            st.info(
                "当前 MAP 更接近先验均值。噪声越大，或先验越窄，估计就越不愿意离开先验。"
            )

    show_fig(posterior_figure(samples, prior, likelihood, posterior, estimate))
    st.caption("三条曲线的纵轴各自缩放。橙色虚线是 MAP。似然画的是 p(g|f) 作为 f 的函数，不是作为 g 的函数。")

    with st.expander("从标量公式走到图像上的优化问题"):
        st.markdown(
            r"""
把标量负对数后验对 $f$ 求导并令其为零，就得到上面的闭式解：

$$
\frac{a(af-g)}{\sigma_n^2}+\frac{f-\mu_f}{\sigma_f^2}=0.
$$

图像的高斯似然同样来自 $n\sim\mathcal N(0,\sigma_n^2 I)$：

$$
p(g\mid f)\propto \exp\left(-\frac{1}{2\sigma_n^2}\lVert g-Af\rVert^2\right).
$$

若先验改成惩罚梯度，

$$
p(f)\propto \exp\left(-\frac{1}{2\sigma_f^2}\lVert Df\rVert^2\right),
$$

负对数后验（略去常数）就是

$$
\frac{1}{2\sigma_n^2}\lVert Af-g\rVert^2+\frac{1}{2\sigma_f^2}\lVert Df\rVert^2.
$$

乘上 $2\sigma_n^2$ 之后，最小值不变，得到 $\lVert Af-g\rVert^2+\lambda\lVert Df\rVert^2$，其中 $\lambda=\sigma_n^2/\sigma_f^2$。
再除以像素数 $N$ 仍然不改变最小值，只把梯度缩小 $N$ 倍，所以第 3 页的学习率要按 $1/N$ 理解。
周期边界让这个二次函数能在每个频率上单独配方，那就是第 4 页的傅里叶解；标量公式则是只有一个未知数时的同一件事。
梯度下降不需要配方，它只反复使用 $\nabla J$。四件事因此连在一起：

$$
g=Af+n,\qquad
p(f\mid g)\propto p(g\mid f)\,p(f),\qquad
\hat f=\arg\min J(f),\qquad
f_{k+1}=f_k-\eta\nabla J(f_k).
$$
"""
        )
