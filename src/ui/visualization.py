"""Plotly figures and a static Matplotlib cost curve."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import font_manager
from plotly.subplots import make_subplots

from src.ui.i18n import plain, shown_label

_FONT_CANDIDATES = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Arial Unicode MS",
    "Heiti SC",
    "Songti SC",
]


def _use_cjk_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in _FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


_use_cjk_font()


def _base_layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        height=height,
        margin=dict(l=12, r=12, t=46, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


_CMAP = {"Gray": "gray", "RdBu_r": "RdBu_r", "Magma": "magma", "Cividis": "cividis"}


def image_png(
    image: np.ndarray,
    title: str,
    zmin: float = 0.0,
    zmax: float = 1.0,
    colorscale: str = "Gray",
    wide: bool = False,
) -> bytes:
    """Static image for the long scroll page. Plotly heatmaps of every panel exhaust the browser."""
    title = shown_label(title)
    array = np.asarray(image, dtype=np.float64)
    colour = array.ndim == 3 and array.shape[-1] == 3
    if wide:
        aspect = array.shape[1] / max(array.shape[0], 1)
        height_in = 2.2
        width_in = min(9.0, max(3.4, height_in * aspect))
        figure, axis = plt.subplots(figsize=(width_in, height_in), dpi=100)
    else:
        figure, axis = plt.subplots(figsize=(3.4, 3.4), dpi=100)
    if colour:
        axis.imshow(np.clip(array, 0.0, 1.0), vmin=0.0, vmax=1.0, interpolation="nearest")
    else:
        axis.imshow(
            array,
            cmap=_CMAP.get(colorscale, "gray"),
            vmin=zmin,
            vmax=zmax,
            interpolation="nearest",
        )
    axis.set_title(title, fontsize=9)
    axis.set_axis_off()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return buffer.getvalue()


def marked_grid_png(
    image: np.ndarray,
    title: str,
    dots: list[tuple[int, int, str]] | None = None,
    shades: list[tuple[np.ndarray, tuple[float, float, float, float]]] | None = None,
) -> bytes:
    """A pixel grid with optional sample markers. Used for the small super-resolution figures."""
    title = shown_label(title)
    array = np.asarray(image, dtype=np.float64)
    height, width = array.shape
    figure, axis = plt.subplots(figsize=(4.4, 4.4), dpi=110)
    low = float(np.min(array))
    high = float(np.max(array))
    axis.imshow(array, cmap="gray", vmin=min(0.0, low), vmax=max(1.0, high), interpolation="nearest")
    if shades:
        for mask, colour in shades:
            overlay = np.zeros((height, width, 4), dtype=np.float64)
            overlay[np.asarray(mask, dtype=bool)] = colour
            axis.imshow(overlay, interpolation="nearest")
    axis.set_xticks(np.arange(-0.5, width, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, height, 1), minor=True)
    axis.grid(which="minor", color="#66788a", linewidth=0.5)
    axis.set_xticks(np.arange(width))
    axis.set_yticks(np.arange(height))
    axis.tick_params(labelsize=6, length=0)
    if dots:
        grouped: dict[tuple[int, int], list[str]] = {}
        for row, col, colour in dots:
            grouped.setdefault((int(row), int(col)), []).append(colour)
        xs: list[float] = []
        ys: list[float] = []
        colours: list[str] = []
        for (row, col), group in grouped.items():
            for slot, colour in enumerate(group):
                xs.append(col + ((slot % 3) - 1) * 0.22)
                ys.append(row + ((slot // 3) - 1) * 0.22)
                colours.append(colour)
        axis.scatter(xs, ys, s=36, c=colours, marker="o", linewidths=0.5, edgecolors="white", zorder=3)
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(height - 0.5, -0.5)
    axis.set_title(title, fontsize=9)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return buffer.getvalue()


def image_figure(
    image: np.ndarray,
    title: str,
    zmin: float = 0.0,
    zmax: float = 1.0,
    colorscale: str = "Gray",
) -> go.Figure:
    figure = go.Figure(
        data=go.Heatmap(
            z=image,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            hovertemplate=shown_label("行 %{y}<br>列 %{x}<br>值 %{z:.4f}<extra></extra>"),
            colorbar=dict(thickness=12, len=0.75),
        )
    )
    _base_layout(figure, title, 360)
    figure.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    figure.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        autorange="reversed",
        scaleanchor="x",
        scaleratio=1,
    )
    return figure


def display_limits(image: np.ndarray, kind: str = "image") -> tuple[float, float, str]:
    """Choose a display scale. Metrics always use the raw values."""
    finite = np.asarray(image, dtype=np.float64)
    if kind == "residual":
        scale = float(np.percentile(np.abs(finite), 99))
        scale = max(scale, 1e-6)
        return -scale, scale, shown_label("色标按残差的 99% 分位对称设置。")
    if kind == "kernel":
        return float(np.min(finite)), float(np.max(finite)), shown_label("色标覆盖核的最小值到最大值。")
    low = float(np.min(finite))
    high = float(np.max(finite))
    if low >= -0.05 and high <= 1.05:
        return 0.0, 1.0, shown_label("显示范围固定为 [0, 1]。")
    p1, p99 = np.percentile(finite, [1, 99])
    return float(p1), float(p99), plain(
        f"Values leave [0, 1] ({low:.2f} to {high:.2f}); the figure is stretched to the 1%–99% percentiles.",
        f"数值超出 [0, 1]（{low:.2f} 到 {high:.2f}），图按 1%–99% 分位拉伸。",
    )


def loss_figure(iteration: np.ndarray, total: np.ndarray, data: np.ndarray, regularisation: np.ndarray) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=iteration, y=total, name=shown_label("总损失 J(f)"), line=dict(width=2.4)))
    figure.add_trace(go.Scatter(x=iteration, y=data, name=shown_label("数据保真 (1/N)||Af−g||²")))
    figure.add_trace(go.Scatter(x=iteration, y=regularisation, name=shown_label("正则项 λ(1/N)||Df||²")))
    _base_layout(figure, shown_label("损失随迭代的变化"), 380)
    figure.update_layout(legend=dict(orientation="h", y=1.12), hovermode="x unified")
    figure.update_xaxes(title=shown_label("迭代 k"))
    figure.update_yaxes(title=shown_label("损失"))
    return figure


def grad_norm_figure(iteration: np.ndarray, grad_norm: np.ndarray) -> go.Figure:
    figure = go.Figure(go.Scatter(x=iteration, y=grad_norm, name="||∇J||", line=dict(color="#0B3A5B")))
    _base_layout(figure, shown_label("梯度范数"), 300)
    figure.update_xaxes(title=shown_label("迭代 k"))
    figure.update_yaxes(title="||∇J(f_k)||₂", type="log")
    return figure


def loss_matplotlib(iteration: np.ndarray, total: np.ndarray, data: np.ndarray, regularisation: np.ndarray):
    """Static copy of the same cost curves."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=120)
    ax.plot(iteration, total, label=shown_label("总损失 J(f)"), linewidth=2.0, color="#0B3A5B")
    ax.plot(iteration, data, label=shown_label("数据保真"), linewidth=1.6, color="#C4492C")
    ax.plot(iteration, regularisation, label=shown_label("正则项"), linewidth=1.6, color="#2F7D4A")
    ax.set_xlabel(shown_label("迭代 k"))
    ax.set_ylabel(shown_label("损失"))
    ax.set_title(shown_label("MAP 目标函数"))
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def matrix_figure(matrix: np.ndarray, highlight_row: int) -> go.Figure:
    text = [[f"{value:.2f}" for value in row] for row in matrix]
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            hovertemplate=shown_label("输出 i=%{y}<br>输入 j=%{x}<br>A=%{z:.3f}<extra></extra>"),
            colorbar=dict(title=shown_label("权重"), thickness=12),
        )
    )
    _base_layout(figure, shown_label("显式模糊矩阵 A"), 420)
    figure.update_xaxes(title=shown_label("输入像素 j"), dtick=1, constrain="domain")
    figure.update_yaxes(title=shown_label("输出像素 i"), dtick=1, autorange="reversed", scaleanchor="x")
    figure.add_shape(
        type="rect",
        x0=-0.5,
        x1=matrix.shape[1] - 0.5,
        y0=highlight_row - 0.5,
        y1=highlight_row + 0.5,
        line=dict(color="#C4492C", width=3),
        fillcolor="rgba(0,0,0,0)",
    )
    return figure


def vector_bars(values: np.ndarray, title: str, color: str) -> go.Figure:
    labels = [str(i) for i in range(len(values))]
    figure = go.Figure(go.Bar(x=labels, y=values, marker_color=color, hovertemplate=shown_label("像素 %{x}<br>值 %{y:.4f}<extra></extra>")))
    _base_layout(figure, title, 280)
    figure.update_yaxes(range=[-0.05, 1.05], title=shown_label("强度"))
    figure.update_xaxes(title=shown_label("像素下标"))
    return figure


def posterior_figure(
    samples: np.ndarray,
    prior: np.ndarray,
    likelihood: np.ndarray,
    posterior: np.ndarray,
    map_estimate: float,
) -> go.Figure:
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            shown_label("先验 prior  p(f)"),
            shown_label("似然 likelihood  p(g | f)"),
            shown_label("归一化后验 posterior  p(f | g)"),
        ),
    )
    series = (prior, likelihood, posterior)
    colors = ("#0B3A5B", "#C4492C", "#2F7D4A")
    for row, (curve, color) in enumerate(zip(series, colors), start=1):
        figure.add_trace(
            go.Scatter(x=samples, y=curve, mode="lines", line=dict(color=color, width=2.2), showlegend=False),
            row=row,
            col=1,
        )
        figure.add_vline(x=map_estimate, line_dash="dash", line_color="#8A5A00", line_width=1.5, row=row, col=1)
    figure.update_layout(
        height=680,
        margin=dict(l=12, r=12, t=40, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        title=shown_label("虚线是 MAP 估计"),
    )
    figure.update_xaxes(title_text=shown_label("未知量 f"), row=3, col=1)
    return figure


def sweep_figure(
    lambdas: np.ndarray,
    psnr_values: np.ndarray,
    data_values: np.ndarray,
    current_lambda: float,
) -> go.Figure:
    """PSNR can be compared across λ. Total cost J cannot, because λ changes the objective."""
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(x=lambdas, y=psnr_values, name=shown_label("PSNR（相对原图）"), line=dict(color="#0B3A5B", width=2.4)),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(x=lambdas, y=data_values, name=shown_label("解的数据保真项"), line=dict(color="#C4492C", width=2.0)),
        secondary_y=True,
    )
    figure.add_vline(x=current_lambda, line_dash="dash", line_color="#8A5A00")
    figure.update_layout(
        title=shown_label("改变 λ 时，傅里叶 MAP 解的表现"),
        height=380,
        margin=dict(l=12, r=12, t=48, b=12),
        legend=dict(orientation="h", y=1.14),
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    figure.update_xaxes(title="λ", type="log")
    figure.update_yaxes(title_text="PSNR (dB)", secondary_y=False)
    figure.update_yaxes(title_text="(1/N)||Af−g||²", secondary_y=True)
    return figure
