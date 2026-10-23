"""Streamlit 显示小工具。不在导入时绘制任何控件。"""

from __future__ import annotations

import hashlib

import numpy as np
import streamlit as st

from src.ui.i18n import colour_notes, shown_label, t
from src.ui.visualization import display_limits, image_png


def show_fig(fig) -> None:
    # 同一轮脚本里每张图都要有不同的 key。原图会在多个标签页重复出现。
    sequence = int(st.session_state.get("_fig_seq", 0)) + 1
    st.session_state["_fig_seq"] = sequence
    st.plotly_chart(fig, config={"displaylogo": False}, key=f"auto-fig-{sequence}")


def show_image(column, image: np.ndarray, title: str, kind: str = "image") -> None:
    zmin, zmax, _note = display_limits(image, kind)
    if kind == "residual":
        colorscale = "RdBu_r"
    elif kind == "kernel":
        colorscale = "Cividis"
    elif kind == "otf":
        colorscale = "Magma"
    else:
        colorscale = "Gray"
    with column:
        st.image(
            image_png(image, shown_label(title), zmin=zmin, zmax=zmax, colorscale=colorscale),
            width="stretch",
        )
        low = float(np.min(image))
        high = float(np.max(image))
        note_en, note_zh = colour_notes(kind, low, high)
        column.caption(t(f"{note_en} min {low:.3f}, max {high:.3f}.", f"{note_zh} 最小 {low:.3f}，最大 {high:.3f}。"))


def render_images(items: list[tuple[str, np.ndarray, str]], columns_per_row: int = 3) -> None:
    for start in range(0, len(items), columns_per_row):
        batch = items[start : start + columns_per_row]
        columns = st.columns(len(batch))
        for column, (title, image, kind) in zip(columns, batch):
            show_image(column, image, title, kind)


def digest_array(array: np.ndarray) -> str:
    payload = np.ascontiguousarray(array).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def banner(label: str, shape: tuple[int, int], sigma: float, noise_std: float, seed: int, lam: float) -> None:
    name = shown_label(label)
    st.caption(
        t(
            f"Experiment: {name} · {shape[0]}×{shape[1]} · blur σ = {sigma:.2f} · noise σ = {noise_std:.3f} · seed = {seed} · λ = {lam:.4g}",
            f"当前实验：{name} · {shape[0]}×{shape[1]} · blur σ = {sigma:.2f} · noise σ = {noise_std:.3f} · seed = {seed} · λ = {lam:.4g}",
        )
    )
