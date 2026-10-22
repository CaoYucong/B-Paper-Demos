"""Bilingual layout helpers. English is the heading; Chinese is the caption."""

from __future__ import annotations

import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.ui.anchors import section_anchor
from src.ui.i18n import colour_notes, say, shown_label, t
from src.ui.visualization import display_limits, image_png


def experiment_heading(lecture_id: str, item: tuple[str, str, str]) -> None:
    """One experiment in the vertical page. The anchor stays put when the language changes."""
    slug = section_anchor(item)
    legacy = f"{lecture_id}-{item[0]}"
    anchors = [slug] if slug == legacy else [slug, legacy]
    st.markdown("".join(f"<div id='{anchor}'></div>" for anchor in anchors), unsafe_allow_html=True)
    st.header(t(item[1], item[2]))


def lecture_anchor(lecture_id: str) -> None:
    st.markdown(f"<div id='{lecture_id}'></div>", unsafe_allow_html=True)


def _heading(en: str, zh: str) -> None:
    st.markdown(f"##### {t(en, zh)}")


def concept(en: str, zh: str) -> None:
    _heading("Concept", "概念")
    say(en, zh)


def model(latex: str) -> None:
    _heading("Mathematical model", "数学模型")
    st.latex(latex)


def controls_heading() -> None:
    _heading("Interactive controls", "交互控制")


def visual_heading() -> None:
    _heading("Visualisation", "可视化")


def steps(pairs: list[tuple[str, str]]) -> None:
    _heading("Step by step", "逐步说明")
    for index, (en, zh) in enumerate(pairs, start=1):
        say(f"{index}. {en}", f"{index}. {zh}")


def takeaways(pairs: list[tuple[str, str]]) -> None:
    _heading("Key takeaways", "要点")
    for en, zh in pairs:
        say(f"- {en}", f"- {zh}")


def try_this(en: str, zh: str) -> None:
    st.markdown(
        '<div class="b14-note b14-note-info">' + t("Try this", "试一试") + "<br><br>" + t(en, zh) + "</div>",
        unsafe_allow_html=True,
    )


def show_array(image: np.ndarray, title: str, kind: str = "image", wide: bool = False) -> None:
    zmin, zmax, _note = display_limits(image, kind)
    colorscale = {"residual": "RdBu_r", "kernel": "Cividis", "otf": "Magma"}.get(kind, "Gray")
    array = np.asarray(image, dtype=np.float64)
    st.image(
        image_png(array, shown_label(title), zmin=zmin, zmax=zmax, colorscale=colorscale, wide=wide),
        use_container_width=True,
    )
    low = float(np.min(array))
    high = float(np.max(array))
    note_en, note_zh = colour_notes(kind, low, high)
    st.caption(t(f"{note_en} min {low:.4g}, max {high:.4g}.", f"{note_zh} 最小 {low:.4g}，最大 {high:.4g}。"))


def show_row(items: list[tuple[str, np.ndarray, str]]) -> None:
    columns = st.columns(len(items))
    for column, (title, image, kind) in zip(columns, items):
        with column:
            show_array(image, title, kind)


def value_figure(matrix: np.ndarray, title: str) -> go.Figure:
    matrix = np.asarray(matrix, dtype=np.float64)
    text = [[f"{value:.3g}" for value in row] for row in matrix]
    diverging = float(np.min(matrix)) < -1e-8
    heatmap = dict(
        z=matrix,
        text=text,
        texttemplate="%{text}",
        colorscale="RdBu_r" if diverging else "Gray",
        hovertemplate=shown_label("row %{y}<br>col %{x}<br>value %{z:.6g}<extra></extra>"),
        colorbar=dict(thickness=12, len=0.75),
    )
    if diverging:
        heatmap["zmid"] = 0.0
    figure = go.Figure(data=go.Heatmap(**heatmap))
    figure.update_layout(
        title=dict(text=shown_label(title), font=dict(size=15)),
        height=max(280, 36 * matrix.shape[0] + 80),
        margin=dict(l=12, r=12, t=46, b=12),
    )
    figure.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    return figure


def line_figure(x: np.ndarray, series: list[tuple[str, np.ndarray]], title: str, xlabel: str, ylabel: str) -> go.Figure:
    figure = go.Figure()
    for name, values in series:
        figure.add_trace(go.Scatter(x=x, y=values, mode="lines", name=shown_label(name)))
    figure.update_layout(
        title=dict(text=shown_label(title), font=dict(size=15)),
        height=340,
        margin=dict(l=12, r=12, t=46, b=12),
        xaxis_title=shown_label(xlabel),
        yaxis_title=shown_label(ylabel),
        legend=dict(orientation="h"),
    )
    return figure


def png_bytes(image: np.ndarray) -> bytes:
    """8-bit PNG. Values outside [0, 1] are clipped only inside this file."""
    scaled = np.clip(np.rint(np.asarray(image, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(scaled, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def download_png(image: np.ndarray, filename: str, key: str) -> None:
    st.download_button(
        "Download PNG / 下载 PNG",
        data=png_bytes(image),
        file_name=filename,
        mime="image/png",
        key=key,
    )
    st.caption("The file is 8-bit and clipped to [0, 1]. Calculations above are not clipped. / 文件是 8 位并裁剪到 [0, 1]。上面的计算没有裁剪。")


def save_workspace(image: np.ndarray, label: str, key: str) -> None:
    if st.button("Use as workspace image / 用作工作区图像", key=key):
        st.session_state["workspace_image"] = np.array(image, dtype=np.float64, copy=True)
        st.session_state["workspace_label"] = label
        st.success(
            "Saved. Select Workspace / 工作区 in the sidebar to carry this image to another lecture. "
            "已保存。在侧栏选择工作区后，其他讲会使用这幅图。"
        )


def centre_crop(image: np.ndarray, size: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    size = min(int(size), image.shape[0], image.shape[1])
    y0 = (image.shape[0] - size) // 2
    x0 = (image.shape[1] - size) // 2
    return image[y0 : y0 + size, x0 : x0 + size]
