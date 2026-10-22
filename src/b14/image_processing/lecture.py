"""Scrolling laboratory for B14 Image Processing."""

from __future__ import annotations

import json

import numpy as np
import streamlit as st
from scipy.ndimage import zoom

from src.b14.image_processing.experiments import (
    L1_EXPERIMENTS,
    L2_EXPERIMENTS,
    L3_EXPERIMENTS,
    L4_EXPERIMENTS,
)
from src.b14.image_processing.home import render_course_home
from src.b14.image_processing.lecture1.page import render_lecture1
from src.b14.image_processing.lecture2.page import render_lecture2
from src.b14.image_processing.lecture3.page import render_lecture3
from src.b14.image_processing.lecture4.page import render_lecture4
from src.b14.image_processing.paste_image import paste_image, take_clipboard
from src.b14.image_processing.practice.page import render_practice
from src.b14.image_processing.settings import DEFAULTS
from src.b14.image_processing.synthetic_images import IMAGE_BUILDERS, load_grayscale_image, make_image
from src.ui.i18n import forget_translation_keys, language_switch, plain, t
from src.ui.navigation import _contents, _leave, _scroll_to_hash
from src.ui.routes import Route, module_of, paper_of, topic_route

SOURCES = ("Built-in / 内置", "Upload / 上传", "Clipboard / 剪贴板", "Workspace / 工作区")


def _reset_experiment() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    for key in list(st.session_state.keys()):
        if key.startswith("toy_v_") or key.startswith("sr_") or key in {"_toy_n_applied", "gd_result", "compare_iterative", "gd_snapshot_index"}:
            del st.session_state[key]
    forget_translation_keys("image_name")
    st.rerun()

def _fit(image: np.ndarray, size: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.shape == (size, size):
        return image
    factors = (size / image.shape[0], size / image.shape[1])
    resized = zoom(image, factors, order=1)
    fitted = np.zeros((size, size), dtype=np.float64)
    height = min(size, resized.shape[0])
    width = min(size, resized.shape[1])
    fitted[:height, :width] = resized[:height, :width]
    return fitted

def _workspace_image(size: int) -> tuple[np.ndarray, str]:
    stored = st.session_state.get("workspace_image")
    if stored is None:
        name = DEFAULTS["image_name"]
        st.warning(
            t(
                "The workspace is empty, so the built-in sharp-edge image is shown.",
                "工作区是空的，因此显示内置的锐边渐变。",
            )
        )
        return make_image(name, size), name
    return _fit(stored, size), str(st.session_state.get("workspace_label", "workspace"))

def current_image() -> tuple[np.ndarray, str]:
    """Image shared by every lecture. Upload and workspace are resized without clipping the source array."""
    size = int(st.session_state["image_size"])
    source = st.session_state["lab_image_source"]
    if source == "Upload / 上传":
        upload = st.session_state.get("lab_upload")
        name = DEFAULTS["image_name"]
        if upload is None:
            return make_image(name, size), name
        try:
            return load_grayscale_image(upload.getvalue(), size), upload.name
        except Exception as exc:
            st.error(t(f"Could not read the file: {exc}", f"无法读取文件：{exc}"))
            return make_image(name, size), name
    if source == "Clipboard / 剪贴板":
        raw = st.session_state.get("lab_clipboard_bytes")
        name = DEFAULTS["image_name"]
        if not isinstance(raw, (bytes, bytearray)) or not raw:
            st.warning(
                t(
                    "No image has been pasted, so the built-in sharp-edge image is shown.",
                    "还没有粘贴图像，因此显示内置的锐边渐变。",
                )
            )
            return make_image(name, size), name
        try:
            return load_grayscale_image(bytes(raw), size), "Clipboard / 剪贴板"
        except Exception as exc:
            st.error(t(f"Could not read the pasted image: {exc}", f"无法读取粘贴的图像：{exc}"))
            return make_image(name, size), name
    if source == "Workspace / 工作区":
        return _workspace_image(size)
    return make_image(st.session_state["image_name"], size), str(st.session_state["image_name"])

def _image_groups():
    return (
        ("home", "Overview", "总览", ()),
        ("l1", "Lecture 1", "第 1 讲", L1_EXPERIMENTS),
        ("l2", "Lecture 2", "第 2 讲", L2_EXPERIMENTS),
        ("l3", "Lecture 3", "第 3 讲", L3_EXPERIMENTS),
        ("l4", "Lecture 4", "第 4 讲", L4_EXPERIMENTS),
        ("practice", "Practice", "练习", ()),
    )

def _clipboard_source() -> None:
    payload = paste_image(
        prompt=plain(
            "Click this box, then paste an image with ⌘V or Ctrl+V. You can also drop an image onto it.",
            "点这个方框，再用 ⌘V 或 Ctrl+V 粘贴图像。也可以把图像拖到方框上。",
        ),
        button=plain("Read the clipboard", "读取剪贴板"),
        empty=plain("The clipboard has no image.", "剪贴板里没有图像。"),
        blocked=plain(
            "The browser blocked a direct read. Click the box and press ⌘V or Ctrl+V.",
            "浏览器拦截了直接读取。点方框后按 ⌘V 或 Ctrl+V。",
        ),
        done=plain("Pasted.", "已粘贴。"),
        key="lab_clipboard",
    )
    taken = take_clipboard(payload, st.session_state.get("lab_clipboard_token"))
    if taken is not None:
        token, raw = taken
        st.session_state["lab_clipboard_token"] = token
        st.session_state["lab_clipboard_bytes"] = raw
    if isinstance(st.session_state.get("lab_clipboard_bytes"), (bytes, bytearray)):
        st.caption(t("A pasted image is in use.", "正在使用粘贴的图像。"))
        if st.button("Clear pasted image / 清除粘贴的图像", key="clear_clipboard"):
            st.session_state.pop("lab_clipboard_bytes", None)
            token = payload.get("token") if isinstance(payload, dict) else None
            if isinstance(token, str) and token:
                st.session_state["lab_clipboard_token"] = token
            st.rerun()

def _sidebar(route: Route) -> None:
    paper = paper_of(route)
    module = module_of(route)
    with st.sidebar:
        language_switch()
        st.header(t("B14 · Image Processing", "B14 · 图像处理"))
        st.caption(t("B14 Image Processing Lab", "B14 图像处理实验"))
        _leave(f"← {module.title_en} / {module.title_zh}", topic_route(paper, module), "back-modules")
        _contents(_image_groups())
        st.divider()
        st.markdown(f"**{t('Image workspace', '图像工作区')}**")
        st.selectbox("Resolution / 分辨率", [64, 128, 256], key="image_size")
        st.radio("Source / 来源", SOURCES, key="lab_image_source")
        if st.session_state["lab_image_source"] == "Built-in / 内置":
            st.selectbox("Built-in image / 内置图像", list(IMAGE_BUILDERS), key="image_name")
        elif st.session_state["lab_image_source"] == "Upload / 上传":
            st.file_uploader("PNG or JPEG", type=["png", "jpg", "jpeg"], key="lab_upload")
        elif st.session_state["lab_image_source"] == "Clipboard / 剪贴板":
            _clipboard_source()
        st.number_input("Random seed / 随机种子", min_value=0, max_value=100000, step=1, key="seed")
        st.caption("Original, observed, and reconstructed images are labelled separately in each experiment. / 每个实验会分开标出原图、观测和重建。")
        if st.button("Reset experiment values / 恢复实验参数", key="reset_defaults"):
            _reset_experiment()
        if st.button("Clear workspace / 清空工作区", key="clear_workspace"):
            st.session_state.pop("workspace_image", None)
            st.session_state.pop("workspace_label", None)
            st.rerun()
        payload = {key: st.session_state.get(key) for key in list(DEFAULTS) + ["lab_image_source"]}
        st.download_button(
            "Export parameters / 导出参数",
            data=json.dumps(payload, indent=2, default=str),
            file_name="b14_image_lab_parameters.json",
            mime="application/json",
            key="export_parameters",
        )

def render_lab(route: Route) -> None:
    _scroll_to_hash()
    _sidebar(route)
    image, image_label = current_image()
    render_course_home()
    render_lecture1(image, image_label)
    render_lecture2(image, image_label)
    render_lecture3(image, image_label)
    render_lecture4(image, image_label)
    render_practice()

