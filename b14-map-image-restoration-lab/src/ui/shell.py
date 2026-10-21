"""Paper, module, and Image Processing navigation."""

from __future__ import annotations

import json

import numpy as np
import streamlit as st
from scipy.ndimage import zoom

from src.catalog import PAPERS
from src.settings import DEFAULTS
from src.synthetic_images import IMAGE_BUILDERS, load_grayscale_image, make_image
from src.ui.experiments import (
    L1_EXPERIMENTS,
    L2_EXPERIMENTS,
    L3_EXPERIMENTS,
    L4_EXPERIMENTS,
    section_anchor,
)
from src.ui.i18n import forget_translation_keys, language_switch, say, t
from src.ui.home import render_course_home
from src.ui.lecture1_page import render_lecture1
from src.ui.lecture2_page import render_lecture2
from src.ui.lecture3_page import render_lecture3
from src.ui.lecture4_page import render_lecture4
from src.ui.practice_page import render_practice
from src.ui.routes import (
    Route,
    cleaned_path,
    home_route,
    module_of,
    paper_of,
    paper_route,
    parse_path,
    site_url,
    topic_route,
    track_route,
)

SOURCES = ("Built-in / 内置", "Upload / 上传", "Workspace / 工作区")


def initialise_navigation() -> None:
    st.session_state.setdefault("lab_image_source", "Built-in / 内置")
    st.session_state.setdefault("ui_language", "both")


def active_route() -> Route:
    """Read the page from the address bar. A refresh uses the same path."""
    raw = str(st.query_params.get("path", ""))
    route = parse_path(raw)
    if cleaned_path(raw) != route.path:
        st.query_params["path"] = route.path
        st.rerun()
    return route


def open_route(route: Route) -> None:
    st.query_params["path"] = route.path
    st.rerun()


def _leave(label: str, route: Route, key: str) -> None:
    """Back control. Inside the public lecture frame, this leaves to the directory URL."""
    if st.context.is_embedded:
        st.markdown(
            f"<p><a href='{site_url(route)}' target='_top'>{label}</a></p>",
            unsafe_allow_html=True,
        )
        return
    if st.button(label, key=key):
        open_route(route)


def _reset_experiment() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    for key in list(st.session_state.keys()):
        if key.startswith("toy_v_") or key in {"_toy_n_applied", "gd_result", "compare_iterative", "gd_snapshot_index"}:
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
    if source == "Workspace / 工作区":
        return _workspace_image(size)
    return make_image(st.session_state["image_name"], size), str(st.session_state["image_name"])


def render_papers() -> None:
    language_switch()
    st.title("B Paper Demonstrations")
    st.caption("Choose a paper, then a module. Image Processing is the module available now. / 先选课程，再选模块。目前可用的是图像处理。")
    columns = st.columns(3)
    for column, paper in zip(columns, PAPERS):
        with column:
            with st.container(border=True):
                st.subheader(paper.code)
                st.markdown(t(paper.title_en, paper.title_zh))
                say(paper.summary_en, paper.summary_zh)
                if st.button("Open / 进入", key=f"open-paper-{paper.id}", use_container_width=True):
                    open_route(paper_route(paper))


def render_modules(route: Route) -> None:
    language_switch()
    paper = paper_of(route)
    _leave("← Home / 返回首页", home_route(), "back-papers")
    st.title(paper.code)
    st.caption(t(paper.title_en, paper.title_zh))
    if not paper.modules:
        st.info(
            "This paper is reserved. Its title and modules will be added with that course. "
            "这门课仍是占位，名称和模块会在加入时再写。"
        )
        return
    columns = st.columns(2)
    for column, module in zip(columns, paper.modules):
        with column:
            with st.container(border=True):
                st.subheader(t(module.title_en, module.title_zh))
                say(module.summary_en, module.summary_zh)
                if st.button("Open / 进入", key=f"open-module-{module.id}", use_container_width=True):
                    open_route(topic_route(paper, module))


def render_topic(route: Route) -> None:
    language_switch()
    paper = paper_of(route)
    module = module_of(route)
    _leave(f"← {paper.code} / 返回课程", paper_route(paper), "back-paper")
    st.title(f"{paper.code} · {t(module.title_en, module.title_zh)}")
    say(module.summary_en, module.summary_zh)
    tracks = (
        (
            "lecture",
            "Lecture",
            "讲义",
            "Lecture notes and the interactive experiments.",
            "讲义和交互实验。",
            module.status == "available",
        ),
        (
            "sheet",
            "Sheet",
            "习题",
            "Example sheet. Reserved for now.",
            "习题纸。暂时预留。",
            False,
        ),
    )
    columns = st.columns(2)
    for column, (track, title_en, title_zh, body_en, body_zh, available) in zip(columns, tracks):
        with column:
            with st.container(border=True):
                st.subheader(t(title_en, title_zh))
                say(body_en, body_zh)
                label = "Open / 进入" if available else "View placeholder / 查看占位"
                if st.button(label, key=f"open-track-{track}", use_container_width=True):
                    open_route(track_route(paper, module, track))


def render_planned(route: Route) -> None:
    language_switch()
    paper = paper_of(route)
    module = module_of(route)
    _leave(f"← {module.title_en} / {module.title_zh}", topic_route(paper, module), "back-topic-planned")
    st.title(f"{paper.code} · {t(module.title_en, module.title_zh)} · {t('Lecture', '讲义')}")
    st.info(t(module.summary_en, module.summary_zh))
    st.markdown(
        "To add this module later, register it in `src/catalog.py` and give it a render function. "
        "The Image Processing laboratory does not need to be copied."
    )
    st.caption("以后加入这个模块时，在 `src/catalog.py` 注册并提供一个渲染函数即可，不必复制图像处理实验。")


def render_sheet(route: Route) -> None:
    language_switch()
    paper = paper_of(route)
    module = module_of(route)
    _leave(f"← {module.title_en} / {module.title_zh}", topic_route(paper, module), "back-topic-sheet")
    st.title(f"{paper.code} · {t(module.title_en, module.title_zh)} · {t('Sheet', '习题')}")
    st.info(
        t(
            "The example sheet for this topic is reserved. Questions will be added here later.",
            "这个主题的习题纸仍是占位，题目以后加在这里。",
        )
    )


def render_missing(route: Route) -> None:
    language_switch()
    st.title(t("Page not found", "没有这个页面"))
    st.code(route.path)
    _leave("← Home / 返回首页", home_route(), "back-missing")


def _contents() -> None:
    st.markdown(f"**{t('On this page', '本页目录')}**")
    st.caption(t("Scroll in lecture order, or jump to a section.", "按讲义顺序往下读，也可以跳到某一节。"))
    groups = (
        ("home", "Overview", "总览", ()),
        ("l1", "Lecture 1", "第 1 讲", L1_EXPERIMENTS),
        ("l2", "Lecture 2", "第 2 讲", L2_EXPERIMENTS),
        ("l3", "Lecture 3", "第 3 讲", L3_EXPERIMENTS),
        ("l4", "Lecture 4", "第 4 讲", L4_EXPERIMENTS),
        ("practice", "Practice", "练习", ()),
    )
    for lecture_id, en, zh, experiments in groups:
        st.markdown(f"[{t(en, zh)}](#{lecture_id})")
        for item in experiments:
            st.markdown(f"&nbsp;&nbsp;[{t(item[1], item[2])}](#{section_anchor(item)})")


def _sidebar(route: Route) -> None:
    paper = paper_of(route)
    module = module_of(route)
    with st.sidebar:
        language_switch()
        st.header(t("B14 · Image Processing", "B14 · 图像处理"))
        st.caption(t("B14 Image Processing Lab", "B14 图像处理实验"))
        _leave(f"← {module.title_en} / {module.title_zh}", topic_route(paper, module), "back-modules")
        _contents()
        st.divider()
        st.markdown(f"**{t('Image workspace', '图像工作区')}**")
        st.selectbox("Resolution / 分辨率", [64, 128, 256], key="image_size")
        st.radio("Source / 来源", SOURCES, key="lab_image_source")
        if st.session_state["lab_image_source"] == "Built-in / 内置":
            st.selectbox("Built-in image / 内置图像", list(IMAGE_BUILDERS), key="image_name")
        elif st.session_state["lab_image_source"] == "Upload / 上传":
            st.file_uploader("PNG or JPEG", type=["png", "jpg", "jpeg"], key="lab_upload")
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


def _scroll_to_hash() -> None:
    """After a refresh, move to the experiment named in the URL hash."""
    st.html(
        """
        <script>
        (function () {
          if (window.top !== window) {
            const path = new URLSearchParams(window.location.search).get("path");
            if (path && window.location.hash) {
              const next = "/" + path.replace(/^\\/+|\\/+$/g, "") + "/" + window.location.hash;
              const current = window.top.location.pathname + window.top.location.hash;
              if (current !== next) window.top.history.replaceState(null, "", next);
            }
          }
          if (window.__b14ScrollDone) return;
          const hash = window.location.hash;
          if (!hash || hash.length < 2) {
            window.__b14ScrollDone = true;
            return;
          }
          const id = decodeURIComponent(hash.slice(1));
          let tries = 0;
          function go() {
            const el = document.getElementById(id);
            if (el) {
              el.scrollIntoView({block: "start"});
              window.__b14ScrollDone = true;
              return;
            }
            tries += 1;
            if (tries < 30) window.setTimeout(go, 100);
            else window.__b14ScrollDone = true;
          }
          go();
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def render_lab(route: Route) -> None:
    _sidebar(route)
    image, image_label = current_image()
    render_course_home()
    render_lecture1(image, image_label)
    render_lecture2(image, image_label)
    render_lecture3(image, image_label)
    render_lecture4(image, image_label)
    render_practice()
    _scroll_to_hash()


def render_lecture(route: Route) -> None:
    if module_of(route).id == "image-processing":
        render_lab(route)
        return
    render_planned(route)
