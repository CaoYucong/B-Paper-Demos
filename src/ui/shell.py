"""Paper and module navigation.

Choosing Lecture opens that module's own laboratory. Image Processing and
Signal Analysis each keep their scrolling page.
"""

from __future__ import annotations

import streamlit as st

from src.b14.image_processing.lecture import render_lab
from src.b14.signal_analysis.lecture import render_signal_lab
from src.catalog import PAPERS
from src.ui.i18n import language_switch, say, t
from src.ui.navigation import _leave, active_route, initialise_navigation, open_route
from src.ui.routes import (
    Route,
    home_route,
    module_of,
    paper_of,
    paper_route,
    topic_route,
    track_route,
)


def render_papers() -> None:
    language_switch()
    st.title("B Paper Demonstrations")
    st.caption("Choose a paper, then a module. Image Processing and Signal Analysis are available. / 先选课程，再选模块。图像处理和信号分析都可以使用。")
    columns = st.columns(3)
    for column, paper in zip(columns, PAPERS):
        with column:
            with st.container(border=True):
                st.subheader(paper.code)
                st.markdown(t(paper.title_en, paper.title_zh))
                say(paper.summary_en, paper.summary_zh)
                if st.button("Open / 进入", key=f"open-paper-{paper.id}", width="stretch"):
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
                if st.button("Open / 进入", key=f"open-module-{module.id}", width="stretch"):
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
                if st.button(label, key=f"open-track-{track}", width="stretch"):
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

def render_lecture(route: Route) -> None:
    module_id = module_of(route).id
    if module_id == "image-processing":
        render_lab(route)
        return
    if module_id == "signal-processing":
        render_signal_lab(route)
        return
    render_planned(route)

