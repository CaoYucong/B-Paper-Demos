"""B Paper demonstrations. Image Processing is the first available module."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="B Paper Demonstrations",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.settings import DEFAULTS
from src.ui.shell import (
    active_route,
    initialise_navigation,
    render_lecture,
    render_missing,
    render_modules,
    render_papers,
    render_sheet,
    render_topic,
)

CSS = """
<style>
.block-container {padding-top: 1.05rem; padding-bottom: 2.4rem; max-width: 1180px;}
h1, h2, h3 {scroll-margin-top: 0.75rem;}
h1 {font-weight: 650; letter-spacing: -0.03em; margin-bottom: 0.2rem;}
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #d7e0e8;
    border-radius: 10px;
    padding: 0.45rem 0.7rem;
}
footer {visibility: hidden;}
</style>
"""


def _initialise_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
    initialise_navigation()


def main() -> None:
    from src.ui.i18n import install

    install()
    if sys.version_info < (3, 11):
        st.error("Python 3.11 or newer is required. / 需要 Python 3.11 或更高版本。")
        return
    st.markdown(CSS, unsafe_allow_html=True)
    _initialise_state()
    st.session_state["_fig_seq"] = 0
    route = active_route()
    if route.kind == "home":
        render_papers()
    elif route.kind == "paper":
        render_modules(route)
    elif route.kind == "topic":
        render_topic(route)
    elif route.kind == "lecture":
        render_lecture(route)
    elif route.kind == "sheet":
        render_sheet(route)
    else:
        render_missing(route)


main()
