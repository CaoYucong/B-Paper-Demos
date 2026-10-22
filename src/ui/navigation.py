"""Address and sidebar helpers shared by every laboratory page."""

from __future__ import annotations

import streamlit as st

from src.ui.anchors import section_anchor
from src.ui.i18n import language_choice, t
from src.ui.routes import Route, cleaned_path, parse_path, site_url


def initialise_navigation() -> None:
    st.session_state.setdefault("lab_image_source", "Built-in / 内置")
    if "ui_language" not in st.session_state:
        requested = st.query_params.get("lang")
        st.session_state["ui_language"] = language_choice(requested if isinstance(requested, str) else None)

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

def _contents(groups) -> None:
    st.markdown(f"**{t('On this page', '本页目录')}**")
    st.caption(t("Scroll in lecture order, or jump to a section.", "按讲义顺序往下读，也可以跳到某一节。"))
    for lecture_id, en, zh, experiments in groups:
        st.markdown(f"[{t(en, zh)}](#{lecture_id})")
        for item in experiments:
            st.markdown(f"&nbsp;&nbsp;[{t(item[1], item[2])}](#{section_anchor(item)})")

def _scroll_to_hash() -> None:
    """Keep the address bar on the section hash and scroll there as soon as it exists."""
    st.html(
        """
        <script>
        (function () {
          if (window.__b14ScrollWatch) return;
          window.__b14ScrollWatch = true;
          function sectionHash() {
            var own = window.location.hash || "";
            if (own.length > 1) return own;
            try {
              if (window.top !== window) return window.top.location.hash || "";
            } catch (error) {}
            return "";
          }
          var hash = sectionHash();
          if (window.top !== window) {
            var path = new URLSearchParams(window.location.search).get("path");
            if (path && hash.length > 1) {
              var next = "/" + path.replace(/^\\/+|\\/+$/g, "") + "/" + hash;
              var current = window.top.location.pathname + window.top.location.hash;
              if (current !== next) window.top.history.replaceState(null, "", next);
            }
          }
          if (hash.length < 2) return;
          var id = "";
          try { id = decodeURIComponent(hash.slice(1)); }
          catch (error) { id = hash.slice(1); }
          var started = performance.now();
          function align() {
            var el = document.getElementById(id);
            var scroller = document.querySelector('[data-testid="stMain"]');
            if (!el || !scroller) return false;
            var delta = el.getBoundingClientRect().top - scroller.getBoundingClientRect().top;
            if (Math.abs(delta) > 2) scroller.scrollTop += delta;
            var after = el.getBoundingClientRect().top - scroller.getBoundingClientRect().top;
            return Math.abs(after) <= 24;
          }
          if (align()) return;
          var observer = new MutationObserver(function () { if (align()) observer.disconnect(); });
          if (document.body) observer.observe(document.body, {childList: true, subtree: true});
          var timer = window.setInterval(function () {
            if (align() || performance.now() - started > 20000) {
              window.clearInterval(timer);
              observer.disconnect();
            }
          }, 50);
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )

