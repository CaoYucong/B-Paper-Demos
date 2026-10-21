"""Publish the laboratory at directory addresses on port 8501.

A section address looks like ``/home/B14/ImageProcessing/Lecture/#Bayersampling``.
The Image Processing lecture is still the Streamlit laboratory, served under
``/_lab`` and shown inside that directory page.
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    import streamlit as st

    from src.ui.site import site_routes

    app = st.App(Path(__file__).with_name("app.py"), routes=site_routes())
    app.run(
        config={
            "server.address": "0.0.0.0",
            "server.port": 8501,
            "server.baseUrlPath": "_lab",
            "server.headless": True,
            "browser.gatherUsageStats": False,
        }
    )


if __name__ == "__main__":
    main()
