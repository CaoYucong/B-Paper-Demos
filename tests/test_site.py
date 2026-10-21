"""Public directory addresses."""

from __future__ import annotations

import asyncio

from starlette.requests import Request

from src.ui.site import lab_frame_src, serve_page
from src.ui.routes import parse_path, site_url


def _request(path: str, query: str = "") -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": query.encode(),
            "headers": [],
            "client": ("127.0.0.1", 123),
            "server": ("127.0.0.1", 8501),
        }
    )


def _get(path: str, query: str = ""):
    return asyncio.run(serve_page(_request(path, query)))


def test_directories_keep_a_trailing_slash_before_the_hash():
    route = parse_path("home/B14/ImageProcessing/Lecture")
    assert site_url(route) == "/home/B14/ImageProcessing/Lecture/"
    assert lab_frame_src(route).startswith("/_lab/?embed=true&path=")


def test_root_and_short_paths_redirect_to_the_directory():
    assert _get("/").headers["location"] == "/home/"
    assert _get("/", "path=home/B14/ImageProcessing/Lecture").headers["location"] == (
        "/home/B14/ImageProcessing/Lecture/"
    )
    moved = _get("/home/b14/imageprocessing/lecture")
    assert moved.status_code == 308
    assert moved.headers["location"] == "/home/B14/ImageProcessing/Lecture/"


def test_signal_lecture_directory_is_a_laboratory_frame():
    response = _get("/home/B14/SignalAnalysis/Lecture/")
    assert response.status_code == 200
    page = response.body.decode()
    assert 'src="/_lab/?embed=true&amp;path=home%2FB14%2FSignalAnalysis%2FLecture"' in page


def test_lecture_directory_is_the_laboratory_frame():
    response = _get("/home/B14/ImageProcessing/Lecture/")
    assert response.status_code == 200
    page = response.body.decode()
    assert 'src="/_lab/?embed=true&amp;path=home%2FB14%2FImageProcessing%2FLecture"' in page
    assert 'id="lab"' in page


def test_topic_directory_links_to_lecture_and_sheet():
    page = _get("/home/B14/ImageProcessing/").body.decode()
    assert "href='/home/B14/ImageProcessing/Lecture/'" in page
    assert "href='/home/B14/ImageProcessing/Sheet/'" in page
    sheet = _get("/home/B14/SignalAnalysis/Sheet/")
    assert sheet.status_code == 200
    assert "reserved" in sheet.body.decode().lower()


def test_unknown_directory_is_not_found():
    response = _get("/home/B99/")
    assert response.status_code == 404
    assert "Page not found" in response.body.decode()
