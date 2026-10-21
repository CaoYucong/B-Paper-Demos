"""Address paths stay stable across a browser refresh."""

from __future__ import annotations

from src.catalog import PAPERS
from src.ui.experiments import L1_EXPERIMENTS, L2_EXPERIMENTS, L3_EXPERIMENTS, L4_EXPERIMENTS, section_anchor
from src.ui.routes import home_route, paper_route, parse_path, site_url, topic_route, track_route


def test_blank_path_opens_home():
    assert parse_path(None) == home_route()
    assert parse_path("") == home_route()
    assert parse_path("/") == home_route()


def test_every_topic_has_lecture_and_sheet_paths():
    for paper in PAPERS:
        assert parse_path(paper_route(paper).path) == paper_route(paper)
        for module in paper.modules:
            assert parse_path(topic_route(paper, module).path) == topic_route(paper, module)
            for track in ("lecture", "sheet"):
                route = track_route(paper, module, track)
                assert parse_path(route.path) == route
                assert parse_path(route.path.lower()) == route


def test_bayer_anchor_matches_the_address_example():
    item = next(item for item in L1_EXPERIMENTS if item[0] == "bayer")
    assert section_anchor(item) == "Bayersampling"
    route = parse_path("home/B14/ImageProcessing/Lecture")
    assert route.kind == "lecture"
    assert route.path == "home/B14/ImageProcessing/Lecture"
    assert site_url(route) + "#" + section_anchor(item) == "/home/B14/ImageProcessing/Lecture/#Bayersampling"


def test_section_anchors_are_unique():
    anchors = [
        section_anchor(item)
        for group in (L1_EXPERIMENTS, L2_EXPERIMENTS, L3_EXPERIMENTS, L4_EXPERIMENTS)
        for item in group
    ]
    assert len(anchors) == len(set(anchors))
    assert all(anchors)


def test_unknown_path_is_kept_for_the_not_found_page():
    route = parse_path("home/B14/Nope")
    assert route.kind == "missing"
    assert route.path == "home/B14/Nope"


def test_signal_analysis_sheet_is_a_distinct_page():
    route = parse_path("home/B14/SignalAnalysis/Sheet")
    assert route.kind == "sheet"
    assert route.module_id == "signal-processing"
    assert route.path == "home/B14/SignalAnalysis/Sheet"
    lecture = parse_path("home/B14/SignalAnalysis/Lecture")
    assert lecture.kind == "lecture"
    assert lecture.path == "home/B14/SignalAnalysis/Lecture"
