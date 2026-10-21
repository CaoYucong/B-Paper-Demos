"""Address paths for the paper menu.

The public address is a directory path, with the section after the hash:

``/home/B14/ImageProcessing/Lecture/#Bayersampling``

``serve.py`` publishes those paths. The lecture experiments still run in
Streamlit, mounted at ``/_lab``, because that server cannot own a nested
pathname itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import unquote

from src.catalog import PAPERS, Module, Paper, module_by_id, paper_by_id

TRACK_LECTURE = "Lecture"
TRACK_SHEET = "Sheet"


@dataclass(frozen=True)
class Route:
    kind: str
    path: str
    paper_id: str | None = None
    module_id: str | None = None


def module_slug(module: Module) -> str:
    return "".join(part.capitalize() for part in module.title_en.split())


def cleaned_path(raw: str | None) -> str:
    return unquote(raw or "").strip().strip("/")


def site_url(route: Route) -> str:
    """Public directory URL. The section hash is appended by the browser."""
    return "/" + route.path.strip("/") + "/"


def home_route() -> Route:
    return Route("home", "home")


def paper_route(paper: Paper) -> Route:
    return Route("paper", f"home/{paper.code}", paper.id)


def topic_route(paper: Paper, module: Module) -> Route:
    return Route("topic", f"home/{paper.code}/{module_slug(module)}", paper.id, module.id)


def track_route(paper: Paper, module: Module, track: str) -> Route:
    segment = TRACK_LECTURE if track == "lecture" else TRACK_SHEET
    kind = "lecture" if track == "lecture" else "sheet"
    return Route(kind, f"home/{paper.code}/{module_slug(module)}/{segment}", paper.id, module.id)


def paper_of(route: Route) -> Paper:
    if route.paper_id is None:
        raise KeyError(route.path)
    return paper_by_id(route.paper_id)


def module_of(route: Route) -> Module:
    if route.paper_id is None or route.module_id is None:
        raise KeyError(route.path)
    return module_by_id(route.paper_id, route.module_id)


def _paper_by_code(code: str) -> Paper | None:
    for paper in PAPERS:
        if paper.code.lower() == code.lower():
            return paper
    return None


def _module_by_slug(paper: Paper, slug: str) -> Module | None:
    folded = slug.lower()
    for module in paper.modules:
        if module_slug(module).lower() == folded or module.id.lower() == folded:
            return module
    return None


def parse_path(raw: str | None) -> Route:
    parts = [part for part in cleaned_path(raw).split("/") if part]
    if not parts:
        return home_route()
    requested = "/".join(parts)
    if parts[0].lower() != "home":
        return Route("missing", requested)
    if len(parts) == 1:
        return home_route()
    paper = _paper_by_code(parts[1])
    if paper is None:
        return Route("missing", requested)
    if len(parts) == 2:
        return paper_route(paper)
    module = _module_by_slug(paper, parts[2])
    if module is None:
        return Route("missing", requested)
    if len(parts) == 3:
        return topic_route(paper, module)
    if len(parts) == 4 and parts[3].lower() == TRACK_LECTURE.lower():
        return track_route(paper, module, "lecture")
    if len(parts) == 4 and parts[3].lower() == TRACK_SHEET.lower():
        return track_route(paper, module, "sheet")
    return Route("missing", requested)
