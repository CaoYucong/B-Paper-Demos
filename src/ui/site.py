"""Public directory pages.

The browser address is ``/home/B14/ImageProcessing/Lecture/#Bayersampling``.
Menu pages are rendered here. The Image Processing lecture frame loads the
Streamlit laboratory from ``/_lab``, where a nested pathname cannot be used.
"""

from __future__ import annotations

import html
from urllib.parse import quote

from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.routing import Route

from src.catalog import PAPERS
from src.ui.routes import Route as LabRoute
from src.ui.routes import home_route, module_of, paper_of, paper_route, parse_path, site_url, topic_route, track_route

LAB_PREFIX = "/_lab"

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%20100%20100'%3E%3Ctext%20x='50%25'%20y='54%25'%20text-anchor='middle'%20dominant-baseline='central'%20font-size='82'%3E%F0%9F%94%AC%3C/text%3E%3C/svg%3E">
<style>
:root {
  color-scheme: light;
  --ink: #1B242C;
  --muted: #4d5c68;
  --line: #d7e0e8;
  --paper: #F6F4F0;
  --card: #ffffff;
  --wash: #E7EEF3;
  --accent: #0B3A5B;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font: 16px/1.5 "Source Sans 3", "Source Sans Pro", "Segoe UI", sans-serif;
}
main { max-width: 1080px; margin: 0 auto; padding: 1.4rem 1.2rem 3rem; }
a { color: var(--accent); }
h1 { font-size: 2rem; line-height: 1.15; margin: 0.2rem 0; letter-spacing: -0.03em; }
.back { display: inline-block; margin-bottom: 0.8rem; text-decoration: none; }
.lede, .zh { margin: 0.15rem 0 1rem; }
.zh { color: var(--muted); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.9rem; }
.card {
  display: block;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 1rem 1rem 0.9rem;
  text-decoration: none;
  color: inherit;
}
.card:hover { border-color: var(--accent); }
.card h2 { margin: 0 0 0.35rem; font-size: 1.25rem; }
.card p { margin: 0.2rem 0; }
.note {
  background: var(--wash);
  border-radius: 10px;
  padding: 0.8rem 1rem;
}
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
</style>
</head>
<body>
<main>
__BODY__
</main>
</body>
</html>
"""

_LECTURE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%20100%20100'%3E%3Ctext%20x='50%25'%20y='54%25'%20text-anchor='middle'%20dominant-baseline='central'%20font-size='82'%3E%F0%9F%94%AC%3C/text%3E%3C/svg%3E">
<style>
html, body { margin: 0; height: 100%; background: #F6F4F0; overflow: hidden; }
iframe { display: block; width: 100%; height: 100%; border: 0; }
html[data-pending] iframe { opacity: 0; pointer-events: none; }
#opening {
  display: none;
  position: fixed;
  inset: 0;
  margin: 0;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  color: #4d5c68;
  font: 16px/1.5 "Source Sans 3", "Source Sans Pro", "Segoe UI", sans-serif;
  opacity: 0;
}
html[data-pending] #opening {
  display: flex;
  animation: b14-opening 0s 350ms forwards;
}
@keyframes b14-opening { to { opacity: 1; } }
html[data-lang="en"] #opening .zh,
html[data-lang="zh"] #opening .en { display: none; }
</style>
<script>
(function () {
  var pinned = window.location.hash || "";
  window.__b14Pin = pinned;
  var lang = "";
  try { lang = localStorage.getItem("b14-language") || ""; } catch (error) {}
  if (lang !== "zh" && lang !== "en") lang = "both";
  window.__b14Lang = lang;
  document.documentElement.setAttribute("data-lang", lang);
  if (pinned.length > 1) document.documentElement.setAttribute("data-pending", "");
  function keep(url) {
    pinned = window.__b14Pin || "";
    if (!pinned || url == null) return url;
    var next = new URL(String(url), window.location.href);
    if (next.pathname !== window.location.pathname) return url;
    if (!next.hash) next.hash = pinned;
    return next.pathname + next.search + next.hash;
  }
  var replace = history.replaceState.bind(history);
  var push = history.pushState.bind(history);
  history.replaceState = function (state, title, url) {
    return replace(state, title, url === undefined ? url : keep(url));
  };
  history.pushState = function (state, title, url) {
    return push(state, title, url === undefined ? url : keep(url));
  };
})();
</script>
</head>
<body>
<iframe id="lab" title="__TITLE__" data-src="__SRC__"></iframe>
<p id="opening"><span class="en">Opening this section…</span><span class="zh">正在打开这一节…</span></p>
<script>
(function () {
  var frame = document.getElementById("lab");
  var base = frame.getAttribute("data-src");
  var lang = window.__b14Lang || "";
  if (lang === "zh" || lang === "en" || lang === "both") {
    base += (base.indexOf("?") === -1 ? "?" : "&") + "lang=" + lang;
  }
  frame.src = base + (window.__b14Pin || "");

  function sectionId() {
    var hash = window.__b14Pin || window.location.hash || "";
    if (hash.length < 2) return "";
    try { return decodeURIComponent(hash.slice(1)); }
    catch (error) { return hash.slice(1); }
  }

  function align() {
    var id = sectionId();
    if (!id) return true;
    var doc;
    try { doc = frame.contentDocument; }
    catch (error) { return false; }
    if (!doc) return false;
    var el = doc.getElementById(id);
    var scroller = doc.querySelector('[data-testid="stMain"]');
    if (!el || !scroller) return false;
    var delta = el.getBoundingClientRect().top - scroller.getBoundingClientRect().top;
    if (Math.abs(delta) > 2) scroller.scrollTop += delta;
    var after = el.getBoundingClientRect().top - scroller.getBoundingClientRect().top;
    var maxScroll = scroller.scrollHeight - scroller.clientHeight;
    var stuckAtEnd = maxScroll > 0 && scroller.scrollTop >= maxScroll - 2 && after > 24;
    return Math.abs(after) <= 24 || stuckAtEnd;
  }

  var stable = 0;
  var revealed = !document.documentElement.hasAttribute("data-pending");
  var revealedAt = revealed ? performance.now() : 0;
  var started = performance.now();
  var userMoved = false;

  function reveal() {
    if (revealed) return;
    revealed = true;
    revealedAt = performance.now();
    document.documentElement.removeAttribute("data-pending");
  }

  function tick(countStable) {
    if (userMoved) return;
    if (revealed && !countStable) return;
    var ready = align();
    if (revealed || !countStable) return;
    if (ready) stable += 1;
    else stable = 0;
    if (stable >= 3 || performance.now() - started > 20000) reveal();
  }

  var lastMirrored = window.__b14Pin || "";

  function mirror() {
    var hash = "";
    try { hash = frame.contentWindow.location.hash; }
    catch (error) { return; }
    if (!hash) return;
    if (hash !== window.location.hash) {
      window.__b14Pin = hash;
      history.replaceState(null, "", window.location.pathname + window.location.search + hash);
    }
    if (hash === lastMirrored) return;
    lastMirrored = hash;
    userMoved = false;
    var n = 0;
    var follow = window.setInterval(function () {
      align();
      n += 1;
      if (n > 12) window.clearInterval(follow);
    }, 50);
  }

  frame.addEventListener("load", function () {
    var doc;
    try { doc = frame.contentDocument; }
    catch (error) { doc = null; }
    if (doc) {
      new MutationObserver(function () { tick(false); }).observe(doc.documentElement, {childList: true, subtree: true});
      doc.addEventListener("wheel", function () { if (revealed) userMoved = true; }, {passive: true, capture: true});
      doc.addEventListener("touchstart", function () { if (revealed) userMoved = true; }, {passive: true, capture: true});
    }
    tick(true);
    window.setInterval(mirror, 200);
  });

  var hold = window.setInterval(function () {
    if (userMoved) { window.clearInterval(hold); return; }
    tick(true);
    if (revealed && performance.now() - revealedAt > 1600) window.clearInterval(hold);
  }, 50);
})();
</script>
</body>
</html>
"""


def lab_frame_src(route: LabRoute) -> str:
    return f"{LAB_PREFIX}/?embed=true&path={quote(route.path, safe='')}"


def _page(title: str, body: str) -> str:
    return _PAGE.replace("__TITLE__", html.escape(title)).replace("__BODY__", body)


def _text(en: str, zh: str) -> str:
    return f"<p class='lede'>{html.escape(en)}</p><p class='zh'>{html.escape(zh)}</p>"


def _card(href: str, title_en: str, title_zh: str, body_en: str, body_zh: str, action: str) -> str:
    return (
        f"<a class='card' href='{html.escape(href)}'>"
        f"<h2>{html.escape(title_en)}</h2>"
        f"<p>{html.escape(title_zh)}</p>"
        f"<p>{html.escape(body_en)}</p>"
        f"<p class='zh'>{html.escape(body_zh)}</p>"
        f"<p><strong>{html.escape(action)}</strong></p>"
        "</a>"
    )


def _back(href: str, label: str) -> str:
    return f"<a class='back' href='{html.escape(href)}'>{html.escape(label)}</a>"


def menu_document(route: LabRoute) -> str:
    if route.kind == "home":
        cards = []
        for paper in PAPERS:
            cards.append(
                _card(
                    site_url(paper_route(paper)),
                    paper.code,
                    f"{paper.title_en} / {paper.title_zh}",
                    paper.summary_en,
                    paper.summary_zh,
                    "Open / 进入",
                )
            )
        body = (
            "<h1>B Paper Demonstrations</h1>"
            + _text(
                "Choose a paper, then a module. Image Processing and Signal Analysis are available.",
                "先选课程，再选模块。图像处理和信号分析都可以使用。",
            )
            + f"<div class='grid'>{''.join(cards)}</div>"
        )
        return _page("B Paper Demonstrations", body)

    if route.kind == "paper":
        paper = paper_of(route)
        body = _back(site_url(home_route()), "← Home / 返回首页")
        body += f"<h1>{html.escape(paper.code)}</h1>" + _text(paper.title_en, paper.title_zh)
        if not paper.modules:
            body += (
                "<div class='note'><p>This paper is reserved. Its title and modules will be added with that course.</p>"
                "<p class='zh'>这门课仍是占位，名称和模块会在加入时再写。</p></div>"
            )
        else:
            cards = [
                _card(
                    site_url(topic_route(paper, module)),
                    module.title_en,
                    module.title_zh,
                    module.summary_en,
                    module.summary_zh,
                    "Open / 进入",
                )
                for module in paper.modules
            ]
            body += f"<div class='grid'>{''.join(cards)}</div>"
        return _page(paper.code, body)

    if route.kind == "topic":
        paper = paper_of(route)
        module = module_of(route)
        lecture = track_route(paper, module, "lecture")
        sheet = track_route(paper, module, "sheet")
        lecture_action = "Open / 进入" if module.status == "available" else "View placeholder / 查看占位"
        body = _back(site_url(paper_route(paper)), f"← {paper.code} / 返回课程")
        body += f"<h1>{html.escape(paper.code)} · {html.escape(module.title_en)}</h1>"
        body += _text(module.summary_en, module.summary_zh)
        body += "<div class='grid'>" + _card(
            site_url(lecture),
            "Lecture",
            "讲义",
            "Lecture notes and the interactive experiments.",
            "讲义和交互实验。",
            lecture_action,
        ) + _card(
            site_url(sheet),
            "Sheet",
            "习题",
            "Example sheet. Reserved for now.",
            "习题纸。暂时预留。",
            "View placeholder / 查看占位",
        ) + "</div>"
        return _page(f"{paper.code} · {module.title_en}", body)

    if route.kind == "lecture":
        paper = paper_of(route)
        module = module_of(route)
        body = _back(site_url(topic_route(paper, module)), f"← {module.title_en} / {module.title_zh}")
        body += f"<h1>{html.escape(paper.code)} · {html.escape(module.title_en)} · Lecture</h1>"
        body += _text(module.summary_en, module.summary_zh)
        body += (
            "<div class='note'><p>To add this module later, register it in <code>src/catalog.py</code> "
            "and give it a render function. The Image Processing laboratory does not need to be copied.</p>"
            "<p class='zh'>以后加入这个模块时，在 src/catalog.py 注册并提供一个渲染函数即可，不必复制图像处理实验。</p></div>"
        )
        return _page(f"{paper.code} · {module.title_en} · Lecture", body)

    if route.kind == "sheet":
        paper = paper_of(route)
        module = module_of(route)
        body = _back(site_url(topic_route(paper, module)), f"← {module.title_en} / {module.title_zh}")
        body += f"<h1>{html.escape(paper.code)} · {html.escape(module.title_en)} · Sheet</h1>"
        body += (
            "<div class='note'><p>The example sheet for this topic is reserved. Questions will be added here later.</p>"
            "<p class='zh'>这个主题的习题纸仍是占位，题目以后加在这里。</p></div>"
        )
        return _page(f"{paper.code} · {module.title_en} · Sheet", body)

    body = _back(site_url(home_route()), "← Home / 返回首页")
    body += "<h1>Page not found</h1>" + _text("There is no page at this address.", "没有这个页面。")
    body += f"<p><code>{html.escape(route.path)}</code></p>"
    return _page("Page not found", body)


def lecture_document(route: LabRoute) -> str:
    module = module_of(route)
    title = f"{module.title_en} · Lecture"
    return (
        _LECTURE.replace("__TITLE__", html.escape(title))
        .replace("__SRC__", html.escape(lab_frame_src(route), quote=True))
    )


def _redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=308, headers={"Cache-Control": "no-cache"})


async def serve_page(request: Request) -> Response:
    if request.url.path in {"", "/"}:
        legacy = request.query_params.get("path")
        if legacy:
            route = parse_path(legacy)
            if route.kind != "missing":
                return _redirect(site_url(route))
        return _redirect(site_url(home_route()))

    route = parse_path(request.url.path)
    if route.kind != "missing" and request.url.path != site_url(route):
        return _redirect(site_url(route))
    if route.kind == "lecture" and module_of(route).status == "available":
        return HTMLResponse(lecture_document(route), headers={"Cache-Control": "no-cache"})
    status = 404 if route.kind == "missing" else 200
    return HTMLResponse(menu_document(route), status_code=status, headers={"Cache-Control": "no-cache"})


def site_routes() -> list[Route]:
    return [
        Route("/", serve_page),
        Route("/home", serve_page),
        Route("/home/{rest:path}", serve_page),
    ]
