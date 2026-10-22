"""Address-bar hashes shared by both laboratories."""

from __future__ import annotations

import re

def section_anchor(item: tuple[str, str, str]) -> str:
    """Hash used in the address bar. ``Bayer sampling`` becomes ``Bayersampling``."""
    return re.sub(r"[^A-Za-z0-9]", "", item[1]) or item[0]


def option(item: tuple[str, str, str]) -> str:
    return f"{item[1]} / {item[2]}"


def option_list(items: tuple[tuple[str, str, str], ...]) -> list[str]:
    return [option(item) for item in items]


def experiment_id(items: tuple[tuple[str, str, str], ...], selected: str) -> str:
    for item in items:
        if option(item) == selected:
            return item[0]
    return items[0][0]
