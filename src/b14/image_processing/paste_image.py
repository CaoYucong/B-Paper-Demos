"""Paste an image from the clipboard into the shared workspace."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT = components.declare_component(
    "b14_paste_image",
    path=str(Path(__file__).resolve().parent / "paste_image_component"),
)


def paste_image(*, prompt: str, button: str, empty: str, blocked: str, done: str, key: str) -> object:
    """Return the latest paste payload, or None before the first paste."""
    return _COMPONENT(prompt=prompt, button=button, empty=empty, blocked=blocked, done=done, key=key, default=None)


def take_clipboard(payload: object, previous_token: str | None) -> tuple[str, bytes] | None:
    """JPEG bytes for a new paste. The same token is ignored so a rerun does not decode again."""
    if not isinstance(payload, dict):
        return None
    token = payload.get("token")
    data = payload.get("jpeg")
    if not isinstance(token, str) or not token or token == previous_token:
        return None
    if not isinstance(data, str) or not data:
        return None
    try:
        raw = base64.b64decode(data, validate=True)
    except Exception:
        return None
    if not raw:
        return None
    return token, raw
