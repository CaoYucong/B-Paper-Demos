"""Clipboard paste payloads."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from src.b14.image_processing.synthetic_images import load_grayscale_image
from src.b14.image_processing.paste_image import take_clipboard


def _jpeg(value: int) -> str:
    image = Image.new("L", (4, 4), color=value)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_take_clipboard_keeps_only_a_new_jpeg():
    payload = {"token": "1", "jpeg": _jpeg(255)}
    taken = take_clipboard(payload, None)
    assert taken is not None
    token, raw = taken
    assert token == "1"
    assert take_clipboard(payload, token) is None
    assert take_clipboard({"token": "2"}, token) is None
    assert take_clipboard({"token": "2", "jpeg": "not base64!!"}, token) is None
    assert take_clipboard(None, token) is None
    loaded = load_grayscale_image(raw, 8)
    assert loaded.shape == (8, 8)
    assert np.mean(loaded) > 0.9
