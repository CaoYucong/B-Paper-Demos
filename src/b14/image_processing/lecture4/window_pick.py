"""Place a window by clicking or dragging on an image.

``centre`` is a pixel, possibly on the border. ``inside`` is a centre whose
whole window stays on the image. ``origin`` is a top-left corner. ``row``
selects one full row. A neighbour pixel can be dragged as well.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from src.ui.components import png_bytes

_COMPONENT = components.declare_component(
    "b14_median_window",
    path=str(Path(__file__).resolve().parent / "window_pick_component"),
)


def clamp_centre(row: int, col: int, height: int, width: int) -> tuple[int, int]:
    """Keep a window centre on a pixel of an image."""
    height = int(height)
    width = int(width)
    if height < 1 or width < 1:
        raise ValueError("image must have a positive height and width.")
    return (
        min(max(int(row), 0), height - 1),
        min(max(int(col), 0), width - 1),
    )


def clamp_inside(row: int, col: int, height: int, width: int, size: int) -> tuple[int, int]:
    """Keep an odd window's centre where the whole window lies inside the image."""
    height = int(height)
    width = int(width)
    size = int(size)
    if size < 1 or size % 2 == 0 or height < size or width < size:
        raise ValueError("the window must be odd and fit inside the image.")
    half = size // 2
    return (
        min(max(int(row), half), height - half - 1),
        min(max(int(col), half), width - half - 1),
    )


def clamp_origin(row: int, col: int, height: int, width: int, size: int) -> tuple[int, int]:
    """Keep the top-left of a block so the whole block stays inside the image."""
    height = int(height)
    width = int(width)
    size = int(size)
    if size < 1 or height < size or width < size:
        raise ValueError("the block must fit inside the image.")
    return (
        min(max(int(row), 0), height - size),
        min(max(int(col), 0), width - size),
    )


def window_png(image: np.ndarray) -> str:
    """Base64 grayscale PNG whose pixels line up with the image grid."""
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("the neighbourhood image must be two-dimensional.")
    return base64.b64encode(png_bytes(array)).decode("ascii")


def _pair(value: object) -> tuple[int, int] | None:
    if not isinstance(value, dict):
        return None
    row = value.get("row")
    col = value.get("col")
    if isinstance(row, bool) or isinstance(col, bool):
        return None
    if not isinstance(row, (int, float)) or not isinstance(col, (int, float)):
        return None
    return int(row), int(col)


def _clamp_saved(row: int, col: int, height: int, width: int, size: int, anchor: str) -> tuple[int, int]:
    if anchor == "origin":
        return clamp_origin(row, col, height, width, size)
    if anchor == "inside":
        return clamp_inside(row, col, height, width, size)
    if anchor == "row":
        return clamp_centre(row, 0, height, width)
    return clamp_centre(row, col, height, width)


def _fallback(height: int, width: int, anchor: str, default: tuple[int, int] | None) -> tuple[int, int]:
    if default is not None:
        return int(default[0]), int(default[1])
    if anchor == "origin":
        return 0, 0
    return height // 2, width // 2


def _recall(
    key: str,
    height: int,
    width: int,
    size: int,
    anchor: str,
    default: tuple[int, int] | None,
) -> tuple[int, int]:
    fallback = _fallback(height, width, anchor, default)
    saved = _pair(st.session_state.get(key))
    if saved is None:
        saved = _pair(
            {
                "row": st.session_state.get(f"{key}-row", fallback[0]),
                "col": st.session_state.get(f"{key}-col", fallback[1]),
            }
        )
    return _clamp_saved(*(saved or fallback), height, width, size, anchor)


def saved_neighbourhood(
    height: int,
    width: int,
    *,
    size: int,
    key: str,
    anchor: str = "centre",
    default: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Position the picker will use before the picture is drawn."""
    return _recall(key, int(height), int(width), int(size), anchor, default)


def pick_neighbourhood(
    image: np.ndarray,
    *,
    size: int,
    hint: str,
    key: str,
    anchor: str = "centre",
    default: tuple[int, int] | None = None,
    neighbour: tuple[int, int] | None = None,
    max_width: int = 720,
    labelled: bool = False,
) -> tuple[int, int] | tuple[int, int, int, int]:
    """Return the stored pixel. Click places the window; dragging the square moves it.

    Pass ``neighbour`` to also drag a second pixel. The return is then
    ``(row, col, neighbour_row, neighbour_col)``.
    """
    if anchor not in {"centre", "origin", "inside", "row"}:
        raise ValueError("anchor must be centre, origin, inside, or row.")
    height, width = int(image.shape[0]), int(image.shape[1])
    size = 1 if anchor == "row" else int(size)
    row, col = _recall(key, height, width, size, anchor, default)
    q_row, q_col = (0, 0)
    if neighbour is not None:
        raw = st.session_state.get(key)
        if isinstance(raw, dict) and "qRow" in raw and "qCol" in raw:
            q_row, q_col = clamp_centre(int(raw["qRow"]), int(raw["qCol"]), height, width)
        else:
            q_row, q_col = clamp_centre(int(neighbour[0]), int(neighbour[1]), height, width)
    payload = _COMPONENT(
        png=window_png(image),
        rows=height,
        cols=width,
        row=row,
        col=col,
        size=size,
        anchor=anchor,
        hint=hint,
        marker=neighbour is not None,
        labelled=bool(labelled),
        maxWidth=int(max_width),
        qRow=q_row,
        qCol=q_col,
        key=key,
        default=None,
    )
    chosen = _pair(payload)
    if chosen is not None:
        row, col = _clamp_saved(*chosen, height, width, size, anchor)
    if neighbour is not None and isinstance(payload, dict) and "qRow" in payload and "qCol" in payload:
        q_row, q_col = clamp_centre(int(payload["qRow"]), int(payload["qCol"]), height, width)
    st.session_state[f"{key}-row"] = row
    st.session_state[f"{key}-col"] = col
    if neighbour is not None:
        return row, col, q_row, q_col
    return row, col
