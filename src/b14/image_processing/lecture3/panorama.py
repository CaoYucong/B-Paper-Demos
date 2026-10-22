"""Show how the four courtyard photographs land on the registered plane.

Each placement picture warps one more frame onto that plane. The warp is
the alignment of these photographs with the registered panorama. It is an
illustration of the placement, not the non-examinable sketch solution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

FRAME_HEIGHT = 72
FRAME_WIDTH = 64


def facade_scene(height: int, width: int) -> np.ndarray:
    """A wide synthetic facade. It is not the photograph on the slide."""
    height = int(height)
    width = int(width)
    image = np.zeros((height, width), dtype=np.float64)
    sky = max(1, height // 5)
    ground = max(sky + 2, height - height // 5)
    image[:sky] = np.linspace(0.82, 0.95, sky, dtype=np.float64)[:, None]
    image[sky:ground] = 0.58
    image[ground:] = 0.42
    path_top = ground + max(1, (height - ground) // 3)
    image[path_top:, width // 5 : 4 * width // 5] = 0.50
    top = sky + max(2, height // 12)
    bottom = max(top + 1, ground - max(2, height // 12))
    for left in range(6, max(6, width - 6), 16):
        image[top:bottom, left : left + 7] = 0.12
        image[top : top + 2, left : left + 7] = 0.90
    tower = width // 3
    image[1:ground, tower : tower + 8] = 0.30
    image[top:bottom, tower + 2 : tower + 6] = 0.75
    return np.clip(image, 0.0, 1.0)


def frame_origins(n_frames: int, shift: float) -> np.ndarray:
    """Left edge of each frame on the scene axis. Frame 0 starts at 0."""
    return float(shift) * np.arange(int(n_frames), dtype=np.float64)


def sample_positions(origin: float, width: int) -> np.ndarray:
    """Scene coordinate of sample j in one frame: x = origin + j."""
    return float(origin) + np.arange(int(width), dtype=np.float64)


def canvas_width(n_frames: int, frame_width: int, shift: float) -> int:
    origins = frame_origins(n_frames, shift)
    end = float(origins[-1] + int(frame_width) - 1)
    return int(np.ceil(end - 1e-9)) + 1


def sample_frame(scene: np.ndarray, origin: float, width: int) -> np.ndarray:
    """Bilinear sample of ``width`` columns starting at ``origin``."""
    scene = np.asarray(scene, dtype=np.float64)
    columns = scene.shape[1]
    xs = sample_positions(origin, width)
    x0 = np.floor(xs + 1e-9).astype(np.int64)
    frac = np.clip(xs - x0, 0.0, 1.0)
    left = np.clip(x0, 0, columns - 1)
    right = np.clip(x0 + 1, 0, columns - 1)
    return (1.0 - frac) * scene[:, left] + frac * scene[:, right]


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.float64) / 255.0


def load_example_frames() -> list[np.ndarray]:
    """The four photographs used as the panorama frames, in display order."""
    directory = Path(__file__).resolve().parent / "panorama_frames"
    return [_load_rgb(directory / f"frame_{index}.png") for index in range(1, 5)]


def load_placement_steps() -> list[np.ndarray]:
    """Canvas after warping frame 1, then frames 1–2, 1–3, and 1–4."""
    directory = Path(__file__).resolve().parent / "panorama_frames"
    return [_load_rgb(directory / f"place_{index}.png") for index in range(1, 5)]


def load_registered_panorama() -> np.ndarray:
    """The four frames after registration onto one plane.

    The blank corners are canvas that no warped frame covers. This is the
    registered picture, not a horizontal shift of the four frames.
    """
    return _load_rgb(Path(__file__).resolve().parent / "panorama_frames" / "registered.png")


def stitch(frames: list[np.ndarray], origins: np.ndarray) -> np.ndarray:
    """Lay each frame on the canvas and average where deposits overlap.

    A sample at continuous position x splits linearly between floor(x)
    and the next pixel. An integer position stays on that one pixel.
    Colour frames are placed one channel at a time, with the same weights.
    """
    if not frames:
        raise ValueError("stitch needs at least one frame.")
    sample = np.asarray(frames[0])
    if sample.ndim == 3:
        channels = [
            stitch([np.asarray(frame)[..., channel] for frame in frames], origins)
            for channel in range(sample.shape[-1])
        ]
        return np.stack(channels, axis=-1)
    height, width = frames[0].shape
    origins = np.asarray(origins, dtype=np.float64)
    if origins.shape != (len(frames),):
        raise ValueError("each frame needs one origin.")
    end = float(np.max(origins + width - 1))
    canvas_w = int(np.ceil(end - 1e-9)) + 1
    total = np.zeros((height, canvas_w), dtype=np.float64)
    weight = np.zeros((height, canvas_w), dtype=np.float64)
    for frame, origin in zip(frames, origins):
        frame = np.asarray(frame, dtype=np.float64)
        if frame.shape != (height, width):
            raise ValueError("all frames must share one shape.")
        xs = sample_positions(float(origin), width)
        x0 = np.floor(xs + 1e-9).astype(np.int64)
        frac = np.clip(xs - x0, 0.0, 1.0)
        total[:, x0] += frame * (1.0 - frac)
        weight[:, x0] += 1.0 - frac
        split = frac > 1e-8
        if np.any(split):
            right = x0[split] + 1
            total[:, right] += frame[:, split] * frac[split]
            weight[:, right] += frac[split]
    covered = weight > 0.0
    out = np.zeros_like(total)
    out[covered] = total[covered] / weight[covered]
    return out


def min_sample_spacing(origins: np.ndarray, frame_width: int) -> float:
    """Smallest gap between distinct registered sample positions."""
    columns = [sample_positions(float(origin), frame_width) for origin in np.asarray(origins, dtype=np.float64)]
    positions = np.unique(np.round(np.concatenate(columns), decimals=6))
    if positions.size < 2:
        return float("nan")
    return float(np.min(np.diff(positions)))
