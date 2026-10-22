"""Small periodic cameras for the super-resolution teaching sections.

Two forward models share one shift convention. A shift is measured in
low-resolution pixels. Multiplying by the sampling factor lands on a
high-resolution pixel, and the boundary is periodic, as in the MAP section.

Point sampling reads those pixels and nothing else. The realistic model
blurs, averages a square sensor window, and then downsamples. A marker on
that model is the corner of a window, not a claim that the window equals
one high-resolution pixel.

Real images never build the dense matrix. The 1D toy does, so a student
can see ``M = D B T`` with its dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.b14.image_processing.lecture3.operators import (
    apply_otf,
    apply_otf_adjoint,
    gaussian_psf,
    psf_to_otf,
)

COMPLEMENTARY_SHIFTS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.5, 0.0),
    (0.0, 0.5),
    (0.5, 0.5),
)
FRAME_COLOURS: tuple[str, ...] = (
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
)
PATTERN_NAMES: tuple[str, ...] = ("combined", "edge", "diagonal", "checkerboard", "shape")


def make_camera(
    factor: int, horizontal: float, vertical: float, blur_sigma: float = 0.0, realistic: bool = False
) -> Camera:
    """``horizontal`` moves across columns. ``vertical`` moves across rows."""
    return Camera(int(factor), float(vertical), float(horizontal), float(blur_sigma), bool(realistic))


def hr_offset(shift_lr: float, factor: int) -> int:
    """Convert a low-resolution shift into a whole high-resolution pixel."""
    scaled = float(shift_lr) * int(factor)
    rounded = int(np.rint(scaled))
    if abs(scaled - rounded) > 1e-8:
        raise ValueError("the shift must land on a high-resolution pixel.")
    return rounded


@dataclass(frozen=True)
class Camera:
    """One low-resolution view of a periodic high-resolution image."""

    factor: int
    shift_y: float
    shift_x: float
    blur_sigma: float = 0.0
    realistic: bool = False

    def __post_init__(self) -> None:
        if int(self.factor) < 1:
            raise ValueError("the sampling factor must be positive.")
        if float(self.blur_sigma) < 0.0:
            raise ValueError("blur strength cannot be negative.")
        hr_offset(self.shift_y, self.factor)
        hr_offset(self.shift_x, self.factor)

    def offsets(self) -> tuple[int, int]:
        return hr_offset(self.shift_y, self.factor), hr_offset(self.shift_x, self.factor)


def high_resolution_pattern(name: str, size: int = 16) -> np.ndarray:
    """A tiny synthetic scene. It is not a photograph from the lecture."""
    size = int(size)
    if size < 8 or size % 2 != 0:
        raise ValueError("the pattern size must be an even integer of at least 8.")
    if name == "edge":
        image = np.zeros((size, size), dtype=np.float64)
        image[:, size // 2 :] = 1.0
        return image
    if name == "diagonal":
        image = np.full((size, size), 0.15, dtype=np.float64)
        index = np.arange(size)
        image[index, index] = 1.0
        image[index, (index + 1) % size] = 1.0
        return image
    if name == "checkerboard":
        yy, xx = np.indices((size, size))
        return ((yy + xx) % 2 == 0).astype(np.float64)
    if name == "shape":
        image = np.full((size, size), 0.2, dtype=np.float64)
        lo, hi = size // 4, size // 2
        image[lo:hi, lo:hi] = 1.0
        cy, cx, radius = 3.0 * size / 4.0, 3.0 * size / 4.0, max(1.0, size / 8.0)
        yy, xx = np.ogrid[:size, :size]
        image[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2] = 0.0
        return image
    if name == "combined":
        return _combined(size)
    raise ValueError(f"unknown pattern {name!r}.")


def _combined(size: int) -> np.ndarray:
    half = size // 2
    image = np.full((size, size), 0.08, dtype=np.float64)
    image[:half, :half] = 0.15
    image[:half, half // 2 : half] = 0.92
    image[:half, half:] = 0.35
    for row in range(half):
        image[row, half + row] = 1.0
    yy, xx = np.indices((half, half))
    image[half:, :half] = 0.08 + 0.88 * ((yy + xx) % 2)
    image[half:, half:] = 0.22
    cy = cx = half + half / 2.0
    radius = max(1.0, half / 4.0)
    yy_full, xx_full = np.ogrid[:size, :size]
    image[(yy_full - cy) ** 2 + (xx_full - cx) ** 2 <= radius**2] = 0.96
    return image


def low_resolution_shape(shape: tuple[int, int], factor: int) -> tuple[int, int]:
    height, width = int(shape[0]), int(shape[1])
    factor = int(factor)
    if factor < 1 or height % factor or width % factor:
        raise ValueError("the sampling factor must divide both sides of the image.")
    return height // factor, width // factor


def _roll2(image: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    return np.roll(np.roll(np.asarray(image, dtype=np.float64), int(row_shift), axis=0), int(col_shift), axis=1)


def shift_image(image: np.ndarray, row_offset: int, col_offset: int) -> np.ndarray:
    """Move the scene so that the output at (0, 0) reads the input at the offset."""
    return _roll2(image, -int(row_offset), -int(col_offset))


def shift_adjoint(image: np.ndarray, row_offset: int, col_offset: int) -> np.ndarray:
    return _roll2(image, int(row_offset), int(col_offset))


def box_average(image: np.ndarray, factor: int) -> np.ndarray:
    """Mean of the factor-by-factor window that starts at each pixel. Periodic."""
    factor = int(factor)
    image = np.asarray(image, dtype=np.float64)
    if factor <= 1:
        return np.array(image, copy=True)
    total = np.zeros_like(image)
    for row in range(factor):
        for col in range(factor):
            total += _roll2(image, -row, -col)
    return total / float(factor * factor)


def box_average_adjoint(image: np.ndarray, factor: int) -> np.ndarray:
    factor = int(factor)
    image = np.asarray(image, dtype=np.float64)
    if factor <= 1:
        return np.array(image, copy=True)
    total = np.zeros_like(image)
    for row in range(factor):
        for col in range(factor):
            total += _roll2(image, row, col)
    return total / float(factor * factor)


def _gaussian(image: np.ndarray, sigma: float, adjoint: bool) -> np.ndarray:
    if float(sigma) <= 0.0:
        return np.array(image, dtype=np.float64, copy=True)
    otf = psf_to_otf(gaussian_psf(image.shape, float(sigma)))
    if adjoint:
        return apply_otf_adjoint(image, otf)
    return apply_otf(image, otf)


def apply_camera(image: np.ndarray, camera: Camera) -> np.ndarray:
    """Apply ``M = D`` or ``M = D B T``. The result is low-resolution."""
    image = np.asarray(image, dtype=np.float64)
    low_resolution_shape(image.shape, camera.factor)
    row_offset, col_offset = camera.offsets()
    shifted = shift_image(image, row_offset, col_offset)
    if camera.realistic and camera.blur_sigma > 0.0:
        shifted = _gaussian(shifted, camera.blur_sigma, adjoint=False)
    if camera.realistic:
        shifted = box_average(shifted, camera.factor)
    return np.array(shifted[:: camera.factor, :: camera.factor], copy=True)


def apply_camera_adjoint(residual: np.ndarray, shape: tuple[int, int], camera: Camera) -> np.ndarray:
    """Adjoint of ``apply_camera`` for one low-resolution residual."""
    low_h, low_w = low_resolution_shape(shape, camera.factor)
    residual = np.asarray(residual, dtype=np.float64)
    if residual.shape != (low_h, low_w):
        raise ValueError("the residual shape does not match this camera.")
    lifted = np.zeros(shape, dtype=np.float64)
    lifted[:: camera.factor, :: camera.factor] = residual
    if camera.realistic:
        lifted = box_average_adjoint(lifted, camera.factor)
    if camera.realistic and camera.blur_sigma > 0.0:
        lifted = _gaussian(lifted, camera.blur_sigma, adjoint=True)
    row_offset, col_offset = camera.offsets()
    return shift_adjoint(lifted, row_offset, col_offset)


def window_origins(shape: tuple[int, int], camera: Camera) -> list[tuple[int, int]]:
    """Top-left corner, in the original image, of every sensor window."""
    height, width = int(shape[0]), int(shape[1])
    low_h, low_w = low_resolution_shape(shape, camera.factor)
    row_offset, col_offset = camera.offsets()
    origins: list[tuple[int, int]] = []
    for row in range(low_h):
        for col in range(low_w):
            origins.append(
                ((row_offset + row * camera.factor) % height, (col_offset + col * camera.factor) % width)
            )
    return origins


def window_mask(shape: tuple[int, int], camera: Camera) -> np.ndarray:
    """Pixels that a point sample reads, or that a sensor window averages.

    A positive Gaussian blur also mixes pixels outside this mask. The mask
    is the integration window, not the whole Gaussian support.
    """
    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=bool)
    span = camera.factor if camera.realistic else 1
    for row, col in window_origins(shape, camera):
        for d_row in range(span):
            for d_col in range(span):
                mask[(row + d_row) % height, (col + d_col) % width] = True
    return mask


def observe(image: np.ndarray, camera: Camera, noise_std: float, seed: int) -> np.ndarray:
    """One noisy measurement. The noise is not clipped."""
    clean = apply_camera(image, camera)
    if float(noise_std) <= 0.0:
        return clean
    generator = np.random.default_rng(int(seed))
    return clean + float(noise_std) * generator.standard_normal(clean.shape)


def observation_seed(base_seed: int, index: int, camera: Camera) -> int:
    row_offset, col_offset = camera.offsets()
    return int(base_seed) + 1009 * (int(index) + 1) + 17 * row_offset + col_offset


def spread_observations(
    shape: tuple[int, int], cameras: list[Camera], observations: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Average the adjoint of each observation, weighted by how often a pixel is used.

    Point sampling puts the measured number on the sampled pixel. Area
    integration spreads that number across the window. Overlaps are averaged.
    """
    if len(cameras) != len(observations):
        raise ValueError("each observation needs one camera.")
    total = np.zeros(shape, dtype=np.float64)
    weight = np.zeros(shape, dtype=np.float64)
    for camera, observation in zip(cameras, observations):
        total += apply_camera_adjoint(observation, shape, camera)
        weight += apply_camera_adjoint(np.ones_like(observation), shape, camera)
    estimate = np.zeros(shape, dtype=np.float64)
    seen = weight > 1e-12
    estimate[seen] = total[seen] / weight[seen]
    return estimate, weight


def cycle_shifts(count: int) -> list[tuple[float, float]]:
    """Repeat the four complementary shifts. Later copies add noise, not new sites."""
    count = int(count)
    if count < 1:
        raise ValueError("at least one frame is required.")
    return [COMPLEMENTARY_SHIFTS[index % 4] for index in range(count)]


def averaging_curve(
    clean: np.ndarray, noise_std: float, counts: tuple[int, ...], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """RMS error of the average of identical noisy copies, and ``σ / sqrt(N)``."""
    clean = np.asarray(clean, dtype=np.float64)
    counts_array = np.asarray(counts, dtype=int)
    if counts_array.size == 0 or np.any(counts_array < 1):
        raise ValueError("frame counts must be positive.")
    generator = np.random.default_rng(int(seed))
    noise = float(noise_std) * generator.standard_normal((int(counts_array.max()),) + clean.shape)
    empirical = np.empty(counts_array.size, dtype=np.float64)
    for index, count in enumerate(counts_array):
        average = clean + noise[: int(count)].mean(axis=0)
        error = average - clean
        empirical[index] = float(np.sqrt(np.mean(error * error)))
    theory = float(noise_std) / np.sqrt(counts_array.astype(np.float64))
    return empirical, theory


def _matrix_from_columns(outputs: int, inputs: int, apply_one) -> np.ndarray:
    matrix = np.zeros((outputs, inputs), dtype=np.float64)
    basis = np.zeros(inputs, dtype=np.float64)
    for column in range(inputs):
        basis[column] = 1.0
        matrix[:, column] = apply_one(basis)
        basis[column] = 0.0
    return matrix


def _kernel_1d(length: int, sigma: float) -> np.ndarray:
    center = length // 2
    axis = np.arange(length)
    distance = np.minimum(np.abs(axis - center), length - np.abs(axis - center)).astype(np.float64)
    kernel = np.exp(-(distance * distance) / (2.0 * float(sigma) ** 2))
    total = float(kernel.sum())
    if total <= 0.0:
        raise ValueError("the 1D Gaussian kernel vanished.")
    return kernel / total


def convolve_centered_1d(vector: np.ndarray, kernel: np.ndarray, adjoint: bool = False) -> np.ndarray:
    spectrum = np.fft.fft(np.fft.ifftshift(np.asarray(kernel, dtype=np.float64)))
    if adjoint:
        spectrum = np.conj(spectrum)
    transformed = np.fft.fft(np.asarray(vector, dtype=np.float64)) * spectrum
    return np.fft.ifft(transformed).real


def box_average_1d(vector: np.ndarray, factor: int) -> np.ndarray:
    factor = int(factor)
    vector = np.asarray(vector, dtype=np.float64)
    if factor <= 1:
        return np.array(vector, copy=True)
    total = np.zeros_like(vector)
    for offset in range(factor):
        total += np.roll(vector, -offset)
    return total / float(factor)


def box_average_1d_adjoint(vector: np.ndarray, factor: int) -> np.ndarray:
    factor = int(factor)
    vector = np.asarray(vector, dtype=np.float64)
    if factor <= 1:
        return np.array(vector, copy=True)
    total = np.zeros_like(vector)
    for offset in range(factor):
        total += np.roll(vector, offset)
    return total / float(factor)


def factor_matrices_1d(
    length: int, factor: int, shift_lr: float, blur_sigma: float = 0.0, realistic: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``T`` (n×n), ``B`` (n×n), ``D`` (m×n), and ``M = D B T`` (m×n)."""
    length = int(length)
    factor = int(factor)
    if factor < 1 or length % factor:
        raise ValueError("the factor must divide the 1D length.")
    offset = hr_offset(shift_lr, factor)
    kernel = _kernel_1d(length, blur_sigma) if realistic and blur_sigma > 0.0 else None

    def apply_translation(vector: np.ndarray) -> np.ndarray:
        return np.roll(vector, -offset)

    def apply_blur(vector: np.ndarray) -> np.ndarray:
        out = np.asarray(vector, dtype=np.float64)
        if kernel is not None:
            out = convolve_centered_1d(out, kernel, adjoint=False)
        if realistic and factor > 1:
            out = box_average_1d(out, factor)
        return out

    def apply_down(vector: np.ndarray) -> np.ndarray:
        return np.asarray(vector, dtype=np.float64)[::factor]

    translation = _matrix_from_columns(length, length, apply_translation)
    blur = _matrix_from_columns(length, length, apply_blur)
    down = _matrix_from_columns(length // factor, length, apply_down)
    return translation, blur, down, down @ blur @ translation


def apply_operator_1d(
    vector: np.ndarray, factor: int, shift_lr: float, blur_sigma: float = 0.0, realistic: bool = False
) -> np.ndarray:
    _translation, _blur, _down, matrix = factor_matrices_1d(
        vector.size, factor, shift_lr, blur_sigma, realistic
    )
    return matrix @ np.asarray(vector, dtype=np.float64)


@dataclass(frozen=True)
class LinearReport:
    measurements: int
    unknowns: int
    rank: int
    full_column_rank: bool
    more_equations_than_unknowns: bool
    least_squares: np.ndarray
    pseudoinverse: np.ndarray
    fit_mse: float
    reconstruction_mse: float


def report_stacked_system(matrix: np.ndarray, measurements: np.ndarray, truth: np.ndarray) -> LinearReport:
    """Least squares and the Moore–Penrose solution of one stacked toy system.

    ``truth`` is used only to score the reconstruction. The solves see
    ``matrix`` and ``measurements``.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    measurements = np.asarray(measurements, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    least_squares = np.linalg.lstsq(matrix, measurements, rcond=None)[0]
    pseudoinverse = np.linalg.pinv(matrix, rcond=1e-8) @ measurements
    rank = int(np.linalg.matrix_rank(matrix, tol=1e-8))
    residual = matrix @ least_squares - measurements
    return LinearReport(
        measurements=int(matrix.shape[0]),
        unknowns=int(matrix.shape[1]),
        rank=rank,
        full_column_rank=rank == matrix.shape[1],
        more_equations_than_unknowns=matrix.shape[0] > matrix.shape[1],
        least_squares=least_squares,
        pseudoinverse=pseudoinverse,
        fit_mse=float(np.mean(residual * residual)),
        reconstruction_mse=float(np.mean((least_squares - truth) ** 2)),
    )
