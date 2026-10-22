"""Multi-frame MAP, Huber restoration, and a small 1D blind deblurring demo.

The objective uses the same ``1/N`` normalisation as the single-image MAP
section. ``N`` is the number of high-resolution pixels. Each observation
adds a data term. The original image is accepted only as an optional score
and is not used in an update or a linear solve.

``D`` in the penalty is the periodic forward difference. It is not the
downsampling matrix ``D_i`` inside ``M_i``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from src.b14.image_processing.lecture3.operators import apply_gradient_penalty, gradient_squared_norm
from src.b14.image_processing.lecture3.sampling import Camera, apply_camera, apply_camera_adjoint


def _data_terms(
    estimate: np.ndarray, observations: list[np.ndarray], cameras: list[Camera]
) -> tuple[float, np.ndarray]:
    if len(observations) != len(cameras) or not cameras:
        raise ValueError("at least one camera and one matching observation are required.")
    accumulated = np.zeros_like(estimate, dtype=np.float64)
    energy = 0.0
    for observation, camera in zip(observations, cameras):
        residual = apply_camera(estimate, camera) - observation
        energy += float(np.dot(residual.ravel(), residual.ravel()))
        accumulated += apply_camera_adjoint(residual, estimate.shape, camera)
    return energy, accumulated


def evaluate_multiframe(
    estimate: np.ndarray, observations: list[np.ndarray], cameras: list[Camera], lam: float
) -> tuple[float, float, float, np.ndarray]:
    """Return ``J``, the data term, ``λ`` times the penalty, and ``∇J``.

    ```
    J(f) = Σ_i (1/N) ||M_i f - g_i||² + λ (1/N) ||D f||²
    ∇J(f) = (2/N) [ Σ_i M_iᵀ (M_i f - g_i) + λ Dᵀ D f ]
    ```
    """
    if lam < 0.0:
        raise ValueError("λ cannot be negative.")
    estimate = np.asarray(estimate, dtype=np.float64)
    energy, data_gradient = _data_terms(estimate, observations, cameras)
    regulariser = gradient_squared_norm(estimate)
    pixels = int(estimate.size)
    gradient = (2.0 / pixels) * (data_gradient + float(lam) * apply_gradient_penalty(estimate))
    data_term = energy / pixels
    penalty_term = float(lam) * regulariser / pixels
    return data_term + penalty_term, data_term, penalty_term, gradient


def _normal_apply(
    image: np.ndarray, cameras: list[Camera], lam: float
) -> np.ndarray:
    applied = np.zeros_like(image, dtype=np.float64)
    for camera in cameras:
        applied += apply_camera_adjoint(apply_camera(image, camera), image.shape, camera)
    if lam != 0.0:
        applied += float(lam) * apply_gradient_penalty(image)
    return applied


def operator_spectral_radius(
    shape: tuple[int, int], cameras: list[Camera], lam: float, steps: int = 24
) -> float:
    """Power iteration for the largest eigenvalue of ``Σ MᵀM + λ DᵀD``."""
    generator = np.random.default_rng(0)
    vector = generator.standard_normal(shape)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return 0.0
    vector /= norm
    rayleigh = 0.0
    for _ in range(int(steps)):
        transformed = _normal_apply(vector, cameras, lam)
        rayleigh = float(np.sum(vector * transformed))
        norm = float(np.linalg.norm(transformed))
        if norm < 1e-15:
            return 0.0
        vector = transformed / norm
    return max(rayleigh, 0.0)


def lipschitz_constant(shape: tuple[int, int], cameras: list[Camera], lam: float) -> float:
    """Lipschitz constant of ``∇J``. A step ``η = s/L`` needs ``s < 2`` for this quadratic."""
    pixels = int(shape[0]) * int(shape[1])
    radius = operator_spectral_radius(shape, cameras, lam)
    if radius <= 0.0:
        return 0.0
    return (2.0 / pixels) * radius


def _call_cg(operator: LinearOperator, rhs: np.ndarray, maxiter: int, rtol: float):
    kwargs: dict = {"maxiter": maxiter}
    parameters = inspect.signature(cg).parameters
    if "rtol" in parameters:
        kwargs["rtol"] = rtol
    elif "tol" in parameters:
        kwargs["tol"] = rtol
    return cg(operator, rhs, **kwargs)


def solve_normal_equations(
    observations: list[np.ndarray],
    cameras: list[Camera],
    lam: float,
    shape: tuple[int, int],
    rtol: float = 1e-10,
) -> tuple[np.ndarray, int]:
    """Solve ``(Σ Mᵢᵀ Mᵢ + λ Dᵀ D) f = Σ Mᵢᵀ gᵢ`` without forming the matrix.

    The ``1/N`` and the factor 2 cancel, so this is the critical point of
    ``evaluate_multiframe``. ``info == 0`` means conjugate gradients met the tolerance.
    """
    if lam < 0.0:
        raise ValueError("λ cannot be negative.")
    pixels = int(shape[0]) * int(shape[1])

    def matvec(vector: np.ndarray) -> np.ndarray:
        image = np.reshape(vector, shape)
        return _normal_apply(image, cameras, lam).ravel()

    rhs = np.zeros(shape, dtype=np.float64)
    for observation, camera in zip(observations, cameras):
        rhs += apply_camera_adjoint(observation, shape, camera)
    operator = LinearOperator((pixels, pixels), matvec=matvec, dtype=np.float64)
    solution, info = _call_cg(operator, rhs.ravel(), maxiter=max(80, 4 * pixels), rtol=rtol)
    return np.reshape(np.real(solution), shape), int(info)


def initialise_estimate(
    observations: list[np.ndarray], cameras: list[Camera], shape: tuple[int, int], mode: str
) -> np.ndarray:
    if mode == "zeros":
        return np.zeros(shape, dtype=np.float64)
    if mode == "mean":
        value = float(np.mean([np.mean(observation) for observation in observations]))
        return np.full(shape, value, dtype=np.float64)
    if mode == "backproject":
        estimate, weight = _spread_for_init(observations, cameras, shape)
        seen = weight > 1e-8
        if np.any(seen) and np.any(~seen):
            estimate[~seen] = float(np.mean(estimate[seen]))
        return estimate
    raise ValueError(f"unknown initialisation {mode!r}.")


def _spread_for_init(observations, cameras, shape):
    from src.b14.image_processing.lecture3.sampling import spread_observations

    return spread_observations(shape, cameras, observations)


@dataclass
class MultiFrameDescent:
    images: list[np.ndarray]
    total_loss: np.ndarray
    data_loss: np.ndarray
    regularisation_loss: np.ndarray
    grad_norm: np.ndarray
    reconstruction_mse: np.ndarray
    diverged: bool
    step_size: float
    lipschitz: float
    reference: np.ndarray
    reference_info: int


def run_gradient_descent_multiframe(
    observations: list[np.ndarray],
    cameras: list[Camera],
    lam: float,
    step_fraction: float,
    iterations: int,
    initial: np.ndarray,
    truth: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    reference_info: int = -1,
) -> MultiFrameDescent:
    """Gradient descent on the normalised multi-frame objective.

    ``truth`` is written into the MSE curve and is not read by the update.
    ``reference`` is stored for the page; this function does not compute it.
    """
    if step_fraction <= 0.0:
        raise ValueError("the step fraction must be positive.")
    if int(iterations) < 1:
        raise ValueError("at least one iteration is required.")
    image = np.array(initial, dtype=np.float64, copy=True)
    lipschitz = lipschitz_constant(image.shape, cameras, lam)
    if lipschitz <= 0.0:
        raise ValueError("the Lipschitz estimate must be positive.")
    step_size = float(step_fraction) / lipschitz
    totals: list[float] = []
    data_terms: list[float] = []
    penalties: list[float] = []
    norms: list[float] = []
    errors: list[float] = []
    images: list[np.ndarray] = []
    diverged = False
    for _ in range(int(iterations)):
        total, data_term, penalty, gradient = evaluate_multiframe(image, observations, cameras, lam)
        images.append(image.copy())
        totals.append(total)
        data_terms.append(data_term)
        penalties.append(penalty)
        norms.append(float(np.linalg.norm(gradient)))
        if truth is None:
            errors.append(float("nan"))
        else:
            difference = image - truth
            errors.append(float(np.mean(difference * difference)))
        image = image - step_size * gradient
        if not np.isfinite(total) or not np.isfinite(image).all() or abs(total) > 1e6:
            diverged = True
            break
    if not diverged:
        images.append(image.copy())
        total, data_term, penalty, gradient = evaluate_multiframe(image, observations, cameras, lam)
        totals.append(total)
        data_terms.append(data_term)
        penalties.append(penalty)
        norms.append(float(np.linalg.norm(gradient)))
        if truth is None:
            errors.append(float("nan"))
        else:
            difference = image - truth
            errors.append(float(np.mean(difference * difference)))
        if not np.isfinite(total):
            diverged = True
    stored_reference = np.zeros_like(initial) if reference is None else np.array(reference, copy=True)
    return MultiFrameDescent(
        images=images,
        total_loss=np.asarray(totals, dtype=np.float64),
        data_loss=np.asarray(data_terms, dtype=np.float64),
        regularisation_loss=np.asarray(penalties, dtype=np.float64),
        grad_norm=np.asarray(norms, dtype=np.float64),
        reconstruction_mse=np.asarray(errors, dtype=np.float64),
        diverged=diverged,
        step_size=step_size,
        lipschitz=lipschitz,
        reference=stored_reference,
        reference_info=int(reference_info),
    )


def squared_loss(residual: np.ndarray) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    return 0.5 * residual * residual


def absolute_loss(residual: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(residual, dtype=np.float64))


def huber_loss(residual: np.ndarray, delta: float) -> np.ndarray:
    """Huber loss with quadratic ``r²/2`` inside the threshold ``delta``."""
    if delta <= 0.0:
        raise ValueError("the Huber threshold must be positive.")
    residual = np.asarray(residual, dtype=np.float64)
    magnitude = np.abs(residual)
    quadratic = 0.5 * residual * residual
    linear = float(delta) * (magnitude - 0.5 * float(delta))
    return np.where(magnitude <= float(delta), quadratic, linear)


def huber_derivative(residual: np.ndarray, delta: float) -> np.ndarray:
    if delta <= 0.0:
        raise ValueError("the Huber threshold must be positive.")
    return np.clip(np.asarray(residual, dtype=np.float64), -float(delta), float(delta))


def huber_weights(residual: np.ndarray, delta: float) -> np.ndarray:
    """IRLS weights ``ψ(r) / r``. A large residual is weighted by ``delta / |r|``."""
    residual = np.asarray(residual, dtype=np.float64)
    weights = np.ones_like(residual)
    large = np.abs(residual) > float(delta)
    weights[large] = float(delta) / np.maximum(np.abs(residual[large]), 1e-15)
    return weights


def difference_matrix(length: int) -> np.ndarray:
    """Periodic forward difference, ``(D f)[i] = f[i+1] - f[i]``."""
    length = int(length)
    matrix = -np.eye(length, dtype=np.float64)
    for index in range(length):
        matrix[index, (index + 1) % length] = 1.0
    return matrix


def convolution_matrix_1d(kernel_full: np.ndarray) -> np.ndarray:
    length = int(kernel_full.size)
    matrix = np.zeros((length, length), dtype=np.float64)
    basis = np.zeros(length, dtype=np.float64)
    for column in range(length):
        basis[column] = 1.0
        matrix[:, column] = circular_convolve_1d(basis, kernel_full)
        basis[column] = 0.0
    return matrix


def circular_convolve_1d(signal: np.ndarray, kernel_full: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft(np.fft.ifftshift(np.asarray(kernel_full, dtype=np.float64)))
    return np.fft.ifft(np.fft.fft(np.asarray(signal, dtype=np.float64)) * spectrum).real


def place_kernel(length: int, taps: np.ndarray) -> np.ndarray:
    """Put a short kernel so its centre tap sits at index ``length // 2``."""
    taps = np.asarray(taps, dtype=np.float64)
    if taps.ndim != 1 or taps.size < 1 or taps.size > length:
        raise ValueError("the kernel does not fit in the signal.")
    placed = np.zeros(length, dtype=np.float64)
    start = length // 2 - taps.size // 2
    placed[start : start + taps.size] = taps
    return placed


def corrupted_blur_problem(
    length: int = 32, spike: float = 3.0, noise_std: float = 0.01, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """A periodic 1D signal, its blur matrix, a corrupted observation, and the bad index."""
    length = int(length)
    signal = np.zeros(length, dtype=np.float64)
    signal[length // 5 : length // 2] = 1.0
    signal[2 * length // 3 :] = 0.45
    kernel = place_kernel(length, np.array([0.2, 0.6, 0.2], dtype=np.float64))
    matrix = convolution_matrix_1d(kernel)
    generator = np.random.default_rng(int(seed))
    observed = matrix @ signal + float(noise_std) * generator.standard_normal(length)
    corrupt = length // 3
    observed[corrupt] += float(spike)
    return signal, matrix, observed, corrupt


def restore_quadratic_1d(matrix: np.ndarray, observed: np.ndarray, lam: float) -> np.ndarray:
    difference = difference_matrix(matrix.shape[1])
    normal = matrix.T @ matrix + float(lam) * (difference.T @ difference)
    return np.linalg.solve(normal, matrix.T @ np.asarray(observed, dtype=np.float64))


def restore_huber_1d(
    matrix: np.ndarray, observed: np.ndarray, lam: float, delta: float, iterations: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively reweighted least squares. Returns the estimate and the final weights."""
    observed = np.asarray(observed, dtype=np.float64)
    estimate = restore_quadratic_1d(matrix, observed, lam)
    difference = difference_matrix(matrix.shape[1])
    penalty = float(lam) * (difference.T @ difference)
    weights = np.ones(observed.size, dtype=np.float64)
    for _ in range(int(iterations)):
        weights = huber_weights(matrix @ estimate - observed, delta)
        weighted = matrix.T * weights
        estimate = np.linalg.solve(weighted @ matrix + penalty, weighted @ observed)
    return estimate, weights


def map_deconv_1d(observed: np.ndarray, kernel_full: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form 1D periodic MAP with a known kernel and a difference penalty."""
    if lam < 0.0:
        raise ValueError("λ cannot be negative.")
    observed = np.asarray(observed, dtype=np.float64)
    spectrum = np.fft.fft(np.fft.ifftshift(np.asarray(kernel_full, dtype=np.float64)))
    frequency = np.fft.fftfreq(observed.size)
    power = np.abs(np.exp(2j * np.pi * frequency) - 1.0) ** 2
    denominator = np.abs(spectrum) ** 2 + float(lam) * power
    numerator = np.conj(spectrum) * np.fft.fft(observed)
    restored = np.zeros(observed.size, dtype=np.complex128)
    mask = denominator > 1e-15
    restored[mask] = numerator[mask] / denominator[mask]
    return np.fft.ifft(restored).real


def project_kernel(taps: np.ndarray) -> np.ndarray:
    """Nonnegative taps that sum to one. An empty kernel becomes a centred impulse."""
    projected = np.maximum(np.asarray(taps, dtype=np.float64), 0.0)
    total = float(projected.sum())
    if total <= 1e-15:
        projected = np.zeros_like(projected)
        projected[projected.size // 2] = 1.0
        return projected
    return projected / total


def fit_kernel_1d(observed: np.ndarray, signal: np.ndarray, length: int, ridge: float) -> np.ndarray:
    """Least squares for a short kernel, then the nonnegative unit-sum projection."""
    length = int(length)
    observed = np.asarray(observed, dtype=np.float64)
    matrix = np.zeros((observed.size, length), dtype=np.float64)
    center = length // 2
    for tap in range(length):
        matrix[:, tap] = np.roll(signal, tap - center)
    normal = matrix.T @ matrix + float(ridge) * np.eye(length)
    taps = np.linalg.solve(normal, matrix.T @ observed)
    return project_kernel(taps)


@dataclass
class BlindDemo:
    signal: np.ndarray
    true_kernel: np.ndarray
    observed: np.ndarray
    known_image: np.ndarray
    images: list[np.ndarray]
    kernels: list[np.ndarray]


def blind_deblur_1d(
    length: int = 48,
    kernel_length: int = 5,
    lam: float = 1e-3,
    ridge: float = 1e-3,
    noise_std: float = 0.01,
    iterations: int = 20,
    seed: int = 0,
) -> BlindDemo:
    """Alternate a known-form MAP image step with a constrained kernel step.

    The kernel stays nonnegative and sums to one. This does not guarantee
    that the iteration finds the original signal or the original kernel.
    """
    if kernel_length < 1 or kernel_length % 2 == 0:
        raise ValueError("the demonstration kernel length must be odd.")
    length = int(length)
    axis = np.linspace(0.0, 1.0, length, endpoint=False)
    signal = 0.15 + 0.55 * (axis > 0.35).astype(np.float64)
    signal += 0.25 * np.exp(-0.5 * ((axis - 0.72) / 0.06) ** 2)
    true_kernel = project_kernel(np.array([0.05, 0.2, 0.5, 0.2, 0.05], dtype=np.float64))
    if true_kernel.size != kernel_length:
        true_kernel = project_kernel(np.ones(kernel_length, dtype=np.float64))
    observed_clean = circular_convolve_1d(signal, place_kernel(length, true_kernel))
    generator = np.random.default_rng(int(seed))
    observed = observed_clean + float(noise_std) * generator.standard_normal(length)
    known_image = map_deconv_1d(observed, place_kernel(length, true_kernel), lam)
    taps = np.zeros(kernel_length, dtype=np.float64)
    taps[kernel_length // 2] = 1.0
    images: list[np.ndarray] = []
    kernels: list[np.ndarray] = []
    for _ in range(int(iterations)):
        estimate = map_deconv_1d(observed, place_kernel(length, taps), lam)
        taps = fit_kernel_1d(observed, estimate, kernel_length, ridge)
        images.append(estimate)
        kernels.append(taps.copy())
    return BlindDemo(
        signal=signal,
        true_kernel=true_kernel,
        observed=observed,
        known_image=known_image,
        images=images,
        kernels=kernels,
    )
