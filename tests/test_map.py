"""MAP 目标、梯度、下降和直接求解必须使用同一种归一化。"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.b14.image_processing.lecture3.degradation import degrade
from src.b14.image_processing.lecture3.map_estimation import (
    compute_gradient,
    objective,
    objective_parts,
    run_gradient_descent,
    solve_map_cg,
    solve_map_fourier,
    step_size_from_fraction,
)
from src.b14.image_processing.metrics import mse
from src.b14.image_processing.lecture3.operators import apply_gradient_penalty, apply_otf, apply_otf_adjoint, forward_diff_x, forward_diff_y
from src.b14.image_processing.synthetic_images import geometric_shapes, repeated_stripes


def _small_problem(seed: int = 2):
    observation = degrade(geometric_shapes(8), sigma=1.1, noise_std=0.05, seed=seed)
    rng = np.random.default_rng(seed)
    image = rng.random(observation.original.shape)
    return observation, image


def test_objective_is_mean_normalised_and_gradient_has_the_two_over_n_factor():
    observation, image = _small_problem()
    lam = 0.3
    residual = apply_otf(image, observation.otf) - observation.observed
    dx = forward_diff_x(image)
    dy = forward_diff_y(image)
    expected = float(np.mean(residual**2) + lam * np.mean(dx**2 + dy**2))
    total, data, regularisation = objective_parts(image, observation.observed, observation.otf, lam)
    assert total == pytest.approx(expected, rel=0, abs=1e-12)
    assert data == pytest.approx(float(np.mean(residual**2)), rel=0, abs=1e-12)
    assert regularisation == pytest.approx(float(np.mean(dx**2 + dy**2)), rel=0, abs=1e-12)

    bracket = apply_otf_adjoint(residual, observation.otf) + lam * apply_gradient_penalty(image)
    gradient = compute_gradient(image, observation.observed, observation.otf, lam)
    np.testing.assert_allclose(gradient, (2.0 / image.size) * bracket, atol=1e-12)


def test_gradient_matches_central_differences():
    observation, image = _small_problem(seed=1)
    lam = 0.3
    gradient = compute_gradient(image, observation.observed, observation.otf, lam)
    rng = np.random.default_rng(9)
    for _ in range(3):
        direction = rng.normal(size=image.shape)
        direction /= np.linalg.norm(direction)
        eps = 1e-5
        plus = objective(image + eps * direction, observation.observed, observation.otf, lam)
        minus = objective(image - eps * direction, observation.observed, observation.otf, lam)
        numeric = (plus - minus) / (2.0 * eps)
        analytic = float(np.dot(gradient.ravel(), direction.ravel()))
        np.testing.assert_allclose(numeric, analytic, rtol=1e-5, atol=1e-7)

    eps = 1e-5
    plus = objective(image + eps * _pixel(image.shape, 3, 4), observation.observed, observation.otf, lam)
    minus = objective(image - eps * _pixel(image.shape, 3, 4), observation.observed, observation.otf, lam)
    np.testing.assert_allclose((plus - minus) / (2.0 * eps), gradient[3, 4], rtol=1e-4, atol=1e-7)


def _pixel(shape, row: int, col: int) -> np.ndarray:
    direction = np.zeros(shape, dtype=np.float64)
    direction[row, col] = 1.0
    return direction


def test_mean_step_and_rescaled_step_are_the_same_update():
    observation, image = _small_problem(seed=4)
    lam = 0.2
    step = step_size_from_fraction(observation.otf, lam, fraction=0.7)
    gradient = compute_gradient(image, observation.observed, observation.otf, lam)
    residual = apply_otf(image, observation.otf) - observation.observed
    bracket = apply_otf_adjoint(residual, observation.otf) + lam * apply_gradient_penalty(image)
    moved_by_eta = image - step.eta * gradient
    moved_by_alpha = image - 2.0 * step.alpha * bracket
    np.testing.assert_allclose(moved_by_eta, moved_by_alpha, atol=1e-12)
    assert step.alpha == pytest.approx(step.eta / image.size)
    assert step.theoretically_stable


def test_stable_step_decreases_the_objective_and_large_step_does_not():
    observation = degrade(repeated_stripes(16), sigma=1.4, noise_std=0.03, seed=0)
    stable = run_gradient_descent(
        observation.observed,
        observation.otf,
        lam=0.1,
        n_iterations=40,
        step_fraction=1.0,
        initialisation="zeros",
    )
    assert not stable.diverged
    assert stable.total_loss[-1] < stable.total_loss[0]
    assert np.max(np.diff(stable.total_loss)) < 1e-8

    unstable = run_gradient_descent(
        observation.observed,
        observation.otf,
        lam=0.05,
        n_iterations=40,
        step_fraction=4.0,
        initialisation="observed",
    )
    assert unstable.diverged or unstable.total_loss[-1] > unstable.total_loss[0]


def test_gradient_descent_matches_direct_quadratic_solution():
    rng = np.random.default_rng(0)
    observation = degrade(rng.random((12, 12)), sigma=1.0, noise_std=0.02, seed=3)
    lam = 0.25
    direct = solve_map_fourier(observation.observed, observation.otf, lam)
    cg_image, info = solve_map_cg(observation.observed, observation.otf, lam, rtol=1e-10, maxiter=400)
    assert info == 0
    np.testing.assert_allclose(cg_image, direct, atol=1e-6)

    gradient = compute_gradient(direct, observation.observed, observation.otf, lam)
    assert np.linalg.norm(gradient) < 1e-8

    result = run_gradient_descent(
        observation.observed,
        observation.otf,
        lam,
        n_iterations=2500,
        step_fraction=1.0,
        initialisation="zeros",
        max_snapshots=4,
    )
    assert not result.diverged
    assert mse(result.final, direct) < 1e-4
    assert abs(objective(result.final, observation.observed, observation.otf, lam) - objective(direct, observation.observed, observation.otf, lam)) < 1e-6


def test_solvers_do_not_accept_the_ground_truth():
    for function in (run_gradient_descent, solve_map_fourier, solve_map_cg):
        parameters = inspect.signature(function).parameters
        assert "original" not in parameters
        assert "ground_truth" not in parameters
        assert "reference" not in parameters


def test_projection_is_labelled_by_the_result_and_stays_in_the_box():
    observation = degrade(geometric_shapes(12), sigma=1.3, noise_std=0.05, seed=1)
    result = run_gradient_descent(
        observation.observed,
        observation.otf,
        lam=0.05,
        n_iterations=15,
        step_fraction=1.0,
        initialisation="observed",
        project=True,
    )
    assert result.projected
    assert result.final.min() >= -1e-12
    assert result.final.max() <= 1.0 + 1e-12
