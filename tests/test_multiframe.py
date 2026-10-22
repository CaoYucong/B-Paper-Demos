"""Sampling operators, stacked systems, MAP gradients, Huber, and blind-kernel constraints."""

from __future__ import annotations

import numpy as np

from src.b14.image_processing.lecture3.multiframe import (
    blind_deblur_1d,
    corrupted_blur_problem,
    evaluate_multiframe,
    fit_kernel_1d,
    huber_derivative,
    huber_loss,
    huber_weights,
    initialise_estimate,
    map_deconv_1d,
    place_kernel,
    restore_huber_1d,
    restore_quadratic_1d,
    run_gradient_descent_multiframe,
    solve_normal_equations,
)
from src.b14.image_processing.lecture3.sampling import (
    Camera,
    apply_camera,
    apply_camera_adjoint,
    averaging_curve,
    factor_matrices_1d,
    high_resolution_pattern,
    report_stacked_system,
    spread_observations,
    window_mask,
)


def _camera(shift, factor=2, sigma=0.0, realistic=False) -> Camera:
    return Camera(factor, shift[0], shift[1], sigma, realistic)


def test_point_sampling_reads_the_shifted_lattice():
    image = np.arange(16 * 16, dtype=np.float64).reshape(16, 16)
    assert np.allclose(apply_camera(image, _camera((0.0, 0.0))), image[::2, ::2])
    assert np.allclose(apply_camera(image, _camera((0.5, 0.0))), image[1::2, ::2])
    assert np.allclose(apply_camera(image, _camera((0.0, 0.5))), image[::2, 1::2])
    assert np.allclose(apply_camera(image, _camera((0.5, 0.5))), image[1::2, 1::2])
    from src.b14.image_processing.lecture3.sampling import make_camera

    assert np.allclose(apply_camera(image, make_camera(2, horizontal=0.5, vertical=0.0)), image[::2, 1::2])
    even = window_mask(image.shape, _camera((0.0, 0.0)))
    odd = window_mask(image.shape, _camera((0.5, 0.0)))
    assert even.mean() == 0.25
    assert not np.any(even & odd)


def test_camera_adjoints_match_the_inner_product():
    generator = np.random.default_rng(2)
    image = generator.normal(size=(16, 16))
    residual = generator.normal(size=(8, 8))
    for realistic, sigma in ((False, 0.0), (True, 0.0), (True, 0.7)):
        camera = _camera((0.5, 0.5), sigma=sigma, realistic=realistic)
        left = float(np.sum(apply_camera(image, camera) * residual))
        right = float(np.sum(image * apply_camera_adjoint(residual, image.shape, camera)))
        assert abs(left - right) < 1e-8


def test_realistic_window_covers_pixels_that_point_sampling_misses():
    shape = (16, 16)
    point = window_mask(shape, _camera((0.0, 0.0), realistic=False))
    area = window_mask(shape, _camera((0.0, 0.0), sigma=0.4, realistic=True))
    assert area.all()
    assert point.mean() == 0.25
    image = high_resolution_pattern("edge", 16)
    point_view = apply_camera(image, _camera((0.0, 0.0)))
    mixed = apply_camera(image, _camera((0.0, 0.0), sigma=0.8, realistic=True))
    assert point_view.shape == (8, 8)
    assert not np.allclose(point_view, mixed)


def test_patterns_are_small_and_bounded():
    for name in ("combined", "edge", "diagonal", "checkerboard", "shape"):
        image = high_resolution_pattern(name, 16)
        assert image.shape == (16, 16)
        assert image.min() >= 0.0 and image.max() <= 1.0
    board = high_resolution_pattern("checkerboard", 16)
    assert np.allclose(apply_camera(board, _camera((0.0, 0.0))), 1.0)
    assert np.allclose(apply_camera(board, _camera((0.5, 0.0))), 0.0)


def test_explicit_factors_have_the_toy_dimensions_and_multiply():
    translation, blur, down, matrix = factor_matrices_1d(8, 2, 0.0, 0.0, False)
    assert translation.shape == (8, 8)
    assert blur.shape == (8, 8)
    assert down.shape == (4, 8)
    assert matrix.shape == (4, 8)
    assert np.allclose(blur, np.eye(8))
    assert np.allclose(matrix, down @ blur @ translation)
    signal = np.arange(8, dtype=np.float64)
    even = factor_matrices_1d(8, 2, 0.0, 0.0, False)[3]
    odd = factor_matrices_1d(8, 2, 0.5, 0.0, False)[3]
    assert np.allclose(even @ signal, signal[::2])
    assert np.allclose(odd @ signal, signal[1::2])
    _t, box, _d, realistic = factor_matrices_1d(8, 2, 0.0, 0.6, True)
    assert box.shape == (8, 8)
    assert not np.allclose(box, np.eye(8))
    assert np.allclose(realistic, _d @ box @ _t)
    vector = np.random.default_rng(0).normal(size=4)
    assert abs(float(vector @ (even @ signal)) - float(signal @ (even.T @ vector))) < 1e-10


def test_rank_stays_low_when_the_new_rows_repeat():
    signal = np.array([0.1, 0.2, 0.0, 0.9, 0.4, 0.8, 0.3, 0.7])
    even = factor_matrices_1d(8, 2, 0.0, 0.0, False)[3]
    odd = factor_matrices_1d(8, 2, 0.5, 0.0, False)[3]
    complementary = np.vstack([even, odd])
    repeated = np.vstack([even, even])
    assert np.linalg.matrix_rank(complementary, tol=1e-8) == 8
    assert np.linalg.matrix_rank(repeated, tol=1e-8) == 4
    recovered = report_stacked_system(complementary, complementary @ signal, signal)
    assert recovered.measurements == 8 and recovered.unknowns == 8
    assert recovered.full_column_rank
    assert recovered.reconstruction_mse < 1e-16
    assert np.allclose(recovered.least_squares, recovered.pseudoinverse)
    redundant = report_stacked_system(repeated, repeated @ signal, signal)
    assert redundant.measurements == redundant.unknowns
    assert not redundant.full_column_rank
    assert redundant.rank == 4
    assert redundant.reconstruction_mse > 1e-3
    longer = np.vstack([even, even, even])
    over = report_stacked_system(longer, longer @ signal, signal)
    assert over.more_equations_than_unknowns and over.rank == 4


def test_noise_average_follows_one_over_sqrt_n_and_does_not_add_sites():
    clean = high_resolution_pattern("shape", 96)
    empirical, theory = averaging_curve(clean, noise_std=0.2, counts=(1, 4, 16), seed=0)
    assert empirical[0] > empirical[1] > empirical[2]
    assert np.allclose(empirical, theory, rtol=0.12)
    camera = _camera((0.0, 0.0))
    image = high_resolution_pattern("edge", 16)
    one, weight_one = spread_observations((16, 16), [camera], [apply_camera(image, camera)])
    many, weight_many = spread_observations(
        (16, 16), [camera, camera], [apply_camera(image, camera), apply_camera(image, camera)]
    )
    assert np.array_equal(weight_one > 0, weight_many > 0)
    assert np.allclose(one, many)
    shifts = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
    cameras = [_camera(shift) for shift in shifts]
    observations = [apply_camera(image, camera) for camera in cameras]
    fused, weight = spread_observations(image.shape, cameras, observations)
    assert weight.min() == 1.0
    assert np.allclose(fused, image)


def test_map_gradient_matches_a_finite_difference_and_ignores_the_truth_label():
    truth = high_resolution_pattern("combined", 8)
    cameras = [_camera((0.0, 0.0), sigma=0.4, realistic=True), _camera((0.5, 0.5), sigma=0.4, realistic=True)]
    generator = np.random.default_rng(1)
    observations = [apply_camera(truth, camera) + 0.01 * generator.normal(size=(4, 4)) for camera in cameras]
    estimate = generator.normal(size=truth.shape)
    _total, _data, _penalty, gradient = evaluate_multiframe(estimate, observations, cameras, lam=0.05)
    direction = generator.normal(size=truth.shape)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-6
    ahead = evaluate_multiframe(estimate + epsilon * direction, observations, cameras, 0.05)[0]
    behind = evaluate_multiframe(estimate - epsilon * direction, observations, cameras, 0.05)[0]
    assert abs((ahead - behind) / (2.0 * epsilon) - float(np.sum(gradient * direction))) < 1e-6
    reference, info = solve_normal_equations(observations, cameras, 0.05, truth.shape)
    assert info == 0
    _total_ref, _data_ref, _penalty_ref, grad_ref = evaluate_multiframe(reference, observations, cameras, 0.05)
    assert np.linalg.norm(grad_ref) < 1e-6


def test_point_sample_gradient_descent_reaches_the_partition_and_can_diverge():
    truth = high_resolution_pattern("edge", 8)
    cameras = [_camera(shift) for shift in ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))]
    observations = [apply_camera(truth, camera) for camera in cameras]
    initial = initialise_estimate(observations, cameras, truth.shape, "zeros")
    result = run_gradient_descent_multiframe(
        observations, cameras, lam=0.0, step_fraction=1.0, iterations=1, initial=initial, truth=truth
    )
    assert np.allclose(result.images[-1], truth, atol=1e-8)
    assert result.reconstruction_mse[-1] < 1e-14
    other = run_gradient_descent_multiframe(
        observations, cameras, 0.0, 1.0, 1, initial, truth=np.zeros_like(truth)
    )
    assert np.allclose(other.images[-1], result.images[-1])
    assert not np.allclose(other.reconstruction_mse, result.reconstruction_mse)
    blurry = [_camera(shift, sigma=0.5, realistic=True) for shift in ((0.0, 0.0), (0.5, 0.5))]
    noisy = [apply_camera(truth, camera) for camera in blurry]
    start = initialise_estimate(noisy, blurry, truth.shape, "mean")
    descent = run_gradient_descent_multiframe(noisy, blurry, 1e-2, 1.0, 30, start, truth=truth)
    assert descent.total_loss[-1] < descent.total_loss[0]
    assert not descent.diverged
    exploded = run_gradient_descent_multiframe(noisy, blurry, 1e-2, 40.0, 12, start, truth=truth)
    assert exploded.diverged


def test_huber_derivative_and_the_corrupted_sample():
    residual = np.linspace(-3.0, 3.0, 601)
    delta = 0.8
    epsilon = 1e-6
    numeric = (huber_loss(residual + epsilon, delta) - huber_loss(residual - epsilon, delta)) / (2.0 * epsilon)
    assert np.allclose(numeric, huber_derivative(residual, delta), atol=1e-6)
    weights = huber_weights(np.array([0.2, 2.0]), delta)
    assert weights[0] == 1.0 and weights[1] < 1.0
    signal, matrix, observed, corrupt = corrupted_blur_problem(spike=3.0, noise_std=0.01, seed=1)
    least_squares = restore_quadratic_1d(matrix, observed, lam=0.1)
    robust, weights = restore_huber_1d(matrix, observed, lam=0.1, delta=0.5, iterations=30)
    assert weights[corrupt] < 0.5
    least_error = float(np.mean((least_squares - signal) ** 2))
    robust_error = float(np.mean((robust - signal) ** 2))
    assert robust_error < least_error


def test_blind_kernel_constraints_and_known_blur():
    demo = blind_deblur_1d(noise_std=0.0, lam=1e-4, iterations=8, seed=0)
    assert demo.true_kernel.shape == (5,)
    assert abs(float(demo.true_kernel.sum()) - 1.0) < 1e-12
    for kernel in demo.kernels:
        assert kernel.shape == (5,)
        assert np.all(kernel >= -1e-12)
        assert abs(float(kernel.sum()) - 1.0) < 1e-8
    assert np.mean((demo.known_image - demo.signal) ** 2) < 1e-6
    clean = demo.observed
    recovered = fit_kernel_1d(clean, demo.signal, 5, ridge=1e-8)
    assert np.allclose(recovered, demo.true_kernel, atol=1e-6)
    restored = map_deconv_1d(clean, place_kernel(clean.size, demo.true_kernel), lam=0.0)
    assert np.mean((restored - demo.signal) ** 2) < 1e-8
