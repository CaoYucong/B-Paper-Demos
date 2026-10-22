"""前向卷积、伴随和差分算子。"""

from __future__ import annotations

import numpy as np
import pytest

from src.b14.image_processing.lecture3.operators import (
    adjoint_diff_x,
    adjoint_diff_y,
    apply_gradient_penalty,
    apply_otf,
    apply_otf_adjoint,
    averaging_matrix,
    circular_three_point_average,
    forward_diff_x,
    forward_diff_y,
    frequency_gradient_power,
    gaussian_psf,
    impulse_at_center,
    inner_product,
    motion_blur_psf,
    psf_to_otf,
)


def reference_convolve(image: np.ndarray, psf_centered: np.ndarray) -> np.ndarray:
    """圆卷积的直接定义：out[n] = sum_m k0[m] x[n-m]，k0 的零滞后在索引 0。"""
    kernel = np.fft.ifftshift(np.asarray(psf_centered, dtype=np.float64))
    output = np.zeros_like(image, dtype=np.float64)
    for i, j in np.ndindex(kernel.shape):
        weight = kernel[i, j]
        if weight == 0.0:
            continue
        output += weight * np.roll(np.roll(image, int(i), axis=0), int(j), axis=1)
    return output


def reference_correlate(image: np.ndarray, psf_centered: np.ndarray) -> np.ndarray:
    kernel = np.fft.ifftshift(np.asarray(psf_centered, dtype=np.float64))
    output = np.zeros_like(image, dtype=np.float64)
    for i, j in np.ndindex(kernel.shape):
        weight = kernel[i, j]
        if weight == 0.0:
            continue
        output += weight * np.roll(np.roll(image, -int(i), axis=0), -int(j), axis=1)
    return output


@pytest.mark.parametrize("shape", [(8, 8), (9, 9), (8, 9)])
def test_impulse_response_is_centered_psf(shape):
    psf = gaussian_psf(shape, sigma=1.25)
    response = apply_otf(impulse_at_center(shape), psf_to_otf(psf))
    np.testing.assert_allclose(response, psf, atol=1e-10)
    peak = np.unravel_index(int(np.argmax(response)), shape)
    assert peak == (shape[0] // 2, shape[1] // 2)


def test_motion_impulse_matches_asymmetric_psf():
    shape = (12, 16)
    psf = motion_blur_psf(shape, length=6)
    response = apply_otf(impulse_at_center(shape), psf_to_otf(psf))
    np.testing.assert_allclose(response, psf, atol=1e-10)


def test_fft_convolution_matches_spatial_definition():
    rng = np.random.default_rng(0)
    image = rng.normal(size=(8, 8))
    for psf in (gaussian_psf((8, 8), 1.1), motion_blur_psf((8, 8), 3)):
        otf = psf_to_otf(psf)
        np.testing.assert_allclose(apply_otf(image, otf), reference_convolve(image, psf), atol=1e-10)
        np.testing.assert_allclose(apply_otf_adjoint(image, otf), reference_correlate(image, psf), atol=1e-10)


def test_gaussian_otf_is_real_and_preserves_constants():
    otf = psf_to_otf(gaussian_psf((32, 32), 2.2))
    assert np.max(np.abs(otf.imag)) < 1e-8
    assert otf[0, 0].real == pytest.approx(1.0, abs=1e-10)
    image = np.full((32, 32), 0.37)
    np.testing.assert_allclose(apply_otf(image, otf), image, atol=1e-10)


def test_adjoint_identity_for_symmetric_and_asymmetric_kernels():
    rng = np.random.default_rng(4)
    shape = (10, 12)
    x = rng.normal(size=shape)
    y = rng.normal(size=shape)
    for psf in (gaussian_psf(shape, 1.4), motion_blur_psf(shape, 5)):
        otf = psf_to_otf(psf)
        lhs = inner_product(apply_otf(x, otf), y)
        rhs = inner_product(x, apply_otf_adjoint(y, otf))
        assert abs(lhs - rhs) <= 1e-8 * max(1.0, abs(lhs))


def test_gaussian_is_self_adjoint_but_motion_blur_is_not():
    rng = np.random.default_rng(5)
    image = rng.normal(size=(16, 16))
    gaussian = psf_to_otf(gaussian_psf((16, 16), 2.0))
    np.testing.assert_allclose(apply_otf(image, gaussian), apply_otf_adjoint(image, gaussian), atol=1e-10)
    motion = psf_to_otf(motion_blur_psf((16, 16), 7))
    assert np.max(np.abs(apply_otf(image, motion) - apply_otf_adjoint(image, motion))) > 1e-3


def test_finite_difference_adjoints_and_fourier_symbol():
    rng = np.random.default_rng(6)
    image = rng.normal(size=(9, 11))
    field = rng.normal(size=(9, 11))
    for forward, adjoint in ((forward_diff_x, adjoint_diff_x), (forward_diff_y, adjoint_diff_y)):
        lhs = inner_product(forward(image), field)
        rhs = inner_product(image, adjoint(field))
        assert abs(lhs - rhs) <= 1e-9 * max(1.0, abs(lhs))

    power = frequency_gradient_power(image.shape)
    spectral = np.fft.ifft2(np.fft.fft2(image) * power).real
    np.testing.assert_allclose(apply_gradient_penalty(image), spectral, atol=1e-10)


def test_three_point_matrix_is_periodic_averaging():
    matrix = averaging_matrix(6)
    assert matrix.shape == (6, 6)
    np.testing.assert_allclose(matrix.sum(axis=1), np.ones(6))
    assert matrix[0, -1] == pytest.approx(1.0 / 3.0)
    vector = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.4])
    np.testing.assert_allclose(matrix @ vector, circular_three_point_average(vector))
