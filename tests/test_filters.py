"""逆滤波、维纳滤波，以及它们和二次 MAP 的关系。"""

from __future__ import annotations

import numpy as np

from src.b14.image_processing.lecture3.comparison import run_comparison
from src.b14.image_processing.lecture3.degradation import degrade
from src.b14.image_processing.lecture3.operators import gaussian_psf, motion_blur_psf, psf_to_otf
from src.b14.image_processing.synthetic_images import geometric_shapes
from src.b14.image_processing.lecture3.wiener import inverse_filter, wiener_filter


def test_wiener_uses_the_hermitian_formula():
    rng = np.random.default_rng(0)
    observed = rng.random((16, 16))
    otf = psf_to_otf(motion_blur_psf((16, 16), length=5))
    gain = 0.03
    got = wiener_filter(observed, otf, gain)
    spectrum = np.fft.fft2(observed)
    manual = np.fft.ifft2(np.conj(otf) / (np.abs(otf) ** 2 + gain) * spectrum).real
    without_conjugate = np.fft.ifft2(otf / (np.abs(otf) ** 2 + gain) * spectrum).real
    np.testing.assert_allclose(got, manual, atol=1e-12)
    assert np.max(np.abs(without_conjugate - got)) > 1e-6


def test_inverse_matches_wiener_when_threshold_is_inactive_and_k_is_zero():
    rng = np.random.default_rng(1)
    observed = rng.random((24, 24))
    otf = psf_to_otf(gaussian_psf((24, 24), sigma=1.2))
    threshold = float(np.min(np.abs(otf)) * 0.1)
    assert threshold > 0
    inverse, suppressed = inverse_filter(observed, otf, threshold)
    assert suppressed == 0
    np.testing.assert_allclose(inverse, wiener_filter(observed, otf, gain=0.0), atol=1e-8)


def test_inverse_threshold_suppresses_small_gains():
    rng = np.random.default_rng(2)
    observed = rng.random((32, 32))
    otf = psf_to_otf(gaussian_psf((32, 32), sigma=3.5))
    loose, n_loose = inverse_filter(observed, otf, 1e-8)
    tight, n_tight = inverse_filter(observed, otf, 1e-2)
    assert n_tight > n_loose
    assert np.max(np.abs(loose - tight)) > 1e-6


def test_fourier_map_has_the_smallest_objective():
    observation = degrade(geometric_shapes(16), sigma=1.3, noise_std=0.02, seed=0)
    results = run_comparison(
        observation,
        lam=0.1,
        wiener_k=0.01,
        inverse_threshold=1e-3,
        gd_iterations=25,
        step_fraction=1.0,
    )
    by_key = {item.key: item for item in results}
    assert set(by_key) == {"observed", "inverse", "wiener", "gd", "fourier", "cg"}
    np.testing.assert_allclose(by_key["cg"].image, by_key["fourier"].image, atol=1e-5)
    best = by_key["fourier"].objective
    for item in results:
        assert item.objective >= best - 1e-6
        assert np.isfinite(item.mse)
        assert np.isfinite(item.psnr)
