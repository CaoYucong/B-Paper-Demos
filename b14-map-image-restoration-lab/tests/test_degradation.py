"""前向模型 g = Af + n，以及它和反射边界高斯滤波不是同一个算子。"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from src.degradation import degrade
from src.synthetic_images import geometric_shapes


def test_fixed_seed_is_deterministic_and_changes_with_seed():
    image = geometric_shapes(24)
    first = degrade(image, sigma=1.5, noise_std=0.04, seed=7)
    second = degrade(image, sigma=1.5, noise_std=0.04, seed=7)
    third = degrade(image, sigma=1.5, noise_std=0.04, seed=8)
    np.testing.assert_allclose(first.observed, second.observed)
    np.testing.assert_allclose(first.blurred, second.blurred)
    assert not np.allclose(first.observed, third.observed)
    np.testing.assert_allclose(first.observed, first.blurred + first.noise)


def test_noise_is_not_silently_clipped():
    image = np.ones((16, 16), dtype=np.float64)
    observation = degrade(image, sigma=0.8, noise_std=0.2, seed=0)
    assert observation.observed.max() > 1.0
    assert observation.observed.min() < 1.0
    # 常数图像经过归一化高斯核后仍然是常数。
    np.testing.assert_allclose(observation.blurred, image, atol=1e-10)


def test_periodic_blur_is_not_scipy_reflect_gaussian():
    image = geometric_shapes(32)
    observation = degrade(image, sigma=3.0, noise_std=0.0, seed=0)
    reflected = gaussian_filter(image, sigma=3.0, mode="reflect")
    assert np.max(np.abs(reflected - observation.blurred)) > 1e-3


def test_blur_reduces_the_jump_of_a_stripe_image():
    from src.operators import forward_diff_x, gradient_squared_norm
    from src.synthetic_images import repeated_stripes

    image = repeated_stripes(48)
    observation = degrade(image, sigma=2.0, noise_std=0.0, seed=0)
    assert gradient_squared_norm(observation.blurred) < gradient_squared_norm(image)
    assert np.std(forward_diff_x(observation.blurred)) < np.std(forward_diff_x(image))
