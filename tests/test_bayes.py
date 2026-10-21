"""标量高斯 MAP：闭式解、归一化后验，以及噪声大小对估计的拉动。"""

from __future__ import annotations

import numpy as np

from src.bayes import (
    gaussian_pdf,
    grid_map,
    normalised_posterior,
    scalar_map,
    scalar_posterior_variance,
)


def test_scalar_map_matches_the_derived_formula():
    observation = 1.2
    coefficient = 0.8
    noise_std = 0.35
    prior_mean = 0.2
    prior_std = 0.9
    estimate = scalar_map(observation, coefficient, noise_std, prior_mean, prior_std)
    numerator = coefficient * observation / noise_std**2 + prior_mean / prior_std**2
    denominator = coefficient**2 / noise_std**2 + 1.0 / prior_std**2
    assert estimate == numerator / denominator


def test_numerical_posterior_matches_the_gaussian_formula():
    observation, coefficient = 1.2, 0.8
    noise_std, prior_mean, prior_std = 0.35, 0.2, 0.9
    mean = scalar_map(observation, coefficient, noise_std, prior_mean, prior_std)
    std = np.sqrt(scalar_posterior_variance(coefficient, noise_std, prior_std))
    samples = np.linspace(mean - 10.0 * std, mean + 10.0 * std, 20001)
    posterior = normalised_posterior(samples, observation, coefficient, noise_std, prior_mean, prior_std)
    analytical = gaussian_pdf(samples, mean, std)
    integral = np.trapezoid(posterior, samples) if hasattr(np, "trapezoid") else np.trapz(posterior, samples)
    assert abs(float(integral) - 1.0) < 1e-4
    assert np.max(np.abs(posterior - analytical)) < 1e-4
    assert abs(grid_map(samples, posterior) - mean) < 5e-3


def test_low_noise_moves_map_toward_the_observation():
    low_noise = scalar_map(1.5, 1.0, 0.05, 0.0, 1.0)
    high_noise = scalar_map(1.5, 1.0, 2.0, 0.0, 1.0)
    assert abs(low_noise - 1.5) < abs(high_noise - 1.5)
    assert abs(high_noise - 0.0) < abs(low_noise - 0.0)
    assert low_noise == scalar_map(1.5, 1.0, 0.05, 0.0, 1.0)


def test_zero_coefficient_returns_the_prior_mean():
    assert scalar_map(3.0, 0.0, 0.4, 0.25, 1.2) == 0.25
