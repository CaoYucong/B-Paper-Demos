"""Lecture 2: aliasing, the five-point parabolic filter, and the unit circle."""

from __future__ import annotations

import math

import numpy as np

# Causal five-point smoother from the notes:
# y[k] = (1/35) (-3 x[k] + 12 x[k-1] + 17 x[k-2] + 12 x[k-3] - 3 x[k-4]).
PARABOLIC_TAPS = np.array([-3.0, 12.0, 17.0, 12.0, -3.0], dtype=np.float64) / 35.0


def alias_pair(sample_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Samples of cos(π t / (2T)) and cos(5π t / (2T)) at t = kT.

    The notes show that these two continuous signals produce the same samples.
    """
    k = np.asarray(sample_index, dtype=np.float64)
    low = np.cos(math.pi * k / 2.0)
    high = np.cos(5.0 * math.pi * k / 2.0)
    return low, high


def parabolic_centre(window: np.ndarray) -> float:
    """s0 for one five-point window ordered as x[-2], x[-1], x[0], x[1], x[2]."""
    values = np.asarray(window, dtype=np.float64)
    if values.shape != (5,):
        raise ValueError("the parabolic fit uses five samples")
    weights = np.array([-3.0, 12.0, 17.0, 12.0, -3.0], dtype=np.float64)
    return float(np.dot(weights, values) / 35.0)


def apply_fir(samples: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Causal convolution. taps[0] multiplies the current sample."""
    return np.convolve(np.asarray(samples, dtype=np.float64), np.asarray(taps, dtype=np.float64), mode="full")


def fir_frequency_response(taps: np.ndarray, digital_omega: np.ndarray) -> np.ndarray:
    """G(e^{jω}) = sum_i a_i e^{-j ω i}, with ω in radians per sample."""
    coefficients = np.asarray(taps, dtype=np.float64)
    omega = np.asarray(digital_omega, dtype=np.float64)
    response = np.zeros(omega.shape, dtype=np.complex128)
    for index, coefficient in enumerate(coefficients):
        response += coefficient * np.exp(-1j * omega * index)
    return response


def parabolic_closed_form(digital_omega: np.ndarray) -> np.ndarray:
    """G(e^{jωT}) = (1/35) e^{-2jωT} (17 + 24 cos ωT − 6 cos 2ωT)."""
    omega = np.asarray(digital_omega, dtype=np.float64)
    real = 17.0 + 24.0 * np.cos(omega) - 6.0 * np.cos(2.0 * omega)
    return np.exp(-2j * omega) * real / 35.0


def recursive_lowpass_response(alpha: float, digital_omega: np.ndarray) -> np.ndarray:
    """G(z) = 1/(z − α) evaluated at z = e^{jω}.

    The notes derive this for y[k] = x[k−1] + α y[k−1]. The unit-circle
    values are the frequency response only when the pole is inside the
    unit circle.
    """
    omega = np.asarray(digital_omega, dtype=np.float64)
    return 1.0 / (np.exp(1j * omega) - float(alpha))


def causal_stable_alpha(alpha: float) -> bool:
    """The notes require 0 < α < 1 for this real, positive coefficient."""
    return 0.0 < float(alpha) < 1.0


def step_z_transform_gain(z: complex) -> complex:
    """Closed form F(z) = 1/(1 − z^{−1}) for the causal unit step.

    The notes say the geometric series is strictly valid for |z^{−1}| < 1.
    """
    if abs(z) <= 1.0:
        raise ValueError("the step-function series in the notes needs |z| > 1")
    return 1.0 / (1.0 - z**-1)
