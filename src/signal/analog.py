"""Lecture 1: sampling, convolution, the ideal low-pass, and Butterworth filters."""

from __future__ import annotations

import math

import numpy as np

# Denominator coefficients of G(s) = 1 / sum_k a_k (s/ωc)^k, with a0 = 1.
# Rows are the lecture table for orders 1 through 8. The leading 1 is a0.
BUTTERWORTH_TABLE: dict[int, tuple[float, ...]] = {
    1: (1.0, 1.0),
    2: (1.0, 1.4141, 1.0),
    3: (1.0, 2.0, 2.0, 1.0),
    4: (1.0, 2.6131, 3.4142, 2.6131, 1.0),
    5: (1.0, 3.2361, 5.2361, 5.2361, 3.2361, 1.0),
    6: (1.0, 3.8637, 7.4641, 9.1416, 7.4641, 3.8637, 1.0),
    7: (1.0, 4.4940, 10.0978, 14.5918, 14.5918, 10.0978, 4.4940, 1.0),
    8: (1.0, 5.1258, 13.1371, 21.8462, 25.6884, 21.8462, 13.1371, 5.1258, 1.0),
}


def sampled_sinusoid(
    amplitude: float,
    frequency_hz: float,
    phase_rad: float,
    sample_rate_hz: float,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous-time sinusoid sampled at t = k/fs.

    The notes write xa(t) = A cos(2π fa t + φ) and xd[n] = A cos(2π fd n + φ)
    with fd = fa/fs.
    """
    times = np.arange(count, dtype=np.float64) / float(sample_rate_hz)
    digital_frequency = float(frequency_hz) / float(sample_rate_hz)
    samples = amplitude * np.cos(2.0 * math.pi * digital_frequency * np.arange(count) + phase_rad)
    return times, np.asarray(samples, dtype=np.float64)


def uniform_quantise(samples: np.ndarray, levels: int, amplitude: float) -> np.ndarray:
    """Uniform mid-rise quantiser on [-amplitude, amplitude].

    The signal-types slide names quantisation. It does not give this rule,
    so callers should label the result as an illustration.
    """
    levels = int(levels)
    if levels < 2:
        raise ValueError("levels must be at least 2")
    span = float(amplitude)
    if span <= 0:
        raise ValueError("amplitude must be positive")
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -span, span)
    step = (2.0 * span) / levels
    indices = np.floor((clipped + span) / step)
    indices = np.clip(indices, 0, levels - 1)
    return -span + (indices + 0.5) * step


def first_difference(samples: np.ndarray) -> np.ndarray:
    """y[k] = x[k] - x[k-1], with x[-1] taken as 0.

    The lecture writes the continuous example y(t) = x(t) - x(t-1).
    This is the same operation on a sampled sequence.
    """
    values = np.asarray(samples, dtype=np.float64)
    output = np.empty_like(values)
    previous = 0.0
    for index, value in enumerate(values):
        output[index] = value - previous
        previous = value
    return output


def discrete_convolution(signal: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    """y[n] = sum_k x[k] g[n-k], the discrete convolution in the notes."""
    return np.convolve(np.asarray(signal, dtype=np.float64), np.asarray(impulse, dtype=np.float64))


def ideal_lowpass_kernel(time: np.ndarray, cutoff_rad_s: float) -> np.ndarray:
    """g(t) = (ωc/π) sin(ωc t) / (ωc t), with value ωc/π at t = 0."""
    time = np.asarray(time, dtype=np.float64)
    cutoff = float(cutoff_rad_s)
    kernel = np.empty_like(time)
    zero = np.abs(time) < 1e-12
    kernel[zero] = cutoff / math.pi
    other = ~zero
    kernel[other] = cutoff / math.pi * np.sin(cutoff * time[other]) / (cutoff * time[other])
    return kernel


def butterworth_squared_magnitude(omega: np.ndarray, cutoff_rad_s: float, order: int) -> np.ndarray:
    """|G(jω)|² = 1 / (1 + (ω/ωc)^(2n))."""
    if order < 1:
        raise ValueError("order must be positive")
    ratio = np.asarray(omega, dtype=np.float64) / float(cutoff_rad_s)
    return 1.0 / (1.0 + ratio ** (2 * int(order)))


def butterworth_order(attenuation: float, frequency_ratio: float) -> float:
    """n = log(A² − 1) / (2 log(ωa/ωc)). The notes then take the next integer."""
    if attenuation <= 1.0:
        raise ValueError("attenuation A must exceed 1")
    if frequency_ratio <= 1.0:
        raise ValueError("ωa/ωc must exceed 1")
    return math.log(attenuation**2 - 1.0) / (2.0 * math.log(frequency_ratio))


def butterworth_poles(order: int, cutoff_rad_s: float = 1.0) -> np.ndarray:
    """s = ωc exp j[π/2 + (2k+1)π/(2n)], k = 0 … 2n−1."""
    order = int(order)
    k = np.arange(2 * order, dtype=np.float64)
    angle = math.pi / 2.0 + (2.0 * k + 1.0) * math.pi / (2.0 * order)
    return float(cutoff_rad_s) * np.exp(1j * angle)


def stable_butterworth_poles(order: int, cutoff_rad_s: float = 1.0) -> np.ndarray:
    """Poles in the open left half-plane. The notes assign these to G(s)."""
    poles = butterworth_poles(order, cutoff_rad_s)
    return poles[np.real(poles) < -1e-10]


def exponential_frequency_response(alpha: float, omega: float) -> complex:
    """G(jω) = 1/(α + jω) for the causal exponential g(t) = e^{−α t} u(t), α > 0."""
    if alpha <= 0.0:
        raise ValueError("the causal exponential in the notes has α > 0")
    return 1.0 / (float(alpha) + 1j * float(omega))


def butterworth_polynomial_gain(normalized_s: complex, order: int) -> complex:
    """Evaluate the lecture coefficient table at s/ωc."""
    coefficients = BUTTERWORTH_TABLE[int(order)]
    value = 0j
    power = 1 + 0j
    z = complex(normalized_s)
    for coefficient in coefficients:
        value += coefficient * power
        power *= z
    return 1.0 / value
