"""Lecture 3: Hanning, windowed FIR, bilinear warping, and the Butterworth example."""

from __future__ import annotations

import math

import numpy as np

from src.b14.signal_analysis.lecture2.digital import fir_frequency_response

# y[k] = (1/4) (x[k] + 2 x[k-1] + x[k-2]).
HANNING_TAPS = np.array([0.25, 0.5, 0.25], dtype=np.float64)

# Difference equation printed for the 10 kHz Butterworth example.
EXAMPLE_FIR = np.array([0.20657, 0.41314, 0.20657], dtype=np.float64)
EXAMPLE_RECURSION = np.array([0.36953, -0.19582], dtype=np.float64)
EXAMPLE_SAMPLE_RATE_HZ = 10_000.0
EXAMPLE_CUTOFF_HZ = 2_000.0
EXAMPLE_STOP_HZ = 4_250.0


def endpoint_window(length: int, alpha: float) -> np.ndarray:
    """The notes' cosine window shifted onto a record of this length.

    Endpoints equal 2α − 1, so a Hanning window (α = 0.5) reaches 0 there.
    The lecture uses that taper to reduce spectral leakage.
    """
    length = int(length)
    if length < 2:
        raise ValueError("a leakage window needs at least two samples")
    k = np.arange(length, dtype=np.float64)
    centre = (length - 1) / 2.0
    return float(alpha) + (1.0 - float(alpha)) * np.cos(math.pi * (k - centre) / centre)


def cosine_window(half_width: int, alpha: float) -> np.ndarray:
    """w[k] = α + (1−α) cos(kπ/N) for k = −N … N, as written for Hamming and Hanning.

    α = 0.54 is the Hamming window. α = 0.5 is the Hanning window.
    The notes give this zero-phase definition and say it differs from some textbooks.
    """
    half_width = int(half_width)
    if half_width < 1:
        raise ValueError("N must be at least 1")
    k = np.arange(-half_width, half_width + 1, dtype=np.float64)
    return float(alpha) + (1.0 - float(alpha)) * np.cos(k * math.pi / half_width)


def ideal_lowpass_taps(half_width: int, cutoff_rad_sample: float) -> np.ndarray:
    """Sampled ideal low-pass of height 1 and cutoff ωc radians per sample.

    g[k] = (ωc/π) sin(ωc k) / (ωc k). The lecture's brick-wall example uses
    ωc = π/(4T), which is π/4 radians per sample.
    """
    k = np.arange(-int(half_width), int(half_width) + 1, dtype=np.float64)
    taps = np.empty_like(k)
    zero = k == 0
    taps[zero] = float(cutoff_rad_sample) / math.pi
    other = ~zero
    taps[other] = np.sin(cutoff_rad_sample * k[other]) / (math.pi * k[other])
    return taps


def windowed_fir(half_width: int, cutoff_rad_sample: float, alpha: float | None) -> np.ndarray:
    """Truncate at |k| = N, optionally multiply by the cosine window, then shift to be causal.

    alpha is None for the rectangular truncation. The lecture shifts the
    |k| ≤ 10 truncation by 10 samples, giving 21 coefficients.
    """
    taps = ideal_lowpass_taps(half_width, cutoff_rad_sample)
    if alpha is not None:
        taps = taps * cosine_window(half_width, alpha)
    return taps


def bilinear_analog_frequency(digital_omega: np.ndarray, sample_period: float) -> np.ndarray:
    """ωa = (2/T) tan(ωd/2), with ωd in radians per sample."""
    return (2.0 / float(sample_period)) * np.tan(np.asarray(digital_omega, dtype=np.float64) / 2.0)


def example_prewarp() -> tuple[float, float]:
    """Digital cutoffs and pre-warped analogue frequencies for fs = 10 kHz."""
    period = 1.0 / EXAMPLE_SAMPLE_RATE_HZ
    digital_pass = 2.0 * math.pi * EXAMPLE_CUTOFF_HZ / EXAMPLE_SAMPLE_RATE_HZ
    digital_stop = 2.0 * math.pi * EXAMPLE_STOP_HZ / EXAMPLE_SAMPLE_RATE_HZ
    analog_pass = float(bilinear_analog_frequency(np.array([digital_pass]), period)[0])
    analog_stop = float(bilinear_analog_frequency(np.array([digital_stop]), period)[0])
    return analog_pass, analog_stop


def example_order_estimate(analog_pass: float, analog_stop: float) -> float:
    """n ≈ log10(A) / log10(ωa2/ωa1) with A = 10^(30/20) for 30 dB."""
    attenuation = 10.0 ** (30.0 / 20.0)
    return math.log10(attenuation) / math.log10(analog_stop / analog_pass)


def iir_frequency_response(
    fir: np.ndarray,
    recursion: np.ndarray,
    digital_omega: np.ndarray,
) -> np.ndarray:
    """G(e^{jω}) for y[k] = sum b_i x[k−i] + sum a_i y[k−i].

    recursion[i] is the coefficient of y[k−(i+1)], matching the lecture's
    +0.36953 y[k−1] − 0.19582 y[k−2].
    """
    numerator = fir_frequency_response(fir, digital_omega)
    feedback = np.asarray(recursion, dtype=np.float64)
    omega = np.asarray(digital_omega, dtype=np.float64)
    denominator = np.ones(omega.shape, dtype=np.complex128)
    for index, coefficient in enumerate(feedback, start=1):
        denominator -= coefficient * np.exp(-1j * omega * index)
    return numerator / denominator


def apply_iir(samples: np.ndarray, fir: np.ndarray, recursion: np.ndarray) -> np.ndarray:
    """Run the difference equation. Earlier outputs start at 0."""
    values = np.asarray(samples, dtype=np.float64)
    b = np.asarray(fir, dtype=np.float64)
    a = np.asarray(recursion, dtype=np.float64)
    output = np.zeros(values.shape, dtype=np.float64)
    for k in range(values.size):
        acc = 0.0
        for i, coefficient in enumerate(b):
            if k - i >= 0:
                acc += coefficient * values[k - i]
        for i, coefficient in enumerate(a, start=1):
            if k - i >= 0:
                acc += coefficient * output[k - i]
        output[k] = acc
    return output
