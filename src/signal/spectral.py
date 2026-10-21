"""Lecture 4: DFT, leakage, periodogram, and autoregressive spectra."""

from __future__ import annotations

import math

import numpy as np

# Four samples printed for f(t) = 5 + 2 cos(2πt − 90°) + 3 cos(4πt) at fs = 4 Hz.
DFT_EXAMPLE_SAMPLES = np.array([8.0, 4.0, 8.0, 0.0], dtype=np.float64)
DFT_EXAMPLE_SPECTRUM = np.array([20.0 + 0j, 0.0 - 4j, 12.0 + 0j, 0.0 + 4j], dtype=np.complex128)

# Coefficients the notes report for the AR(4) fit in their figure.
REPORTED_AR4 = np.array([1.46, -1.08, 0.60, -0.186], dtype=np.float64)


def dft(samples: np.ndarray) -> np.ndarray:
    """F[n] = sum_k f[k] exp(−j 2π n k / N). The 1/N factor is on the inverse."""
    values = np.asarray(samples, dtype=np.complex128)
    count = values.size
    k = np.arange(count)
    n = k.reshape(-1, 1)
    return values @ np.exp(-2j * math.pi * n * k / count)


def idft(spectrum: np.ndarray) -> np.ndarray:
    """f[k] = (1/N) sum_n F[n] exp(+j 2π n k / N)."""
    values = np.asarray(spectrum, dtype=np.complex128)
    count = values.size
    n = np.arange(count)
    k = n.reshape(-1, 1)
    return (values @ np.exp(2j * math.pi * k * n / count)) / count


def dft_example_samples() -> np.ndarray:
    """f[k] = 5 + 2 cos(π k/2 − π/2) + 3 cos(π k), k = 0,1,2,3."""
    k = np.arange(4, dtype=np.float64)
    return 5.0 + 2.0 * np.cos(math.pi * k / 2.0 - math.pi / 2.0) + 3.0 * np.cos(math.pi * k)


def dft_matrix(count: int) -> np.ndarray:
    """The matrix with W = exp(−j 2π/N) and entries W^{nk}."""
    index = np.arange(count)
    return np.exp(-2j * math.pi * np.outer(index, index) / count)


def conjugate_pair_term(spectrum: np.ndarray, bin_index: int, sample_index: np.ndarray) -> np.ndarray:
    """Contribution of one conjugate pair, (2/N) |F[n]| cos(2π n k/N + arg F[n])."""
    values = np.asarray(spectrum, dtype=np.complex128)
    count = values.size
    coefficient = values[bin_index]
    k = np.asarray(sample_index, dtype=np.float64)
    amplitude = 2.0 / count * abs(coefficient)
    phase = math.atan2(float(np.imag(coefficient)), float(np.real(coefficient)))
    return amplitude * np.cos(2.0 * math.pi * bin_index * k / count + phase)


def periodogram_bins(samples: np.ndarray) -> np.ndarray:
    """Pxx(n/N) = (1/N) |F[n]|²."""
    values = np.asarray(samples, dtype=np.float64)
    return np.abs(dft(values)) ** 2 / values.size


def biased_autocorrelation(samples: np.ndarray, max_lag: int) -> np.ndarray:
    """rxx(m) = (1/N) sum_{n=0}^{N−m−1} x[n] x[n+m] for m = 0 … max_lag.

    This is the factor written in the notes. The mean is not removed here;
    the AR section of the notes says to subtract the mean before fitting.
    """
    values = np.asarray(samples, dtype=np.float64)
    count = values.size
    lags = np.empty(int(max_lag) + 1, dtype=np.float64)
    for lag in range(lags.size):
        lags[lag] = float(np.dot(values[: count - lag], values[lag:]) / count)
    return lags


def embedding_matrix(samples: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """M and x in x = M a + e. Row 0 predicts x[p] from x[p−1] … x[0]."""
    values = np.asarray(samples, dtype=np.float64)
    order = int(order)
    target = values[order:]
    columns = [values[order - 1 - lag : values.size - 1 - lag] for lag in range(order)]
    return np.column_stack(columns), target


def least_squares_ar(samples: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray, float]:
    """â = (Mᵀ M)⁻¹ Mᵀ x, then the residual variance as the mean square of e."""
    matrix, target = embedding_matrix(samples, order)
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    residual = target - matrix @ coefficients
    return coefficients, residual, float(np.mean(residual**2))


def yule_walker_ar(samples: np.ndarray, order: int) -> np.ndarray:
    """a = R⁻¹ r using the un-normalised lags in the notes' equation (4.18)."""
    values = np.asarray(samples, dtype=np.float64)
    values = values - float(np.mean(values))
    lags = biased_autocorrelation(values, order)
    matrix = np.empty((order, order), dtype=np.float64)
    rhs = np.empty(order, dtype=np.float64)
    for row in range(order):
        rhs[row] = lags[row + 1]
        for column in range(order):
            matrix[row, column] = lags[abs(row - column)]
    return np.linalg.solve(matrix, rhs)


def ar_psd(coefficients: np.ndarray, noise_variance: float, digital_omega: np.ndarray) -> np.ndarray:
    """Pxx(ω) = σe² / |1 − sum_k a_k exp(−j k ω)|², with ω in radians per sample."""
    omega = np.asarray(digital_omega, dtype=np.float64)
    denominator = np.ones(omega.shape, dtype=np.complex128)
    for lag, coefficient in enumerate(np.asarray(coefficients, dtype=np.float64), start=1):
        denominator -= coefficient * np.exp(-1j * lag * omega)
    return float(noise_variance) / np.abs(denominator) ** 2


def synthesise_ar(coefficients: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Run xt = sum_k a_k x_{t−k} + e_t from zero initial conditions."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    driving = np.asarray(noise, dtype=np.float64)
    output = np.zeros(driving.shape, dtype=np.float64)
    order = coefficients.size
    for time in range(driving.size):
        acc = driving[time]
        for lag, coefficient in enumerate(coefficients, start=1):
            if time - lag >= 0:
                acc += coefficient * output[time - lag]
        output[time] = acc
    return output
