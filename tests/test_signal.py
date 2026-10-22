"""Checks against the formulas printed in the Signal Analysis notes."""

from __future__ import annotations

import math

import numpy as np

from src.b14.signal_analysis.lecture1.analog import (
    BUTTERWORTH_TABLE,
    butterworth_order,
    butterworth_polynomial_gain,
    butterworth_squared_magnitude,
    discrete_convolution,
    first_difference,
    ideal_lowpass_kernel,
    sampled_sinusoid,
    stable_butterworth_poles,
)
from src.b14.signal_analysis.lecture3.design import (
    EXAMPLE_FIR,
    EXAMPLE_RECURSION,
    HANNING_TAPS,
    bilinear_analog_frequency,
    cosine_window,
    example_order_estimate,
    example_prewarp,
    ideal_lowpass_taps,
    iir_frequency_response,
    windowed_fir,
)
from src.b14.signal_analysis.lecture2.digital import (
    PARABOLIC_TAPS,
    alias_pair,
    causal_stable_alpha,
    fir_frequency_response,
    parabolic_centre,
    parabolic_closed_form,
)
from src.b14.signal_analysis.practice.problems import PROBLEMS, QUIZ, grade_numeric
from src.b14.signal_analysis.lecture4.spectral import (
    DFT_EXAMPLE_SPECTRUM,
    REPORTED_AR4,
    ar_psd,
    biased_autocorrelation,
    dft,
    dft_example_samples,
    idft,
    least_squares_ar,
    periodogram_bins,
    synthesise_ar,
    yule_walker_ar,
)


def test_alias_pair_shares_samples():
    low, high = alias_pair(np.arange(8))
    assert np.allclose(low, high)


def test_sampling_relation_and_first_difference_superposition():
    _, samples = sampled_sinusoid(2.0, 3.0, 0.4, 40.0, 20)
    index = np.arange(20)
    expected = 2.0 * np.cos(2.0 * math.pi * (3.0 / 40.0) * index + 0.4)
    assert np.allclose(samples, expected)
    first = np.linspace(-1.0, 1.0, 12)
    second = np.cos(np.linspace(0.0, 3.0, 12))
    combined = first_difference(2.0 * first - 0.5 * second)
    separate = 2.0 * first_difference(first) - 0.5 * first_difference(second)
    assert np.allclose(combined, separate)


def test_discrete_convolution_matches_the_sum():
    signal = np.array([1.0, 2.0, 0.0])
    impulse = np.array([1.0, -1.0])
    assert np.allclose(discrete_convolution(signal, impulse), [1.0, 1.0, -2.0, 0.0])


def test_ideal_lowpass_is_the_sinc_in_the_notes():
    cutoff = 2.0
    assert ideal_lowpass_kernel(np.array([0.0]), cutoff)[0] == cutoff / math.pi
    assert abs(ideal_lowpass_kernel(np.array([math.pi / cutoff]), cutoff)[0]) < 1e-12


def test_butterworth_specification_rounds_up_to_seven():
    order = butterworth_order(100.0, 2.0)
    assert 6.6 < order < 6.7
    omega = np.array([0.0, 500.0])
    squared = butterworth_squared_magnitude(omega, 500.0, 7)
    assert squared[0] == 1.0
    assert abs(squared[1] - 0.5) < 1e-12
    angles = np.angle(stable_butterworth_poles(5), deg=True) % 360
    assert np.allclose(sorted(angles), [108.0, 144.0, 180.0, 216.0, 252.0])
    gain = butterworth_polynomial_gain(2j, 7)
    attenuation = 1.0 / abs(gain)
    assert 120.0 < attenuation < 140.0
    assert len(BUTTERWORTH_TABLE[7]) == 8


def test_parabolic_gain_and_centre_coefficient():
    window = np.array([4.0, 1.0, 0.0, 1.0, 4.0])
    assert parabolic_centre(window) == 0.0
    omega = np.linspace(0.0, math.pi, 64)
    direct = fir_frequency_response(PARABOLIC_TAPS, omega)
    closed = parabolic_closed_form(omega)
    assert np.allclose(direct, closed)
    assert abs(closed[0] - 1.0) < 1e-12


def test_hanning_zero_and_window_endpoints():
    nyquist = fir_frequency_response(HANNING_TAPS, np.array([math.pi]))[0]
    assert abs(nyquist) < 1e-12
    hanning = cosine_window(10, 0.5)
    hamming = cosine_window(10, 0.54)
    assert abs(hanning[0]) < 1e-12
    assert abs(hamming[0] - 0.08) < 1e-12
    assert abs(hanning[10] - 1.0) < 1e-12


def test_windowed_lowpass_cutoff_and_ripple():
    cutoff = math.pi / 4.0
    assert abs(ideal_lowpass_taps(10, cutoff)[10] - 0.25) < 1e-12
    omega = np.linspace(0.0, math.pi, 512)
    rectangular = np.abs(fir_frequency_response(windowed_fir(10, cutoff, None), omega))
    hamming = np.abs(fir_frequency_response(windowed_fir(10, cutoff, 0.54), omega))
    passband = omega < cutoff * 0.5
    rect_ripple = float(np.max(rectangular[passband]) - np.min(rectangular[passband]))
    hamm_ripple = float(np.max(hamming[passband]) - np.min(hamming[passband]))
    assert hamm_ripple < rect_ripple


def test_bilinear_example_matches_the_printed_frequencies():
    period = 1e-4
    digital = 0.4 * math.pi
    analog = float(bilinear_analog_frequency(np.array([digital]), period)[0])
    assert abs(analog - 2e4 * math.tan(0.2 * math.pi)) < 1e-6
    analog_pass, analog_stop = example_prewarp()
    assert abs(analog_pass - 14.53e3) < 20.0
    assert 1.9 < example_order_estimate(analog_pass, analog_stop) < 2.0
    assert causal_stable_alpha(0.8)
    assert not causal_stable_alpha(1.0)
    dc = iir_frequency_response(EXAMPLE_FIR, EXAMPLE_RECURSION, np.array([0.0]))[0]
    cutoff_gain = iir_frequency_response(EXAMPLE_FIR, EXAMPLE_RECURSION, np.array([0.4 * math.pi]))[0]
    assert abs(abs(dc) - 1.0) < 1e-4
    assert abs(abs(cutoff_gain) - 0.707) < 0.002


def test_dft_example_and_inverse():
    samples = dft_example_samples()
    assert np.allclose(samples, [8.0, 4.0, 8.0, 0.0])
    spectrum = dft(samples)
    assert np.allclose(spectrum, DFT_EXAMPLE_SPECTRUM)
    assert np.allclose(idft(spectrum), samples)
    bins = periodogram_bins(samples)
    assert np.allclose(bins, np.abs(DFT_EXAMPLE_SPECTRUM) ** 2 / 4.0)


def _frequency_samples(samples: np.ndarray, length: int) -> np.ndarray:
    """Samples of the length-N transform at L equispaced frequencies."""
    values = np.asarray(samples, dtype=np.complex128)
    k = np.arange(values.size)
    n = np.arange(length).reshape(-1, 1)
    return np.exp(-2j * math.pi * n * k / length) @ values


def test_zero_padding_keeps_the_sequence_and_a_short_dft_does_not():
    samples = np.array([1.0, 2.0, 3.0, 4.0])
    padded = np.concatenate([samples, np.zeros(4)])
    recovered = np.real(idft(dft(padded)))
    assert np.allclose(recovered[:4], samples)
    aliased = np.real(idft(_frequency_samples(samples, 2)))
    assert np.allclose(aliased, [4.0, 6.0])


def test_ar_fit_recovers_a_synthesised_model():
    coefficients = np.array([0.6, -0.3])
    noise = np.zeros(12)
    noise[0] = 1.0
    series = synthesise_ar(coefficients, noise)
    estimated, residual, variance = least_squares_ar(series, 2)
    assert np.allclose(estimated, coefficients, atol=1e-8)
    assert variance == float(np.mean(residual**2))
    # A long, nearly noiseless series lets Yule-Walker approach the same coefficients.
    rng = np.random.default_rng(0)
    long_noise = 0.01 * rng.normal(size=4000)
    long_series = synthesise_ar(coefficients, long_noise)
    walker = yule_walker_ar(long_series, 2)
    assert np.allclose(walker, coefficients, atol=0.05)
    omega = np.linspace(0.0, math.pi, 32)
    spectrum = ar_psd(REPORTED_AR4, 0.079, omega)
    assert spectrum.shape == omega.shape
    assert np.all(spectrum > 0.0)
    lags = biased_autocorrelation(np.array([1.0, 0.0, 0.0, 0.0]), 2)
    assert np.allclose(lags, [0.25, 0.0, 0.0])


def test_practice_anchors_exist_on_the_signal_page():
    from src.b14.signal_analysis.experiments import (
        SA_L1_EXPERIMENTS,
        SA_L2_EXPERIMENTS,
        SA_L3_EXPERIMENTS,
        SA_L4_EXPERIMENTS,
    )
    from src.ui.anchors import section_anchor
    drawn = {
        section_anchor(item)
        for group in (SA_L1_EXPERIMENTS, SA_L2_EXPERIMENTS, SA_L3_EXPERIMENTS, SA_L4_EXPERIMENTS)
        for item in group
    }
    assert len(drawn) == 6 + 6 + 5 + 5
    for problem in PROBLEMS:
        assert problem.anchor in drawn


def test_signal_practice_is_original_and_graded():
    assert len(PROBLEMS) == 8
    assert len(QUIZ) == 8
    for problem in PROBLEMS:
        assert grade_numeric(problem.id, problem.answer)
        assert not grade_numeric(problem.id, problem.answer + problem.tolerance + 1.0)
    assert all(0 <= item.correct < len(item.choices_en) for item in QUIZ)
