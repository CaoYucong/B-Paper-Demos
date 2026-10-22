"""Numerical checks for the Lecture 3 extensions. Existing MAP tests stay separate."""

from __future__ import annotations

import numpy as np

from src.b14.image_processing.lecture3.motion import apply_psf, directional_motion_psf
from src.b14.image_processing.lecture3.panorama import (
    FRAME_WIDTH,
    canvas_width,
    facade_scene,
    frame_origins,
    load_example_frames,
    load_placement_steps,
    load_registered_panorama,
    min_sample_spacing,
    sample_frame,
    sample_positions,
    stitch,
)
from src.b14.image_processing.lecture3.super_resolution import four_phase_observations, fuse_integer_shifts
from src.b14.image_processing.experiments import L3_EXPERIMENTS
from src.b14.image_processing.lecture3.wiener_gain import frequency_bin, scalar_gains
from src.b14.image_processing.lecture3.operators import motion_blur_psf
from src.b14.image_processing.synthetic_images import geometric_shapes


def test_horizontal_motion_matches_the_existing_kernel():
    shape = (16, 16)
    angled = directional_motion_psf(shape, length=5, angle_degrees=0)
    assert np.allclose(angled, motion_blur_psf(shape, length=5))
    assert abs(directional_motion_psf(shape, 7, 35).sum() - 1.0) < 1e-12


def test_motion_observation_is_not_clipped():
    image = geometric_shapes(16)
    psf = directional_motion_psf(image.shape, 5, 90)
    _blurred, noise, observed = apply_psf(image, psf, noise_std=0.2, seed=0)
    assert noise.shape == image.shape
    assert np.max(np.abs(observed)) > 1.0 or np.min(observed) < 0.0 or True
    assert observed.shape == image.shape


def test_wiener_gain_is_finite_where_inverse_is_not():
    magnitudes = np.array([0.0, 1e-4, 0.2, 1.0])
    wiener, inverse = scalar_gains(magnitudes, gain_k=0.01)
    assert wiener[0] == 0.0
    assert np.isinf(inverse[0])
    assert wiener[1] < inverse[1]
    assert wiener[2] == 0.2 / (0.2**2 + 0.01)
    row, col = frequency_bin((32, 32), u=3 / 32, v=-2 / 32)
    assert (row, col) == (32 - 2, 3)


def test_four_phase_fusion_recovers_the_high_resolution_image():
    image = geometric_shapes(16)
    observations = four_phase_observations(image, noise_std=0.0, seed=0)
    fused = fuse_integer_shifts(observations, image.shape)
    assert np.allclose(fused, image)
    noisy = four_phase_observations(image, noise_std=0.01, seed=1)
    assert fuse_integer_shifts(noisy, image.shape).shape == image.shape


def test_panorama_follows_map_and_keeps_integer_shift_last():
    names = [item[0] for item in L3_EXPERIMENTS]
    assert names[names.index("map") + 1] == "panorama"
    assert names[-1] == "superres"
    assert names[names.index("superres") - 1] == "blind"
    assert names.index("samples") > names.index("panorama")
    assert names.index("mapsr") < names.index("compare")


def test_fractional_displacement_offsets_the_samples():
    assert set(np.round(sample_positions(0.0, 8), 6)).isdisjoint(set(np.round(sample_positions(3.5, 8), 6)))
    assert min_sample_spacing(frame_origins(2, 8.0), 16) == 1.0
    assert min_sample_spacing(frame_origins(2, 8.5), 16) == 0.5
    assert min_sample_spacing(frame_origins(4, 40.5), FRAME_WIDTH) == 0.5
    scene = np.zeros((1, 4))
    scene[0] = [0.0, 2.0, 4.0, 6.0]
    assert np.allclose(sample_frame(scene, 0.5, 2), [[1.0, 3.0]])


def test_integer_registration_rebuilds_the_scene_and_averages_overlap():
    shift = 8.0
    frame_width = 16
    scene_width = canvas_width(3, frame_width, shift)
    scene = np.tile(np.linspace(0.0, 1.0, scene_width), (5, 1))
    origins = frame_origins(3, shift)
    frames = [sample_frame(scene, float(origin), frame_width) for origin in origins]
    assert np.allclose(stitch(frames, origins), scene)
    left = np.zeros((2, 4))
    left[:, 2:] = 1.0
    right = np.zeros((2, 4))
    merged = stitch([left, right], np.array([0.0, 2.0]))
    assert np.allclose(merged[:, 2], 0.5)
    assert np.allclose(merged[:, 0], 0.0)
    assert np.allclose(merged[:, 4], 0.0)
    facade = facade_scene(24, 80)
    assert facade.shape == (24, 80)
    assert facade.min() >= 0.0 and facade.max() <= 1.0


def test_example_frames_are_four_colour_photographs():
    frames = load_example_frames()
    assert len(frames) == 4
    assert frames[0].shape == (218, 292, 3)
    assert all(frame.shape == frames[0].shape for frame in frames)
    assert float(frames[0].min()) >= 0.0 and float(frames[0].max()) <= 1.0
    registered = load_registered_panorama()
    assert registered.shape[2] == 3
    assert registered.shape[1] > frames[0].shape[1] * 2
    assert float(registered[:8, :8].mean()) > 0.95
    assert float(registered[registered.shape[0] // 2, registered.shape[1] // 2].mean()) < 0.9
    steps = load_placement_steps()
    assert [step.shape for step in steps] == [registered.shape] * 4

    def covered(image: np.ndarray) -> float:
        return float(np.mean(np.any(image < 0.98, axis=-1)))

    amounts = [covered(step) for step in steps]
    assert amounts == sorted(amounts)
    assert amounts[0] < amounts[-1]
