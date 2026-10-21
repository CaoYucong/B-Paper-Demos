"""Integer-shift super-resolution by interleaving known samples.

The lecture introduces a generative model for super-resolution and marks
the full sketch solution as not examinable. This file is a supplementary
illustration: each observation keeps one pixel phase of a high-resolution
image,

    g_{a,b}[p, q] = f[2p+a, 2q+b] + n_{a,b}[p, q].

It is not a camera-motion or panorama solver.
"""

from __future__ import annotations

import numpy as np


def four_phase_observations(
    image: np.ndarray, noise_std: float, seed: int
) -> dict[tuple[int, int], np.ndarray]:
    image = np.asarray(image, dtype=np.float64)
    if image.shape[0] % 2 or image.shape[1] % 2:
        raise ValueError("the high-resolution image must have even side lengths.")
    rng = np.random.default_rng(int(seed))
    observations: dict[tuple[int, int], np.ndarray] = {}
    for row_phase in (0, 1):
        for col_phase in (0, 1):
            sample = image[row_phase::2, col_phase::2]
            observations[(row_phase, col_phase)] = sample + float(noise_std) * rng.standard_normal(sample.shape)
    return observations


def fuse_integer_shifts(
    observations: dict[tuple[int, int], np.ndarray], shape: tuple[int, int]
) -> np.ndarray:
    """Place each phase back onto its high-resolution coordinates.

    With one observation per phase this is the maximum-likelihood estimate
    under independent Gaussian noise, because the samples do not overlap.
    """
    fused = np.zeros(shape, dtype=np.float64)
    for (row_phase, col_phase), sample in observations.items():
        fused[row_phase::2, col_phase::2] = sample
    return fused
