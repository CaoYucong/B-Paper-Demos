"""合成图像、上传解码和损失曲线图。"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.b14.image_processing.lecture3.bayes import suggested_axis
from src.b14.image_processing.synthetic_images import IMAGE_BUILDERS, load_grayscale_image, make_image
from src.ui.visualization import loss_matplotlib, posterior_figure


def test_builtin_images_are_normalised_and_not_blank():
    for name in IMAGE_BUILDERS:
        image = make_image(name, 48)
        assert image.shape == (48, 48)
        assert image.dtype == np.float64
        assert image.min() >= 0.0
        assert image.max() <= 1.0
        assert np.std(image) > 0.05


def test_png_upload_becomes_square_grayscale():
    picture = Image.new("RGB", (20, 10), (20, 180, 40))
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    array = load_grayscale_image(buffer.getvalue(), size=16)
    assert array.shape == (16, 16)
    assert array.min() >= 0.0
    assert array.max() <= 1.0
    assert array.mean() > 0.2


def test_loss_and_posterior_figures_build():
    iteration = np.arange(5)
    figure = loss_matplotlib(iteration, np.linspace(1.0, 0.4, 5), np.linspace(0.8, 0.3, 5), np.linspace(0.2, 0.1, 5))
    assert len(figure.axes[0].lines) == 3
    plt.close(figure)

    samples = suggested_axis(1.0, 1.0, 0.4, 0.0, 1.0)
    interactive = posterior_figure(samples, np.ones_like(samples), np.ones_like(samples), np.ones_like(samples), 0.4)
    assert len(interactive.data) == 3
