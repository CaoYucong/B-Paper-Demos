"""界面默认值。恢复默认参数时回到这里。"""

from __future__ import annotations

DEFAULTS = {
    "image_size": 128,
    "image_source": "内置图像",
    "image_name": "锐边渐变",
    "sigma": 2.0,
    "noise_std": 0.03,
    "seed": 0,
    "log10_lambda": -1.3,
    "lr_fraction": 1.0,
    "n_iters": 300,
    "init_mode": "观测图像 g",
    "use_projection": False,
    "show_psf": True,
    "show_noise": False,
    "wiener_log10_k": -2.0,
    "inverse_log10_threshold": -3.0,
    "compare_iters": 250,
    "bayes_g": 1.5,
    "bayes_a": 1.0,
    "bayes_sigma_n": 0.35,
    "bayes_mu": 0.0,
    "bayes_sigma_f": 1.0,
    "toy_n": 6,
    "toy_row": 0,
    "show_motion": False,
}

INIT_TO_CODE = {
    "观测图像 g": "observed",
    "全零": "zeros",
}
