"""在同一幅观测、同一个算子 A 上比较几种复原。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.b14.image_processing.lecture3.degradation import Observation
from src.b14.image_processing.lecture3.map_estimation import objective, objective_parts, run_gradient_descent, solve_map_cg, solve_map_fourier
from src.b14.image_processing.metrics import mse, psnr
from src.b14.image_processing.lecture3.wiener import inverse_filter, wiener_filter


@dataclass
class MethodResult:
    key: str
    name: str
    image: np.ndarray
    seconds: float
    mse: float
    psnr: float
    objective: float
    note: str


def _pack(
    key: str,
    name: str,
    image: np.ndarray,
    seconds: float,
    observation: Observation,
    lam: float,
    note: str,
) -> MethodResult:
    return MethodResult(
        key=key,
        name=name,
        image=np.asarray(image, dtype=np.float64),
        seconds=float(seconds),
        mse=mse(image, observation.original),
        psnr=psnr(image, observation.original),
        objective=objective(image, observation.observed, observation.otf, lam),
        note=note,
    )


def run_comparison(
    observation: Observation,
    lam: float,
    wiener_k: float,
    inverse_threshold: float,
    gd_iterations: int,
    step_fraction: float = 1.0,
) -> list[MethodResult]:
    """五种思路：不复原、逆滤波、维纳、MAP 梯度下降、MAP 直接求解。

    直接求解同时给出傅里叶闭式解和矩阵自由共轭梯度。
    只有周期卷积加周期差分时，傅里叶解才是法方程的精确解。
    原图只在这里计算 MSE / PSNR，不进入任何一个求解器。
    """
    observed = observation.observed
    otf = observation.otf
    results: list[MethodResult] = []

    results.append(
        _pack(
            "observed",
            "不复原（观测 g）",
            observed,
            0.0,
            observation,
            lam,
            "直接把 g 当作估计。它不是 J 的最小值。",
        )
    )

    started = time.perf_counter()
    inverse, suppressed = inverse_filter(observed, otf, inverse_threshold)
    results.append(
        _pack(
            "inverse",
            "朴素逆滤波（inverse filter）",
            inverse,
            time.perf_counter() - started,
            observation,
            lam,
            f"硬阈值 {inverse_threshold:.2e}，抑制了 {suppressed} 个频率。",
        )
    )

    started = time.perf_counter()
    wiener = wiener_filter(observed, otf, wiener_k)
    results.append(
        _pack(
            "wiener",
            "维纳滤波（Wiener）",
            wiener,
            time.perf_counter() - started,
            observation,
            lam,
            f"常数 K = {wiener_k:.3e}，对应白先验而不是梯度先验。",
        )
    )

    started = time.perf_counter()
    gd = run_gradient_descent(
        observed,
        otf,
        lam,
        n_iterations=int(gd_iterations),
        step_fraction=step_fraction,
        initialisation="observed",
        project=False,
        max_snapshots=4,
    )
    gd_note = f"普通梯度下降 {gd.iteration[-1]} 步，步长比例 {step_fraction:.2f}。"
    if gd.diverged:
        gd_note += " 迭代发散。"
    results.append(
        _pack(
            "gd",
            "MAP 梯度下降（gradient descent）",
            gd.final,
            time.perf_counter() - started,
            observation,
            lam,
            gd_note,
        )
    )

    started = time.perf_counter()
    fourier = solve_map_fourier(observed, otf, lam)
    results.append(
        _pack(
            "fourier",
            "MAP 傅里叶闭式解（Fourier MAP）",
            fourier,
            time.perf_counter() - started,
            observation,
            lam,
            "周期算子下二次目标的精确最小值。",
        )
    )

    started = time.perf_counter()
    cg_image, info = solve_map_cg(observed, otf, lam)
    cg_note = "矩阵自由共轭梯度，解的是同一条法方程。"
    if info == 0:
        cg_note += " 已达到容差。"
    elif info > 0:
        cg_note += f" 在 {info} 次迭代内未完全收敛。"
    else:
        cg_note += f" 求解器返回 info = {info}。"
    results.append(
        _pack(
            "cg",
            "MAP 共轭梯度（conjugate gradient）",
            cg_image,
            time.perf_counter() - started,
            observation,
            lam,
            cg_note,
        )
    )
    return results


def lambda_sweep(observation: Observation, lambdas: np.ndarray) -> dict[str, np.ndarray]:
    """对一组 λ 计算傅里叶 MAP。PSNR 用原图评估，求解器只看到 g。

    不同 λ 定义了不同的目标 J，所以这里记录的是解的数据保真项，而不是 J 本身。
    """
    psnr_values = []
    data_values = []
    mse_values = []
    for lam in np.asarray(lambdas, dtype=np.float64):
        estimate = solve_map_fourier(observation.observed, observation.otf, float(lam))
        _total, data, _regularisation = objective_parts(estimate, observation.observed, observation.otf, float(lam))
        psnr_values.append(psnr(estimate, observation.original))
        data_values.append(data)
        mse_values.append(mse(estimate, observation.original))
    return {
        "lambda": np.asarray(lambdas, dtype=np.float64),
        "psnr": np.asarray(psnr_values, dtype=np.float64),
        "data": np.asarray(data_values, dtype=np.float64),
        "mse": np.asarray(mse_values, dtype=np.float64),
    }
