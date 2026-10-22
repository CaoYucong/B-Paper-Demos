"""二次 MAP 目标、解析梯度、梯度下降，以及两条直接求解路径。

平均损失（除以像素数 N）：

    J(f) = (1/N) ||A f - g||^2 + λ (1/N) ||D f||^2

解析梯度：

    ∇J(f) = (2/N) [ A^T (A f - g) + λ D^T D f ]

普通梯度下降：

    f <- f - η ∇J(f)

因为梯度里已经有 1/N，η 的数值会比较大。等价的重缩放更新是

    f <- f - 2 α [ A^T (A f - g) + λ D^T D f ]，   α = η / N

二者路径相同，最小值也相同。本模块只使用第一种，避免把平均损失
和未平均的梯度混在一起。

在周期边界下，A 和 D 都是循环矩阵，傅里叶基把法方程对角化：

    F_hat = H* G / (|H|^2 + λ (|D_x|^2 + |D_y|^2))

共轭梯度解的是同一个法方程，只是不把 A 写成矩阵。
原图 f 从不进入这些函数。
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from src.b14.image_processing.lecture3.operators import (
    adjoint_diff_x,
    adjoint_diff_y,
    apply_otf,
    apply_otf_adjoint,
    forward_diff_x,
    forward_diff_y,
    frequency_gradient_power,
)


@dataclass
class StepSize:
    """由 Hessian 的精确谱半径换算得到的步长。"""

    eta: float
    alpha: float
    lipschitz: float
    spectral_radius: float
    fraction: float
    stable_limit: float
    theoretically_stable: bool


@dataclass
class GDResult:
    """梯度下降的轨迹。snapshots 是稀疏保存的 float32 图像。"""

    final: np.ndarray
    snapshot_iterations: np.ndarray
    snapshots: list[np.ndarray]
    iteration: np.ndarray
    total_loss: np.ndarray
    data_loss: np.ndarray
    regularisation_loss: np.ndarray
    grad_norm: np.ndarray
    step: StepSize
    projected: bool
    diverged: bool
    initialisation: str


def evaluate(image: np.ndarray, observed: np.ndarray, otf: np.ndarray, lam: float):
    """同时计算 J 的三个部分和 ∇J。热循环里只走这一条路径。"""
    residual = apply_otf(image, otf) - observed
    data_grad = apply_otf_adjoint(residual, otf)
    dx = forward_diff_x(image)
    dy = forward_diff_y(image)
    reg_grad = adjoint_diff_x(dx) + adjoint_diff_y(dy)
    n_pixels = image.size
    data = float(np.dot(residual.ravel(), residual.ravel()) / n_pixels)
    regularisation = float((np.dot(dx.ravel(), dx.ravel()) + np.dot(dy.ravel(), dy.ravel())) / n_pixels)
    gradient = (2.0 / n_pixels) * (data_grad + float(lam) * reg_grad)
    total = data + float(lam) * regularisation
    return total, data, regularisation, gradient


def objective_parts(image: np.ndarray, observed: np.ndarray, otf: np.ndarray, lam: float):
    total, data, regularisation, _gradient = evaluate(image, observed, otf, lam)
    return total, data, regularisation


def objective(image: np.ndarray, observed: np.ndarray, otf: np.ndarray, lam: float) -> float:
    total, _data, _regularisation, _gradient = evaluate(image, observed, otf, lam)
    return total


def compute_gradient(image: np.ndarray, observed: np.ndarray, otf: np.ndarray, lam: float) -> np.ndarray:
    _total, _data, _regularisation, gradient = evaluate(image, observed, otf, lam)
    return gradient


def hessian_spectral_radius(otf: np.ndarray, lam: float) -> float:
    """A^T A + λ D^T D 的最大特征值。周期算子在傅里叶基下对角化，所以这个值是精确的。"""
    power = frequency_gradient_power(otf.shape)
    return float(np.max(np.abs(otf) ** 2 + float(lam) * power))


def lipschitz_constant(otf: np.ndarray, lam: float) -> float:
    """∇J 的 Lipschitz 常数 L = (2/N) ρ。普通梯度下降要求 0 < η < 2/L。"""
    n_pixels = otf.size
    return (2.0 / n_pixels) * hessian_spectral_radius(otf, lam)


def step_size_from_fraction(otf: np.ndarray, lam: float, fraction: float) -> StepSize:
    """fraction = 1 表示 η = 1/L；fraction = 2 是稳定边界。"""
    rho = hessian_spectral_radius(otf, lam)
    n_pixels = int(otf.size)
    lipschitz = (2.0 / n_pixels) * rho
    if lipschitz <= 0:
        raise ValueError("Lipschitz 常数必须为正。")
    eta = float(fraction) / lipschitz
    alpha = eta / n_pixels
    return StepSize(
        eta=eta,
        alpha=alpha,
        lipschitz=lipschitz,
        spectral_radius=rho,
        fraction=float(fraction),
        stable_limit=2.0 / lipschitz,
        theoretically_stable=float(fraction) < 2.0,
    )


def solve_map_fourier(observed: np.ndarray, otf: np.ndarray, lam: float) -> np.ndarray:
    """周期二次 MAP 的闭式解。λ 与平均损失里的 λ 是同一个数，1/N 和系数 2 都会消掉。"""
    if lam < 0:
        raise ValueError("λ 不能为负。")
    power = frequency_gradient_power(observed.shape)
    denominator = np.abs(otf) ** 2 + float(lam) * power
    numerator = np.conj(otf) * np.fft.fft2(observed)
    restored = np.zeros_like(numerator)
    mask = denominator > 1e-15
    restored[mask] = numerator[mask] / denominator[mask]
    return np.fft.ifft2(restored).real


def apply_normal_operator(image: np.ndarray, otf: np.ndarray, lam: float) -> np.ndarray:
    """法方程左边的矩阵自由算子 (A^T A + λ D^T D) f。"""
    return apply_otf_adjoint(apply_otf(image, otf), otf) + float(lam) * (
        adjoint_diff_x(forward_diff_x(image)) + adjoint_diff_y(forward_diff_y(image))
    )


def _call_cg(operator: LinearOperator, rhs: np.ndarray, maxiter: int, rtol: float):
    """兼容 SciPy 把 tol 改名为 rtol 的两个版本。"""
    kwargs: dict = {"maxiter": maxiter}
    parameters = inspect.signature(cg).parameters
    if "rtol" in parameters:
        kwargs["rtol"] = rtol
    elif "tol" in parameters:
        kwargs["tol"] = rtol
    return cg(operator, rhs, **kwargs)


def solve_map_cg(
    observed: np.ndarray,
    otf: np.ndarray,
    lam: float,
    rtol: float = 1e-8,
    maxiter: int | None = None,
) -> tuple[np.ndarray, int]:
    """用共轭梯度解 (A^T A + λ D^T D) f = A^T g。不构造矩阵。

    返回 (图像, info)。info 为 0 表示达到容差；正数表示用完了迭代次数。
    """
    if lam < 0:
        raise ValueError("λ 不能为负。")
    shape = observed.shape
    n_pixels = observed.size
    if maxiter is None:
        maxiter = int(min(800, max(100, 4 * max(shape))))

    def matvec(vector: np.ndarray) -> np.ndarray:
        image = np.reshape(vector, shape)
        return apply_normal_operator(image, otf, lam).ravel()

    operator = LinearOperator((n_pixels, n_pixels), matvec=matvec, dtype=np.float64)
    rhs = apply_otf_adjoint(observed, otf).ravel()
    solution, info = _call_cg(operator, rhs, maxiter=maxiter, rtol=rtol)
    return np.reshape(np.real(solution), shape), int(info)


def run_gradient_descent(
    observed: np.ndarray,
    otf: np.ndarray,
    lam: float,
    n_iterations: int,
    step_fraction: float = 1.0,
    initialisation: str = "observed",
    project: bool = False,
    max_snapshots: int = 48,
    on_iterate: Callable[[int, int, np.ndarray, bool], None] | None = None,
) -> GDResult:
    """最小化平均损失 J。ground truth 不是参数，也不能从观测里反推出来。

    project=True 时，每一步之后把像素裁剪到 [0, 1]。
    那是投影梯度下降，稳态一般不再等于无约束傅里叶解。
    """
    if n_iterations < 1:
        raise ValueError("迭代次数至少为 1。")
    if lam < 0:
        raise ValueError("λ 不能为负。")
    if initialisation == "zeros":
        current = np.zeros(observed.shape, dtype=np.float64)
    elif initialisation == "observed":
        current = np.array(observed, dtype=np.float64, copy=True)
    else:
        raise ValueError("initialisation 只能是 'zeros' 或 'observed'。")

    step = step_size_from_fraction(otf, lam, step_fraction)
    if project:
        current = np.clip(current, 0.0, 1.0)

    frame_count = min(max(int(max_snapshots), 1), int(n_iterations) + 1)
    if frame_count == 1:
        keep_steps = {0}
    else:
        keep_steps = {int(i * int(n_iterations) / (frame_count - 1)) for i in range(frame_count)}
    iterations: list[int] = []
    totals: list[float] = []
    datas: list[float] = []
    regs: list[float] = []
    norms: list[float] = []
    snapshot_iterations: list[int] = []
    snapshots: list[np.ndarray] = []
    diverged = False
    last_finite = np.array(current, dtype=np.float64, copy=True)

    for k in range(int(n_iterations) + 1):
        total, data, regularisation, gradient = evaluate(current, observed, otf, lam)
        grad_norm = float(np.linalg.norm(gradient))
        # 直流分量往往是曲率最大的方向，但模糊不改变平均值，所以略大于 2/L 的步长
        # 可能先下降。损失或图像一旦大到失去教学意义，就停止并标记为发散。
        image_scale = float(np.max(np.abs(current))) if current.size else 0.0
        if not np.isfinite(total) or not np.isfinite(grad_norm) or total > 1e6 or image_scale > 1e3:
            diverged = True
            break
        last_finite = np.array(current, dtype=np.float64, copy=True)
        iterations.append(k)
        totals.append(total)
        datas.append(data)
        regs.append(float(lam) * regularisation)
        norms.append(grad_norm)
        saved = k in keep_steps
        if saved:
            snapshot_iterations.append(k)
            snapshots.append(np.array(current, dtype=np.float32))
        if on_iterate is not None:
            on_iterate(k, int(n_iterations), current, saved)
        if k == int(n_iterations):
            break
        nxt = current - step.eta * gradient
        if project:
            nxt = np.clip(nxt, 0.0, 1.0)
        if not np.isfinite(nxt).all():
            diverged = True
            break
        current = nxt

    if not iterations:
        raise RuntimeError("梯度下降未能完成第一次评估。请检查观测和步长。")

    if not snapshot_iterations or snapshot_iterations[-1] != iterations[-1]:
        snapshot_iterations.append(iterations[-1])
        snapshots.append(np.array(last_finite, dtype=np.float32))

    return GDResult(
        final=last_finite,
        snapshot_iterations=np.asarray(snapshot_iterations, dtype=int),
        snapshots=snapshots,
        iteration=np.asarray(iterations, dtype=int),
        total_loss=np.asarray(totals, dtype=np.float64),
        data_loss=np.asarray(datas, dtype=np.float64),
        regularisation_loss=np.asarray(regs, dtype=np.float64),
        grad_norm=np.asarray(norms, dtype=np.float64),
        step=step,
        projected=bool(project),
        diverged=diverged,
        initialisation=initialisation,
    )


def snapshot_image(result: GDResult, index: int) -> np.ndarray:
    """最后一帧返回 float64 终值，中间帧返回节省内存的 float32 快照。"""
    iteration = int(result.snapshot_iterations[index])
    if iteration == int(result.iteration[-1]):
        return result.final
    return np.asarray(result.snapshots[index], dtype=np.float64)
