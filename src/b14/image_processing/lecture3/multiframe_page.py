"""Interactive pages for multi-frame super-resolution and a 1D blind demo.

The photographs on the panorama page stay there. These sections use a small
synthetic grid. Gradient descent runs only after its button.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.b14.image_processing.experiments import L3_EXPERIMENTS
from src.b14.image_processing.lecture3.multiframe import (
    blind_deblur_1d,
    corrupted_blur_problem,
    huber_loss,
    absolute_loss,
    initialise_estimate,
    lipschitz_constant,
    restore_huber_1d,
    restore_quadratic_1d,
    run_gradient_descent_multiframe,
    solve_normal_equations,
    squared_loss,
)
from src.b14.image_processing.lecture3.sampling import (
    FRAME_COLOURS,
    apply_camera,
    averaging_curve,
    factor_matrices_1d,
    high_resolution_pattern,
    make_camera,
    observe,
    observation_seed,
    report_stacked_system,
    spread_observations,
    window_mask,
    window_origins,
)
from src.ui.anchors import section_anchor
from src.ui.common import show_fig
from src.ui.components import (
    concept,
    controls_heading,
    line_figure,
    model,
    show_array,
    show_row,
    steps,
    takeaways,
    try_this,
    value_figure,
    visual_heading,
)
from src.ui.i18n import forget_translation_keys, say, shown_label, t
from src.ui.visualization import marked_grid_png

_PATTERNS = (
    ("combined", "Edge, line, checks, and a disk / 边缘、斜线、棋盘和圆盘"),
    ("edge", "Sharp edge / 锐边"),
    ("diagonal", "Diagonal line / 斜线"),
    ("checkerboard", "Checkerboard / 棋盘"),
    ("shape", "Square and disk / 方块和圆盘"),
)
_COLOUR_NAMES = (
    ("red", "红"),
    ("blue", "蓝"),
    ("green", "绿"),
    ("orange", "橙"),
    ("purple", "紫"),
    ("brown", "棕"),
    ("pink", "粉"),
    ("grey", "灰"),
)
_TOY = np.array([0.1, 0.1, 0.1, 0.9, 0.15, 0.85, 0.2, 0.8], dtype=np.float64)
_GUIDE = (
    ("samples", "One frame leaves high-resolution pixels unmeasured.", "一帧测不到全部高分辨率像素。"),
    ("benefits", "A shift can add a new sample. A repeat only averages noise.", "位移能补上新样本。重复只平均噪声。"),
    ("measure", "M is the matrix that turns f into one observation.", "M 是把 f 变成一幅观测的矩阵。"),
    ("stacked", "Stacking the matrices gives more equations. Rank decides if they are new.", "把矩阵叠起来会多出方程。秩决定它们是不是新信息。"),
    ("mapsr", "Least squares fits the equations. MAP adds a penalty, and gradient descent is one way to minimise it.", "最小二乘拟合这些方程。MAP 再加一项惩罚，梯度下降是一种最小化的方法。"),
    ("huber", "Huber loss stops a single huge residual from dominating the fit.", "Huber 损失让一个特别大的残差不能主导拟合。"),
    ("blind", "Blind deblurring estimates the blur as well as the image. The lecture marks it not examinable.", "盲去模糊要同时估计模糊和图像。讲义把它标成非考试内容。"),
)


def _seed() -> int:
    return int(st.session_state.get("seed", 0))


def _heading(en: str, zh: str) -> None:
    st.markdown(f"##### {t(en, zh)}")


def _looking(en: str, zh: str) -> None:
    _heading("What am I looking at?", "我在看什么？")
    say(en, zh)


def _misconception(en: str, zh: str) -> None:
    _heading("Common misconception", "常见误解")
    say(en, zh)


def _guide() -> None:
    _heading("Guided sequence", "学习顺序")
    say(
        "The panorama above is the lecture's courtyard. The sections below use a synthetic grid so the samples can be counted. Read them in this order.",
        "上面的全景是讲义里的庭院。下面几节用合成网格，这样数得清样本。按这个顺序读。",
    )
    for item_id, en, zh in _GUIDE:
        item = next(item for item in L3_EXPERIMENTS if item[0] == item_id)
        anchor = section_anchor(item)
        say(f"- [{en}](#{anchor})", f"- [{zh}](#{anchor})")


def _pattern_box(key: str) -> str:
    codes = [code for code, _label in _PATTERNS]
    labels = {code: label for code, label in _PATTERNS}
    return st.selectbox("High-resolution pattern / 高分辨率图案", codes, format_func=labels.get, key=key)


def _camera(factor: int, horizontal: float, vertical: float, blur: float, realistic: bool):
    return make_camera(factor, horizontal, vertical, blur if realistic else 0.0, realistic)


def _legend(count: int) -> None:
    english = ", ".join(en for en, _zh in _COLOUR_NAMES[:count])
    chinese = "、".join(zh for _en, zh in _COLOUR_NAMES[:count])
    say(
        f"Frame colours, in order: {english}.",
        f"各帧颜色按顺序是：{chinese}。",
    )


def _grid(image: np.ndarray, title: str, dots=None, shades=None) -> None:
    st.image(marked_grid_png(image, shown_label(title), dots=dots, shades=shades), use_container_width=True)


def render_samples() -> None:
    concept(
        "The intuitive model treats an image as point samples. One low-resolution camera reads only some of those points. A second camera, shifted by half a low-resolution pixel, reads a different set. The realistic mode then replaces the point sample by blur and a sensor window, so the same shift no longer reveals a single original pixel.",
        "直观模型把图像当成点样本。一台低分辨率相机只读到其中一部分点。再移半个低分辨率像素，读到的是另一组点。真实模式改成模糊加上传感器窗口，同一次位移就不再对应某一个原始像素。",
    )
    _guide()
    controls_heading()
    pattern = _pattern_box("sr-pattern")
    left, middle, right = st.columns(3)
    with left:
        size = int(st.selectbox("Grid size / 网格边长", [8, 16], index=1, key="sr-size"))
        factors = [1, 2, 4]
        if "sr-factor" in st.session_state and int(st.session_state["sr-factor"]) not in factors:
            st.session_state["sr-factor"] = 2
            forget_translation_keys("sr-factor")
        factor = int(
            st.selectbox(
                "Sampling factor / 采样倍数",
                factors,
                index=1,
                format_func=lambda value: t("1, no downsampling", "1，不降采样") if value == 1 else str(value),
                key="sr-factor",
            )
        )
    with middle:
        shifts = [0.0, 0.5] if factor > 1 else [0.0]
        for shift_key in ("sr-hx", "sr-vy"):
            if shift_key in st.session_state and float(st.session_state[shift_key]) not in shifts:
                st.session_state[shift_key] = 0.0
                forget_translation_keys(shift_key)
        horizontal = float(st.selectbox("Horizontal shift / 水平位移", shifts, key="sr-hx"))
        vertical = float(st.selectbox("Vertical shift / 垂直位移", shifts, key="sr-vy"))
    with right:
        noise = float(st.slider("Noise σ / 噪声", 0.0, 0.15, 0.0, 0.01, key="sr-noise"))
        blur = float(st.slider("Blur σ / 模糊", 0.0, 1.5, 0.0, 0.1, key="sr-blur"))
    realistic = st.radio(
        "Formation model / 成像模型",
        ["point", "realistic"],
        format_func=lambda code: t("Point samples", "点采样") if code == "point" else t("Blur and area integration", "模糊与面积积分"),
        horizontal=True,
        key="sr-model",
    ) == "realistic"
    signature = (pattern, size, factor, realistic, round(blur, 5) if realistic else 0.0)
    if st.session_state.get("sr_sig") != signature:
        st.session_state["sr_sig"] = signature
        st.session_state["sr_frames"] = [] if st.session_state.get("sr_ready") else [[0.0, 0.0]]
        st.session_state["sr_ready"] = True
    frames: list[list[float]] = st.session_state["sr_frames"]
    add, clear = st.columns(2)
    with add:
        if st.button("Add frame / 增加一帧", key="sr-add") and len(frames) < 8:
            frames.append([horizontal, vertical])
            st.session_state["sr-keep"] = len(frames)
    with clear:
        if st.button("Clear frames / 清空各帧", key="sr-clear"):
            frames = []
            st.session_state["sr_frames"] = frames
            st.session_state.pop("sr-keep", None)
    used = len(frames)
    if used > 1:
        if int(st.session_state.get("sr-keep", used)) > used:
            st.session_state["sr-keep"] = used
        if int(st.session_state.get("sr-keep", used)) < 1:
            st.session_state["sr-keep"] = 1
        keep = int(st.slider("Frames used, from the first saved frame / 使用前几帧", 1, used, used, key="sr-keep"))
        frames = frames[:keep]
    else:
        st.caption(t("No frame is saved yet. The current camera is only a preview.", "还没有保存任何一帧。当前相机只是预览。"))
    if realistic:
        say(
            "Blur applies only in this mode. The tint is the sensor window. A Gaussian wider than that window also mixes unmarked pixels, because a Gaussian never becomes exactly zero.",
            "只有这个模式使用模糊。色块是传感器窗口。比窗口更宽的高斯还会混入没有标出的像素，因为高斯不会精确变成零。",
        )
    else:
        if factor == 1:
            say(
                "Point sampling ignores the blur slider. Factor 1 keeps every pixel. The camera does not downsample, so one frame already measures the whole grid.",
                "点采样不用模糊滑块。倍数为 1 就是不降采样，每个像素都保留。一帧就已经测完整张网格。",
            )
        else:
            step = factor // 2
            say(
                "Point sampling ignores the blur slider. A shift of 0.5 low-resolution pixels is "
                f"{step} high-resolution pixel{'s' if step != 1 else ''}.",
                f"点采样不用模糊滑块。倍数为 {factor} 时，0.5 个低分辨率像素等于 {step} 个高分辨率像素。",
            )
    truth = high_resolution_pattern(pattern, size)
    current = _camera(factor, horizontal, vertical, blur, realistic)
    preview = observe(truth, current, noise, _seed() + 17)
    saved_cameras = [_camera(factor, frame[0], frame[1], blur, realistic) for frame in frames]
    saved_views = [
        observe(truth, camera, noise, observation_seed(_seed(), index, camera))
        for index, camera in enumerate(saved_cameras)
    ]
    spread, weight = (
        spread_observations(truth.shape, saved_cameras, saved_views)
        if saved_cameras
        else (np.zeros_like(truth), np.zeros_like(truth))
    )
    visual_heading()
    _legend(max(1, len(frames)))
    panels = st.columns(4)
    with panels[0]:
        _grid(truth, "Ground truth / 真值")
    with panels[1]:
        _grid(
            truth,
            "Current camera / 当前相机",
            dots=[(row, col, FRAME_COLOURS[0]) for row, col in window_origins(truth.shape, current)],
            shades=[(window_mask(truth.shape, current), (0.84, 0.15, 0.16, 0.28))],
        )
    with panels[2]:
        _grid(preview, "Current observation / 当前观测")
    with panels[3]:
        dots = []
        for index, camera in enumerate(saved_cameras):
            colour = FRAME_COLOURS[index % len(FRAME_COLOURS)]
            dots.extend((row, col, colour) for row, col in window_origins(truth.shape, camera))
        _grid(
            spread,
            "Saved measurements / 已保存的测量",
            dots=dots,
            shades=[(weight <= 1e-12, (0.75, 0.16, 0.16, 0.38))] if saved_cameras else None,
        )
    covered = int(np.count_nonzero(weight > 1e-12))
    st.write(
        t(
            f"Saved frames {len(frames)}. High-resolution pixels touched by a saved window: {covered} of {truth.size}.",
            f"已保存 {len(frames)} 帧。被已保存窗口碰到的高分辨率像素：{covered} / {truth.size}。",
        )
    )
    _looking(
        "Panel 1 is the synthetic scene, one cell per pixel. Panel 2 lays the current camera on that scene: a dot is a point sample, and a tint is the integration window. Panel 3 is the low-resolution picture that camera records. Panel 4 puts the saved frames back on the grid. Red empty cells have not been measured. In the realistic mode the number painted across a window is the window's measurement, not the value of each pixel inside it.",
        "第 1 格是合成场景，一格一个像素。第 2 格把当前相机放在场景上：圆点是点采样，色块是积分窗口。第 3 格是这台相机记录的低分辨率图。第 4 格把已保存的各帧放回网格。红色空格还没有被测到。在真实模式里，涂满一个窗口的数是这个窗口的测量，不是窗口里每个像素自己的值。",
    )
    model(r"g_i = M_i f + n_i")
    with st.expander(t("How this page builds M / 这一页怎样构造 M", "这一页怎样构造 M")):
        say(
            "The slide's generative model draws M1 to M4 from the high-resolution image and names registrations, lighting, and blur. The sketch says each M_i combines registration, lighting, and down-sampling. This laboratory splits that description into a shift T, a blur-and-window B, and downsampling D, with lighting left at 1. The next section shows the matrices on eight pixels. Point sampling uses B = I.",
            "讲义的生成模型从高分辨率图像画出 M1 到 M4，并写了配准、光照和模糊。草图解说每个 M_i 包含配准、光照和降采样。本实验把这个描述拆成位移 T、模糊加窗口 B、以及降采样 D，光照取 1。下一节在八个像素上展示这些矩阵。点采样时 B 是单位阵。",
        )
        model(r"M_i = D_i B_i T_i")
    steps(
        [
            ("Choose a pattern and look at which cells the first camera misses.", "选一个图案，先看第一台相机漏掉哪些格子。"),
            ("Change the horizontal or vertical shift and watch the dots move.", "改变水平或垂直位移，看圆点移动。"),
            ("Press Add frame. The new colour appears only in the last panel.", "按增加一帧。新的颜色只出现在最后一格。"),
            ("Switch to blur and area integration. The tint grows into a window, and one dot is no longer one original pixel.", "换成模糊与面积积分。色块变成窗口，一个圆点不再是一个原始像素。"),
        ]
    )
    try_this(
        "Use the checkerboard, factor 2, and point sampling. Shift (0, 0) sees only the bright squares. Add horizontal shift 0.5 and the dark squares appear. Then set the factor to 4 and add the same 0.5 shift: the new dots land two high-resolution pixels away, so half the squares stay missing.",
        "用棋盘、倍数 2、点采样。位移 (0, 0) 只能看到亮格。加上水平位移 0.5，暗格出现。再把倍数改成 4，仍用 0.5：新圆点隔了两个高分辨率像素，还有一半方格没被测到。",
    )
    _misconception(
        "A half-pixel shift does not, by itself, uncover the original pixel values once the camera blurs and averages an area. In the point-sampling model the complementary shifts do land on the other pixels. In the realistic model each measurement is a mixture.",
        "一旦相机先模糊再对一块面积取平均，半个像素的位移本身并不会揭开原来的像素值。点采样模型里，互补位移确实落在其余像素上。真实模型里，每个测量都是混合物。",
    )
    takeaways(
        [
            ("Different camera positions measure different parts of the same scene.", "不同的相机位置测量同一场景的不同部分。"),
            ("Changing the pattern, the factor, or the formation model clears the saved frames, because those frames would no longer match.", "改变图案、倍数或成像模型会清空已保存的帧，因为那些帧不再匹配。"),
        ]
    )


def render_benefits() -> None:
    concept(
        "The same slide names two different benefits of more images: higher resolution, and lower noise. Complementary shifts change which pixels are read. Repeating one camera, with a new draw of noise each time, only improves the average. A repeated frame does not guarantee a sharper grid.",
        "同一页讲义把多幅图像的好处分成两件：提高分辨率，以及降低噪声。互补位移改变读到的像素。同一台相机再拍一次，只是换了一份噪声，只能改善平均。重复的一帧并不保证网格更细。",
    )
    controls_heading()
    noise = float(st.slider("Noise σ for the repeated camera / 重复拍摄的噪声", 0.01, 0.3, 0.08, 0.01, key="sr-benefit-noise"))
    visual_heading()
    left, right = st.columns(2)
    scene = high_resolution_pattern("combined", 16)
    with left:
        _heading("Increase resolution", "提高分辨率")
        count = int(st.slider("Complementary frames / 互补帧数", 1, 4, 1, key="sr-benefit-count"))
        shifts = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))[:count]
        cameras = [make_camera(2, horizontal, vertical, 0.0, False) for vertical, horizontal in shifts]
        views = [apply_camera(scene, camera) for camera in cameras]
        fused, weight = spread_observations(scene.shape, cameras, views)
        show_array(fused, f"Fused samples, {count} frames / 拼回的样本，{count} 帧", "image")
        error = float(np.mean((fused - scene) ** 2))
        say(
            f"Fraction of pixels read: {weight.mean():.2f}. Mean squared error against the original: {error:.3g}.",
            f"读到的像素比例：{weight.mean():.2f}。相对原图的均方误差：{error:.3g}。",
        )
    with right:
        _heading("Reduce noise", "降低噪声")
        camera = make_camera(2, 0.0, 0.0, 0.0, False)
        clean = apply_camera(scene, camera)
        counts = (1, 2, 4, 8, 16)
        generator = np.random.default_rng(_seed() + 3)
        draws = clean + noise * generator.standard_normal((16,) + clean.shape)
        averages = [draws[:count].mean(axis=0) for count in counts]
        show_row([(f"N = {count}", average, "image") for count, average in zip(counts, averages)])
        empirical, theory = averaging_curve(clean, noise, counts, _seed() + 9)
        show_fig(
            line_figure(
                np.asarray(counts, dtype=float),
                [("Measured RMS / 测得的均方根", empirical), ("σ / sqrt(N)", theory)],
                "Noise against the number of identical frames / 相同帧数与噪声",
                "Frames N / 帧数",
                "RMS error / 均方根误差",
            )
        )
        say(
            "Every one of these averages uses the same sample locations. The unread high-resolution pixels stay unread.",
            "这些平均用的是同一批采样位置。没读到的高分辨率像素仍然没读到。",
        )
    _looking(
        "The left picture starts as a quarter of the pixels and fills in as the four shifts arrive. Unread pixels are drawn as black. The right pictures are the same low-resolution view, averaged. The curve is the size of the remaining noise. Independent noise falls like σ / sqrt(N). That formula is for the average of repeated measurements, not for a shift that sees a new pixel.",
        "左边从四分之一的像素开始，四种位移到齐后才填满。没读到的像素画成黑色。右边是同一幅低分辨率图的平均。曲线是剩下的噪声有多大。独立噪声按 σ / sqrt(N) 下降。这个公式针对重复测量的平均，不是针对看到新像素的位移。",
    )
    model(r"\mathrm{RMS} \approx \sigma / \sqrt{N}")
    try_this(
        "Move the left slider from 1 to 4 and watch the black gaps close. Then look at the right-hand curve: 16 frames make the noise smaller and leave the picture the same size.",
        "把左边的滑块从 1 拨到 4，看黑色的空缺合上。再看右边的曲线：16 帧让噪声变小，画面尺寸并没有变大。",
    )
    _misconception(
        "More frames are not automatically more resolution. If every frame hits the same pixels, the average gets cleaner and the missing pixels stay missing.",
        "帧数变多并不自动等于分辨率变高。如果每帧都打在同一批像素上，平均会更干净，缺掉的像素仍然缺着。",
    )
    takeaways(
        [
            ("Complementary sampling and noise averaging solve different problems.", "互补采样和噪声平均解决的是不同的问题。"),
            ("The slide lists both, under the heading of treating images as point samples.", "讲义在“把图像当点样本”下面把两件事都列了出来。"),
        ]
    )


def _toy_matrices(offset: float, blur: float, realistic: bool):
    return factor_matrices_1d(8, 2, offset, blur, realistic)


def render_measure() -> None:
    concept(
        "A previous section wrote one image as g = A f + n. Here each low-resolution view has its own matrix. Eight high-resolution samples become four low-resolution samples. Frame 1 reads the even indices. Frame 2, shifted by 0.5, reads the odd indices. The highlighted row is one output pixel, and the products show which original samples it uses.",
        "前面把一幅图写成 g = A f + n。这里每一幅低分辨率视图有自己的矩阵。八个高分辨率样本变成四个低分辨率样本。第 1 帧读偶数下标。位移 0.5 的第 2 帧读奇数下标。高亮的一行是一个输出像素，乘积显示它用了哪些原始样本。",
    )
    model(r"g_i = M_i f + n_i, \qquad M_i = D_i B_i T_i")
    controls_heading()
    blur = float(st.slider("Blur σ in B / B 里的模糊", 0.0, 1.2, 0.0, 0.05, key="sr-m-blur"))
    realistic = st.radio(
        "Operator / 算子",
        ["point", "realistic"],
        format_func=lambda code: t("Point sampling, B = I", "点采样，B = I")
        if code == "point"
        else t("Blur and a 2-sample window", "模糊和长度为 2 的窗口"),
        horizontal=True,
        key="sr-m-model",
    ) == "realistic"
    row = int(st.slider("Highlight output row / 高亮输出行", 0, 3, 0, key="sr-m-row"))
    use_blur = blur if realistic else 0.0
    _t1, b1, d1, m1 = _toy_matrices(0.0, use_blur, realistic)
    _t2, _b2, _d2, m2 = _toy_matrices(0.5, use_blur, realistic)
    g1 = m1 @ _TOY
    g2 = m2 @ _TOY
    visual_heading()
    say(
        "f has 8 entries. T and B are 8×8. D is 4×8. M is 4×8. g has 4 entries. A 16×16 image would make M about 64×256, and a photograph would make it millions by millions. The laboratory applies T, B, and D in turn and never stores that matrix. The toy stores it so the multiplication can be read.",
        "f 有 8 个数。T 和 B 是 8×8。D 是 4×8。M 是 4×8。g 有 4 个数。16×16 的图像会让 M 大约是 64×256，一张照片则会到百万乘百万。实验按顺序做 T、B、D，不保存那张大矩阵。这个玩具把它存下来，是为了能读乘法。",
    )
    show_fig(line_figure(np.arange(8), [("f", _TOY)], "High-resolution samples f / 高分辨率样本 f", "Index / 下标", "Value / 值"))
    show_fig(value_figure(m1, "M1, offset 0 / M1，偏移 0"))
    show_fig(value_figure(m2, "M2, offset 0.5 / M2，偏移 0.5"))
    products = m1[row] * _TOY
    show_fig(
        line_figure(
            np.arange(8),
            [(f"Row {row} of M1, times f / M1 第 {row} 行乘 f", products)],
            "One output is a weighted sum of f / 一个输出是 f 的加权和",
            "Index of f / f 的下标",
            "Product / 乘积",
        )
    )
    st.write(t(f"g1[{row}] = {g1[row]:.4g}. Sum of the products {products.sum():.4g}.", f"g1[{row}] = {g1[row]:.4g}。乘积之和 {products.sum():.4g}。"))
    st.write(t(f"g1 = {np.array2string(g1, precision=3)}", f"g1 = {np.array2string(g1, precision=3)}"))
    st.write(t(f"g2 = {np.array2string(g2, precision=3)}", f"g2 = {np.array2string(g2, precision=3)}"))
    show_fig(value_figure(_t1, "T for offset 0, 8×8 / 偏移 0 的 T，8×8"))
    show_fig(value_figure(b1, "B, 8×8 / B，8×8"))
    show_fig(value_figure(d1, "D, 4×8 / D，4×8"))
    _looking(
        "Each row of M1 makes one low-resolution sample. In point sampling that row contains a single 1, so the output copies one entry of f. After blur and the window, the same row contains several weights and the output is a mixture. Drag the row slider to move the highlight. g2 uses the other phase.",
        "M1 的每一行产生一个低分辨率样本。点采样时这一行只有一个 1，输出就是 f 的一个数。加上模糊和窗口之后，同一行有好几个权重，输出是混合物。拖动行滑块就移动高亮。g2 用的是另一个相位。",
    )
    with st.expander(t("What the slide calls M / 讲义把 M 叫什么", "讲义把 M 叫什么")):
        say(
            "The sketch, which the lecture marks not examinable, writes g_i = M_i f + noise and says M_i combines registration, lighting, and down-sampling. The diagram also includes blur. T is the shift used here as a known registration. B is blur and the sensor window. D keeps every second sample. Lighting is 1, so this page does not estimate it.",
            "讲义标成非考试的草图解写成 g_i = M_i f + 噪声，并说 M_i 包含配准、光照和降采样。图上还有模糊。这里的 T 是已知的位移，当作配准。B 是模糊和传感器窗口。D 每隔一个样本取一个。光照取 1，这一页不估计光照。",
        )
    try_this(
        "Leave the operator on point sampling and compare g1 with the even entries of f. Switch on blur and the window, set blur to 0, and watch each row of M1 gain two weights of one half. The output is then an average, not a copy.",
        "先保持点采样，把 g1 和高分辨率 f 的偶数项比较。再打开模糊和窗口，把模糊设为 0，看 M1 的每一行变成两个二分之一。这时输出是平均，不是拷贝。",
    )
    _misconception(
        "Seeing the matrix does not mean a program builds it for a photograph. The factors above are the definition. The image experiments apply the same factors one after another.",
        "看见这个矩阵，并不表示程序会对一张照片把它建出来。上面的因子是定义。图像实验按同样的因子一个接一个地做。",
    )
    takeaways(
        [
            ("M1 and M2 ask different questions about the same f.", "M1 和 M2 对同一个 f 问的是不同的问题。"),
            ("D here is downsampling. The MAP penalty later uses D for a derivative. The sections name which one they mean.", "这里的 D 是降采样。后面的 MAP 惩罚用 D 表示导数。每一节都会说明自己用的是哪一个。"),
        ]
    )


def render_stacked() -> None:
    concept(
        "The sketch stacks the separate equations into one system. It describes that system as over-determined. More rows than columns is not enough: if the new frame repeats an old shift, the rank stays where it was and the missing samples stay free.",
        "草图解把各自的方程叠成一个方程组，并把它说成超定。行数多于列数还不够：如果新的一帧重复旧的位移，秩停在原地，缺掉的样本仍然自由。",
    )
    model(r"\begin{bmatrix} g_1 \\ g_2 \end{bmatrix} = \begin{bmatrix} M_1 \\ M_2 \end{bmatrix} f + \begin{bmatrix} n_1 \\ n_2 \end{bmatrix}")
    controls_heading()
    offset = float(st.selectbox("Offset of the frame to add / 要增加的那一帧的偏移", [0.0, 0.5], key="sr-stack-offset"))
    noise = float(st.slider("Noise σ / 噪声", 0.0, 0.1, 0.0, 0.005, key="sr-stack-noise"))
    if "sr_toy" not in st.session_state:
        st.session_state["sr_toy"] = [0.0]
    offsets: list[float] = st.session_state["sr_toy"]
    add, clear = st.columns(2)
    with add:
        if st.button("Add frame / 增加一帧", key="sr-stack-add") and len(offsets) < 6:
            offsets.append(offset)
    with clear:
        if st.button("Clear frames / 清空各帧", key="sr-stack-clear"):
            offsets = []
            st.session_state["sr_toy"] = offsets
    visual_heading()
    if not offsets:
        say("Add a frame to build the stack.", "增加一帧，才能叠起方程组。")
    else:
        generator = np.random.default_rng(_seed() + 11)
        blocks = []
        measurements = []
        for index, item in enumerate(offsets):
            matrix = factor_matrices_1d(8, 2, item, 0.0, False)[3]
            blocks.append(matrix)
            measurements.append(matrix @ _TOY + noise * generator.standard_normal(4))
        stacked = np.vstack(blocks)
        measured = np.concatenate(measurements)
        report = report_stacked_system(stacked, measured, _TOY)
        show_fig(value_figure(stacked, "Stacked M / 叠放的 M"))
        say(
            "Rows are grouped by frame, four rows each, in the order the frames were added. Offset 0 and offset 0.5 are the two point-sampling phases.",
            "行按帧分组，每帧四行，顺序就是增加帧的顺序。偏移 0 和偏移 0.5 是点采样的两个相位。",
        )
        one, two, three, four = st.columns(4)
        one.metric(t("Unknowns", "未知数"), report.unknowns)
        two.metric(t("Measurements", "测量数"), report.measurements)
        three.metric(t("Rank", "秩"), report.rank)
        four.metric(t("Full column rank", "列满秩"), t("yes", "是") if report.full_column_rank else t("no", "否"))
        if report.full_column_rank and report.more_equations_than_unknowns:
            say(
                "There are more equations than unknowns, and the rank reaches the unknowns. Least squares has one fit.",
                "方程比未知数多，而且秩达到了未知数的个数。最小二乘有一个拟合。",
            )
        elif report.full_column_rank:
            say(
                "The rank equals the number of unknowns, so the samples determine f.",
                "秩等于未知数的个数，所以这些样本能确定 f。",
            )
        else:
            say(
                "The rank is still short of the eight unknowns. Adding another copy of a shift that is already present makes the measurement count larger and leaves the rank unchanged.",
                "秩仍然够不到八个未知数。把一个已经有的位移再抄一遍，测量数会变大，秩不变。",
            )
        show_fig(
            line_figure(
                np.arange(8),
                [
                    ("f, evaluation only / f，只用于评价", _TOY),
                    ("Least squares / 最小二乘", report.least_squares),
                    ("Pseudoinverse / 伪逆", report.pseudoinverse),
                ],
                "Estimated samples / 估计出的样本",
                "Index / 下标",
                "Value / 值",
            )
        )
        gap = float(np.max(np.abs(report.least_squares - report.pseudoinverse)))
        st.write(
            t(
                f"Fit MSE {report.fit_mse:.3g}. Reconstruction MSE {report.reconstruction_mse:.3g}. Largest gap between least squares and the pseudoinverse {gap:.3g}.",
                f"拟合均方误差 {report.fit_mse:.3g}。重建均方误差 {report.reconstruction_mse:.3g}。最小二乘和伪逆的最大差距 {gap:.3g}。",
            )
        )
    _looking(
        "The heatmap is the stacked matrix. Eight columns are the eight unknown samples. The two curves that sit on top of each other are the two solvers: numpy's least-squares solution is the minimum-norm solution, which is what the pseudoinverse returns. The pale curve is the original f, used only to draw the error.",
        "热图就是叠起来的矩阵。八列是八个未知样本。重合的两条曲线是两个求解器：numpy 的最小二乘给出最小范数解，也就是伪逆给出的那个。淡色曲线是原始 f，只用来画误差。",
    )
    try_this(
        "Start from one frame. The rank is 4 and the estimate is zero on the unread samples. Add the other offset. The rank becomes 8 and, with zero noise, the estimate matches f. Add the first offset again and the measurement count grows while the rank does not.",
        "从一帧开始。秩是 4，没读到的样本上估计是零。加上另一个偏移。秩变成 8；噪声为零时，估计和 f 重合。再把第一个偏移加一次，测量数增加，秩不增加。",
    )
    _misconception(
        "An over-determined count of equations does not repair a repeated measurement. The sketch calls the stacked system over-determined because the intended frames add new rows that increase the rank.",
        "方程个数变成超定，也补不好一次重复的测量。草图解把叠放的系统称为超定，是因为想用的那些帧会增加秩。",
    )
    takeaways(
        [
            ("Compare the rank with the number of unknowns, not only the number of rows.", "拿秩和未知数的个数比，不要只比行数。"),
            ("The original f is not passed to the solver.", "原始 f 没有传给求解器。"),
        ]
    )


def _map_problem():
    pattern = _pattern_box("sr-map-pattern")
    factor = 2
    size = 16
    frames = int(st.slider("Frames / 帧数", 1, 8, 2, key="sr-map-frames"))
    realistic = st.radio(
        "Formation model / 成像模型",
        ["point", "realistic"],
        format_func=lambda code: t("Point samples", "点采样") if code == "point" else t("Blur and area integration", "模糊与面积积分"),
        horizontal=True,
        key="sr-map-model",
    ) == "realistic"
    noise = float(st.slider("Noise σ / 噪声", 0.0, 0.1, 0.01, 0.005, key="sr-map-noise"))
    blur = float(st.slider("Blur σ / 模糊", 0.0, 1.2, 0.4, 0.05, key="sr-map-blur"))
    log_lambda = float(st.slider("log₁₀(λ)", -4.0, 0.0, -2.0, 0.1, key="sr-map-lambda"))
    step_fraction = float(st.slider("Learning-rate fraction s, with η = s/L / 学习率比例 s，η = s/L", 0.2, 1.8, 1.0, 0.1, key="sr-map-step"))
    iterations = int(st.slider("Iterations / 迭代次数", 1, 80, 30, key="sr-map-iters"))
    init_code = st.selectbox(
        "Initialisation / 初始化",
        ["backproject", "mean", "zeros"],
        format_func=lambda code: {
            "backproject": "Back-projection / 反投影",
            "mean": "Mean of the observations / 观测的均值",
            "zeros": "Zeros / 全零",
        }[code],
        key="sr-map-init",
    )
    truth = high_resolution_pattern(pattern, size)
    pairs = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
    cameras = []
    observations = []
    for index in range(frames):
        vertical, horizontal = pairs[index % 4]
        camera = _camera(factor, horizontal, vertical, blur, realistic)
        cameras.append(camera)
        observations.append(observe(truth, camera, noise, observation_seed(_seed(), index, camera)))
    lam = float(10.0 ** log_lambda)
    return {
        "truth": truth,
        "cameras": cameras,
        "observations": observations,
        "lam": lam,
        "step": step_fraction,
        "iterations": iterations,
        "init": init_code,
        "token": (pattern, frames, realistic, round(blur, 5), round(noise, 5), round(lam, 8), round(step_fraction, 3), iterations, init_code, _seed()),
    }


def render_map_sr() -> None:
    concept(
        "Guess a high-resolution image. Simulate every camera, compare the simulations with the observations, and step downhill on that error plus a penalty on the differences of neighbouring pixels. The original image is drawn beside the result and is not an input. The single-image MAP section uses the same 1/N scaling, so λ means the same kind of thing per pixel. Summing over frames lets each independent observation add evidence.",
        "先猜一幅高分辨率图像。模拟每一台相机，拿模拟和观测比较，再沿着这个误差往下走，并惩罚相邻像素的差。原图画在结果旁边，不是输入。单幅 MAP 用同一个 1/N，所以 λ 对每个像素的含义相同。对帧求和，是让每次独立观测都提供证据。",
    )
    model(r"J(f)=\sum_i \frac{1}{N}\|M_i f-g_i\|^2+\lambda \frac{1}{N}\|Df\|^2")
    controls_heading()
    problem = _map_problem()
    lipschitz = lipschitz_constant(problem["truth"].shape, problem["cameras"], problem["lam"])
    step_size = problem["step"] / lipschitz if lipschitz > 0.0 else float("nan")
    st.caption(
        t(
            f"N = {problem['truth'].size}. L ≈ {lipschitz:.4g}. η = s/L ≈ {step_size:.4g}. Stable steps use s < 2. Frames after the fourth repeat a shift and add a new noise draw.",
            f"N = {problem['truth'].size}。L ≈ {lipschitz:.4g}。η = s/L ≈ {step_size:.4g}。稳定步长用 s < 2。第四帧之后会重复位移，并重新抽一次噪声。",
        )
    )
    say(
        "In the penalty, D is the periodic forward difference from the MAP section. It is not the downsampling matrix.",
        "惩罚项里的 D 是 MAP 那一节的周期前向差分。它不是降采样矩阵。",
    )
    if st.button("Run gradient descent / 运行梯度下降", key="sr-map-run"):
        reference, info = solve_normal_equations(
            problem["observations"], problem["cameras"], problem["lam"], problem["truth"].shape
        )
        initial = initialise_estimate(problem["observations"], problem["cameras"], problem["truth"].shape, problem["init"])
        result = run_gradient_descent_multiframe(
            problem["observations"],
            problem["cameras"],
            problem["lam"],
            problem["step"],
            problem["iterations"],
            initial,
            truth=problem["truth"],
            reference=reference,
            reference_info=info,
        )
        st.session_state["sr_map"] = {"token": problem["token"], "result": result}
        st.session_state["sr-map-snapshot"] = len(result.images) - 1
    saved = st.session_state.get("sr_map")
    result = saved["result"] if isinstance(saved, dict) and saved.get("token") == problem["token"] else None
    if result is None:
        estimate = initialise_estimate(problem["observations"], problem["cameras"], problem["truth"].shape, problem["init"])
        say("The picture below is the initial guess. Press the button to optimise. Changing a control asks for a new run.", "下面是初始猜测。按按钮才会优化。改了控制量就需要重新运行。")
    else:
        last = len(result.images) - 1
        if last == 0:
            index = 0
        else:
            if int(st.session_state.get("sr-map-snapshot", 0)) > last:
                st.session_state["sr-map-snapshot"] = last
            index = int(st.slider("Saved iteration / 已保存的迭代", 0, last, last, key="sr-map-snapshot"))
        estimate = result.images[index]
        if result.diverged:
            say("The step became too large and the iteration stopped. Lower s.", "步长太大，迭代停了。把 s 调小。")
    visual_heading()
    panels = [
        ("Ground truth, evaluation only / 真值，只用于评价", problem["truth"], "image"),
        ("Current estimate / 当前估计", estimate, "image"),
    ]
    if result is not None:
        panels.append(("Reference normal equations / 法方程的参考解", result.reference, "image"))
    show_row(panels)
    shown = min(4, len(problem["observations"]))
    show_row([(f"g{index + 1}", problem["observations"][index], "image") for index in range(shown)])
    predictions = [apply_camera(estimate, camera) for camera in problem["cameras"][:shown]]
    show_row([(f"M{index + 1} f", prediction, "image") for index, prediction in enumerate(predictions)])
    show_row(
        [
            (f"g{index + 1} − M f", problem["observations"][index] - predictions[index], "residual")
            for index in range(shown)
        ]
    )
    if len(problem["observations"]) > shown:
        say(
            f"The loss uses all {len(problem['observations'])} frames. The strips show the first four.",
            f"损失使用全部 {len(problem['observations'])} 帧。图条只显示前四帧。",
        )
    if result is not None:
        axis = np.arange(result.total_loss.size)
        show_fig(
            line_figure(
                axis,
                [
                    ("Total J / 总损失 J", result.total_loss),
                    ("Data / 数据项", result.data_loss),
                    ("Penalty / 惩罚项", result.regularisation_loss),
                ],
                "Loss / 损失",
                "Stored point / 保存的点",
                "Loss / 损失",
            )
        )
        show_fig(
            line_figure(
                axis,
                [("Reconstruction MSE / 重建均方误差", result.reconstruction_mse), ("Gradient norm / 梯度范数", result.grad_norm)],
                "Error and gradient / 误差和梯度",
                "Stored point / 保存的点",
                "Value / 值",
            )
        )
        st.caption(
            t(
                f"Reference solver info {result.reference_info}. Zero means conjugate gradients reached the tolerance. The reconstruction MSE uses the original image and the solver does not.",
                f"参考求解器的 info 是 {result.reference_info}。零表示共轭梯度达到了容差。重建均方误差用了原图，求解器没有用。",
            )
        )
    _looking(
        "The top row is the hidden scene, the current guess, and the direct numerical solution of the normal equations. The next rows are what the cameras recorded, what the current guess predicts, and the difference. A dark residual means that camera is already explained.",
        "最上一行是隐藏的场景、当前猜测，以及法方程的直接数值解。下面几行是相机记录的、当前猜测预测的，以及两者的差。残差暗，表示这台相机已经被解释了。",
    )
    with st.expander(t("Gradient / 梯度", "梯度")):
        model(r"\nabla J(f)=\frac{2}{N}\left[\sum_i M_i^{\mathsf T}(M_i f-g_i)+\lambda D^{\mathsf T} D f\right]")
        say(
            "The slide writes a sum of squares plus a prior, without the 1/N. Dividing by N matches the existing MAP page. It does not move the minimiser if λ and the step size are scaled with it. The factor 2 is the derivative of a square. Each M_i transpose pushes that camera's residual back onto the high-resolution grid.",
            "讲义写的是平方和加上先验，没有 1/N。除以 N 是为了和已有的 MAP 页一致。只要 λ 和步长一起缩放，最小值的位置不变。系数 2 来自平方的导数。每个 M_i 的转置把那台相机的残差送回高分辨率网格。",
        )
    steps(
        [
            ("Start from the chosen initialisation. It is built from the observations only.", "从所选的初始化开始。它只用观测来构造。"),
            ("Apply each camera to the guess.", "对猜测施加每一台相机。"),
            ("Subtract the prediction from the observation.", "用观测减去预测。"),
            ("Add the data term and the difference penalty.", "把数据项和差分惩罚加起来。"),
            ("Step by η times the gradient.", "沿梯度走 η 倍。"),
            ("Repeat. The reference button-path solves the same quadratic directly.", "重复。同一次按钮还会直接解同一个二次问题，作为参考。"),
        ]
    )
    try_this(
        "Use point sampling, two frames, and λ near 0.01. Run, then raise the frame count to 4 and run again. The black gaps in the estimate fill in. With one frame, the reference solution still has to invent the unread pixels, and the penalty is what makes that invention smooth.",
        "用点采样、两帧，λ 靠近 0.01。运行，再把帧数加到 4 重新运行。估计里的黑色空缺会补上。只有一帧时，参考解仍要编出没读到的像素，惩罚项让这种编造保持平滑。",
    )
    _misconception(
        "Gradient descent is not the definition of MAP. It is one optimiser. The normal-equation reference is the same quadratic, solved directly. Neither one is allowed to see the original image.",
        "梯度下降不是 MAP 的定义。它是一种优化方法。法方程的参考解是同一个二次问题的直接解。两者都不能看原图。",
    )
    takeaways(
        [
            ("Each new complementary frame adds a residual that the guess has to explain.", "每增加一帧互补的观测，猜测就多一个必须解释的残差。"),
            ("A repeated shift mostly asks the guess to agree with another noisy copy.", "重复的位移主要是要求猜测去符合另一份带噪声的拷贝。"),
        ]
    )


def render_huber() -> None:
    concept(
        "Squared error grows very fast, so one corrupted sample can drag the whole estimate. The lecture names Huber loss as an example that reduces the effect of the squared term when the error is large. Absolute error grows in a straight line. Huber follows the square near zero and then follows a straight line.",
        "平方误差涨得很快，所以一个坏样本能把整个估计拖走。讲义把 Huber 损失当作一个例子：误差大的时候，减弱平方项的作用。绝对误差沿直线增长。Huber 在零附近沿平方走，之后改沿直线走。",
    )
    model(r"\rho(r)=\begin{cases} r^2/2 & |r|\le \delta \\ \delta(|r|-\delta/2) & |r|>\delta \end{cases}")
    controls_heading()
    delta = float(st.slider("Huber threshold δ / Huber 阈值 δ", 0.2, 2.0, 0.5, 0.05, key="sr-huber-delta"))
    lam = float(st.slider("Regularisation λ / 正则 λ", 0.0, 1.0, 0.1, 0.01, key="sr-huber-lambda"))
    axis = np.linspace(-4.0, 4.0, 401)
    visual_heading()
    show_fig(
        line_figure(
            axis,
            [
                ("Squared / 平方", squared_loss(axis)),
                ("Absolute / 绝对", absolute_loss(axis)),
                ("Huber / Huber", huber_loss(axis, delta)),
            ],
            "Three penalties on one residual / 一个残差上的三种惩罚",
            "Residual r / 残差 r",
            "Penalty / 惩罚",
        )
    )
    signal, matrix, observed, corrupt = corrupted_blur_problem(spike=3.0, noise_std=0.01, seed=_seed())
    least_squares = restore_quadratic_1d(matrix, observed, lam)
    robust, weights = restore_huber_1d(matrix, observed, lam, delta, iterations=25)
    show_fig(
        line_figure(
            np.arange(signal.size),
            [
                ("Original, evaluation only / 原信号，只用于评价", signal),
                ("Corrupted observation / 被破坏的观测", observed),
                ("Squared-error restoration / 平方误差复原", least_squares),
                ("Huber restoration / Huber 复原", robust),
            ],
            "One corrupted sample in a blurred 1D signal / 模糊一维信号里的一个坏样本",
            "Index / 下标",
            "Value / 值",
        )
    )
    show_fig(line_figure(np.arange(weights.size), [("Huber weight / Huber 权重", weights)], "Weight on each residual / 每个残差的权重", "Index / 下标", "Weight / 权重"))
    least_error = float(np.mean((least_squares - signal) ** 2))
    robust_error = float(np.mean((robust - signal) ** 2))
    st.write(
        t(
            f"Corrupted index {corrupt}. Its Huber weight is {weights[corrupt]:.3f}. Squared-error MSE {least_error:.3g}. Huber MSE {robust_error:.3g}.",
            f"被破坏的下标是 {corrupt}。它的 Huber 权重是 {weights[corrupt]:.3f}。平方误差的均方误差 {least_error:.3g}。Huber 的均方误差 {robust_error:.3g}。",
        )
    )
    _looking(
        "The first plot compares the three curves. Huber bends away from the square once the residual passes δ. The second plot is a blurred step with one sample pushed up by 3. The weights are 1 while the residual is inside δ, and δ/|r| outside it. A small weight means that sample has stopped pulling as hard as a square would pull.",
        "第一张图比较三条曲线。残差超过 δ 之后，Huber 离开平方、改走直线。第二张图是一个模糊的阶跃，其中一个样本被抬高了 3。残差在 δ 以内时权重是 1，以外是 δ/|r|。权重小，表示这个样本不再像平方那样用力拉估计。",
    )
    with st.expander(t("Derivative / 导数", "导数")):
        say(
            "The derivative of r²/2 is r. The derivative of this Huber function is r inside the threshold and δ times the sign of r outside it. The influence of a residual therefore stops growing. The restoration uses iteratively reweighted least squares with those weights. The formula is the standard Huber function used in this laboratory. The slide names Huber and does not write this expression.",
            "r²/2 的导数是 r。这个 Huber 函数的导数在阈值内是 r，在阈值外是 δ 乘以 r 的符号。残差的影响因此不再继续变大。复原用这些权重做迭代重加权最小二乘。这个公式是本实验使用的标准 Huber 函数。讲义点了 Huber 的名字，没有写出这个式子。",
        )
    try_this(
        "Set λ near 0. The squared restoration chases the spike, the residual at that sample becomes small, and Huber agrees with it because there is no large residual left to down-weight. Raise λ to about 0.1. The smooth penalty refuses the spike, the corrupted residual stays large, and its weight drops.",
        "把 λ 调到接近 0。平方复原会去追那个尖峰，该样本的残差变小，Huber 也就和它一致，因为已经没有大残差可降权。把 λ 调到大约 0.1。平滑惩罚拒绝尖峰，坏样本的残差仍然大，权重下降。",
    )
    _misconception(
        "Huber loss does not delete the corrupted sample, and it does not know which sample was corrupted. It only limits how hard a large residual can pull.",
        "Huber 损失不会删掉坏样本，也不知道哪一个样本是坏的。它只限制一个大残差能拉多用力。",
    )
    takeaways(
        [
            ("A large residual keeps a bounded influence under Huber loss.", "在 Huber 损失下，大残差的影响有上限。"),
            ("The comparison uses one known blur. It is not the blind problem.", "这个比较用的是已知模糊。它不是盲去模糊。"),
        ]
    )


def render_examples() -> None:
    concept(
        "The lecture shows two super-resolution examples after the filter comparison, then blind-deblurring photographs. Those image files are not in this laboratory. The pictures below are synthetic stand-ins, labelled as such. They are not the Mars frames and they are not frames from the film.",
        "讲义在滤波器比较之后给了两个超分辨率例子，然后是盲去模糊的照片。那些图像文件不在本实验里。下面的图是合成的替代，并标明了这一点。它们不是火星的画面，也不是电影的画面。",
    )
    scene = high_resolution_pattern("combined", 32)
    camera = make_camera(2, 0.0, 0.0, 0.0, False)
    clean = apply_camera(scene, camera)
    generator = np.random.default_rng(_seed() + 21)
    repeats = clean + 0.06 * generator.standard_normal((16,) + clean.shape)
    shifts = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
    cameras = [make_camera(2, horizontal, vertical, 0.0, False) for vertical, horizontal in shifts]
    views = [apply_camera(scene, item) + 0.02 * generator.standard_normal(clean.shape) for item in cameras]
    fused, _weight = spread_observations(scene.shape, cameras, views)
    visual_heading()
    _heading("Mars, as described on the slide", "讲义上的火星例子")
    say(
        "The slide says there are 25 JPEG images from different sweeps of a rotating camera, and it shows an original frame, an average, and a super-resolution result. The row below is a synthetic version of that comparison: one noisy view, the average of 16 copies of that same view, and four complementary shifts placed back on the grid.",
        "这一页说有 25 张 JPEG，来自旋转相机的不同扫描，并展示了原始一帧、平均，以及超分辨率结果。下面这一行是这个比较的合成版本：一幅带噪声的视图、同一视图 16 份拷贝的平均，以及四种互补位移放回网格。",
    )
    show_row(
        [
            ("Synthetic single view / 合成的单幅", repeats[0], "image"),
            ("Synthetic average / 合成的平均", repeats.mean(axis=0), "image"),
            ("Synthetic complementary samples / 合成的互补样本", fused, "image"),
        ]
    )
    _heading("Run Lola Run, as described on the slides", "讲义上的罗拉快跑")
    say(
        "The slides show a low-resolution input and a result. This laboratory does not include the film. The single view above is the kind of coarse input those slides start from, and the complementary-sample picture is only an illustration of combining shifted samples.",
        "这几页展示了低分辨率输入和结果。本实验不收录该电影。上面的单幅是那种从粗画面出发的例子，互补样本那一幅只说明怎样合并有位移的样本。",
    )
    _heading("Blind-deblurring photographs", "盲去模糊的照片")
    say(
        "The slides show a blurred photograph, an estimated blur, and a restored photograph. The next section is a one-dimensional stand-in with a known short kernel. It is not those photographs, and the lecture marks blind deblurring not examinable.",
        "这几页展示了模糊的照片、估计出的模糊，以及复原的照片。下一节是一个一维替代，核是短的并且形式已知。它不是那些照片。讲义把盲去模糊标成非考试内容。",
    )
    _looking(
        "Three synthetic panels stand in for the Mars comparison of one frame, an average, and a reconstruction from shifted samples. Nothing in this row was taken from the lecture PDF.",
        "三个合成面板用来代替火星例子里的一帧、平均，以及由位移样本得到的重建。这一行没有一张图是从讲义 PDF 里取的。",
    )
    try_this(
        "Compare the average with the complementary-sample picture. The average is cleaner than one frame and still blocky. The complementary samples fill the finer grid. That is the resolution-versus-noise distinction from the earlier section.",
        "把平均和互补样本比较。平均比单帧干净，但仍然发块。互补样本填上更细的网格。这就是前面那一节里分辨率和噪声的区别。",
    )
    _misconception(
        "A synthetic panel that looks sharper than its average is not the Mars result and is not a result from the film.",
        "一幅看起来比平均更清晰的合成图，不是火星的结果，也不是电影的结果。",
    )
    takeaways(
        [
            ("The lecture's examples use many real frames. This page only shows the measurement pattern those examples rely on.", "讲义的例子用了很多真实画面。这一页只展示那些例子所依赖的测量方式。"),
        ]
    )


def render_blind() -> None:
    concept(
        "Not examinable. Up to this point the blur h is known, and the estimate is the image. Blind deblurring has only the observation. The slide estimates both the image and the blur, with a penalty on each. This demonstration is one-dimensional, the kernel has a fixed odd length, and every iterate is forced to be nonnegative and to sum to one. That does not guarantee recovery of the original signal or the original kernel.",
        "非考试内容。到这里为止，模糊 h 是已知的，要估计的是图像。盲去模糊只有观测。讲义同时估计图像和模糊，并各加一项惩罚。这个演示是一维的，核的奇数长度是固定的，每次迭代都强迫核非负且和为 1。这并不能保证找回原来的信号或原来的核。",
    )
    model(r"\min_{f,h}\ (g-A(h)f)^2+\lambda p_f(f)+\mu p_h(h)")
    controls_heading()
    lam = float(st.slider("Image penalty λ / 图像惩罚 λ", 0.0, 0.05, 0.001, 0.001, key="sr-blind-lambda"))
    noise = float(st.slider("Noise σ / 噪声", 0.0, 0.05, 0.0, 0.005, key="sr-blind-noise"))
    iterations = int(st.slider("Alternating iterations / 交替迭代次数", 1, 30, 12, key="sr-blind-iters"))
    demo = blind_deblur_1d(lam=lam, noise_std=noise, iterations=iterations, seed=_seed())
    if iterations == 1:
        index = 0
    else:
        if int(st.session_state.get("sr-blind-which", 1)) > iterations:
            st.session_state["sr-blind-which"] = iterations
        index = int(st.slider("Show iteration / 查看第几次", 1, iterations, iterations, key="sr-blind-which")) - 1
    visual_heading()
    axis = np.arange(demo.signal.size)
    show_fig(
        line_figure(
            axis,
            [
                ("Original signal / 原信号", demo.signal),
                ("Blurred observation / 模糊观测", demo.observed),
                ("Known blur restored / 已知模糊的复原", demo.known_image),
                ("Blind estimate / 盲估计", demo.images[index]),
            ],
            "Known blur beside blind deblurring / 已知模糊与盲去模糊",
            "Index / 下标",
            "Value / 值",
        )
    )
    taps = np.arange(demo.true_kernel.size)
    show_fig(
        line_figure(
            taps,
            [
                ("True kernel / 真实的核", demo.true_kernel),
                ("Estimated kernel / 估计的核", demo.kernels[index]),
            ],
            "Blur kernel / 模糊核",
            "Tap / 抽头",
            "Weight / 权重",
        )
    )
    known_error = float(np.mean((demo.known_image - demo.signal) ** 2))
    blind_error = float(np.mean((demo.images[index] - demo.signal) ** 2))
    kernel = demo.kernels[index]
    st.write(
        t(
            f"Known-blur MSE {known_error:.3g}. Blind MSE {blind_error:.3g}. Kernel sum {kernel.sum():.4f}. Kernel minimum {kernel.min():.3g}.",
            f"已知模糊的均方误差 {known_error:.3g}。盲估计的均方误差 {blind_error:.3g}。核的和 {kernel.sum():.4f}。核的最小值 {kernel.min():.3g}。",
        )
    )
    _looking(
        "The signal plot has four curves: the original, the blurred observation, the restoration that was given the true kernel, and the restoration that had to estimate the kernel as well. The kernel plot shows that a close image can still sit on a kernel that is not the true one. The estimate often becomes a bit sharper than the kernel that made the data.",
        "信号图有四条曲线：原信号、模糊观测、拿到真实核之后的复原，以及必须同时估计核的复原。核的图说明：图像可以已经很接近，核却仍然不是真的那个。估计出的核常常比产生数据的核更尖一点。",
    )
    say(
        "The image step is the same periodic MAP solve used when the kernel is known. The kernel step is least squares on the five taps, then a projection onto nonnegative weights that sum to one. The blur penalty in the slide is represented here by that projection and a small ridge, not by a claim that the projection is the lecture's formula.",
        "图像这一步就是核已知时用的那个周期 MAP 求解。核这一步对五个抽头做最小二乘，再投影到非负且和为 1 的权重上。讲义里的模糊惩罚，在这里由这个投影和一个小的岭项代替，并不是说投影就是讲义的公式。",
    )
    try_this(
        "Set the iteration to 1. The kernel starts as an impulse, so the blind estimate is close to the blurred observation. Move the iteration forward and compare it with the known-blur curve. Then raise the noise and watch the estimated taps move.",
        "把迭代设为 1。核从脉冲开始，所以盲估计接近模糊观测。把迭代往后拨，再和已知模糊的曲线比较。然后加大噪声，看估计出的抽头怎样移动。",
    )
    _misconception(
        "A nonnegative kernel that sums to one is a constraint we imposed. It is not evidence that the original kernel has been found. The lecture does not claim that blind deblurring has a unique answer, and this page does not either.",
        "非负且和为 1 是我们加上的约束。它不能证明原来的核已经被找到。讲义没有说盲去模糊有唯一答案，这一页也没有。",
    )
    takeaways(
        [
            ("Known blur estimates f. Blind blur estimates f and h.", "已知模糊估计 f。盲去模糊估计 f 和 h。"),
            ("This section is not examinable, and it is not a two-dimensional restoration of the lecture photographs.", "这一节是非考试内容，也不是对讲义照片做的二维复原。"),
        ]
    )
