"""Lecture 1 experiments."""

from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.b14.image_processing.lecture1.fundamentals import (
    bayer_rggb,
    demosaic_bilinear,
    downsample,
    quantise,
    synthetic_rgb,
    temporal_average,
    upsample_nearest,
    bright_border,
)
from src.b14.image_processing.lecture1.gaussian import (
    gaussian_filter,
    gaussian_filter_separable,
    gaussian_kernel_2d,
    multiplications_per_pixel,
)
from src.b14.image_processing.lecture1.histogram import bin_counts, coin_image, equalise_image, equalise_regions
from src.b14.image_processing.lecture1.matched_filter import add_noise, matched_response, place_template, template_image
from src.b14.image_processing.lecture1.point_operations import apply_point, histogram, mapping_curve, negative_count
from src.b14.image_processing.lecture1.sobel import rotated_step, sobel, sobel_separable, threshold_magnitude
from src.b14.image_processing.lecture1.spatial_filters import (
    KERNELS,
    convolve2d,
    correlate2d,
    kernel_products,
    neighbourhood,
    valid_output_shape,
)
from src.b14.image_processing.lecture4.window_pick import pick_neighbourhood, saved_neighbourhood
from src.b14.image_processing.metrics import mse
from src.ui.components import (
    centre_crop,
    concept,
    controls_heading,
    download_png,
    experiment_heading,
    lecture_anchor,
    line_figure,
    model,
    save_workspace,
    show_array,
    show_row,
    steps,
    takeaways,
    try_this,
    value_figure,
    visual_heading,
)
from src.b14.image_processing.experiments import L1_EXPERIMENTS
from src.ui.i18n import plain, shown_label, t
from src.ui.common import show_fig


def render_lecture1(image: np.ndarray, label: str) -> None:
    lecture_anchor("l1")
    st.title(t("Lecture 1 — Image Enhancement", "第 1 讲 — 图像增强"))
    st.caption(shown_label(label))
    st.caption(
        t(
            "Sections follow the lecture: sampling, Bayer, point operations, histograms, temporal averaging, then spatial filters.",
            "各节按讲义排列：采样、Bayer、点运算、直方图、时间平均，然后是空间滤波。",
        )
    )
    runners = {
        "fundamentals": _fundamentals,
        "bayer": _bayer,
        "point": _point,
        "histogram": _histogram,
        "temporal": _temporal,
        "convolution": _convolution,
        "padding": _padding,
        "sobel": _sobel,
        "gaussian": _gaussian,
        "matched": _matched,
    }
    for item in L1_EXPERIMENTS:
        experiment_heading("l1", item)
        if item[0] in {"histogram", "bayer"}:
            runners[item[0]]()
        else:
            runners[item[0]](image)


def _fundamentals(image: np.ndarray) -> None:
    concept(
        "Spatial resolution is the number of samples. Intensity resolution is how many distinct values each sample can take. They destroy different information.",
        "空间分辨率是采样个数。强度分辨率是每个样本能取多少个不同的值。它们丢掉的是不同的信息。",
    )
    model(r"f[y,x]\in\{0,1/(L-1),\ldots,1\},\qquad L=2^{b}")
    controls_heading()
    factor = int(st.select_slider("Spatial factor / 空间抽样间隔", options=[1, 2, 4, 8], value=4))
    method = st.radio("Sampling / 抽样", ["point", "area"], horizontal=True, format_func=lambda name: "Point / 点抽样" if name == "point" else "Area average / 块平均")
    bits = int(st.slider("Bit depth / 位深", 1, 8, 3, key="l1-bits"))
    small = downsample(image, factor, method=method)
    restored = upsample_nearest(small, factor, image.shape)
    _codes, quantised = quantise(image, bits)
    visual_heading()
    st.markdown(f"**{shown_label('Original / 原图')}**")
    row, col = pick_neighbourhood(
        image,
        size=1,
        hint=plain(
            "Click a pixel to inspect it. Drag the red pixel to move it.",
            "点击一个像素查看它。拖动红色像素来移动它。",
        ),
        key="l1-inspector",
    )
    st.caption(t(f"Inspector is row {row}, column {col}.", f"查看位置是第 {row} 行、第 {col} 列。"))
    show_row(
        [
            ("Sampled and repeated / 抽样后重复", restored, "image"),
            (f"{bits}-bit reconstruction / {bits} 位重建", quantised, "image"),
        ]
    )
    crop = centre_crop(image[max(0, row - 4) : row + 4, max(0, col - 4) : col + 4], 8)
    show_fig(value_figure(crop, "Pixel values around the inspector / 查看位置附近的像素值"))
    st.write(
        f"Sampling MSE {mse(restored, image):.4g}. Quantisation MSE {mse(quantised, image):.4g}. "
        f"The sampled array is {small.shape[0]}×{small.shape[1]} before it is repeated for display."
    )
    st.caption(f"抽样均方误差 {mse(restored, image):.4g}。量化均方误差 {mse(quantised, image):.4g}。抽样数组在放大显示前是 {small.shape[0]}×{small.shape[1]}。")
    if mse(quantised, image) < 1e-12 and bits < 8:
        st.caption(
            t(
                "This image already sits on the quantisation levels, so lowering the bit depth does not change it. "
                "The built-in sharp-edge image does.",
                "这幅图的强度已经落在量化级别上，所以降低位深不会改变它。内置的「锐边渐变」会变。",
            )
        )
    steps(
        [
            ("Point sampling keeps one pixel from each block. Area sampling replaces the block by its mean.", "点抽样保留每块中的一个像素。块平均用均值代替整块。"),
            ("Nearest-neighbour repetition shows the coarse grid at the original size. It does not invent the missing samples.", "最近邻重复只是把粗网格画回原尺寸，并没有恢复丢失的样本。"),
            ("Quantisation rounds each intensity to the nearest of L levels. The spatial grid is unchanged.", "量化把每个强度舍入到 L 个级别中最近的一个。空间网格不变。"),
        ]
    )
    takeaways(
        [
            ("A blurry-looking block image can still have 256 grey levels. A 1-bit image can still be full resolution.", "看起来成块的图仍可以有 256 级灰度。1 位图像仍可以是满分辨率。"),
        ]
    )
    try_this(
        "Set the spatial factor to 8 with point sampling, then switch to area sampling. Compare that with 1-bit quantisation at factor 1.",
        "把空间间隔设为 8 并使用点抽样，再改成块平均。然后在间隔为 1 时做 1 位量化，比较两种损失。",
    )
    save_workspace(quantised, "quantised", "ws-quant")
    download_png(quantised, "quantised.png", "dl-quant")


def _point(image: np.ndarray) -> None:
    concept(
        "A point operation reads one pixel and writes one pixel. The same mapping T is used everywhere.",
        "点运算读一个像素、写一个像素。整幅图使用同一个映射 T。",
    )
    model(r"g(x,y)=T(f(x,y))")
    controls_heading()
    kind = st.selectbox(
        "Mapping / 映射",
        ["brightness", "contrast", "invert", "gamma", "threshold"],
        format_func=lambda name: {
            "brightness": "Brightness / 亮度",
            "contrast": "Contrast / 对比度",
            "invert": "Invert / 反相",
            "gamma": "Gamma / 伽马",
            "threshold": "Threshold / 阈值",
        }[name],
    )
    brightness = float(st.slider("Brightness offset / 亮度偏移", -0.5, 0.5, 0.1, 0.01))
    contrast = float(st.slider("Contrast scale / 对比度缩放", 0.2, 3.0, 1.4, 0.05))
    gamma = float(st.slider("Gamma / 伽马", 0.2, 3.0, 0.6, 0.05))
    threshold = float(st.slider("Threshold / 阈值", 0.0, 1.0, 0.5, 0.01))
    output = apply_point(image, kind, brightness=brightness, contrast=contrast, gamma=gamma, threshold=threshold)
    visual_heading()
    show_row([("Input / 输入", image, "image"), ("T(f) / 输出", output, "image")])
    if negative_count(image) and kind == "gamma":
        st.warning("Gamma is undefined for negative intensities. Those pixels were left unchanged. / 负强度的伽马没有定义，这些像素保持原值。")
    x, y = mapping_curve(kind, brightness=brightness, contrast=contrast, gamma=gamma, threshold=threshold)
    show_fig(line_figure(x, [("T(f)", y)], "Input-output mapping / 输入输出映射", "f", "T(f)"))
    before_x, before_y = histogram(image)
    after_x, after_y = histogram(output, value_range=(float(min(np.min(output), 0.0)), float(max(np.max(output), 1.0))))
    figure = go.Figure()
    figure.add_bar(x=before_x, y=before_y, name="Before / 处理前")
    figure.add_bar(x=after_x, y=after_y, name="After / 处理后")
    figure.update_layout(barmode="overlay", title="Histograms / 直方图", height=320)
    show_fig(figure)
    steps(
        [
            ("Brightness adds a constant. Contrast scales the distance from 0.5.", "亮度加一个常数。对比度缩放相对 0.5 的距离。"),
            ("Gamma is a power. Threshold replaces the sample by 0 or 1.", "伽马是乘方。阈值把样本换成 0 或 1。"),
            ("The result is not clipped. A displayed grey scale may be stretched when values leave [0, 1].", "结果不裁剪。数值离开 [0, 1] 时，显示用的灰度可能被拉伸。"),
        ]
    )
    takeaways([("No neighbour appears in T. Sharpening and blurring cannot be point operations.", "T 里没有邻域。锐化和模糊都不可能是点运算。")])
    try_this("Raise the contrast until values leave [0, 1], then look at the caption under the image. / 把对比度加大到超出 [0, 1]，再看图像下面的说明。", "Compare the histogram, not only the picture. / 不要只看图像，也要看直方图。")
    save_workspace(output, f"point-{kind}", "ws-point")


def _histogram() -> None:
    concept(
        "Global histogram equalisation uses one mapping for every pixel. A dark object and a bright background then share the same set of output levels.",
        "全局直方图均衡对所有像素使用同一个映射。暗物体和亮背景会共用同一组输出级别。",
    )
    model(r"s(k)=\frac{1}{N}\sum_{j=0}^{k} n_j")
    controls_heading()
    size = int(st.session_state.get("image_size", 128))
    image, disk = coin_image(size)
    levels = int(st.slider("Histogram bins / 直方图箱数", 8, 256, 64, 8))
    st.caption("This page builds a synthetic coin so the two regions are known. It does not use the workspace image. / 这一页自己生成硬币图，以便知道两个区域。它不使用工作区图像。")
    global_image, counts, cdf = equalise_image(image, levels=levels)
    regional = equalise_regions(image, disk, levels=levels)
    visual_heading()
    show_row(
        [
            ("Coin / 硬币图", image, "image"),
            ("Global equalisation / 全局均衡", global_image, "image"),
            ("Separate disk and background / 圆盘与背景分开", regional, "image"),
        ]
    )
    bins = np.arange(levels)
    figure = go.Figure()
    figure.add_bar(x=bins, y=counts, name="Counts / 计数")
    figure.add_scatter(x=bins, y=cdf * np.max(counts), name="CDF, scaled to the histogram / CDF，按直方图缩放", mode="lines")
    figure.update_layout(title="Histogram and CDF of the coin / 硬币图的直方图和 CDF", height=320)
    show_fig(figure)
    st.caption("The CDF curve is scaled only so it can share the count axis. The mapping uses the unscaled CDF. / CDF 曲线只是为了和计数共用坐标轴才做了缩放。映射使用的是未缩放的 CDF。")
    centres = (bins + 0.5) / levels
    after_counts = bin_counts(global_image, levels)
    compared = go.Figure()
    compared.add_bar(x=centres, y=counts, name=shown_label("Before / 均衡前"), opacity=0.7)
    compared.add_bar(x=centres, y=after_counts, name=shown_label("After / 均衡后"), opacity=0.7)
    compared.update_layout(
        barmode="overlay",
        title=shown_label("Histogram before and after equalisation / 均衡前后的直方图"),
        height=360,
        margin=dict(l=12, r=12, t=46, b=64),
        xaxis_title=shown_label("Intensity / 强度"),
        yaxis_title=shown_label("Counts / 计数"),
        legend=dict(orientation="h", yanchor="top", y=-0.22),
    )
    show_fig(compared)
    st.caption(
        t(
            "Both histograms use the same bins on [0, 1]. Equalisation moves each occupied bin to its CDF value. It does not split the pixels inside a bin.",
            "两幅直方图用 [0, 1] 上的同一组箱。均衡把每个有像素的箱移到它的 CDF 值上，不会拆开箱里的像素。",
        )
    )
    separated = go.Figure()
    separated.add_bar(x=centres, y=bin_counts(regional[disk], levels), name=shown_label("Disk / 圆盘"), opacity=0.7)
    separated.add_bar(x=centres, y=bin_counts(regional[~disk], levels), name=shown_label("Background / 背景"), opacity=0.7)
    separated.update_layout(
        barmode="overlay",
        title=shown_label("Histogram with the disk and background separated / 圆盘与背景分开后的直方图"),
        height=360,
        margin=dict(l=12, r=12, t=46, b=64),
        xaxis_title=shown_label("Intensity / 强度"),
        yaxis_title=shown_label("Counts / 计数"),
        legend=dict(orientation="h", yanchor="top", y=-0.22),
    )
    show_fig(separated)
    st.caption(
        t(
            "These counts come from the third image. The disk and the background were equalised with their own histograms, then drawn on the same intensity bins.",
            "这些计数来自第三幅图。圆盘和背景各自用自己的直方图做了均衡，再画在同一组强度箱上。",
        )
    )
    steps(
        [
            ("Count how many pixels fall into each intensity bin.", "统计每个强度箱里有多少像素。"),
            ("The CDF says which fraction of pixels are at or below that bin.", "CDF 表示有多少比例的像素落在这个箱或更低的箱。"),
            ("Replace each pixel by that fraction. Positions do not move.", "把每个像素换成这个比例。位置不动。"),
            ("The region-aware result equalises the disk and the background with two separate histograms.", "分区结果用两份直方图，分别均衡圆盘和背景。"),
        ]
    )
    takeaways(
        [
            ("Global equalisation can reveal the emblem and also stretch the smooth background, because both compete for the same output levels.", "全局均衡会显出纹章，也会拉伸平滑背景，因为两者在争同一组输出级别。"),
            ("This is not a claim that the lecture specifies this particular coin picture. The picture is a constructed counterexample.", "这不是说讲义指定了这幅硬币图。这幅图是为了展示反例而构造的。"),
        ]
    )
    try_this(
        "Lower the bin count to 8 and watch the emblem break into flat bands.",
        "把箱数降到 8，观察纹章变成平坦的色带。",
    )


def _kernel_from_controls() -> np.ndarray:
    name = st.selectbox(
        "Kernel / 核",
        ["identity", "box", "gaussian", "horizontal", "vertical", "sobel_x", "sobel_y", "sharpen", "custom"],
    )
    if name == "gaussian":
        sigma = float(st.slider("3×3 Gaussian σ", 0.3, 2.0, 0.8, 0.05))
        return gaussian_kernel_2d(3, sigma)
    if name != "custom":
        return KERNELS[name]
    columns = st.columns(3)
    values = []
    for index in range(9):
        with columns[index % 3]:
            values.append(float(st.number_input(f"k{index // 3}{index % 3}", value=1.0 if index == 4 else 0.0, step=0.1, key=f"custom-k-{index}")))
    return np.array(values, dtype=np.float64).reshape(3, 3)


def _convolution(image: np.ndarray) -> None:
    concept(
        "Correlation multiplies the kernel as it is drawn. Convolution flips that kernel by 180° and then does the same sum. Sobel masks in this lab are correlation masks.",
        "相关按画出的核做乘法。卷积先把核旋转 180°，再做同一个求和。本实验里的 Sobel 模板是相关模板。",
    )
    model(r"(f\star h)[y,x]=\sum_{i,j} h[i,j]\,f[y+i,x+j],\qquad (f*h)[y,x]=(f\star h_{\mathrm{flipped}})[y,x]")
    controls_heading()
    kernel = _kernel_from_controls()
    operation = st.radio("Operation / 运算", ["correlation", "convolution"], horizontal=True)
    boundary = st.selectbox("Boundary / 边界", ["reflect", "zero", "replicate", "periodic"], key="conv-boundary")
    visual_heading()
    st.markdown(f"**{shown_label('Original / 原图')}**")
    row, col = pick_neighbourhood(
        image,
        size=int(kernel.shape[0]),
        hint=plain(
            "Click a pixel to place the kernel. Drag the red square to move it. Yellow marks the centre.",
            "点击一个像素放置核。拖动红色方块来移动它。黄色标出中心。",
        ),
        key="l1-convolution",
    )
    window = neighbourhood(image, row, col, kernel.shape, boundary)
    used, products, total = kernel_products(window, kernel, flipped=operation == "convolution")
    if operation == "correlation":
        filtered = correlate2d(image, kernel, boundary=boundary)
    else:
        filtered = convolve2d(image, kernel, boundary=boundary)
    st.caption(t(f"Kernel centre is row {row}, column {col}. Output {total:.6g}.", f"核的中心是第 {row} 行、第 {col} 列。输出 {total:.6g}。"))
    left, middle, right = st.columns(3)
    with left:
        show_fig(value_figure(window, "Neighbourhood, after padding / 填充后的邻域"))
    with middle:
        show_fig(value_figure(used, "Kernel actually used / 实际使用的核"))
    with right:
        show_fig(value_figure(products, "Products / 逐项乘积"))
    show_array(filtered, "Filtered image / 滤波结果", "kernel")
    st.caption(f"Valid output for this image and kernel would be {valid_output_shape(image.shape, kernel.shape)} if no padding were added. The picture above uses same-size padding. / 若不填充，有效输出尺寸是 {valid_output_shape(image.shape, kernel.shape)}。上面的图使用了保持尺寸的填充。")
    steps(
        [
            ("Read the 3×3 neighbourhood. At the border, the extra samples come from the selected boundary rule.", "读取 3×3 邻域。在边界上，多出来的样本来自所选的边界规则。"),
            ("For convolution, rotate the kernel before multiplying. The products panel shows that rotated kernel.", "卷积时，先旋转核再相乘。乘积面板显示的是旋转后的核。"),
            ("The output pixel is the sum of the nine products. Move the centre to see the next pixel.", "输出像素是九个乘积之和。移动中心就能看下一个像素。"),
        ]
    )
    takeaways([("An asymmetric kernel makes correlation and convolution disagree. A symmetric Gaussian does not.", "不对称的核会使相关和卷积给出不同结果。对称高斯核则不会。")])
    try_this("Select horizontal derivative and switch between correlation and convolution. The sign of the response flips. / 选择水平导数，在相关和卷积之间切换。响应的符号会反过来。", "Move the centre onto the bright border of a built-in shape. / 把中心移到内置图形的亮边缘上。")
    save_workspace(filtered, f"filtered-{operation}", "ws-conv")


def _gaussian(image: np.ndarray) -> None:
    concept(
        "A discrete Gaussian is a sampled bell curve, then divided by its sum so that a constant image is unchanged. The 2D kernel is an outer product, so the filter separates.",
        "离散高斯是采样后的钟形曲线，再除以总和，使常数图像保持不变。二维核是外积，所以滤波可以分离。",
    )
    model(r"h[i,j]\propto \exp\left(-\frac{i^2+j^2}{2\sigma^2}\right),\qquad O(k^2)\ \mathrm{vs}\ O(2k)")
    controls_heading()
    sigma = float(st.slider("σ", 0.4, 4.0, 1.5, 0.1))
    size = int(st.slider("Kernel size / 核大小", 3, 15, 7, 2))
    boundary = st.selectbox("Boundary / 边界", ["reflect", "zero", "replicate", "periodic"], key="gauss-boundary")
    visual_heading()
    started = time.perf_counter()
    direct, kernel = gaussian_filter(image, size, sigma, boundary=boundary)
    direct_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    separable, factor = gaussian_filter_separable(image, size, sigma, boundary=boundary)
    separable_ms = (time.perf_counter() - started) * 1000
    direct_cost, separable_cost = multiplications_per_pixel(size)
    show_row(
        [
            ("Kernel / 核", kernel, "kernel"),
            ("Direct 2D / 直接二维", direct, "image"),
            ("Separable / 分离", separable, "image"),
        ]
    )
    surface = go.Figure(data=go.Surface(z=kernel, colorscale="Cividis", showscale=False))
    surface.update_layout(title="Gaussian surface / 高斯曲面", height=420, margin=dict(l=0, r=0, t=46, b=0))
    show_fig(surface)
    st.markdown(f"**{shown_label('Original / 原图')}**")
    row, _col = pick_neighbourhood(
        image,
        size=1,
        anchor="row",
        hint=plain(
            "Click the image to choose the cross-section row. Drag the red line up or down.",
            "点击图像选择剖面所在行。上下拖动红色横线。",
        ),
        key="l1-gauss-row",
    )
    st.caption(t(f"Cross-section is row {row}.", f"剖面在第 {row} 行。"))
    show_fig(
        line_figure(
            np.arange(image.shape[1]),
            [("Input / 输入", image[row]), ("Filtered / 滤波后", direct[row])],
            "Intensity cross-section / 强度剖面",
            "column / 列",
            "intensity / 强度",
        )
    )
    difference = float(np.max(np.abs(direct - separable)))
    st.write(
        f"Kernel sum {kernel.sum():.6f}. One-dimensional factor sum {factor.sum():.6f}. "
        f"Max |direct − separable| = {difference:.3g}. Time {direct_ms:.2f} ms vs {separable_ms:.2f} ms. "
        f"Multiplications per pixel: {direct_cost} vs {separable_cost}."
    )
    st.caption(f"核的和是 {kernel.sum():.6f}。一维因子的和是 {factor.sum():.6f}。直接法与分离法的最大差是 {difference:.3g}。耗时 {direct_ms:.2f} ms 对 {separable_ms:.2f} ms。每像素乘法次数：{direct_cost} 对 {separable_cost}。")
    st.caption("The timing is one measurement on this machine, not a complexity proof. The operation counts are exact for a dense kernel. / 时间只是这台机器上的一次测量，不是复杂度证明。运算次数对稠密核是精确的。")
    steps(
        [
            ("Build the 1D Gaussian and normalise it. The 2D kernel is its outer product, which is already normalised.", "构造一维高斯并归一化。二维核是它的外积，因此也已经归一化。"),
            ("Pad the image once, filter horizontally, then vertically. Padding the intermediate image again would change the border.", "图像只填充一次，先横向再纵向。如果对中间结果再次填充，边界会改变。"),
            ("A larger σ needs a larger window before the discarded tails become small.", "更大的 σ 需要更大的窗口，丢掉的尾部才会变小。"),
        ]
    )
    takeaways([("Noise falls and edges spread together. A linear Gaussian cannot remove noise only at smooth pixels.", "噪声下降时，边缘也一起变宽。线性高斯不能只在平滑像素上去噪。")])
    try_this("Increase σ and watch the cross-section lose its steps. Then compare the two runtimes at kernel size 15. / 增大 σ，看剖面里的阶跃消失。再把核大小设为 15，比较两次运行时间。", "The operation-count ratio is k/2. The measured time ratio will not match it exactly. / 运算次数比是 k/2。测到的时间比不会精确等于它。")
    save_workspace(direct, "gaussian", "ws-gaussian")


def _padding(image: np.ndarray) -> None:
    concept(
        "Same-size filtering has to invent samples outside the image. The invention becomes visible when the real border is bright.",
        "保持尺寸的滤波必须在图像外编造样本。真实边界很亮时，这种编造会看得见。",
    )
    model(r"n_{\mathrm{valid}}=n-m+1")
    controls_heading()
    use_border = st.checkbox("Use a synthetic bright frame / 使用合成的亮边框", value=True)
    source = bright_border(image.shape[0]) if use_border else image
    mode = st.selectbox("Same-size boundary / 保持尺寸的边界", ["valid", "zero", "replicate", "reflect", "periodic"])
    kernel = gaussian_kernel_2d(9, 1.5)
    visual_heading()
    if mode == "valid":
        output = correlate2d(source, kernel, mode="valid")
        st.write(f"Valid shape {output.shape}. Formula gives {valid_output_shape(source.shape, kernel.shape)}.")
        st.caption(f"有效尺寸 {output.shape}。公式给出 {valid_output_shape(source.shape, kernel.shape)}。")
    else:
        output = correlate2d(source, kernel, mode="same", boundary=mode)
        st.write(f"Same-size output {output.shape}, using {mode} padding of {(kernel.shape[0] - 1) // 2} pixels.")
        st.caption(f"保持尺寸的输出是 {output.shape}，{mode} 填充 {(kernel.shape[0] - 1) // 2} 个像素。")
    show_row([("Input / 输入", source, "image"), (t(f"Result · {mode}", f"结果 · {mode}"), output, "image")])
    steps(
        [
            ("Valid mode only writes a pixel when the whole kernel lies inside the image.", "只有整个核都落在图像内时，有效模式才写出一个像素。"),
            ("Zero padding mixes the border with black. Replicate copies the edge. Reflect mirrors it. Periodic wraps the opposite side in.", "零填充把边界和黑色混合。复制填充沿用边缘。反射填充做镜像。周期填充把对边卷进来。"),
            ("Lecture 3 uses periodic convolution because that is the operation diagonalised by the DFT.", "第 3 讲使用周期卷积，因为这正是 DFT 对角化的运算。"),
        ]
    )
    takeaways([("A dark halo at a bright border is often the padding, not a feature of the scene.", "亮边界外的暗圈常常来自填充，不是场景里的结构。")])
    try_this("Switch between zero and periodic on the bright frame. / 在亮边框上切换零填充和周期填充。", "Turn the synthetic frame off and use an uploaded image with a bright edge. / 关掉合成边框，改用一张边缘很亮的上传图像。")


def _sobel(image: np.ndarray) -> None:
    concept(
        "Sobel Gx responds to a horizontal change of intensity. Gy responds to a vertical change. The gradient direction is perpendicular to the edge.",
        "Sobel Gx 响应强度的水平变化。Gy 响应垂直变化。梯度方向垂直于边缘。",
    )
    model(r"G_x=\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix},\quad \theta=\operatorname{atan2}(G_y,G_x)")
    controls_heading()
    synthetic = st.checkbox("Use a rotated synthetic step / 使用旋转的合成阶跃", value=True)
    angle = float(st.slider("Normal angle, degrees / 法线角度（度）", -180.0, 180.0, 30.0, 1.0))
    level = float(st.slider("Magnitude threshold / 幅度阈值", 0.0, 8.0, 1.0, 0.1))
    source = rotated_step(image.shape[0], angle) if synthetic else image
    gx, gy, magnitude, orientation = sobel(source)
    separable_x, separable_y = sobel_separable(source)
    edges = threshold_magnitude(magnitude, level)
    visual_heading()
    show_row(
        [
            ("Input / 输入", source, "image"),
            ("Gx", gx, "residual"),
            ("Gy", gy, "residual"),
        ]
    )
    show_row(
        [
            ("Magnitude / 幅度", magnitude, "kernel"),
            ("Thresholded edges / 阈值后的边缘", edges, "image"),
            ("|separable − direct|", np.hypot(gx - separable_x, gy - separable_y), "residual"),
        ]
    )
    step = max(6, source.shape[0] // 12)
    xs: list[float | None] = []
    ys: list[float | None] = []
    for y in range(step // 2, source.shape[0], step):
        for x in range(step // 2, source.shape[1], step):
            if magnitude[y, x] < level:
                continue
            length = step * 0.45
            xs.extend([x, x + length * np.cos(orientation[y, x]), None])
            ys.extend([y, y + length * np.sin(orientation[y, x]), None])
    arrows = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#0B3A5B", width=2)))
    arrows.update_yaxes(autorange="reversed", scaleanchor="x")
    arrows.update_layout(title="Gradient direction, subsampled / 抽稀后的梯度方向", height=420)
    show_fig(arrows)
    steps(
        [
            ("Gx is the outer product [1, 2, 1]ᵀ [-1, 0, 1]. Gy swaps those factors. The separable result matches the 3×3 mask.", "Gx 是 [1, 2, 1]ᵀ 与 [-1, 0, 1] 的外积。Gy 交换这两个因子。分离结果与 3×3 模板一致。"),
            ("atan2(Gy, Gx) points across the edge. Add 90° to describe the edge itself.", "atan2(Gy, Gx) 指向跨过边缘的方向。给它加 90° 才是边缘本身的方向。"),
            ("Thresholding the magnitude keeps strong responses and drops smooth regions.", "对幅度取阈值会留下强响应，去掉平滑区域。"),
        ]
    )
    takeaways([("Rotating the step from 0° toward 90° moves the response from Gx to Gy.", "把阶跃从 0° 转到 90°，响应会从 Gx 移到 Gy。")])
    try_this("Set the normal angle to 0°, then 90°, then 45°, and watch which component is large. / 把法线角度设为 0°、90° 和 45°，看哪个分量变大。", "Turn off the synthetic step and use the workspace image. / 关掉合成阶跃，改用工作区图像。")


def _matched(image: np.ndarray) -> None:
    concept(
        "Additional explanation. Correlation with a template scores how well each location matches that template. Under white noise, this is the matched filter for a known shape.",
        "补充说明。与模板做相关，是在给每个位置和该模板的相似程度打分。在白噪声下，这就是已知形状的匹配滤波器。",
    )
    model(r"r = f \star h")
    controls_heading()
    kind = st.selectbox("Template / 模板", ["horizontal", "vertical", "disk", "diagonal"])
    noise = float(st.slider("Noise σ / 噪声", 0.0, 0.8, 0.15, 0.01, key="l1-match-noise"))
    template = template_image(kind, 11)
    if image.shape[0] < 11 or image.shape[1] < 11:
        st.warning(t("The image is smaller than the 11×11 template.", "图像小于 11×11 的模板。"))
        return
    planted = (image.shape[0] // 3, image.shape[1] // 2)
    row, col = saved_neighbourhood(image.shape[0], image.shape[1], size=11, key="l1-matched", anchor="inside", default=planted)
    hidden = place_template(image.shape, template, row, col)
    noisy = add_noise(hidden, noise, int(st.session_state.get("seed", 0)))
    visual_heading()
    st.markdown(f"**{shown_label('Noisy image / 含噪图像')}**")
    row, col = pick_neighbourhood(
        noisy,
        size=11,
        anchor="inside",
        default=planted,
        hint=plain(
            "Click to place the template. Drag the red square to move it. Yellow marks its centre.",
            "点击放置模板。拖动红色方块来移动它。黄色标出中心。",
        ),
        key="l1-matched",
    )
    hidden = place_template(image.shape, template, row, col)
    noisy = add_noise(hidden, noise, int(st.session_state.get("seed", 0)))
    response = matched_response(noisy, template)
    peak = np.unravel_index(int(np.argmax(response)), response.shape)
    show_row(
        [
            ("Template, mean removed / 去均值的模板", template, "residual"),
            ("Correlation / 相关响应", response, "kernel"),
        ]
    )
    st.write(f"Maximum at row {peak[0]}, column {peak[1]}. Planted at row {row}, column {col}.")
    st.caption(f"最大值在第 {peak[0]} 行、第 {peak[1]} 列。放入的位置是第 {row} 行、第 {col} 列。")
    steps(
        [
            ("The template is mean-zero, so a flat region scores near zero.", "模板均值为零，所以平坦区域的分数接近零。"),
            ("Each output is the inner product of the template and one neighbourhood.", "每个输出都是模板与一个邻域的内积。"),
            ("The largest inner product is the detection. It is not a probability.", "最大的内积就是检测结果。它不是概率。"),
        ]
    )
    takeaways([("A template that matches the line direction peaks more sharply than one at the wrong orientation.", "与线条方向一致的模板，峰值比方向错误的模板更尖锐。")])
    try_this("Hide a vertical line and score it with the horizontal template. The peak should move or weaken. / 隐藏一条竖线，再用水平模板去评分。峰值应会移动或变弱。", "Increase the noise until the reported peak leaves the planted coordinate. / 增大噪声，直到报告的峰值离开放入的坐标。")


def _temporal(image: np.ndarray) -> None:
    concept(
        "The lecture includes temporal filtering by averaging frames. Independent noise falls as more frames are averaged; a static scene does not.",
        "这一讲包含用平均帧做时间滤波。独立噪声会随帧数增加而下降；静止场景不会。",
    )
    model(r"\bar f=\frac{1}{T}\sum_{t=1}^{T}(f+n_t)")
    controls_heading()
    frames = int(st.slider("Frames / 帧数", 1, 30, 8, key="l1-frames"))
    noise = float(st.slider("Noise σ per frame / 每帧噪声", 0.0, 0.4, 0.1, 0.01))
    stack, mean = temporal_average(image, frames, noise, int(st.session_state.get("seed", 0)))
    visual_heading()
    show_row(
        [
            ("One noisy frame / 一帧含噪图像", stack[0], "image"),
            (f"Mean of {frames} / {frames} 帧平均", mean, "image"),
            ("Original / 原图", image, "image"),
        ]
    )
    st.write(f"MSE of one frame {mse(stack[0], image):.4g}. MSE of the mean {mse(mean, image):.4g}.")
    st.caption(f"单帧均方误差 {mse(stack[0], image):.4g}。平均后均方误差 {mse(mean, image):.4g}。")
    steps(
        [
            ("Each frame adds a new noise draw. The scene is the same.", "每一帧加入一次新的噪声。场景不变。"),
            ("The variance of the average of T independent draws is the single-frame variance divided by T.", "T 次独立抽样的平均，其方差是单帧方差除以 T。"),
        ]
    )
    takeaways([("Averaging does not undo blur. It only attacks noise that changes from frame to frame.", "平均不能撤销模糊。它只对付帧与帧之间变化的噪声。")])
    try_this("Compare 1 frame with 30 frames at the same noise level. / 在相同噪声下比较 1 帧和 30 帧。", "This average is not clipped, so a large noise value can leave [0, 1]. / 这个平均没有裁剪，所以很大的噪声可以离开 [0, 1]。")
    save_workspace(mean, "temporal-mean", "ws-temporal")


def _bayer() -> None:
    concept(
        "A Bayer grid measures one colour at each pixel. The other two colours are not observed there. The reconstruction below is a simple neighbour average, not a camera pipeline.",
        "Bayer 网格在每个像素只测量一种颜色。另外两种颜色在那里没有被观测到。下面的重建是简单的邻域平均，不是相机流程。",
    )
    model(r"\mathrm{RGGB}:\ R_{2i,2j},\; G_{2i,2j+1},\; G_{2i+1,2j},\; B_{2i+1,2j+1}")
    controls_heading()
    size = int(st.session_state.get("image_size", 128))
    rgb = synthetic_rgb(size)
    mosaic = bayer_rggb(rgb)
    restored = demosaic_bilinear(mosaic)
    visual_heading()
    show_row(
        [
            ("Red channel / 红通道", rgb[:, :, 0], "image"),
            ("Green channel / 绿通道", rgb[:, :, 1], "image"),
            ("Blue channel / 蓝通道", rgb[:, :, 2], "image"),
        ]
    )
    show_row(
        [
            ("Mosaic / 马赛克", mosaic, "image"),
            ("Filled red / 补全的红", restored[:, :, 0], "image"),
            ("Filled blue / 补全的蓝", restored[:, :, 2], "image"),
        ]
    )
    steps(
        [
            ("Write the measured sample into its channel and leave the other channel values unknown.", "把测到的样本写进它自己的通道，其他通道在该像素上是未知的。"),
            ("Fill an unknown sample by the average of known neighbours in a 3×3 window. Known samples are not changed.", "未知样本用 3×3 窗口里已知邻居的平均来填。已知样本不改。"),
        ]
    )
    takeaways([("The mosaic is not a grayscale photograph of the scene. It is three interlaced measurements.", "马赛克不是场景的灰度照片。它是三份交错的测量。")])
    try_this("The mosaic is the measurement. The filled colours are interpolated, so they are not new measurements.", "马赛克才是测量。补全的颜色是插值，不是新的测量。")
