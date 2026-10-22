"""Lecture 4 experiments."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.b14.image_processing.lecture1.gaussian import gaussian_filter
from src.b14.image_processing.lecture4.bilateral import bilateral_filter, bilateral_weight_maps, reflect_window
from src.b14.image_processing.lecture4.compression import rle_decode, rle_encode, rle_ratio, storage_report
from src.b14.image_processing.lecture4.dct import basis_montage, dct2, dct_matrix, idct2, keep_frequencies
from src.b14.image_processing.lecture4.huffman import average_length, build_tree, codes_from_tree, decode, encode, frequencies, tree_lines
from src.b14.image_processing.lecture4.inpainting import diffuse, rectangle_mask
from src.b14.image_processing.lecture4.jpeg_demo import jpeg_like, zigzag
from src.b14.image_processing.lecture4.median import median, salt_and_pepper, sorted_median, window
from src.b14.image_processing.lecture4.window_pick import pick_neighbourhood
from src.b14.image_processing.lecture4.nonlocal_means import (
    BAR,
    DIAGONAL_A,
    DIAGONAL_B,
    MARK_SIZE,
    compare_patches,
    matching_patches,
    nonlocal_means,
    repeated_marks,
    weight_map,
    weighted_value,
)
from src.b14.image_processing.metrics import format_db, mse, psnr
from src.ui.components import (
    centre_crop,
    concept,
    controls_heading,
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
from src.b14.image_processing.experiments import L4_EXPERIMENTS
from src.ui.i18n import plain, say, shown_label, t
from src.ui.common import show_fig


def render_lecture4(image: np.ndarray, label: str) -> None:
    lecture_anchor("l4")
    st.title(t("Lecture 4 — Compression and Nonlinear Filters", "第 4 讲 — 压缩与非线性滤波"))
    st.caption(shown_label(label))
    st.caption(
        t(
            "Sections follow the lecture: storage, run-length coding, Huffman, the DCT and a JPEG-like pipeline, then median, bilateral, non-local means, and inpainting.",
            "各节按讲义排列：存储、游程编码、霍夫曼、离散余弦变换和类 JPEG 流程，然后是中值、双边、非局部均值和修补。",
        )
    )
    runners = {
        "storage": _storage,
        "rle": _rle,
        "huffman": _huffman,
        "dct": _dct,
        "jpeg": _jpeg,
        "median": _median,
        "bilateral": _bilateral,
        "nlm": _nlm,
        "inpaint": _inpaint,
    }
    for item in L4_EXPERIMENTS:
        experiment_heading("l4", item)
        runners[item[0]](image)


def _storage(image: np.ndarray) -> None:
    del image
    concept(
        "An uncompressed raster stores every sample. A compression ratio here means uncompressed bits divided by compressed bits. It is not a measured file size.",
        "未压缩的光栅会存下每一个样本。这里的压缩比是未压缩位数除以压缩后位数。它不是测得的文件大小。",
    )
    model(r"\mathrm{bits}=WH\,b\,C\,T")
    controls_heading()
    width = int(st.number_input("Width / 宽", 1, 7680, 640))
    height = int(st.number_input("Height / 高", 1, 4320, 480))
    depth = int(st.select_slider("Bit depth / 位深", options=[1, 8, 10, 12, 16], value=8, key="l4-bits"))
    channels = int(st.select_slider("Channels / 通道数", options=[1, 3, 4], value=1))
    frames = int(st.number_input("Frames / 帧数", 1, 100000, 1, key="l4-frames"))
    ratio = float(st.slider("Compression ratio / 压缩比", 1.0, 50.0, 8.0, 0.5))
    rate_megabits = float(st.slider("Transfer rate, Mbit/s / 传输速率", 0.1, 100.0, 10.0, 0.1))
    report = storage_report(width, height, depth, channels, frames, ratio, rate_megabits * 1_000_000)
    visual_heading()
    left, right = st.columns(2)
    left.metric("Uncompressed MB, decimal / 十进制 MB", f"{report['uncompressed_mb']:.3f}")
    right.metric("Uncompressed KiB / 二进制 KiB", f"{report['uncompressed_kib']:.1f}")
    left.metric("Compressed decimal MB / 压缩后十进制 MB", f"{report['compressed_mb']:.3f}")
    right.metric("Transfer time, seconds / 传输秒数", f"{report['transfer_seconds']:.3f}")
    st.caption(
        f"Uncompressed bytes {report['uncompressed_bytes']:.0f}. "
        "MB is 10⁶ bytes. KiB is 1024 bytes. The compressed size divides by the ratio; it is not the length of a coded file. "
        f"未压缩字节数 {report['uncompressed_bytes']:.0f}。MB 是 10⁶ 字节。KiB 是 1024 字节。压缩后的大小是除以压缩比，不是编码文件的长度。"
    )
    steps(
        [
            ("Multiply width, height, bits per sample, channels, and frames.", "把宽、高、每样本位数、通道数和帧数相乘。"),
            ("Divide by 8 for bytes.", "除以 8 得到字节。"),
            ("Divide the bit count by the ratio, then by the transfer rate in bits per second.", "把位数除以压缩比，再除以每秒位数，得到传输时间。"),
        ]
    )
    takeaways([("A video at 25 frames for 10 seconds is 250 frames, not 25. Set Frames yourself; the control is not a hidden frame rate.", "25 帧/秒、10 秒是 250 帧，不是 25。帧数由你填写，控件里没有隐藏的帧率。")])
    try_this("Compare one 8-bit grey frame with a 1920×1080, 3-channel, 250-frame clip at the same ratio. / 比较一帧 8 位灰度，和同样压缩比下 1920×1080、3 通道、250 帧的片段。", "Lossless and lossy methods are separate experiments below. This page only does the arithmetic. / 无损和有损方法在下面的实验里。这一页只做算术。")


def _rle(image: np.ndarray) -> None:
    concept(
        "Run-length encoding stores a value and how many times it repeats. It shrinks long constant runs and expands noise.",
        "游程编码存储一个值和它重复的次数。它能压缩长的常数段，却会让噪声变长。",
    )
    model(r"(v_1,c_1),(v_2,c_2),\ldots")
    controls_heading()
    source = st.radio(
        "Sequence / 序列",
        ["image row", "constant", "alternating"],
        horizontal=True,
        format_func=lambda name: {"image row": "One image row / 图像的一行", "constant": "Constant / 常数", "alternating": "Alternating / 交替"}[name],
    )
    visual_heading()
    if source == "image row":
        st.markdown(f"**{shown_label('Original / 原图')}**")
        row, _col = pick_neighbourhood(
            image,
            size=1,
            anchor="row",
            hint=plain(
                "Click the image to choose the row. Drag the red line up or down.",
                "点击图像选择这一行。上下拖动红色横线。",
            ),
            key="l4-rle-row",
        )
        st.caption(t(f"This sequence is row {row}.", f"这段序列是第 {row} 行。"))
        sequence = np.rint(np.clip(image[row], 0, 1) * 15).astype(int).tolist()
    elif source == "constant":
        sequence = [3] * 32
    else:
        sequence = [0, 1] * 16
    runs = rle_encode(sequence)
    decoded = rle_decode(runs)
    show_fig(line_figure(np.arange(len(sequence)), [("Samples / 样本", np.array(sequence, dtype=float))], "Sequence / 序列", "index", "value"))
    st.write(runs)
    st.write(f"Lossless check {decoded == sequence}. Accounting ratio {rle_ratio(sequence):.3f}. Runs {len(runs)}.")
    st.caption("每个游程按两个符号计。这是记账模型，不是文件格式。无损检查为真表示解码序列与原序列相同。")
    steps(
        [
            ("Walk the sequence. Start a new run when the value changes.", "沿序列走。数值变化时开始新的游程。"),
            ("Store the value and the count.", "存下数值和计数。"),
            ("The ratio uses two stored symbols per run, so alternating data has ratio 0.5.", "每个游程算两个符号，所以交替数据的比是 0.5。"),
        ]
    )
    takeaways([("The image-row option quantises to 16 levels only for this display. That quantisation is not part of RLE.", "图像行这个选项只为了显示才量化到 16 级。这个量化不属于游程编码。")])
    try_this("Compare the constant sequence with the alternating sequence. / 比较常数序列和交替序列。", "A row of a flat region should beat a row through the checker texture. / 平坦区域的一行应该胜过穿过棋盘纹理的一行。")


def _huffman(image: np.ndarray) -> None:
    del image
    concept(
        "Huffman coding assigns short bit strings to frequent symbols. Decoding follows the same tree, so this demonstration is lossless.",
        "霍夫曼编码给常见符号分配短的比特串。解码沿着同一棵树走，所以这个演示是无损的。",
    )
    model(r"L=\sum_s p(s)\,|c(s)|")
    controls_heading()
    text = st.text_input("Symbols / 符号", value="ABRACADABRA")
    if not text:
        st.warning("Enter at least one symbol. / 至少输入一个符号。")
        return
    counts = frequencies(text)
    tree = build_tree(counts)
    codes = codes_from_tree(tree)
    bits = encode(text, codes)
    restored = decode(bits, tree)
    visual_heading()
    st.write(codes)
    st.code("\n".join(tree_lines(tree)))
    st.write(
        f"Bits {len(bits)}. Fixed 8-bit length would be {8 * len(text)}. "
        f"Average length {average_length(counts, codes):.3f} bits/symbol. "
        f"Lossless {restored == text}."
    )
    st.caption(f"比特数 {len(bits)}。若固定用 8 位，长度是 {8 * len(text)}。平均码长 {average_length(counts, codes):.3f} 比特/符号。无损检查 {restored == text}。")
    st.caption("The ratio against 8-bit symbols is an accounting comparison, not a file that contains a Huffman table. / 相对于 8 位符号的比例是记账比较，不是一个含有霍夫曼表的文件。")
    steps(
        [
            ("Count the symbols.", "统计符号。"),
            ("Repeatedly join the two least frequent nodes.", "反复把频率最小的两个节点合并。"),
            ("A left branch writes 0 and a right branch writes 1.", "左支写 0，右支写 1。"),
            ("Decode by walking until a symbol is reached, then returning to the root.", "解码时一直走到一个符号，然后回到根。"),
        ]
    )
    takeaways([("A single repeated symbol still needs a code. The tree gives it one bit in this implementation.", "只有一个重复符号时仍然需要一个码。这个实现给它一位。")])
    try_this("Change ABRACADABRA to a string of identical letters and watch the average length. / 把 ABRACADABRA 换成一串相同的字母，观察平均码长。", "The decoded string is shown by the lossless check, which must stay true. / 无损检查必须保持为真，它表示解码字符串与输入相同。")


def _dct(image: np.ndarray) -> None:
    concept(
        "The DCT-II is a real cosine basis related to a Fourier transform of an even extension. JPEG uses an 8×8 block of it. Low indices are slow variations.",
        "DCT-II 是实余弦基，与偶延拓的傅里叶变换有关。JPEG 使用它的 8×8 小块。小的下标是缓慢变化。",
    )
    model(r"C[k,n]=\alpha(k)\cos\left(\frac{\pi k (n+1/2)}{N}\right)")
    if image.shape[0] < 8 or image.shape[1] < 8:
        st.warning("The image is smaller than 8×8. / 图像小于 8×8。")
        return
    max_sum = int(st.session_state.get("l4-dct-max-sum", 4))
    visual_heading()
    show_array(basis_montage(8), "64 DCT basis images, each scaled for display / 64 个 DCT 基图像，每块单独缩放以便显示", "kernel")
    st.markdown(f"**{shown_label('Original image / 原图')}**")
    row, col = pick_neighbourhood(
        image,
        size=8,
        anchor="origin",
        hint=plain(
            "Click a pixel to place the 8×8 block. Drag the red square to move it. Yellow marks its first pixel.",
            "点击一个像素放置 8×8 小块。拖动红色方块来移动它。黄色标出它的第一个像素。",
        ),
        key="l4-dct-block",
    )
    st.caption(t(f"Block starts at row {row}, column {col}.", f"小块从第 {row} 行、第 {col} 列开始。"))
    block = image[row : row + 8, col : col + 8]
    matrix = dct_matrix(8)
    coefficients = dct2(block - 0.5, matrix)
    kept = keep_frequencies(coefficients, max_sum)
    reconstructed = idct2(kept, matrix) + 0.5
    left, right = st.columns(2)
    with left:
        show_fig(value_figure(block, "Input block / 输入小块"))
    with right:
        show_fig(value_figure(coefficients, "DCT coefficients / DCT 系数"))
    show_row(
        [
            ("Kept coefficients / 保留的系数", kept, "residual"),
            ("Reconstruction / 重建", reconstructed, "image"),
            ("Error / 误差", reconstructed - block, "residual"),
        ]
    )
    st.write(f"Block MSE {mse(reconstructed, block):.4g}.")
    st.caption("基图像的每一块除以了自己的最大值，所以这块拼图不能拿来读系数大小。系数矩阵才是未缩放的。")
    controls_heading()
    st.slider("Keep coefficients with k + ℓ ≤ / 保留 k+ℓ 不超过", 0, 14, 4, key="l4-dct-max-sum")
    steps(
        [
            ("Subtract 0.5, the analogue of JPEG's level shift for intensities in [0, 1].", "减去 0.5，这是强度在 [0, 1] 时对应 JPEG 电平平移的做法。"),
            ("Multiply by the orthonormal DCT matrix on the left and its transpose on the right.", "左边乘正交 DCT 矩阵，右边乘它的转置。"),
            ("Zero the high index sum and invert.", "把高下标和置零，再逆变换。"),
        ]
    )
    takeaways([("Keeping only k + ℓ ≤ 1 leaves a smooth ramp. The sharp corners of the block need the discarded coefficients.", "只保留 k+ℓ ≤ 1 会留下平滑的渐变。小块里的尖角需要被丢掉的系数。")])
    try_this("Set the limit to 0, then 14. The error should collapse at 14. / 把限制设为 0，再设为 14。误差在 14 时应消失。", "Move the block onto an edge in the image. / 把小块移到图像的一条边缘上。")


def _jpeg(image: np.ndarray) -> None:
    concept(
        "This is a teaching pipeline: 8×8 blocks, DCT, quantise, inverse DCT, reassemble. It does not write a JPEG file. RLE of one zigzag is shown separately and is not included in the PSNR.",
        "这是教学流程：8×8 小块、DCT、量化、逆 DCT、拼回。它不写出 JPEG 文件。一个 zigzag 的游程编码单独显示，并且不包含在 PSNR 里。",
    )
    model(r"Q_{ij}=\mathrm{round}(C_{ij}/q_{ij})")
    controls_heading()
    strength = float(st.slider("Quantisation strength / 量化强度", 0.05, 4.0, 1.0, 0.05))
    result = jpeg_like(image, strength)
    cropped = result["cropped"]
    reconstructed = result["reconstructed"]
    visual_heading()
    show_row(
        [
            ("Cropped original / 裁到 8 的倍数", cropped, "image"),
            ("Reconstruction / 重建", reconstructed, "image"),
            ("Error / 误差", reconstructed - cropped, "residual"),
        ]
    )
    show_row(
        [
            ("One coefficient block / 一块系数", result["coefficients"][:8, :8], "residual"),
            ("Quantised indices / 量化索引", result["indices"][:8, :8], "residual"),
            ("Quantisation table / 量化表", result["table"], "kernel"),
        ]
    )
    indices = result["indices"][:8, :8]
    runs = rle_encode(zigzag(indices).tolist())
    st.write(
        f"PSNR {format_db(psnr(reconstructed, cropped))}. "
        f"Fraction of zero indices {result['sparsity']:.3f}. "
        f"Zigzag runs in the first block {len(runs)}."
    )
    st.caption(
        "PSNR compares the cropped original with the reconstructed array. "
        "The zero fraction is sparsity of the quantised coefficients, not a file size. "
        "The luminance table is the usual JPEG example table, scaled by the strength. It was not read off a lecture slide. "
        "PSNR 比较的是裁剪后的原图和重建数组。零的比例是量化系数的稀疏度，不是文件大小。亮度表是常用的 JPEG 示例表，再乘以强度。它不是从讲义幻灯片上读出来的。"
    )
    steps(
        [
            ("Split the image into 8×8 blocks, dropping a trailing strip if the size is not a multiple of 8.", "把图像分成 8×8 的块。尺寸不是 8 的倍数时丢掉末尾的一条。"),
            ("DCT each level-shifted block.", "对每个做过电平平移的小块做 DCT。"),
            ("Divide by the table, round, and multiply back.", "除以量化表，四舍五入，再乘回去。"),
            ("Invert the DCT. A separate RLE of the zigzag shows how zeros would be described to an entropy coder.", "逆 DCT。zigzag 的游程编码单独展示零将如何交给熵编码器。"),
        ]
    )
    takeaways([("Strong quantisation makes neighbouring blocks choose different coarse levels. That is the blocking artefact.", "强量化会让相邻小块选择不同的粗糙级别。这就是块效应。")])
    try_this("Raise the strength and watch both the PSNR and the zero fraction. / 提高强度，同时看 PSNR 和零的比例。", "Do not treat the zero fraction as the compression ratio from the storage calculator. / 不要把零的比例当成存储计算器里的压缩比。")
    save_workspace(reconstructed, "jpeg-like", "ws-jpeg")


def _median(image: np.ndarray) -> None:
    concept(
        "The median replaces the centre pixel by the middle sample after sorting. Sorting is not a weighted sum, so the filter is nonlinear.",
        "中值先排序，再用中间那个样本替换中心像素。排序不是加权和，所以这个滤波器是非线性的。",
    )
    model(r"g[y,x]=\mathrm{median}\{f[y+i,x+j]\}")
    controls_heading()
    amount = float(st.slider("Salt-and-pepper fraction / 椒盐比例", 0.0, 0.3, 0.06, 0.01))
    size = int(st.radio("Window / 窗口", [3, 5], horizontal=True))
    noisy = salt_and_pepper(image, amount, int(st.session_state["seed"]))
    rng = np.random.default_rng(int(st.session_state["seed"]) + 1)
    gaussian_noisy = image + 0.05 * rng.standard_normal(image.shape)
    med = median(noisy, size)
    gauss, _kernel = gaussian_filter(noisy, size=5, sigma=1.0)
    visual_heading()
    st.markdown(f"**{shown_label('Salt and pepper / 椒盐')}**")
    row, col = pick_neighbourhood(
        noisy,
        size=size,
        hint=plain(
            "Click a pixel to place the window. Drag the red window to move it. Yellow marks the centre pixel.",
            "点击一个像素放置窗口。拖动红色窗口来移动它。黄色标出中心像素。",
        ),
        key="l4-median-window",
    )
    values = window(noisy, row, col, size)
    ordered, centre = sorted_median(values)
    left, right = st.columns(2)
    with left:
        show_fig(value_figure(values, "Neighbourhood / 邻域"))
    with right:
        show_fig(line_figure(np.arange(ordered.size), [("Sorted / 排序后", ordered)], "Sorted samples / 排序后的样本", "rank", "value"))
    st.caption(
        t(
            f"Window centre is row {row}, column {col}.",
            f"窗口中心是第 {row} 行、第 {col} 列。",
        )
    )
    show_row(
        [
            (f"Median {size}×{size} / 中值", med, "image"),
            ("Gaussian on the same input / 同一输入上的高斯", gauss, "image"),
        ]
    )
    st.write(f"Median {centre:.4g}. It replaces the centre of the window. MSE median {mse(med, image):.4g}. MSE Gaussian {mse(gauss, image):.4g}.")
    st.caption("中值替换窗口中心。两个 MSE 都相对干净的工作图像。椒盐噪声不是高斯噪声。")
    show_array(median(gaussian_noisy, size), "Median of a Gaussian-noised copy / 对高斯噪声副本做中值", "image")
    steps(
        [
            ("Collect the window, including reflected samples at the border.", "收集窗口；在边界上包含反射样本。"),
            ("Sort. The median is the middle entry, index (k² − 1) / 2 for an odd window.", "排序。奇数窗口的中值是中间那一项，下标 (k² − 1) / 2。"),
            ("A bright impulse becomes the last entry and is not chosen unless the window is mostly impulses.", "亮脉冲会变成最后一项，除非窗口里大部分都是脉冲，否则不会被选中。"),
        ]
    )
    takeaways([("The Gaussian smooths the impulses into grey spots. The median deletes isolated ones.", "高斯把脉冲抹成灰斑。中值会去掉孤立的脉冲。")])
    try_this("Set the fraction to 0.2 and compare the two MSE values. / 把比例设为 0.2，比较两个 MSE。", "Then look at the median of the Gaussian-noised copy: the median is not the best filter for every noise. / 再看高斯噪声副本的中值：中值并不是对每种噪声都最好。")
    save_workspace(med, "median", "ws-median")


def _scene(z_title: str, zmin: float, zmax: float, half: int) -> dict:
    return dict(
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.62),
        camera=dict(eye=dict(x=1.35, y=-1.55, z=0.95)),
        uirevision="bilateral-components",
        bgcolor="white",
        xaxis=dict(title="Δx", range=[-half, half], showbackground=False, gridcolor="#e4e4e4"),
        yaxis=dict(title="Δy", range=[-half, half], showbackground=False, gridcolor="#e4e4e4"),
        zaxis=dict(title=z_title, range=[zmin, zmax], showbackground=False, gridcolor="#e4e4e4"),
    )


def _add_height(figure: go.Figure, values: np.ndarray, row: int, col: int, zmin: float, zmax: float) -> None:
    half = values.shape[0] // 2
    offsets = np.arange(-half, half + 1)
    figure.add_trace(
        go.Surface(
            x=offsets,
            y=offsets,
            z=values,
            colorscale="Turbo",
            cmin=zmin,
            cmax=zmax,
            showscale=False,
            lighting=dict(ambient=0.62, diffuse=0.7, specular=0.12, roughness=0.7),
            contours=dict(
                x=dict(show=True, color="rgba(0,0,0,0.28)", width=1, highlight=False),
                y=dict(show=True, color="rgba(0,0,0,0.28)", width=1, highlight=False),
            ),
            hovertemplate="Δx %{x:.0f}<br>Δy %{y:.0f}<br>%{z:.3f}<extra></extra>",
        ),
        row=row,
        col=col,
    )
    lift = 0.035 * (zmax - zmin)
    figure.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[float(values[half, half]) + lift],
            mode="markers+text",
            text=["p"],
            textposition="top center",
            textfont=dict(size=12, color="#111111"),
            marker=dict(size=4, color="#f4d35e", line=dict(color="#111111", width=2)),
            hovertemplate="p<extra></extra>",
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def _bilateral_component_figure(
    window: np.ndarray,
    spatial: np.ndarray,
    range_weight: np.ndarray,
    product: np.ndarray,
    output: np.ndarray,
    q_row: int,
    q_col: int,
) -> go.Figure:
    """Five rotatable surfaces: output, the two Gaussians, their product, and the input."""
    half = window.shape[0] // 2
    q_row = int(np.clip(q_row, -half, half))
    q_col = int(np.clip(q_col, -half, half))
    low = float(min(window.min(), output.min()))
    high = float(max(window.max(), output.max()))
    span = max(high - low, 0.2)
    pad = 0.08 * span
    image_z = (low - pad, high + pad)
    weight_z = (0.0, 1.0)
    figure = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "surface"}, {"type": "surface"}, {"type": "surface"}],
            [{"type": "surface"}, {"type": "surface", "colspan": 2}, None],
        ],
        subplot_titles=[
            plain("Input", "输入"),
            plain("Spatial Gaussian", "空间高斯"),
            plain("Range Gaussian", "强度高斯"),
            plain("Output", "输出"),
            plain("Product", "乘积"),
        ],
        horizontal_spacing=0.03,
        vertical_spacing=0.08,
    )
    panels = [
        (window, 1, 1, image_z),
        (spatial, 1, 2, weight_z),
        (range_weight, 1, 3, weight_z),
        (output, 2, 1, image_z),
        (product, 2, 2, weight_z),
    ]
    for values, row, col, (zmin, zmax) in panels:
        _add_height(figure, values, row, col, zmin, zmax)
    centre = float(window[half, half])
    neighbour = float(window[half + q_row, half + q_col])
    lift = 0.035 * (image_z[1] - image_z[0])
    figure.add_trace(
        go.Scatter3d(
            x=[q_col, q_col],
            y=[q_row, q_row],
            z=[neighbour + lift, centre + lift],
            mode="lines+markers+text",
            text=["q", ""],
            textposition="top center",
            textfont=dict(size=12, color="#111111"),
            line=dict(color="#e0a106", width=6),
            marker=dict(size=4, color="#f4d35e", line=dict(color="#111111", width=2)),
            hovertemplate="|I(p)−I(q)|<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    weight_title = plain("weight", "权重")
    intensity_title = plain("intensity", "强度")
    scenes = {
        "scene": _scene(intensity_title, *image_z, half),
        "scene2": _scene(weight_title, *weight_z, half),
        "scene3": _scene(weight_title, *weight_z, half),
        "scene4": _scene(intensity_title, *image_z, half),
        "scene5": _scene(weight_title, *weight_z, half),
    }
    figure.update_layout(
        height=740,
        margin=dict(l=0, r=0, t=48, b=0),
        showlegend=False,
        uirevision="bilateral-components",
        **scenes,
    )
    figure.update_annotations(font_size=13)
    return figure


def _bilateral(image: np.ndarray) -> None:
    concept(
        "Each neighbour is weighted by a spatial Gaussian and by how close its intensity is to the centre. The weights depend on the image, so the filter is nonlinear.",
        "每个邻居的权重是空间高斯，再乘上它的强度与中心有多接近。权重依赖于图像，所以滤波器是非线性的。",
    )
    say(
        "Non-Linear Image Filters, Image Processing I. The player opens at 7:49.",
        "非线性图像滤波，Image Processing I。播放器从 7:49 开始。",
    )
    st.video(
        "https://www.youtube.com/embed/7FP7ndMEfsc",
        start_time=7 * 60 + 49,
        alt="Non-Linear Image Filters | Image Processing I",
    )
    model(
        r"G_{\sigma_s}(i,j)=\exp\left(-\frac{i^2+j^2}{2\sigma_s^2}\right),\quad "
        r"G_{\sigma_r}(\Delta)=\exp\left(-\frac{\Delta^2}{2\sigma_r^2}\right),\quad "
        r"g[y,x]=\frac{1}{W}\sum_{i,j}G_{\sigma_s}(i,j)\,G_{\sigma_r}(f[y,x]-f[y+i,x+j])\,f[y+i,x+j]"
    )
    controls_heading()
    use_step = st.checkbox("Use a synthetic noisy step / 使用含噪的合成阶跃", value=True)
    if use_step:
        source = np.zeros_like(image)
        source[:, image.shape[1] // 2 :] = 1.0
        rng = np.random.default_rng(int(st.session_state["seed"]))
        source = source + 0.05 * rng.standard_normal(source.shape)
    else:
        source = image
    sigma_s = float(st.slider("Spatial σ / 空间 σ", 0.5, 4.0, 2.0, 0.1))
    sigma_r = float(st.slider("Range σ / 强度 σ", 0.02, 0.5, 0.12, 0.01))
    radius = int(st.slider("Radius / 半径", 1, 4, 2))
    height, width = source.shape
    visual_heading()
    st.markdown(f"**{shown_label('Input / 输入')}**")
    st.caption(
        t(
            "The red box p is the centre pixel. The weights below are computed around p. The gold box q is one neighbour compared with p. Click the image to move p. Drag the gold box to move q.",
            "红框 p 是中心像素，下面的权重都围绕 p 计算。金框 q 是拿来和 p 比较的一个邻居。点击图像移动 p。拖动金框移动 q。",
        )
    )
    centre_col = min(width - 1, width // 2)
    picked = pick_neighbourhood(
        source,
        size=1,
        hint=plain(
            "Click to move p. Drag the gold box to move q.",
            "点击移动 p。拖动金框移动 q。",
        ),
        key="l4-bilateral-p",
        default=(height // 2, centre_col),
        neighbour=(height // 2, max(0, centre_col - 8)),
        max_width=420,
        labelled=True,
    )
    row_p, col_p, q_abs_row, q_abs_col = picked
    q_row = int(q_abs_row) - int(row_p)
    q_col = int(q_abs_col) - int(col_p)
    gauss, _kernel = gaussian_filter(source, size=2 * radius + 1, sigma=sigma_s)
    filtered = bilateral_filter(source, sigma_s, sigma_r, radius)
    half = max(8, int(np.ceil(3 * sigma_s)))
    window, spatial, range_weight, product = bilateral_weight_maps(source, row_p, col_p, sigma_s, sigma_r, half)
    output = reflect_window(filtered, row_p, col_p, half)
    show_row(
        [
            ("Gaussian / 高斯", gauss, "image"),
            ("Bilateral / 双边", filtered, "image"),
        ]
    )
    show_array(filtered - gauss, "Bilateral minus Gaussian / 双边减高斯", "residual")
    say(
        "Supplementary. The five surfaces are the terms drawn around pixel p: the filtered patch, the spatial Gaussian, the range Gaussian, their product before dividing by W, and the input patch. x is the column offset and y is the row offset. The gold segment on the input is |I(p)−I(q)|. Drag a surface to rotate it. This picture is not taken from a listed slide title.",
        "补充说明。五块曲面是像素 p 周围的各项：滤波后的小块、空间高斯、强度高斯、除以 W 之前的乘积，以及输入小块。x 是列偏移，y 是行偏移。输入上的金色线段是 |I(p)−I(q)|。拖动曲面可以旋转。这张图并不来自已列出的幻灯片标题。",
    )
    st.plotly_chart(
        _bilateral_component_figure(window, spatial, range_weight, product, output, q_row, q_col),
        config={"displaylogo": False, "scrollZoom": True},
        key="l4-bilateral-surfaces",
    )
    centre = half
    q_row_c = int(np.clip(q_row, -half, half))
    q_col_c = int(np.clip(q_col, -half, half))
    intensity_p = float(window[centre, centre])
    intensity_q = float(window[centre + q_row_c, centre + q_col_c])
    spatial_q = float(spatial[centre + q_row_c, centre + q_col_c])
    range_q = float(range_weight[centre + q_row_c, centre + q_col_c])
    support = product[centre - radius : centre + radius + 1, centre - radius : centre + radius + 1]
    weight_sum = float(support.sum())
    say(
        f"I(p) = {intensity_p:.3f} at row {row_p}, column {col_p}. "
        f"I(q) = {intensity_q:.3f} at offset ({q_row_c}, {q_col_c}), so |I(p)−I(q)| = {abs(intensity_p - intensity_q):.3f}. "
        f"There the spatial factor is {spatial_q:.3f}, the range factor is {range_q:.3f}, and the product is {spatial_q * range_q:.3f}. "
        f"W, the sum of the product inside the radius, is {weight_sum:.3f}.",
        f"p 在第 {row_p} 行、第 {col_p} 列，I(p) = {intensity_p:.3f}。"
        f"q 的偏移是 ({q_row_c}, {q_col_c})，I(q) = {intensity_q:.3f}，所以 |I(p)−I(q)| = {abs(intensity_p - intensity_q):.3f}。"
        f"那里的空间因子是 {spatial_q:.3f}，强度因子是 {range_q:.3f}，乘积是 {spatial_q * range_q:.3f}。"
        f"W 是半径内乘积的和，等于 {weight_sum:.3f}。",
    )
    row = image.shape[0] // 2
    show_fig(
        line_figure(
            np.arange(source.shape[1]),
            [("Input", source[row]), ("Gaussian", gauss[row]), ("Bilateral", filtered[row])],
            "Cross-section / 剖面",
            "column",
            "intensity",
        )
    )
    steps(
        [
            ("The spatial factor is the ordinary Gaussian weight.", "空间因子就是普通的高斯权重。"),
            ("The range factor is near zero when a neighbour belongs to the other side of an edge.", "邻居落在边缘另一侧时，强度因子接近零。"),
            ("The product is tall only where a neighbour is both nearby and similar in intensity.", "只有又近、强度又接近的邻居，乘积才高。"),
            ("The weights are renormalised for every pixel. The sum uses integer offsets inside the radius.", "每个像素都重新归一化权重。求和只用半径内的整数偏移。"),
        ]
    )
    takeaways([("A large range σ makes the range factor almost constant, and the filter returns toward a Gaussian.", "很大的强度 σ 会让强度因子几乎变成常数，滤波器就退回高斯。")])
    try_this(
        "On the synthetic step, drag the surfaces, then move p from the bright side of the edge onto the flat region. The range surface should change from a cliff to a plateau. Increase the range σ to 0.5 and the cliff flattens; the cross-section should start to look like the Gaussian.",
        "在合成阶跃上拖动曲面，再把 p 从边缘的亮侧移到平坦区域。强度曲面应从一道悬崖变成一块平台。把强度 σ 增到 0.5，悬崖会变平，剖面应开始像高斯。",
    )
    save_workspace(filtered, "bilateral", "ws-bilateral")


def _patch_compare_figure(patch_p: np.ndarray, patch_q: np.ndarray) -> go.Figure:
    """Three small matrices: the patch at p, the patch at q, and their squared difference."""
    difference = (patch_p - patch_q) ** 2
    difference_max = max(float(difference.max()), 0.2)
    panels = (
        (patch_p, plain("Patch around p", "p 周围的小块"), 0.0, 1.0, "Gray"),
        (patch_q, plain("Patch around q", "q 周围的小块"), 0.0, 1.0, "Gray"),
        (difference, plain("Squared difference", "平方差"), 0.0, difference_max, "Magma"),
    )
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[title for _matrix, title, *_rest in panels],
        horizontal_spacing=0.08,
    )
    text_size = {3: 16, 5: 13, 7: 11}.get(patch_p.shape[0], 10)
    for index, (matrix, _title, zmin, zmax, scale) in enumerate(panels, start=1):
        count = matrix.shape[0]
        figure.add_trace(
            go.Heatmap(
                z=matrix,
                x=list(range(count)),
                y=list(range(count)),
                colorscale=scale,
                zmin=zmin,
                zmax=zmax,
                showscale=index == 3,
                hovertemplate=plain(
                    "row %{y}<br>column %{x}<br>%{z:.4f}<extra></extra>",
                    "行 %{y}<br>列 %{x}<br>%{z:.4f}<extra></extra>",
                ),
                colorbar=dict(thickness=12, len=0.75),
            ),
            row=1,
            col=index,
        )
        span = max(zmax - zmin, 1e-9)
        xs: list[int] = []
        ys: list[int] = []
        labels: list[str] = []
        colours: list[str] = []
        for row_index, row_values in enumerate(matrix):
            for col_index, value in enumerate(row_values):
                xs.append(col_index)
                ys.append(row_index)
                labels.append(f"{float(value):.2f}")
                colours.append("#1b242c" if (float(value) - zmin) / span > 0.55 else "#f7f4ef")
        figure.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="text",
                text=labels,
                textfont=dict(size=text_size, color=colours),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=index,
        )
        axis = "x" if index == 1 else f"x{index}"
        figure.update_yaxes(
            autorange="reversed",
            scaleanchor=axis,
            scaleratio=1,
            range=[count - 0.5, -0.5],
            row=1,
            col=index,
        )
        figure.update_xaxes(dtick=1, range=[-0.5, count - 0.5], row=1, col=index)
    figure.update_layout(height=360, margin=dict(l=8, r=8, t=48, b=8))
    figure.update_annotations(font_size=14)
    return figure


def _weight_picture(
    image: np.ndarray,
    weights: np.ndarray,
    row: int,
    col: int,
    search: int,
    patch: int,
) -> np.ndarray:
    """Dim pixels outside the search window and paint matching patches yellow."""
    grey = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    height, width = grey.shape
    half = int(search) // 2
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    inside = (np.abs(rows - int(row)) <= half) & (np.abs(cols - int(col)) <= half)
    shown = np.where(inside, grey, grey * 0.35)
    alpha = np.clip(matching_patches(weights, patch), 0.0, 1.0)
    yellow = np.array([0.96, 0.75, 0.12], dtype=np.float64)
    painted = shown[..., None] * (1.0 - 0.7 * alpha[..., None]) + yellow * (0.7 * alpha[..., None])
    return np.clip(painted, 0.0, 1.0)


def _landmark_sentence(
    name_en: str,
    name_zh: str,
    place: tuple[int, int],
    weights: np.ndarray,
    row_p: int,
    col_p: int,
    search: int,
) -> tuple[str, str]:
    row, col = place
    half = int(search) // 2
    inside = abs(row - row_p) <= half and abs(col - col_p) <= half and 0 <= row < weights.shape[0] and 0 <= col < weights.shape[1]
    if not inside:
        return (
            f"The {name_en} at row {row}, column {col} is outside the search window, so its weight is 0.",
            f"第 {row} 行、第 {col} 列的{name_zh}在搜索窗外面，所以权重是 0。",
        )
    weight = float(weights[row, col])
    return (
        f"The {name_en} at row {row}, column {col} has weight {weight:.3f}.",
        f"第 {row} 行、第 {col} 列的{name_zh}权重是 {weight:.3f}。",
    )


def _nlm(image: np.ndarray) -> None:
    concept(
        "To replace pixel p, take the small patch around it and look through the search window for pixels whose own patches match. Matching pixels get a large weight. Pixels whose patches differ get a weight near zero. The new value at p is the weighted average, so another copy of the same diagonal can contribute even though it is many columns away.",
        "替换像素 p 时，先取它周围的一小块，再在搜索窗里找周围小块与它相像的像素。相像的权重大，不像的权重接近 0。p 的新值是这个加权平均，所以另一条同样的斜线即使隔着很多列，也可以参加进来。",
    )
    model(
        r"d(p,q)=\mathrm{mean}\big(P(p)-P(q)\big)^2,\quad "
        r"w(p,q)=\exp\big(-d(p,q)/h^2\big),\quad "
        r"g(p)=\sum_q w(p,q)\,f(q)\big/\sum_q w(p,q)"
    )
    say(
        "Supplementary. The sum runs over the search window. The laboratory uses the mean of the squared differences on a square patch. The slides name non-local means and do not write this expression.",
        "补充说明。求和范围是搜索窗。实验室用的是正方形小块上平方差的平均。幻灯片列出了非局部均值，没有写出这个式子。",
    )
    controls_heading()
    use_marks = st.checkbox("Repeated marks / 重复图案", value=True)
    patch = int(st.select_slider("Patch size / 小块边长", options=[3, 5, 7], value=5))
    search = int(st.select_slider("Search window / 搜索窗", options=[15, 33, 61], value=33))
    strength = float(st.slider("Strength h / 强度 h", 0.02, 0.5, 0.08, 0.01))
    seed = int(st.session_state["seed"])
    if use_marks:
        noisy = repeated_marks(noise=0.04, seed=seed)
        default_p, default_q = DIAGONAL_A, DIAGONAL_B
        landmarks = (
            (DIAGONAL_A, "first diagonal", "第一条斜线"),
            (DIAGONAL_B, "other diagonal", "另一条斜线"),
            (BAR, "horizontal bar", "横线"),
        )
    else:
        crop = centre_crop(image, MARK_SIZE)
        rng = np.random.default_rng(seed)
        noisy = np.clip(crop + 0.04 * rng.standard_normal(crop.shape), 0.0, 1.0)
        default_p = (noisy.shape[0] // 2, noisy.shape[1] // 2)
        default_q = (default_p[0], min(noisy.shape[1] - 1, default_p[1] + 8))
        landmarks = ()
    visual_heading()
    if use_marks:
        say(
            "Supplementary. The picture has two copies of one diagonal and one different bar, plus a little noise. It is not a slide figure.",
            "补充说明。图里有两条一样的斜线、一条不一样的横线，再加一点噪声。这不是幻灯片上的图。",
        )
    else:
        say(
            "This is the centre of the workspace image, with noise from the current seed. Yellow still means the patch matches p.",
            "这是工作区图像的中心，噪声由当前的种子决定。黄色仍然表示小块和 p 相像。",
        )
    say(
        "The red box is the patch around p. The gold pixel is a candidate q; its patch has the same size and is centred on that pixel. Click to move p. Drag the gold pixel to move q.",
        "红框是 p 周围的小块。金点是候选像素 q，q 的小块和红框一样大，中心就在这个金点上。点击移动 p。拖动金点移动 q。",
    )
    picked = pick_neighbourhood(
        noisy,
        size=patch,
        hint=plain(
            "Click to move p. Drag the gold pixel to move q.",
            "点击移动 p。拖动金点移动 q。",
        ),
        key="l4-nlm-marks" if use_marks else "l4-nlm-photo",
        default=default_p,
        neighbour=default_q,
        max_width=520,
    )
    row_p, col_p, row_q, col_q = picked
    patch_p, patch_q, distance, weight = compare_patches(noisy, row_p, col_p, row_q, col_q, patch, strength)
    weights = weight_map(noisy, row_p, col_p, patch, search, strength)
    average = weighted_value(noisy, weights)
    show_fig(_patch_compare_figure(patch_p, patch_q))
    if weight >= 0.35:
        verdict_en = "The two patches match, so q is kept in the average."
        verdict_zh = "两块很像，所以 q 会算进平均。"
    else:
        verdict_en = "The two patches differ, so q barely changes p."
        verdict_zh = "两块不像，所以 q 几乎不改变 p。"
    say(
        f"Mean squared difference d = {distance:.4f}. Weight of q: w = exp(−d / h²) = {weight:.3f}. {verdict_en} "
        f"The noisy value at p is {float(noisy[row_p, col_p]):.3f}. The average that replaces it is {average:.3f}. "
        "p itself always has weight 1, because a patch matches itself.",
        f"平均平方差 d = {distance:.4f}。q 的权重 w = exp(−d / h²) = {weight:.3f}。{verdict_zh}"
        f"p 的含噪值是 {float(noisy[row_p, col_p]):.3f}。用来替换它的平均是 {average:.3f}。"
        "p 自己的权重总是 1，因为小块和自己完全一样。",
    )
    show_array(
        _weight_picture(noisy, weights, row_p, col_p, search, patch),
        "Patches that match p / 和 p 相像的小块",
        "image",
    )
    say(
        "Pixels outside the search window are dimmed. Yellow covers each pixel whose weight is above 0.2 and spreads across that pixel's patch, so you see the similar patch rather than one sample. The average uses the centre of each yellow patch. Grey means the window looked there and the patch did not match. This picture belongs only to the p you picked.",
        "搜索窗外面的像素被压暗。黄色盖住权重大于 0.2 的像素，并铺到它的小块上，所以你看到的是相像的那一块，而不是一个点。平均用的是每个黄色小块的中心像素。灰色表示窗里已经比过、但不像。这张图只属于你选中的那个 p。",
    )
    for place, name_en, name_zh in landmarks:
        if place == (row_p, col_p):
            continue
        en, zh = _landmark_sentence(name_en, name_zh, place, weights, row_p, col_p, search)
        say(en, zh)
    strong = weights >= 0.2
    count = int(strong.sum())
    rows, cols = np.nonzero(strong)
    if count == 0:
        far_row, far_col = 0, 0
    else:
        reach = np.maximum(np.abs(rows - row_p), np.abs(cols - col_p))
        chosen = int(np.argmax(reach))
        far_row = int(rows[chosen] - row_p)
        far_col = int(cols[chosen] - col_p)
    if far_row == 0 and far_col == 0:
        say(
            "Every other pixel in the search window has weight below 0.2, so the average barely leaves p itself.",
            "搜索窗里其他像素的权重都低于 0.2，所以这个平均几乎离不开 p 自己。",
        )
    else:
        away_en = []
        away_zh = []
        if far_row:
            away_en.append(f"{abs(far_row)} rows")
            away_zh.append(f"{abs(far_row)} 行")
        if far_col:
            away_en.append(f"{abs(far_col)} columns")
            away_zh.append(f"{abs(far_col)} 列")
        say(
            f"{count} pixels have weight above 0.2. The farthest of them is {' and '.join(away_en)} from p.",
            f"权重大于 0.2 的像素有 {count} 个。其中最远的一个离 p 有 {'、'.join(away_zh)}。",
        )
    if st.button("Run non-local means on the whole picture / 对整幅图运行非局部均值", type="primary"):
        restored, _centre_weights = nonlocal_means(noisy, patch, search, strength)
        st.session_state["nlm_cache"] = {
            "noisy": noisy,
            "restored": restored,
            "patch": patch,
            "search": search,
            "strength": strength,
        }
    cached = st.session_state.get("nlm_cache")
    fresh = (
        isinstance(cached, dict)
        and cached.get("patch") == patch
        and cached.get("search") == search
        and abs(float(cached.get("strength", -1.0)) - strength) < 1e-9
        and isinstance(cached.get("noisy"), np.ndarray)
        and cached["noisy"].shape == noisy.shape
        and np.array_equal(cached["noisy"], noisy)
    )
    if not fresh:
        st.info(
            t(
                "The yellow picture is the average for this one p. The button repeats that average at every pixel. It uses the patch, search window, and h shown above.",
                "上面的黄斑只说明这一个 p。按钮会在每个像素上做同样的平均，用的是上面的小块、搜索窗和 h。",
            )
        )
    else:
        restored = cached["restored"]
        gauss, _kernel = gaussian_filter(noisy, size=7, sigma=1.5)
        show_row(
            [
                ("Noisy / 含噪", noisy, "image"),
                ("Gaussian / 高斯", gauss, "image"),
                ("Non-local means / 非局部均值", restored, "image"),
            ]
        )
        if use_marks:
            clean = repeated_marks(noise=0.0)
            stroke = clean > 0.5
            gauss_error = mse(gauss[stroke], clean[stroke])
            nlm_error = mse(restored[stroke], clean[stroke])
            if nlm_error < gauss_error:
                reading_en = "The Gaussian mixes each stroke with the grey beside it. Non-local means mixes each diagonal with the other copy of that diagonal."
                reading_zh = "高斯把笔画和旁边的灰色混在一起。非局部均值把一条斜线和另一条同样的斜线混在一起。"
            else:
                reading_en = "At this h, unlike patches still receive weight, so the strokes are pulled toward the background. Lower h and run the button again."
                reading_zh = "这个 h 下，不像的小块也有权重，笔画会被拉向背景。把 h 调小后再按一次按钮。"
            say(
                f"On the bright strokes, the Gaussian MSE is {gauss_error:.4f} and the non-local-means MSE is {nlm_error:.4f}. {reading_en}",
                f"在亮的笔画上，高斯的 MSE 是 {gauss_error:.4f}，非局部均值的 MSE 是 {nlm_error:.4f}。{reading_zh}",
            )
        save_workspace(restored, "nlm", "ws-nlm")
    steps(
        [
            ("Read the red patch around p.", "读出 p 周围的红色小块。"),
            ("Read the patch of the same size centred on the gold pixel q.", "读出以金点 q 为中心、同样大小的小块。"),
            ("The weight is exp(−d / h²), and d is the mean squared difference of those two patches.", "权重是 exp(−d / h²)，d 是这两块的平均平方差。"),
            ("A yellow patch matches p. The average uses the centre of that patch. Dim pixels were outside the search window.", "黄色小块和 p 相像。平均用的是这块的中心。压暗的像素在搜索窗外面。"),
        ]
    )
    takeaways(
        [
            (
                "A large h makes unlike patches yellow as well, and the average starts to mix the diagonal with the bar and the background.",
                "很大的 h 会让不像的小块也变黄，平均就开始把斜线、横线和背景混在一起。",
            ),
            (
                "The button repeats the same average at every pixel. The yellow picture is only the map for the p you picked.",
                "按钮在每个像素上重复同一种平均。黄色的图只是你选中的那个 p 的权重。",
            ),
        ]
    )
    try_this(
        "Gold q starts on the other diagonal. The two patches should look alike, the weight should stay high, and both diagonals should be yellow. Drag q onto the horizontal bar: the squared-difference panel brightens, the weight falls near zero, and the bar stays grey. Then set the search window to 15. The other diagonal dims, because it now lies outside the window. The bar is closer, so it stays inside and grey.",
        "金点一开始在另一条斜线上。两块应当差不多，权重应当比较高，两条斜线都应当是黄的。把金点拖到横线上：平方差变亮，权重掉到接近 0，横线保持灰色。再把搜索窗改成 15。另一条斜线会被压暗，因为它已经在窗外。横线更近，所以还在窗里，并且仍然是灰的。",
    )


def _inpaint(image: np.ndarray) -> None:
    concept(
        "Inpainting fills pixels that were never observed. Denoising changes pixels that were observed but are noisy. These are different missing-data problems.",
        "修补填的是从未被观测的像素。去噪改变的是被观测过但含噪的像素。这是两种不同的缺失数据问题。",
    )
    model(r"f \leftarrow \mathrm{mean\ of\ 4\ neighbours},\ \mathrm{on\ the\ missing\ set}")
    controls_heading()
    iterations = int(st.slider("Iterations / 迭代", 20, 500, 200, 20))
    hole = min(16, int(image.shape[0]), int(image.shape[1]))
    visual_heading()
    st.markdown(f"**{shown_label('Original / 原图')}**")
    row, col = pick_neighbourhood(
        image,
        size=hole,
        anchor="origin",
        default=(image.shape[0] // 3, image.shape[1] // 3),
        hint=plain(
            "Click to place the hole. Drag the red square to move it.",
            "点击放置空洞。拖动红色方块来移动它。",
        ),
        key="l4-inpaint-hole",
    )
    missing = rectangle_mask(image.shape, row, col, hole, hole)
    damaged = image.copy()
    damaged[missing] = 0.0
    restored = diffuse(damaged, missing, iterations)
    st.caption(t(f"Hole starts at row {row}, column {col}.", f"空洞从第 {row} 行、第 {col} 列开始。"))
    mask_image = missing.astype(np.float64)
    show_row(
        [
            ("Damaged / 损坏", damaged, "image"),
            ("Mask / 掩模", mask_image, "image"),
        ]
    )
    show_array(restored, "Diffusion result / 扩散结果", "image")
    st.caption(
        "Known pixels are put back after every iteration. The hole is filled by the mean-value property, not by copying a unique original. "
        "A different prior would fill it differently. "
        "每个已知像素在每次迭代后都被放回。空洞是靠平均值性质填上的，不是复制出一个唯一的原图。不同的先验会填出不同的结果。"
    )
    steps(
        [
            ("Mark the missing rectangle.", "标出缺失的矩形。"),
            ("Replace each missing pixel by the average of its four neighbours.", "把每个缺失像素换成四个邻居的平均。"),
            ("Stop after the chosen number of iterations. This is not proved to be the original picture.", "在选定的迭代次数后停止。这并没有被证明就是原来的画面。"),
        ]
    )
    takeaways([("If the missing region is large, many images agree with the same boundary. The algorithm returns one of them.", "缺失区域很大时，许多图像都和同一条边界相符。算法只返回其中一幅。")])
    try_this("Place the hole on an edge and increase the iterations from 20 to 500. / 把空洞放在一条边缘上，把迭代从 20 增到 500。", "Compare this with the median page: there, every pixel was observed. / 和中值那一页比较：那里每个像素都被观测过。")
    save_workspace(restored, "inpainted", "ws-inpaint")
