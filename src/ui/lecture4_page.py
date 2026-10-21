"""Lecture 4 experiments."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.lecture1.gaussian import gaussian_filter
from src.lecture4.bilateral import bilateral_filter
from src.lecture4.compression import rle_decode, rle_encode, rle_ratio, storage_report
from src.lecture4.dct import basis_montage, dct2, dct_matrix, idct2, keep_frequencies
from src.lecture4.huffman import average_length, build_tree, codes_from_tree, decode, encode, frequencies, tree_lines
from src.lecture4.inpainting import diffuse, rectangle_mask
from src.lecture4.jpeg_demo import jpeg_like, zigzag
from src.lecture4.median import median, salt_and_pepper, sorted_median, window
from src.lecture4.nonlocal_means import nonlocal_means
from src.metrics import format_db, mse, psnr
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
from src.ui.experiments import L4_EXPERIMENTS
from src.ui.i18n import shown_label, t
from src.ui_common import show_fig


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
    row = int(st.slider("Row / 行", 0, image.shape[0] - 1, image.shape[0] // 2))
    if source == "image row":
        sequence = np.rint(np.clip(image[row], 0, 1) * 15).astype(int).tolist()
    elif source == "constant":
        sequence = [3] * 32
    else:
        sequence = [0, 1] * 16
    runs = rle_encode(sequence)
    decoded = rle_decode(runs)
    visual_heading()
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
    controls_heading()
    row = int(st.slider("Block row / 小块的行", 0, max(0, image.shape[0] - 8), 0, 8))
    col = int(st.slider("Block column / 小块的列", 0, max(0, image.shape[1] - 8), 0, 8))
    max_sum = int(st.slider("Keep coefficients with k + ℓ ≤ / 保留 k+ℓ 不超过", 0, 14, 4))
    block = image[row : row + 8, col : col + 8]
    if block.shape != (8, 8):
        st.warning("The image is smaller than 8×8. / 图像小于 8×8。")
        return
    matrix = dct_matrix(8)
    coefficients = dct2(block - 0.5, matrix)
    kept = keep_frequencies(coefficients, max_sum)
    reconstructed = idct2(kept, matrix) + 0.5
    visual_heading()
    show_array(basis_montage(8), "64 DCT basis images, each scaled for display / 64 个 DCT 基图像，每块单独缩放以便显示", "kernel")
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
    row = int(st.slider("Neighbourhood row / 邻域行", 0, image.shape[0] - 1, image.shape[0] // 2))
    col = int(st.slider("Neighbourhood column / 邻域列", 0, image.shape[1] - 1, image.shape[1] // 2))
    noisy = salt_and_pepper(image, amount, int(st.session_state["seed"]))
    rng = np.random.default_rng(int(st.session_state["seed"]) + 1)
    gaussian_noisy = image + 0.05 * rng.standard_normal(image.shape)
    med = median(noisy, size)
    gauss, _kernel = gaussian_filter(noisy, size=5, sigma=1.0)
    visual_heading()
    show_row(
        [
            ("Salt and pepper / 椒盐", noisy, "image"),
            (f"Median {size}×{size} / 中值", med, "image"),
            ("Gaussian on the same input / 同一输入上的高斯", gauss, "image"),
        ]
    )
    values = window(noisy, row, col, size)
    ordered, centre = sorted_median(values)
    left, right = st.columns(2)
    with left:
        show_fig(value_figure(values, "Neighbourhood / 邻域"))
    with right:
        show_fig(line_figure(np.arange(ordered.size), [("Sorted / 排序后", ordered)], "Sorted samples / 排序后的样本", "rank", "value"))
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


def _bilateral(image: np.ndarray) -> None:
    concept(
        "Each neighbour is weighted by a spatial Gaussian and by how close its intensity is to the centre. The weights depend on the image, so the filter is nonlinear.",
        "每个邻居的权重是空间高斯，再乘上它的强度与中心有多接近。权重依赖于图像，所以滤波器是非线性的。",
    )
    model(r"w_{ij}=\exp\left(-\frac{i^2+j^2}{2\sigma_s^2}\right)\exp\left(-\frac{(f-f_{ij})^2}{2\sigma_r^2}\right)")
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
    gauss, _kernel = gaussian_filter(source, size=2 * radius + 1, sigma=sigma_s)
    filtered = bilateral_filter(source, sigma_s, sigma_r, radius)
    visual_heading()
    show_row(
        [
            ("Input / 输入", source, "image"),
            ("Gaussian / 高斯", gauss, "image"),
            ("Bilateral / 双边", filtered, "image"),
        ]
    )
    show_array(filtered - gauss, "Bilateral minus Gaussian / 双边减高斯", "residual")
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
            ("The weights are renormalised for every pixel.", "每个像素都重新归一化权重。"),
        ]
    )
    takeaways([("A large range σ makes the range factor almost constant, and the filter returns toward a Gaussian.", "很大的强度 σ 会让强度因子几乎变成常数，滤波器就退回高斯。")])
    try_this("Increase the range σ to 0.5 on the synthetic step. The cross-section should start to look like the Gaussian. / 在合成阶跃上把强度 σ 增到 0.5。剖面应开始像高斯。", "Then turn the synthetic step off and use a workspace image. / 然后关掉合成阶跃，改用工作区图像。")
    save_workspace(filtered, "bilateral", "ws-bilateral")


def _nlm(image: np.ndarray) -> None:
    concept(
        "Non-local means averages pixels whose surrounding patches look alike, even if those pixels are far apart.",
        "非局部均值平均的是周围小块看起来相似的像素，即使这些像素相距很远。",
    )
    model(r"w(p,q)=\exp\left(-\|P(p)-P(q)\|^2/h^2\right)")
    controls_heading()
    patch = int(st.select_slider("Patch size / 小块边长", options=[3, 5], value=3))
    search = int(st.select_slider("Search window / 搜索窗", options=[5, 7, 11], value=7))
    strength = float(st.slider("Strength h / 强度 h", 0.02, 0.4, 0.08, 0.01))
    if st.button("Run non-local means / 运行非局部均值", type="primary"):
        source = centre_crop(image, 48)
        rng = np.random.default_rng(int(st.session_state["seed"]))
        noisy = source + 0.05 * rng.standard_normal(source.shape)
        restored, weights = nonlocal_means(noisy, patch, search, strength)
        st.session_state["nlm_cache"] = (noisy, restored, weights)
    visual_heading()
    cached = st.session_state.get("nlm_cache")
    if cached is None:
        st.info("The search compares patches, so this experiment runs only when you press the button. The crop is 48×48. / 搜索要比较小块，所以只有按下按钮才运行。裁剪尺寸是 48×48。")
        return
    noisy, restored, weights = cached
    show_row(
        [
            ("Noisy crop / 含噪裁剪", noisy, "image"),
            ("Restored / 恢复", restored, "image"),
            ("Weights at the centre pixel / 中心像素的权重", weights, "kernel"),
        ]
    )
    st.caption("The weight map is the search window around the centre pixel, not a second image of the scene. / 权重图是中心像素周围的搜索窗，不是场景的另一幅图像。")
    steps(
        [
            ("Shift the image to a candidate offset.", "把图像移到一个候选偏移。"),
            ("The patch distance is the local average of the squared difference.", "小块距离是平方差的局部平均。"),
            ("Distant pixels with a similar patch receive a large weight.", "远处但小块相似的像素会得到大的权重。"),
        ]
    )
    takeaways([("A repeated texture can contribute evidence from far away. A Gaussian only looks at geometric neighbours.", "重复的纹理可以从远处提供证据。高斯只看几何上的邻居。")])
    try_this("Increase h until the weight map becomes almost flat. / 增大 h，直到权重图几乎变平。", "The button recomputes. Changing the workspace image does nothing until you press it again. / 按钮会重新计算。在再次按下之前，更换工作图像不会生效。")
    save_workspace(restored, "nlm", "ws-nlm")


def _inpaint(image: np.ndarray) -> None:
    concept(
        "Inpainting fills pixels that were never observed. Denoising changes pixels that were observed but are noisy. These are different missing-data problems.",
        "修补填的是从未被观测的像素。去噪改变的是被观测过但含噪的像素。这是两种不同的缺失数据问题。",
    )
    model(r"f \leftarrow \mathrm{mean\ of\ 4\ neighbours},\ \mathrm{on\ the\ missing\ set}")
    controls_heading()
    iterations = int(st.slider("Iterations / 迭代", 20, 500, 200, 20))
    row = int(st.slider("Hole row / 空洞的行", 0, max(0, image.shape[0] - 16), image.shape[0] // 3))
    col = int(st.slider("Hole column / 空洞的列", 0, max(0, image.shape[1] - 16), image.shape[1] // 3))
    missing = rectangle_mask(image.shape, row, col, min(16, image.shape[0]), min(16, image.shape[1]))
    damaged = image.copy()
    damaged[missing] = 0.0
    restored = diffuse(damaged, missing, iterations)
    visual_heading()
    mask_image = missing.astype(np.float64)
    show_row(
        [
            ("Original / 原图", image, "image"),
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
