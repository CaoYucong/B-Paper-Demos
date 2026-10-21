# B Paper Demonstrations

B Paper Demonstrations is a local laboratory for Cambridge Engineering B-paper material. The public pages are directories. Refreshing the browser stays on the same directory, and a section name after `#` stays on that section.

B Paper Demonstrations 是剑桥工程 B 类课程的本地实验。公开页面是目录地址。刷新浏览器会留在同一个目录；`#` 后面的名字会留在那一节。

The only implemented module is **B14 Information Engineering → Image Processing**. Its Lecture page is one scrolling laboratory: an overview, four lectures, and practice. Signal Processing, B15, and B16 are reserved and do not invent course content. Every topic has a Sheet page; those sheets are placeholders.

目前实现的只有 **B14 信息工程 → 图像处理**。它的 Lecture 页是一条往下读的实验：总览、四讲和练习。信号处理、B15 和 B16 是预留，不编造课程内容。每个主题都有 Sheet 页，这些习题页仍是占位。

## Addresses

`python serve.py` listens on port 8501. Open:

`python serve.py` 监听 8501 端口。打开：

| Page | Address |
| --- | --- |
| Home / 首页 | `/home/` |
| B14 | `/home/B14/` |
| Image Processing / 图像处理 | `/home/B14/ImageProcessing/` |
| Lecture / 讲义 | `/home/B14/ImageProcessing/Lecture/` |
| Bayer sampling | `/home/B14/ImageProcessing/Lecture/#Bayersampling` |
| Sheet / 习题 | `/home/B14/ImageProcessing/Sheet/` |
| Signal Processing lecture / 信号处理讲义 | `/home/B14/SignalProcessing/Lecture/` |
| Signal Processing sheet / 信号处理习题 | `/home/B14/SignalProcessing/Sheet/` |
| B15, B16 | `/home/B15/`, `/home/B16/` |

The hash is the English section title with spaces and punctuation removed. `Bayer sampling` is `#Bayersampling`.

`#` 后面是英文小节名，去掉空格和标点。`Bayer sampling` 就是 `#Bayersampling`。

## Install and launch

Python 3.11 or newer. From this directory:

需要 Python 3.11 或更高版本。在本目录：

```bash
source .venv/bin/activate
pip install -r requirements.txt
pytest
python serve.py
```

Dependencies are NumPy, SciPy, Matplotlib, Pillow, Plotly, Streamlit, and pytest. No GPU and no PyTorch. The program does not call an external model service.

依赖是 NumPy、SciPy、Matplotlib、Pillow、Plotly、Streamlit 和 pytest。不需要 GPU，也不用 PyTorch。程序不会调用外部模型服务。

The interface language is 中文, English, or Bilingual. The default is Bilingual.

界面语言是中文、English 或 Bilingual。默认是 Bilingual。

## What the pages contain

**Home / 首页.** Choose B14, B15, or B16.

**B14.** Choose Image Processing or Signal Processing. Image Processing is available. Signal Processing is reserved.

**Image Processing / 图像处理.** Choose Lecture or Sheet. Lecture opens the laboratory. Sheet says the example sheet is reserved.

**Lecture / 讲义.** Overview, Lectures 1–4, and Practice, in that order. The sidebar jumps to a section.

**首页** 选择 B14、B15 或 B16。**B14** 再选择图像处理或信号处理。图像处理可用，信号处理预留。**图像处理** 再选择讲义或习题。讲义打开实验，习题页写明习题纸仍是占位。**讲义** 按顺序是总览、第 1 到第 4 讲和练习。侧栏可以跳到某一节。

### Lecture 1 — Image Enhancement / 第 1 讲 — 图像增强

Sampling and quantisation, Bayer sampling, point operations, histogram equalisation, temporal averaging, the convolution playground, padding and boundaries, Sobel edges, Gaussian filtering, and matched filtering. Matched filtering is an additional example.

采样与量化、Bayer 采样、点运算、直方图均衡、时间平均、卷积实验台、填充与边界、Sobel 边缘、高斯滤波、匹配滤波。匹配滤波是补充例子。

### Lecture 2 — Fourier Analysis / 第 2 讲 — 傅里叶分析

Spatial frequency, Fourier decomposition, Fourier properties, magnitude and phase, frequency-domain filtering, the convolution theorem, sampling and aliasing, and hybrid images.

空间频率、傅里叶分解、傅里叶性质、幅度与相位、频域滤波、卷积定理、采样与混叠、混合图像。

### Lecture 3 — Image Restoration / 第 3 讲 — 图像复原

Degradation model, operator A, inverse filtering, Wiener filtering, directional motion blur, scalar Bayesian MAP, MAP gradient descent, the comparison of MAP, Wiener, and inverse filtering, and supplementary super-resolution. The original operators remain in `src/operators.py`, `src/map_estimation.py`, and `src/wiener.py`. Notes from the earlier write-up are in `docs/lecture3_map_notes.md`.

退化模型、算子 A、逆滤波、维纳滤波、有方向的运动模糊、标量贝叶斯 MAP、MAP 梯度下降，以及 MAP、维纳与逆滤波的比较，另有补充的超分辨率。原来的算子仍在 `src/operators.py`、`src/map_estimation.py` 和 `src/wiener.py`。较早的说明在 `docs/lecture3_map_notes.md`。

### Lecture 4 — Compression and Nonlinear Filters / 第 4 讲 — 压缩与非线性滤波

Storage calculator, run-length encoding, Huffman coding, the discrete cosine transform, a JPEG-like pipeline, median filtering, bilateral filtering, non-local means, and inpainting. The JPEG-like page reconstructs an array. It does not write a JPEG file.

存储量计算、游程编码、霍夫曼编码、离散余弦变换、类 JPEG 流程、中值滤波、双边滤波、非局部均值、修补。类 JPEG 页面重建的是数组，不会写出 JPEG 文件。

### Practice / 练习

Eight original numerical problems and a ten-question quiz, on the Lecture page after Lecture 4. These are not the official example sheet. The Sheet page is the reserved place for that sheet. See `docs/source_alignment.md`.

讲义页在第 4 讲之后有八道原创数值题和十道小测。这些不是官方例题。Sheet 页才是给官方习题纸留的位置。说明见 `docs/source_alignment.md`。

## Shared workspace

The sidebar chooses a built-in image, an upload, or the workspace. **Use as workspace image / 用作工作区图像** stores the current result. Switch the source to Workspace before using it in another section. **Export parameters / 导出参数** writes the slider values as JSON. It does not write an image. If the workspace is empty, the lab shows the built-in sharp-edge image, 锐边渐变, and says so.

侧栏选择内置图像、上传或工作区。**用作工作区图像** 保存当前结果。在另一节使用之前，把来源换成工作区。**导出参数** 把滑块值写成 JSON，不写图像。工作区是空的时候，实验显示内置的锐边渐变，并说明这一点。

## Conventions

Read `docs/mathematical_conventions.md` before changing a formula.

改公式之前先读 `docs/mathematical_conventions.md`。

- Correlation does not flip the kernel. Convolution does. 相关不翻转核，卷积翻转。
- Lecture 3 convolution is circular, through the FFT. 第 3 讲的卷积是圆周卷积，走 FFT。
- Intermediate images are not clipped to `[0, 1]`. PNG download is. 中间图像不裁到 `[0, 1]`，下载的 PNG 会裁。
- A compression ratio in the calculator is not a measured file size. 计算器里的压缩比不是测到的文件大小。
- Solvers do not receive the original image. 求解器拿不到原图。

## Suggested path

1. Lecture 1: sampling, then the convolution playground. 第 1 讲：采样，然后卷积实验台。
2. Lecture 1 Gaussian separability, then Lecture 2 convolution theorem. 第 1 讲高斯可分离性，然后第 2 讲卷积定理。
3. Lecture 2 magnitude and phase, then a low-pass filter. 第 2 讲幅度与相位，然后低通滤波。
4. Lecture 3 degradation, inverse filtering, Wiener, then MAP. 第 3 讲退化、逆滤波、维纳，然后 MAP。
5. Lecture 4 JPEG-like page, then median against salt-and-pepper noise. 第 4 讲类 JPEG 页面，然后用中值对付椒盐噪声。
6. Practice, and use each problem's link to return to the experiment. 练习，并用每题的链接回到实验。

## Troubleshooting

- If port 8501 is taken, stop the old `python serve.py` before starting another. This project's config does not reload on save. 如果 8501 已被占用，先停掉旧的 `python serve.py`。本项目的配置不会在保存时自动重载。
- Non-local means runs on a 48×48 crop and only after its button. Gradient descent and the comparison page also wait for a button. 非局部均值只用 48×48 的裁块，并且要点按钮。梯度下降和比较页也要点按钮。
- Plots that show `log10(1+|F|)` are display compressions. The calculation uses the complex DFT. 图上的 `log10(1+|F|)` 是显示压缩。计算用的是复数 DFT。
- CJK text in a Matplotlib figure needs a font such as PingFang SC. Plotly labels do not. Matplotlib 图里的中文需要 PingFang SC 这类字体。Plotly 的标签不需要。

## Tests

```bash
pytest
```

The suite covers the original MAP operators, Lectures 1–4, the catalog, practice grading, and the public directory addresses. A module is only claimed to pass after `pytest` has been run on that code.

测试覆盖原来的 MAP 算子、第 1 到第 4 讲、课程目录、练习评分和公开目录地址。某个模块只有在对它跑过 `pytest` 之后才说它通过。

## Layout

`serve.py` publishes the directories. `app.py` is the Streamlit laboratory and does not contain numerical code. `src/catalog.py` lists the papers and topics. `src/lecture1/` through `src/lecture4/` and `src/practice/` hold the calculations. `src/ui/` is the bilingual interface. `docs/architecture.md` says how to add another paper or topic. `docs/implementation_plan.md` records what is in this version.

`serve.py` 发布目录地址。`app.py` 是 Streamlit 实验，里面没有数值代码。`src/catalog.py` 列出课程和主题。`src/lecture1/` 到 `src/lecture4/` 以及 `src/practice/` 是计算。`src/ui/` 是中英界面。`docs/architecture.md` 说明怎样再加一门课或一个主题。`docs/implementation_plan.md` 记录这一版有什么。
