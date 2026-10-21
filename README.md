# B Paper Demonstrations

B14 Image Processing Lab is the first module. The window opens on a paper menu: choose **B14**, then **Image Processing**, then a lecture. Signal Processing, B15, and B16 are registered as placeholders so later modules can be added without a second application.

中文：打开后先选课程，再选模块，再选讲次。目前可用的是 B14 的图像处理。信号处理、B15 和 B16 是占位。

The Lecture 3 MAP laboratory is still the code under `src/operators.py`, `src/map_estimation.py`, `src/wiener.py`, and the original pages. Its previous write-up is `docs/lecture3_map_notes.md`.

## Install and launch

From this directory, using the existing virtual environment:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pytest
python serve.py
```

Python 3.11 or newer. The numerical stack is NumPy, SciPy, Pillow, and pytest. The interface is Streamlit with Plotly. No GPU, PyTorch, or network service is required.

`serve.py` listens on port 8501. Open `/home/`. A lecture section address is `/home/B14/ImageProcessing/Lecture/#Bayersampling`.

## What each lecture contains

**Home.** How the four lectures connect, and how the shared image workspace is used.

**Lecture 1 — Image Enhancement.** Sampling against quantisation, point operations and histograms, global versus region-aware histogram equalisation, a 3×3 correlation/convolution playground, separable Gaussian filtering, padding, Sobel gradients, matched filtering (additional), temporal averaging, and a Bayer mosaic.

**Lecture 2 — Fourier Analysis.** Sinusoids, keeping conjugate DFT pairs, magnitude/phase swaps, ideal and Gaussian frequency masks, linear versus circular convolution, DFT properties, aliasing and Moiré, hybrid images.

**Lecture 3 — Image Restoration.** The original forward model, operator `A`, MAP gradient descent, method comparison, and scalar Bayes page, plus inverse-filter and Wiener visualisations, directional motion blur, and a supplementary four-phase super-resolution picture.

**Lecture 4 — Compression and Nonlinear Filters.** A storage calculator, run-length encoding, Huffman coding, an 8×8 DCT, a JPEG-like pipeline that does not write a JPEG file, median, bilateral, non-local means, and diffusion inpainting.

**Practice.** Eight original numerical problems and a ten-question quiz. These are not the official example sheet. See `docs/source_alignment.md`.

## Shared workspace

The sidebar selects a built-in image, an upload, or the workspace. **Use as workspace image / 用作工作区图像** stores the current result. Change the source to Workspace before opening another lecture. **Export parameters** writes the slider values as JSON. It does not write an image.

## Conventions

Read `docs/mathematical_conventions.md` before changing a formula. In particular:

- Correlation does not flip the kernel. Convolution does.
- Lecture 3 convolution is circular, through the FFT.
- Intermediate images are not clipped to `[0, 1]`. PNG download is.
- A compression ratio in the calculator is not a measured file size.
- Solvers do not receive the original image.

## Suggested path

1. Lecture 1, sampling and the convolution playground.
2. Lecture 1, Gaussian separability, then Lecture 2, the convolution theorem.
3. Lecture 2, magnitude and phase, then a low-pass filter.
4. Lecture 3, degradation, inverse filtering, Wiener, then MAP.
5. Lecture 4, the JPEG-like page, then median against salt-and-pepper noise.
6. Practice, and use the link on each problem to return to the experiment.

## Troubleshooting

- If a port is already taken, stop the old `streamlit run` before starting another. This project's config does not reload on save.
- If the workspace source is empty, the lab falls back to the geometric image and says so.
- Non-local means runs on a 48×48 crop and only after its button. Gradient descent and the comparison page also wait for a button.
- Plots that show `log10(1+|F|)` are display compressions. The calculation uses the complex DFT.
- CJK text in a Matplotlib figure needs a font such as PingFang SC. Plotly labels do not.

## Tests

```bash
pytest
```

The suite covers the original MAP operators and the new lecture routines. A module is only claimed to pass after `pytest` has been run on that code.

## Layout

`docs/architecture.md` says how to add another paper or module. `docs/implementation_plan.md` records what is in this version.
