# Architecture

`serve.py` is the public server. `app.py` is the Streamlit laboratory and
does not contain numerical code.

## Navigation

`src/catalog.py` is the list of papers and modules. `src/ui/routes.py`
turns that list into a directory address:

`/home/B14/ImageProcessing/Lecture/#Bayersampling`

`serve.py` publishes those directories. A refresh keeps the directory and
the hash. The hash is the experiment name with spaces and punctuation
removed. The Image Processing lecture page frames the Streamlit laboratory
at `/_lab`, because Streamlit cannot own a nested pathname.

- B14 is available. Its Image Processing module is available and opens the
  laboratory. Signal Processing is registered as planned.
- B15 and B16 are reserved papers with no invented course titles.

Each module has two tracks. `Lecture` opens the laboratory when that
module is available, and a reserved page otherwise. `Sheet` is a
placeholder on every module.

Adding a future module means:

1. Add a `Module` to the paper in `src/catalog.py`.
2. When its status becomes `available`, point `render_lecture` at a new
   `render_*` function. Do not copy the Image Processing pages.
3. Give that module its own section list if it has lectures.

Image Processing sections are Home, Lectures 1–4, and Practice. The sidebar
links jump to those sections on the Lecture page.

## Where the numerics live

Lecture 3's original operators stay importable at their old paths, because
the existing tests use them:

- `src/operators.py`, `src/degradation.py`, `src/map_estimation.py`
- `src/wiener.py`, `src/comparison.py`, `src/bayes.py`
- `src/ui_forward.py` and the other original pages

New code is organised by lecture:

- `src/lecture1/` sampling, point operations, histograms, correlation,
  Gaussian filters, Sobel, matched filtering
- `src/lecture2/` sinusoids, DFT reconstructions, frequency masks,
  linear-versus-circular convolution, aliasing
- `src/lecture3/` directional motion, scalar Wiener gain, integer-phase
  super-resolution
- `src/lecture4/` storage and RLE, Huffman, DCT, the JPEG-like pipeline,
  median, bilateral, non-local means, inpainting
- `src/practice/problems.py` original exercises and the quiz

`src/ui/` is the bilingual shell. `src/ui/experiments.py` holds the
experiment names used both by the select boxes and by the practice links.

## Shared image

The sidebar chooses a built-in image, an upload, or the workspace.
`Use as workspace image` stores a `float64` array. Another lecture sees it
only after the source is set to Workspace. Resizing a stored array uses a
linear zoom and does not change the stored copy.

Restoration solvers still receive only the observation, the operator, and
the regularisation parameter.

## Tests

`pytest` collects the original MAP tests and `tests/test_lecture1.py`
through `tests/test_lecture4.py`, plus the catalog and practice grader.
A module is not described as tested unless that file has been run.
