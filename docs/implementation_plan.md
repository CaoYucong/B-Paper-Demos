# Implementation plan

Status after the menu, four lectures, practice section, tests, and a browser check of the navigation.

## Completed

- Phase 0. The existing MAP code and its tests stay on their original import paths. The app opens on a paper menu, then a module menu, then the Image Processing lectures. Signal Processing, B15, and B16 are placeholders.
- Phase 1. Lecture 1: sampling and quantisation, point operations, histogram equalisation with a synthetic coin, the correlation/convolution playground, separable Gaussians, padding, Sobel, matched filtering (labelled additional), temporal averaging, and Bayer sampling.
- Phase 2. Lecture 2: sinusoids, conjugate-pair reconstruction, magnitude and phase, frequency masks, linear versus circular convolution, DFT properties, aliasing and Moiré, hybrid images.
- Phase 3. Lecture 3 calls the original forward, operator, MAP, comparison, and scalar-Bayes pages. New pages cover inverse filtering, Wiener filtering with a single-frequency gain, directional motion blur, and supplementary integer-phase super-resolution. Blind deblurring is not implemented; the extracted slide title marks it not examinable.
- Phase 4. Lecture 4: storage calculator, RLE, Huffman, 8×8 DCT, JPEG-like reconstruction (not a JPEG file), median, bilateral, non-local means, diffusion inpainting. Texture-synthesis and object-removal slides are not implemented.
- Phase 5. Shared workspace, parameter export, eight original numerical problems, and a ten-question quiz. The example-sheet PDF had no extractable text, so the problems are not the official sheet.
- Phase 6. `pytest` was run and passed: 59 tests, including the original MAP tests. A Streamlit AppTest walked Home and Lectures 1–4 plus Practice without an exception. The browser was opened at `http://localhost:8501` and the path B14 → Image Processing → Lecture 1 (sampling and quantisation) rendered images, both languages, and the reported sampling MSE.

## Known limits

- The running Streamlit config does not reload on save. Restart `streamlit run app.py` after code changes.
- Non-local means uses a 48×48 crop and a Run button.
- The original Lecture 3 pages keep their Chinese explanations, with an English caption added at the top of each.
- The JPEG luminance table is the usual example table, not a table transcribed from a slide.
- Inpainting is diffusion on a rectangle. It does not copy texture from elsewhere in the image.

## Next steps, when another module is added

1. Register it in `src/catalog.py`.
2. Add a render function and call it from `src/ui/shell.py` when that module is available.
3. Do not duplicate this Image Processing package.
