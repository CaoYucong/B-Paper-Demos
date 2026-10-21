---
name: b14-image-processing-lab
description: >-
  Extend or fix the B14 laboratories at the repository root:
  Image Processing and Signal Analysis. Use when editing either lab,
  adding a B14 lecture or experiment, changing navigation, language
  (中文/English/Bilingual), the scrolling page, MAP restoration, Wiener
  filtering, analogue or digital filters, the DFT, or practice problems.
---

# B14 Image Processing Lab

The lab is the repository root: `serve.py`, `app.py`, and `src/`. One app, one `.venv`. Do not add a second Streamlit app.

Lecture PDFs are outside the repo. `docs/source_alignment.md` is the authority for what the slides actually say. Do not invent slide-specific claims. Unverified additions stay labelled additional or supplementary.

## Navigation

Menu → paper → module → one scrolling lab page.

- B14 Image Processing and B14 Signal Analysis are the available modules. B15 and B16 stay placeholders titled Reserved / 预留. Do not invent their titles.
- Signal Analysis is the Michaelmas course in `B14_SP_MT26` lectures 1–4 and the notes. Its directory is `/home/B14/SignalAnalysis/Lecture/`. The module id stays `signal-processing`. Do not retitle it Signal Processing. The sheet stays a placeholder.
- Signal Analysis formulas come from `docs/source_alignment.md`. Do not add Chebyshev, elliptic, Kaiser, impulse-invariance, Welch, or Levinson–Durbin procedures; the notes name them and do not supply those procedures. The uniform quantiser is labelled as an illustration. Practice problems stay original.
- Do not restore a section selectbox or an experiment selectbox. Parameter controls (resolution, kernel, boundary, image source) stay as widgets.

The lab page is one scroll, in slide order: Home, Lecture 1, Lecture 2, Lecture 3, Lecture 4, Practice. Order is the tuples in `src/ui/experiments.py`. Each lecture has `lecture_anchor("l1")`. Each experiment has `experiment_heading(lecture_id, item)` so the anchor is `l1-fundamentals` and does not change with language. The sidebar is a list of links to those anchors, not a menu that hides sections. Practice links are the same anchors (`PRACTICE_LINKS` values are strings like `l1-gaussian`).

Padding stays immediately after the convolution playground. Operator A stays with the degradation model, before inverse filtering. Matched filtering and integer-shift super-resolution stay last and labelled additional or supplementary. Blind deblurring, texture synthesis, and the official example sheet are not implemented.

## Language

`src/ui/i18n.py`. Session key `ui_language` is `zh`, `en`, or `both`. Default `both`. The radio labels stay exactly 中文, English, and Bilingual. Do not put `ui_language` in `DEFAULTS`; reset must not change language.

- Stored widget values stay language-independent. Translate display with `t()`, `say()`, `shown_label()`, or a `format_func`.
- `L()` splits only the first ` / ` when the left side has no CJK and the right side has CJK. Put numbers in both halves with `t(f"Score: {n}", f"得分：{n}")`. Do not wrap `t()` inside markdown asterisks in a way that leaves a dangling `**`.
- Selectbox element id is its key, so a closed selectbox keeps the previous language. Only selectboxes get a key rewritten to `{key}@@{language}` in `_wrap_label`. When something else writes the canonical key, call `forget_translation_keys`. Do not suffix the language radio.

## Images and cost

The whole page runs on every full rerun. Image panels use `image_png` and `st.image`. Do not put a Plotly heatmap on every panel; a page of them stalls the browser. Line charts and the small teaching matrix may stay Plotly. `show_fig` assigns a new key each call. Do not pass `key` to `st.pyplot`.

Gradient descent, conjugate gradients, and non-local means run only after their buttons. Lecture 3 builds one shared observation (`sigma`, `noise_std`, `log10_lambda`) and passes it into the Gaussian experiments. Motion blur and super-resolution keep their own noise controls. Duplicate labels across lectures need explicit keys.

`.streamlit/config.toml` sets `fileWatcherType = "none"`. After editing, kill the listener on port 8501 and start `streamlit run app.py` again. Verify the changed page in the browser: no section or experiment dropdown, headings in tuple order, language switch changes visible text, no exception widget.

## MAP code

Keep import paths `src.operators`, `src.map_estimation`, `src.wiener`, `src.degradation`, and the existing Lecture 3 pages. Ground truth never enters a solver.

- Periodic Gaussian PSF via ifftshift then FFT. `A(f) = ifft(H · fft(f))`. `Aᵀ` uses `conj(H)`.
- `J = (1/N)||Af−g||² + λ(1/N)||Df||²`, with forward differences, periodic boundaries, and exact adjoints.
- Gradient descent uses `η = s/L` and stops when the loss or image explodes.
- Constant-K Wiener is quadratic MAP with a white penalty on `f`, not `λ||Df||²`.

## Practice and tests

Practice problems are original. The example-sheet PDF had no extractable text. Do not present them as the official sheet.

Run tests with the lab `.venv`: `python -m pytest -q`. `pytest.ini` sets `pythonpath = .`. Do not import `app.py` from a test; import runs `main()`.
