# Source alignment

The Image Processing section below is an earlier extraction. It is not a
reading of `B14_SP_MT26_L1.pdf`, `L2.pdf`, `L3.pdf`, `L4.pdf`, or
`B14_SP_MT26_Notes.pdf`. Those five files are the Signal Analysis course
and are recorded in the Signal Analysis section at the end of this file.

The lecture PDFs were read from the course folder outside this repository.
Only slide titles that the extractor actually returned are treated as
lecture content. Body text of most slides is compressed in the PDF and was
not recovered, so formulas in the lab follow the conventions in
`mathematical_conventions.md` rather than an unverified transcription.

The example-sheet PDF returned no extractable text. Practice problems are
original and are labelled as such in the interface. They are not the
official example sheet.

## Lecture 1 titles used

Image Enhancement; spatial sampling; intensity quantisation; colour sensing
and the Bayer grid; point operations; power-law transformations; contrast
stretching; histogram equalisation; image histograms; temporal filtering and
time-averaged frames; advanced equalisation; spatial operations; box filter;
sharpening; Sobel; rectangular filter; Gaussian noise; smoothing before a
derivative; filter separability.

The lab's region-aware coin is a constructed counterexample for one global
mapping. The slides title "Advanced Equalisation Challenges" but the
extracted titles do not specify this picture.

Matched filtering is additional. It is not in the extracted titles.

A slide titled "Convolutional Neural Network Kernel" is present. This
laboratory does not implement a neural network.

## Lecture 2 titles used

2D Fourier transforms; Fourier series; Fourier transform as a change of
basis; frequency filters; magnitude versus phase; filtering versus
convolution; the convolution theorem; discrete sampling; the sampling
theorem and aliasing in one and two dimensions; anti-aliasing; hybrid images.

The hybrid-image page is the filtering application named on the slides.
It is not a measurement of human vision.

## Lecture 3 titles used

Image Restoration; degradations and the degradation model; the inverse
filter and noise amplification; the Wiener filter; motion deblurring;
generative model and inverse problem; maximum a posteriori estimation;
comparing filters; panorama super-resolution.

The extracted title "Sketch solution (Not examinable)" belongs to the
super-resolution section. The integer-phase interleaving page is therefore
marked supplementary. It is not the panorama solver.

"Blind Deblurring (not examinable)" is not implemented.

The quadratic MAP objective, adjoint tests, gradient descent, conjugate
gradients, and the Fourier solution are the existing laboratory. Their
derivation is in `lecture3_map_notes.md`.

## Lecture 4 titles used

Image Compression and Non-linear Filters; reasons to compress; lossless and
lossy compression; run-length encoding; coding redundancy; JPEG;
discrete cosine transform; JPEG quantisation; median filter, including salt
and pepper and the comparison with a Gaussian; bilateral filter; non-local
means; inpainting, hole filling, and texture synthesis.

The luminance quantisation table is the usual JPEG example table. Its
entries were not read from a slide. The page does not write a JPEG file.
Huffman coding is included as the standard construction behind the slide
title "Coding Redundancy"; the extracted title does not show a particular
tree.

Texture synthesis and object-removal examples from the later inpainting
slides are not implemented. The lab uses diffusion on a rectangle and says
the completion is not unique.

## Not claimed

No statement in the interface should be read as "the lecture proves this
with this exact formula" unless this file lists the corresponding slide
title. Where the lab adds a derivation or a picture, the page says
"additional" or "supplementary".

## Signal Analysis

Read from `B14_SP_MT26_Notes.pdf` (62 pages) and the four slide decks
`B14_SP_MT26_L1.pdf` through `L4.pdf`. The notes and lectures 1, 3, and 4
are titled Michaelmas Term 2026, Steve Roberts. The lecture 2 cover says
Michaelmas Term 2025. The notes' lecture 2 matches that deck.

The module title on the slides is Signal Analysis. The public directory is
`/home/B14/SignalAnalysis/`. The sheet page stays a placeholder. Practice
questions are original checks of formulas printed below. They are not an
official example sheet.

### Lecture 1 — Introduction and analogue filters

Used from the notes: superposition; frequency preservation; time
invariance; causality (`g(t) = 0` for `t < 0`); BIBO stability;
convolution, including the discrete sum; `e^{st}` as an eigenfunction;
`G(s) = 1/(s+α)` for `g(t) = e^{-αt} u(t)`, `α > 0`; the Fourier transform
as the Laplace transform on `s = jω`; the ideal low-pass
`g(t) = (ωc/π) sin(ωc t)/(ωc t)` and the delayed copy with phase `−ωT`;
`|G(jω)|² = 1 / (1 + (ω/ωc)^{2n})`; the pole angles
`π/2 + (2k+1)π/(2n)`; the coefficient table for orders 1–8; and
`n = log(A²−1) / (2 log(ωa/ωc))`. The worked specification is 40 dB at
1 kHz and 3 dB at 500 Hz, with `A = 100`, giving `n = 7`, and the notes'
check `A = 128` at `ωa/ωc = 2` on the `n = 7` row.

The signal-types slide names sampling and quantisation. The uniform
quantiser on the page is an illustration; the slides do not give its rule.

Chebyshev Type I, Chebyshev Type II, and elliptic filters are named. Their
formulas are not in the extracted text, so they are not computed.

The sampled first difference is the lecture example `y(t) = x(t) − x(t−1)`
with the delay taken as one sample.

### Lecture 2 — Digital filters

Used from the notes: `fd = fa/fs`; unique digital frequencies in
`[−1/2, 1/2]`; the Nyquist frequency `fs/2`; the alias pair
`cos(π t/(2T))` and `cos(5π t/(2T))`; spectrum repetition at intervals of
`fs`; the anti-aliasing filter; zero-order hold as a step interpolation,
described as convolution with a box; the five-point weights
`(−3, 12, 17, 12, −3)/35` and
`|G| = |17 + 24 cos ωT − 6 cos 2ωT|/35`; FIR versus IIR, including the
note that a recursive equation can still have a finite impulse response;
the step series `1/(1−z^{−1})` for `|z^{−1}| < 1`; poles inside the unit
circle; and `y[k] = x[k−1] + α y[k−1]` with stability `0 < α < 1`.
Frequency response is `G(z)` on `z = e^{jωT}`. The distance reading uses
the printed pole at `z = 0.8`.

### Lecture 3 — Design of digital filters

Used from the notes: the three-point Hanning filter
`y[k] = (1/4)(x[k] + 2x[k−1] + x[k−2])` and its double zero at `z = −1`;
linear phase for symmetric or antisymmetric FIR coefficients, and phase
`−2ωT` for the causal parabolic filter apart from sign changes of `π`;
the brick-wall cutoff `ωc = π/(4T)`, truncation at `|k| = 10`, and a shift
by 10; the Gibbs overshoot, which the notes call about 9% in their figure;
`w[k] = α + (1−α) cos(kπ/N)` with `α = 0.54` (Hamming) and `α = 0.5`
(Hanning). Kaiser is named and has no formula in the extract.

The bilinear map is `s = (2/T)(1−z^{−1})/(1+z^{−1})` and
`ωa = (2/T) tan(ωd/2)`. For `T = 2` the notes plot this against `ωa = ωd/T`.
Impulse invariance is named as a method that aliases. It is not carried out.

The worked IIR example is a digital Butterworth low-pass, 3 dB at 2 kHz,
30 dB at 4.25 kHz, `fs = 10 kHz`. Pre-warping gives edges near
`14.53×10³` and `83.3×10³` rad/s. The order estimate is about 1.98, so
`n = 2`. The page uses the printed coefficients `0.20657`, `0.41314`,
`0.36953`, and `0.19582`. The notes record `|G(1)| = 1` and `|G| = 0.707`
at `ωd = 0.4π`.

### Lecture 4 — Spectral estimation

Used from the notes: the biased autocorrelation with factor `1/N`; the
periodogram `(1/N)|X|²`; the DFT and its inverse with `1/N` on the
inverse; `F[N−n] = F*[n]` for a real sequence; the pair contribution
`(2/N)|F[n]| cos(2π n k/N + arg F[n])`; the four-point example
`f(t) = 5 + 2 cos(2πt − 90°) + 3 cos(4πt)` at `fs = 4 Hz`, samples
`8, 4, 8, 0`, and `F = (20, −j4, 12, j4)`.

Zero-padding (`L > N`) redisplays the spectrum. `L < N` produces the
time-domain aliasing sum in the notes. Spectral leakage is the non-integer
cycle case. The same Hamming and Hanning family tapers the record.

The AR model is `x = M a + e` with
`â = (MᵀM)⁻¹ Mᵀ x`, and the Yule–Walker system `r = R a` in the
un-normalised form. The notes name Levinson–Durbin; the page solves the
linear system and does not run that recursion. The coefficients
`[1.46, −1.08, 0.60, −0.186]` are the values reported for their figure.
The page synthesises a series from them. It does not reproduce that figure's
samples. The AR spectrum is
`Pxx(ω) = σe² / |1 − Σ a_k e^{−jkωT}|²`. An AR(`p`) model can show `p/2`
peaks. The notes' comparison figure uses Welch's method, which is not
implemented; the page plots the periodogram instead.

### Not computed

Chebyshev, elliptic, Kaiser, impulse invariance, Welch, and the
Levinson–Durbin recursion. No statement on the Signal Analysis page should
be read as a slide formula unless it is listed above.
