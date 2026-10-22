# Source alignment

The Image Processing section below is an earlier extraction. It is not a
reading of `B14_SP_MT26_L1.pdf`, `L2.pdf`, `L3.pdf`, `L4.pdf`, or
`B14_SP_MT26_Notes.pdf`. Those five files are the Signal Analysis course
and are recorded in `docs/b14/signal_analysis/source_alignment.md`.

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

Slides 35–53 of `B14_IP_MT26_L3.pdf` were read with PDFKit. The sentences
below are from that text. The rest of the deck is still titles only.

The slide "Example 3: Panorama Super resolution" says there are multiple
images of the same scene, each displaced spatially, and that after
registration the samples are not coincident and may be used to defeat the
Nyquist limit. The panorama page shows those four frames, then each one
warped onto the registered plane in order, then the wide picture. A single
horizontal shift is not that registration. The page does not estimate the warp.

The intuitive-model slide says to treat images as point samples, and that
more images can increase resolution and reduce noise. The generative-model
slide draws M1–M4 and names registrations, lighting, and blur. The sketch,
marked not examinable, writes one view as g_i = M_i f + noise, says M_i
combines registration, lighting, and down-sampling, and stacks the views
into a linear system it calls over-determined. A later slide names Huber
loss as a cost that reduces the effect of the squared term for a large
error. The factorisation M = D B T, the 1/N scaling, and the rank warning
are laboratory explanations of those sentences. They are not written out
on the slides. The small periodic demos are not a solver for the courtyard
photographs.

The Mars slide says there are 25 JPEG images from different sweeps of a
rotating camera, and it shows an original frame, an average, and a
super-resolution result. The Run Lola Run slides show a low-resolution
input and a result. Those pictures are not in this repository. The
examples section uses a labelled synthetic stand-in.

"Blind Deblurring (not examinable)" is a one-dimensional alternating
demonstration. The slides show a blurred photograph, an estimated kernel,
and a restored photograph, and the cost has a data term plus an image
prior and a blur prior. The laboratory does not use those photographs and
does not claim that the iteration recovers the kernel.

The quadratic MAP objective, adjoint tests, gradient descent, conjugate
gradients, and the Fourier solution are the existing laboratory. Their
derivation is in `lecture3/map_notes.md`.

## Lecture 4 titles used

Image Compression and Non-linear Filters; reasons to compress; lossless and
lossy compression; run-length encoding; coding redundancy; JPEG;
discrete cosine transform; JPEG quantisation; median filter, including salt
and pepper and the comparison with a Gaussian; bilateral filter; non-local
means; inpainting, hole filling, and texture synthesis.

The luminance quantisation table is the usual JPEG example table. Its
entries were not read from a slide. The page does not write a JPEG file.
The bilateral section draws the spatial Gaussian, the range Gaussian, and
their product as rotatable surfaces. That picture is labelled supplementary.
The extracted titles name the bilateral filter and do not list this figure.
The same section embeds the video Non-Linear Image Filters
(https://www.youtube.com/embed/7FP7ndMEfsc?start=469), opening at 7:49.
The video is not a slide.
The non-local means section draws a supplementary picture: two copies of a
diagonal and one horizontal bar. Yellow marks the search-window pixels whose
patch matches the selected pixel. The slides name non-local means and do not
list this figure. The laboratory weight is the mean of squared differences
on a square patch; that expression is not taken from a slide.
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
