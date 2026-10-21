# Source alignment

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
