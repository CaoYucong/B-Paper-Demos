# Mathematical conventions

This note records the conventions actually used in the code. Lecture 3's
MAP objective is unchanged from the original laboratory; the longer
derivation is in `lecture3/map_notes.md`.

## Images

Working images are `float64` arrays. A stored photograph is divided by 255
and lies in `[0, 1]`. Filter outputs, noise, and reconstructions are not
clipped. An 8-bit PNG download is the exception: it is clipped and rounded,
and the button says so.

The original image used to build an observation is not an input to inverse
filtering, Wiener filtering, gradient descent, conjugate gradients, or the
Fourier MAP solver.

## Correlation and convolution

For an odd kernel, correlation is

```
(f ★ h)[y, x] = Σ_{i,j} h[i, j] f[y + i - c, x + j - c]
```

with `c` the centre index. `h[0, 0]` multiplies the top-left sample of the
neighbourhood. Convolution flips `h` by 180° and then uses this sum.

Sobel masks are correlation masks. A symmetric kernel makes the two
operations agree.

`same` mode pads once. Separable filtering also pads once; padding the
intermediate result again would not match the two-dimensional kernel at
the border.

Valid output size for an `n×n` image and an `m×m` kernel is `(n-m+1)×(n-m+1)`.

Boundary modes are zero, replicate (`edge`), reflect, and periodic
(`wrap`). Lecture 3 does not use these spatial paddings. It uses circular
convolution through the FFT.

## Gaussian kernel

```
h[t] ∝ exp(-t² / (2σ²)),    Σ_t h[t] = 1
H[i, j] = h[i] h[j]
```

A direct `k×k` correlation uses `k²` multiplications per pixel. The
separable form uses `2k`. Those counts ignore the border. A measured
runtime is not the same statement.

## Sobel

```
G_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
G_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
|G| = hypot(G_x, G_y)
θ = atan2(G_y, G_x)
```

`θ` is the gradient direction, perpendicular to the edge. `G_x` is the
outer product of `[1, 2, 1]` and `[-1, 0, 1]`.

## Fourier transform

The DFT is `numpy.fft.fft2`: unnormalised, with frequency `k/N` cycles per
pixel and DC at index 0. Pictures of spectra use `fftshift`. A displayed
`log10(1+|F|)` is a display compression, not the DFT.

A real image satisfies `F[k] = conj(F[-k])`. Reconstructions that keep
frequency components keep conjugate pairs so the inverse stays real.
A coefficient with numerical magnitude below `1e-12` is given phase 0
before a phase-only or swapped reconstruction.

Translation by `(Δy, Δx)` multiplies `F` by `exp(-2πi (v Δy + u Δx))`.

## Frequency masks and the convolution theorem

Masks are defined on the unshifted DFT grid. Cutoffs are cycles per pixel.
An ideal low-pass is the disk `|frequency| ≤ cutoff`. A Gaussian low-pass is
`exp(-½ (|frequency|/cutoff)²)`.

`ifft(fft(f) * fft(h))` without padding is circular convolution. Linear
convolution pads to shape `(n+m-1, n+m-1)` before the product. The two
agree numerically in `linear_convolve_direct` and `scipy.signal.fftconvolve`.
They are not the same operation as reflect-padded spatial correlation.

## Degradation, inverse filtering, Wiener, MAP

The periodic Gaussian PSF is centred, even, and normalised. Its OTF is

```
H = fft(ifftshift(psf))
A f = ifft(H fft(f)).real
Aᵀ r = ifft(conj(H) fft(r)).real
```

```
g = A f + n
```

Noise is not clipped. Inverse filtering divides by `H` where `|H| ≥ τ` and
sets the other bins to zero. That threshold is a hard safeguard.

```
W = conj(H) / (|H|² + K)
```

For a real nonnegative `H`, the gain is `H / (H² + K)`, which is smaller
than `1/H` when `K > 0`. Constant-`K` Wiener filtering is the quadratic MAP
estimator for a white penalty on `f` itself. The image MAP penalty is
`λ ||D f||²`, and its Fourier denominator is `|H|² + λ(|D_x|² + |D_y|²)`.
Those two filters are not the same.

The normalised MAP objective and its gradient are unchanged:

```
J(f) = (1/N) ||A f - g||² + λ (1/N) ||D f||²
∇J(f) = (2/N) [Aᵀ(A f - g) + λ Dᵀ D f]
```

Directional motion blur is a unit-sum line of pixels. Angle 0 matches the
existing horizontal kernel. It is not even, so `H` is complex and `Aᵀ ≠ A`.

The super-resolution page uses

```
g_{a,b}[p, q] = f[2p+a, 2q+b] + n
```

and puts the four phases back on the even grid. It is not a camera-motion
solver.

The panorama page shows the four frames and the picture after registration
onto one plane. It does not place those frames with one horizontal
displacement.

The multi-frame sections use a separate periodic camera on a small
synthetic image. A shift is in low-resolution pixels and must land on a
high-resolution pixel. Point sampling reads that pixel. The realistic
camera applies a periodic Gaussian, then the mean of a factor-by-factor
window, then decimation. Boundaries match the MAP section.

```
M = D B T
J(f) = Σ_i (1/N) ||M_i f - g_i||² + λ (1/N) ||D f||²
∇J(f) = (2/N) [ Σ_i M_iᵀ (M_i f - g_i) + λ Dᵀ D f ]
```

`N` is the number of high-resolution pixels. `D` in the penalty is the
forward difference. `D` inside `M` is downsampling. The original image is
not an input to gradient descent or to the normal-equation solve.

Huber loss is `r²/2` for `|r| ≤ δ` and `δ(|r| - δ/2)` otherwise. Its
derivative is clipped to `[-δ, δ]`. The 1D blind kernel is projected to
nonnegative taps that sum to one. That projection is a laboratory
constraint, not a claim of unique recovery.

## Histogram equalisation

With `N` pixels and bin counts `n_k`,

```
s(k) = (1/N) Σ_{j=0}^{k} n_j
```

A region that occupies a single bin is left unchanged. Global equalisation
uses one table. The region-aware coin experiment uses two tables. Neither
moves pixels.

## DCT and the JPEG-like pipeline

The 8×8 matrix is orthonormal DCT-II:

```
C[k, n] = α(k) cos(π k (n + ½) / N)
α(0) = sqrt(1/N),   α(k>0) = sqrt(2/N)
```

```
coefficients = C (block - 0.5) Cᵀ
reconstruction = Cᵀ coefficients C + 0.5
```

Quantisation uses the usual JPEG luminance table scaled by a positive
strength, with entries at least 1. Indices are `round(coefficient / q)`.
The reconstruction multiplies the indices by `q` and inverts the DCT.
This is not a JPEG file. A zigzag run-length display is a separate
accounting of one block. Sparsity is the fraction of zero indices, not a
file size.

## Storage, RLE, Huffman

Uncompressed bits are `width × height × bit depth × channels × frames`.
A byte is 8 bits. MB in the calculator is `10⁶` bytes. KiB is 1024 bytes.
The compressed size divides by the chosen ratio. It is not a measured file.

RLE stores `(value, count)` pairs. The reported ratio counts each pair as
two symbols.

Huffman codes are read from the tree that was built, and decoding uses that
same tree. The average length is the frequency-weighted code length.

## Nonlinear filters

The median is the middle sample of the sorted window, with reflection at
the border. It is not a linear kernel.

The bilateral weight is a spatial Gaussian times a range Gaussian, then
normalised at each pixel. Reflection shifts are used. Because the range
weight depends on the image, the operation is nonlinear.

```
G_σs(i, j) = exp( −(i² + j²) / (2 σ_s²) )
G_σr(Δ) = exp( −Δ² / (2 σ_r²) )
g[y, x] = (1 / W) Σ G_σs(i, j) G_σr(f[y, x] − f[y+i, x+j]) f[y+i, x+j]
```

The sum runs over the square radius. `W` is the sum of the product on that
square. The page also draws the unnormalised factors, their product, the
input window, and the output window as surfaces around a chosen pixel.
Samples outside the image use the same reflection as the filter. That
drawing is a supplementary illustration of this formula.

Non-local means uses

```
w = exp( - (local mean of (f - shifted f)²) / h² )
```

over a search window. The weight map on the page is the window around the
centre pixel.

Inpainting replaces missing pixels by the average of four edge-padded
neighbours and writes the known pixels back after every iteration. The
result is one completion of the boundary values, not a unique recovery of
the hidden pixels.

## Seeds

Noise uses `numpy.random.default_rng(seed)`. Changing the seed changes the
noise and nothing else. The same seed is shared from the sidebar.
