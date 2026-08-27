# FFTLog evaluation of the bin-averaged ΔΣ covariance

This note derives, carefully and completely, the replacement of the
brute-force trapezoidal $\ln\ell$ integration of the Gaussian
$\Delta\Sigma$ covariance (Wu et al. 2019, `cluster-lensing-cov`
`cov_DeltaSigma.py`) by an exact FFTLog evaluation, as implemented in
`clenspy.utils.fftlog_cov`.

References: Talman (1978); Hamilton (2000) for FFTLog; Fang, Eifler &
Krause (2020, MNRAS 497, 2699) for the covariance-by-FFTLog strategy of
summing Mellin kernels before the inverse transform; Gradshteyn & Ryzhik
6.574.1 for the double-Bessel Mellin transform.

## 1. The covariance integral as implemented

The Gaussian covariance of the stacked excess surface density between two
radial (angular) bins $i$ and $j$ of one $(z, \lambda)$ cluster bin is

$$
\mathrm{Cov}_{ij}
 = \frac{1}{4\pi f_{\rm sky}} \int_0^\infty \frac{\ell\,d\ell}{2\pi}\,
   \bar J_2(\ell;\theta_{i,-},\theta_{i,+})\,
   \bar J_2(\ell;\theta_{j,-},\theta_{j,+})\;
   C_{\rm tot}(\ell),
$$

with

$$
C_{\rm tot}(\ell)
 = \Big( C_\ell^{hh} + \tfrac{1}{\bar n_h} \Big)
   \Big( C_\ell^{\Sigma\Sigma} + N_{\rm shape} \Big)
 + \big( C_\ell^{h\Sigma} \big)^2,
 \qquad
 N_{\rm shape} = \frac{\sigma_\gamma^2 \langle\Sigma_{\rm crit}\rangle^2}
                      {n_{\rm src}\, f_{\rm src}(z_h)} ,
$$

where $\bar n_h$ is the halo surface density per steradian,
$n_{\rm src} f_{\rm src}$ the effective (behind-the-lens) source density
per steradian, $\theta = r_p/\chi(z_{\rm mid})$, and the final matrix is
converted to $(M_\odot/{\rm pc}^2)^2$ by the $10^{-24}$ Mpc→pc factor.
The legacy code evaluates this with $d\ln\ell = 10^{-3}$ over ~8 decades
per matrix element: $\sim 10^5$ Bessel evaluations × $n_{\rm rp}(n_{\rm
rp}+1)/2$ pairs, per cluster bin, per term.

## 2. The annulus-averaged kernel $\bar J_2$

The estimator averages $\gamma_t$ over the annulus $[a, b]$, so the
kernel is the area-weighted average of $J_2$:

$$
\bar J_2(\ell; a, b)
 \equiv \frac{2}{b^2 - a^2} \int_a^b \theta\, J_2(\ell\theta)\, d\theta .
$$

Using the recurrence $x J_2(x) = 2 J_1(x) - x J_0(x)$ and
$\int x J_0\,dx = x J_1$, $\int J_1\,dx = -J_0$:

$$
\int x J_2(x)\, dx = -2 J_0(x) - x J_1(x) + \text{const},
$$

hence the exact closed form (the `j2_bin` of
`bessel_for_cov_theta.py`, reproduced as `j2_bin_averaged`):

$$
\boxed{\;
\bar J_2(\ell; a, b)
 = \frac{2}{\ell^2 (b^2 - a^2)}
   \Big[ 2J_0(\ell a) - 2J_0(\ell b)
         + \ell a\, J_1(\ell a) - \ell b\, J_1(\ell b) \Big] .
\;}
$$

**Small-argument behaviour.** Define $\psi(x) \equiv 2J_0(x) + xJ_1(x)$,
so that $\bar J_2 = 2[\psi(\ell a) - \psi(\ell b)] / [\ell^2(b^2-a^2)]$.
From the Taylor series of $J_0$, $J_1$,

$$
\psi(x) = 2 - \frac{x^4}{32} + \frac{x^6}{576} - \dots
$$

— the $x^0$ and $x^2$ terms cancel *inside* $\psi$, and the leading
constants cancel *between* $\psi(\ell a)$ and $\psi(\ell b)$, leaving

$$
\bar J_2(\ell; a, b)
 = \frac{\ell^2 (a^2 + b^2)}{16}
 - \frac{\ell^4 (a^4 + a^2 b^2 + b^4)}{288} + O(\ell^6),
$$

i.e. $\bar J_2 = O(\ell^2)$, as required for convergence of the
covariance integral at low $\ell$.  Numerically the closed form loses
all significance for $\ell b \lesssim 10^{-2}$ (the bracket is
$O((\ell b)^4/32) \lesssim 10^{-10}$ against terms of order 2, so
float64 leaves noise below $\ell b \sim 10^{-3}$); the implementation
switches to the series there.  (The legacy `j2_bin` has this latent
defect; it is harmless in the legacy integral only because those $\ell$
contribute negligibly.)

**Large-argument behaviour.** $x J_1(x) \sim \sqrt{2x/\pi}\cos(x -
3\pi/4)$ dominates $\psi$, so $\bar J_2 \sim \ell^{-3/2}$ (oscillatory)
and the product kernel decays as $\ell^{-3}$ times oscillations.

## 3. Analytic separation of the white-noise term

Write $C_{\rm tot} = C_{\rm smooth}(\ell) + N_{\rm white}$ with the pure
constant

$$
N_{\rm white} = \frac{N_{\rm shape}}{\bar n_h}
$$

(shape noise × halo shot noise) and

$$
C_{\rm smooth} = C^{hh} C^{\Sigma\Sigma}
 + N_{\rm shape}\, C^{hh}
 + \frac{1}{\bar n_h}\, C^{\Sigma\Sigma}
 + \big(C^{h\Sigma}\big)^2 ,
$$

every term of which decays at high $\ell$ (Limber-smooth).  The constant
term has an exact closed form.  Insert the definition of $\bar J_2$ as an
annulus average and use the Bessel closure relation
$\int_0^\infty \ell\, J_2(\ell\theta) J_2(\ell\theta')\, d\ell
= \delta(\theta - \theta')/\theta$:

$$
\int_0^\infty \ell\, \bar J_2^{(i)} \bar J_2^{(j)}\, d\ell
 = \frac{4}{(b_i^2 - a_i^2)(b_j^2 - a_j^2)}
   \int_{[a_i,b_i] \cap [a_j,b_j]} \theta\, d\theta
 = \delta_{ij}\, \frac{2}{b_i^2 - a_i^2}
$$

for disjoint annuli.  Hence the white-noise covariance is **exactly
diagonal**:

$$
\boxed{\;
\mathrm{Cov}^{\rm white}_{ij}
 = \delta_{ij}\,
   \frac{N_{\rm white}}{4\pi^2 f_{\rm sky} (b_i^2 - a_i^2)}
 = \delta_{ij}\,
   \frac{\sigma_\gamma^2 \langle\Sigma_{\rm crit}\rangle^2}
        {n_{\rm src}^{\rm eff}\, A_i\, N_{\rm cl}} ,
\;}
$$

the familiar shape-noise-per-annulus-per-cluster formula ($A_i =
\pi(b_i^2 - a_i^2)$ the annulus solid angle, $N_{\rm cl} = \bar n_h
\cdot 4\pi f_{\rm sky}$) — a built-in sanity check.  **Only
$C_{\rm smooth}$ goes through FFTLog**: this removes the one
non-decaying component (which FFTLog handles worst) and the
$\delta$-function part of the kernel product.

## 4. Exact FFTLog mapping

### 4.1 Fixed-ratio reduction

For strictly geometric radial bins, $b_i = \rho\, a_i$ with a single edge
ratio $\rho$, and $a_j = \alpha_d\, a_i$ with $\alpha_d = \rho^d$ for the
pair $(i, j = i+d)$.  Then, with $u = \ell a_i$,

$$
\bar J_2(\ell; a_i, \rho a_i)
 = \frac{2\,[\psi(u) - \psi(\rho u)]}{(\rho^2 - 1)\, u^2},
$$

and the **product kernel depends only on $u$ and the offset $d$**:

$$
K_d(u) \equiv
 \bar J_2(\ell; a_i, \rho a_i)\, \bar J_2(\ell; \alpha_d a_i, \rho\alpha_d a_i)
 = \frac{4\,[\psi(u) - \psi(\rho u)]\,[\psi(\alpha_d u) - \psi(\rho\alpha_d u)]}
        {(\rho^2 - 1)^2\, \alpha_d^2\, u^4}.
$$

Therefore

$$
G_d(\theta) \equiv \int_0^\infty d\ell\, \ell\, C_{\rm smooth}(\ell)\,
K_d(\ell\theta)
 = \int_0^\infty \frac{dx}{x}\, F(x)\, K_d(x\theta),
 \qquad F(\ell) = \ell^2 C_{\rm smooth}(\ell)
$$

is a multiplicative convolution — **one FFTLog transform per diagonal
offset $d$ yields $\mathrm{Cov}(i, i+d)$ for all $i$ simultaneously**
(read $G_d$ at $\theta = a_i$).  With $n_{\rm rp} = 15$ that is 15
transforms per cluster bin per covariance term, and

$$
\mathrm{Cov}_{i,i+d} = \frac{G_d(a_i)}{8\pi^2 f_{\rm sky}}
 + \delta_{d0}\, \mathrm{Cov}^{\rm white}_{ii}.
$$

### 4.2 The Mellin kernel: 16-term exact sum

FFTLog evaluates $G(y) = \int F(x) K(xy)\, dx/x$ through the Mellin
transform of the kernel, $U(s) = \int_0^\infty t^{s-1} K(t)\, dt$.
Expanding $\psi(c_1 u)\, \psi(c_2 u)$ with $\psi(x) = 2J_0(x) + xJ_1(x)$:

$$
\psi(c_1 u)\psi(c_2 u)
 = 4 J_0 J_0 + 2 c_2 u\, J_0 J_1 + 2 c_1 u\, J_1 J_0 + c_1 c_2 u^2\, J_1 J_1
$$

(arguments $c_1 u$ and $c_2 u$ respectively), over the four scale pairs
$(c_1, c_2) \in \{1, \rho\} \times \{\alpha_d, \rho\alpha_d\}$ with signs
$s_1 s_2$ ($+$ for $c_1 = 1$, $-$ for $c_1 = \rho$; likewise for
$c_2$).  Each elementary term is $u^{p-4} J_\mu(c_1 u) J_\nu(c_2 u)$ with
$p \in \{0, 1, 1, 2\}$, and its Mellin transform follows from the master
formula (GR 6.574.1; `mcfit.kernels.Mellin_DoubleBesselJ`, implemented
with `loggamma` and mpmath $_2F_1$, with separate branches for the ratio
$\beta \lessgtr 1$ and the exactly-coincident $\beta = 1$ case):

$$
\int_0^\infty u^{\sigma-1} J_\mu(c_1 u)\, J_\nu(c_2 u)\, du
 = c_1^{-\sigma}\, M_{\mu\nu}^{(\beta)}(\sigma),
 \qquad \beta = c_2/c_1,
$$

$$
M_{\mu\nu}^{(\beta)}(\sigma)
 = \frac{2^{\sigma-1} \beta^{\nu}\,
         \Gamma\!\big(\tfrac{\mu+\nu+\sigma}{2}\big)}
        {\Gamma\!\big(\tfrac{2+\mu-\nu-\sigma}{2}\big)\, \Gamma(1+\nu)}\;
   {}_2F_1\!\Big(
     \tfrac{\nu-\mu+\sigma}{2},\, \tfrac{\nu+\mu+\sigma}{2};\,
     1+\nu;\, \beta^2 \Big), \qquad 0 < \beta < 1
$$

(and the analytically continued $\beta > 1$, $\beta = 1$ branches).
The summed kernel is then

$$
\boxed{\;
U_d(s) = \frac{4}{(\rho^2-1)^2 \alpha_d^2}
 \sum_{\substack{c_1 \in \{1,\rho\} \\ c_2 \in \{\alpha_d, \rho\alpha_d\}}}
 \!\! s_1 s_2 \Big[
   4\, c_1^{-(s-4)} M_{00}^{(\beta)}(s-4)
 + 2 c_2\, c_1^{-(s-3)} M_{01}^{(\beta)}(s-3)
 + 2 c_1\, c_1^{-(s-3)} M_{10}^{(\beta)}(s-3)
 + c_1 c_2\, c_1^{-(s-2)} M_{11}^{(\beta)}(s-2)
 \Big].
\;}
$$

### 4.3 Why the summation must precede the inverse FFT

Individually, the $u^{-4} J_0 J_0$-type terms behave as $u^{-4}$ at small
$u$: their Mellin integrals diverge for $\mathrm{Re}\,s \le 4$ and their
inverse transforms are dominated by enormous low-$\ell$ tails.  These
tails cancel **only in the sum** — the small-$u$ expansion of $K_d$
begins at $u^{+4}$:

$$
K_d(u) = \frac{(\rho^2+1)^2 \alpha_d^2}{256}\, u^4 + O(u^6),
$$

eight powers of $u$ above the individual terms.  Performing the sum at
the level of the analytic Mellin coefficients realizes this cancellation
*exactly* (analytic continuation is unique: the sum of continuations is
the continuation of the sum), whereas transforming the 16 terms
separately and summing afterwards subtracts sixteen huge, oscillatory
arrays and loses up to eight digits.  This is the same strategy as the
2D-FFTLog covariance treatment of Fang, Eifler & Krause (2020).

### 4.4 Analyticity strip and tilt

From $K_d(u) = O(u^4)$ at small $u$ and the $u^{-3}$ oscillatory
envelope at large $u$, the summed $U_d(s)$ is analytic in the strip

$$
-4 < \mathrm{Re}\, s < 3 .
$$

The *individual* continued terms, however, have poles on the real axis
at $s \in \{0, 2\}$ (from $\Gamma\big(\tfrac{\mu+\nu+\sigma}{2}\big)$ at
$\sigma = s - p - \dots \in -2\mathbb{N}_0$) and, through the $\beta = 1$
branch's $\Gamma(1-\sigma)$, at $s \ge 3$.  Since FFTLog samples $U_d$
on the vertical line $\mathrm{Re}\,s = q$, the tilt must keep away from
those poles: **$q = 1.0$** (the default) maximizes the distance to both
$s = 0$ and $s = 2$ and keeps every sampled value finite and moderate.
With $F(\ell) = \ell^2 C_{\rm smooth}$, $F \sim \ell^{2+n_s}$ at low
$\ell$ and decaying at high $\ell$, the double-sided power-law
extrapolation (`extrap=True`) is well behaved at this tilt.

## 5. Sampling, ringing, caching

- **Grid**: log-spaced $\ell$, $N \sim 2048$–4096 over
  $\ell \in [10^{-1}, 10^{6.5}]$, padded by `extrap=True` (power-law
  continuation of $F$ on both sides).  The legacy grid reached
  $2\times10^7$; the FFTLog result is insensitive to the exact cut
  because $C_{\rm smooth}$ has already decayed and the white part is
  handled analytically.
- **`lowring=True`**: fixes the output grid phase to the low-ringing
  condition, suppressing the FFT periodicity artefact.
- **Convergence diagnostic**: doubling $N$ changes the covariance
  diagonal by $< 10^{-4}$ (regression-tested); the Mellin coefficients
  of the (smooth) integrand decay fast, as expected for a
  Limber-integrated $C_\ell$.
- **Geometry-only caching**: $U_d(s)$ depends only on
  $(\rho, \alpha_d, N, q)$ — never on cosmology or the cluster bin — so
  the mpmath $_2F_1$ evaluations are performed once per binning geometry
  and reused for every $(z, \lambda)$ bin, every covariance term and
  every cosmology.

## 6. Validation recipe (implemented in `tests/test_fftlog_cov.py`)

1. **Constant $C$** → the closed-form diagonal of §3 to $10^{-3}$
   (trapz reference; off-diagonal correlation residues $< 10^{-3}$).
2. **Pure power law** $C(\ell) = \ell^{-p}$ → $G_d(y) = y^{p-2}
   U_d(2-p)$ exactly; the FFT pipeline reproduces the direct Mellin
   evaluation to $\lesssim 10^{-5}$ (measured $2\times10^{-7}$ on the
   diagonal, $8\times10^{-6}$ at offset $d = 3$).
3. **Full smooth model vs legacy trapz** ($d\ln\ell = 10^{-3}$, closed
   form $\bar J_2$): diagonal relative agreement $\le 10^{-3}$;
   correlation-matrix elements agree to atol $10^{-3}$.
4. **$N$-doubling stability** $\le 10^{-4}$.
5. **Small-$x$ kernel**: closed form ↔ Taylor series branch continuity
   across the $x = 10^{-2}$ switch; $\bar J_2$ matches direct numerical
   annulus averaging of $J_2$ to $10^{-6}$.

## 7. Rejected alternatives (recorded for posterity)

- **Point-kernel double Bessel + numerical annulus averaging**: the
  unaveraged $J_2 J_2$ product is quasi-singular near $\theta_1 =
  \theta_2$ for broadband $C_\ell$; annulus quadrature then needs
  adaptive refinement per bin pair.  Kept only as a diagnostic idea.
- **Per-row Hankel transforms** ($\alpha = 1$ only): covers only the
  diagonal.
- **Full 2D-FFTLog** (Fang et al. 2020): correct but heavier than
  needed when $C(\ell)$ is a single precomputed 1D function (products of
  Limber $C_\ell$'s), not a $\chi$-integral of separable kernels.
- **Legacy trapz over $\ln\ell$**: $\sim 10^5$ integrand evaluations per
  matrix element and a latent small-$x$ cancellation defect in
  `j2_bin`; retired to a test-only reference
  (`GaussianCovFFTLog.covariance_trapz_reference`).
