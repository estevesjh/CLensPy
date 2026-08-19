# Einasto Profile: Mathematics and Current Implementation

This document reflects the actual state of `src/clenspy/halo/einasto.py` as of
this writing. It supersedes the old `einasto_pitfalls.md`, which described a
design (hard rejection of `n <= 3/2`) that is no longer true. See
{doc}`development` for how this fits the module layout, and the GitHub issue
tracker for open follow-up work (linked at the bottom).

## Definition

```{math}
\rho(r) = \rho_0 \exp\!\left[-(r/h)^{1/n}\right], \qquad n = 1/\alpha, \quad
h = \frac{r_s}{(2n)^n}
```

`alpha`, `rho_0`, `r_s` are `EinastoProfile`'s constructor parameters; `n`
(the Einasto index) and `h` (the natural length scale) are derived.

## Regime map

The 3D quantities (`density`, `enclosed_mass`, `total_mass`) use the
incomplete-gamma closed form and are **exact for any `n > 0`** — nothing
below affects them.

The projected quantities (`sigma`, `deltasigma`, `enclosed_mass_2D`) and the
Fourier transform (`power_spectrum`/`fourier`) are more involved, because no
single elementary closed form covers every `n`. `EinastoProfile.__init__`
only rejects `n <= 0`; everything else dispatches internally:

```{list-table}
:header-rows: 1

* - Regime
  - `sigma` / `deltasigma`
  - `enclosed_mass_2D`
  - `power_spectrum` / `fourier`
* - `n = 1/2` (Gaussian)
  - exact closed form
  - Catalan series (n>3/2 only) — see below
  - exact closed form
* - `n = 1` (exponential)
  - exact closed form (Bessel K)
  - Catalan series (n>3/2 only)
  - exact closed form
* - `0 < n <= 3/2`, other
  - numerical (Abel projection / cumtrapz)
  - **raises `NotImplementedError`**
  - numerical (FFTLog)
* - `n > 3/2`
  - Catalan series
  - Catalan series
  - analytic series (+ GL quadrature fallback)
```

`self._series = (n_index > 1.5)` is the flag the class uses internally to
pick between the Catalan series and the numerical fallback for
`sigma`/`deltasigma`; the `n = 1/2` and `n = 1` exact forms are checked
*before* that flag and bypass both paths entirely, at any `n_index` (though
in practice they only matter in the `n <= 3/2` regime, since for `n > 3/2`
the Catalan series is already fast and exact-in-the-limit).

## 3D quantities (exact for any `n > 0`)

```{math}
M_{\rm 3D}(r) = 4\pi \rho_0\, n\, h^3\, \gamma\!\left(3n,\, (r/h)^{1/n}\right),
\qquad
M_{\rm tot} = 4\pi \rho_0\, n\, h^3\, \Gamma(3n)
```

where `gamma` is the lower incomplete gamma function. Implemented directly
with `scipy.special.gammainc`/`gamma` — no series, no restriction on `n`.

## Catalan series (`n > 3/2`)

For `n = 1/alpha > 3/2`, `sigma`, `deltasigma`, and `enclosed_mass_2D` use
the projected-density series (`docs/einasto_proj_density.tex`). With
`x = (R/h)^{1/n}`, `c_k = \mathrm{Cat}_k/4^k` (Catalan numbers over powers of
4), and `nu_k = 2kn - n + 1`:

```{math}
\Sigma(R) = 2\rho_0 n R \sum_{k \ge 0} (k+1)\, c_k\, E_{\nu_k}(x)
```

```{math}
M_{\rm 2D}(R) = M_{\rm 3D}(R) + 2\pi\rho_0 n R^3 \sum_{k \ge 0} c_k\, E_{\nu_k}(x)
```

```{math}
\Delta\Sigma(R) = \frac{M_{\rm 3D}(R)}{\pi R^2}
- 2\rho_0 n R \sum_{k \ge 1} k\, c_k\, E_{\nu_k}(x)
```

where `E_nu` is the generalized exponential integral.

### Why `n > 3/2` specifically

These series converge algebraically (~`K^{-1/2}`), not geometrically. Series
order `K` needed for 1%/0.1% relative error, from `docs/einasto_pitfalls.md`'s
original measurements:

| n (shape) | K for 1% | K for 0.1% |
|-----------|----------|------------|
| 0.5       | impractical | — |
| 1.0       | ~5000 | >50000 |
| 2.0       | ~1500 | ~15000 |
| 4.0       | ~160  | ~15000 |
| 5.0       | ~40   | ~4000 |
| 6.0       | ~10   | ~900 |

`n <= 3/2` is where this becomes impractical — hence the cutoff. `n = 4-6`
(spiral haloes) and `n = 2-3` (clusters) are comfortably in the "workable"
zone.

### `E_nu(x)` dispatch

`expn_fast` picks one of four branches (no external dependency needed for
integer `n`):

1. Integer `nu >= 1`: `scipy.special.expn` (exact, vectorized, ~microseconds)
2. `nu >= nu_asymp(rtol)`: DLMF 8.20 uniform asymptotic expansion (polynomial
   recurrence, vectorized)
3. `nu < 1`: `E_p(z) = z^{p-1}\, \Gamma(1-p, z)` via `scipy.special.gammaincc`
   (only valid for `a = 1-p > 0`)
4. Otherwise (non-integer `nu` in `(1, nu_asymp)`): `mpmath.expint` (~0.25
   ms/point — the only branch requiring `mpmath`, and only hit for
   non-integer `n`, e.g. `n = 4.5`)

For integer `n` (4, 5, 6 - the physically common cases), all `nu_k` are
integers, so only branches 1-2 ever fire and `mpmath` is never called.

### `DeltaSigma`'s small-`z` cancellation

Below `z = (R/h)^{1/n} < _DS_ASYMP_ZMAX = 0.15`, the native `DeltaSigma`
series suffers catastrophic cancellation (`M_3D/(pi R^2)` and the k-sum are
both `~z^n` and nearly cancel; the true result is `~z^{n+1}`). Below that
threshold, `_deltasigma_asymp` is used instead — a small-`z` asymptotic
series (4 terms, ~1% accurate out to `z ~ 0.15`) derived from the "dual form"
in `docs/einasto_proj_density_v2.tex`:

```{math}
\Delta\Sigma(R) = \sum_{p \ge 1} C_p\, z^{n+p}, \qquad
C_p = -A_p \frac{n+p}{3n+p}, \qquad
A_p = \frac{2\rho_0 n h (-1)^p}{p!}\, \Phi_\Sigma(p)
```

```{math}
\Phi_\Sigma(p) = \sum_{k} \frac{(k+1)\, c_k}{2nk - n - p}
```

(`Phi_Sigma` converges only as `K^{-1/2}` too, so it's evaluated with a
dedicated `K=200000`-term sum, independent of the profile's own `order`.)

### `order_for_tol`: automatic order selection

Since the series converges algebraically, a last-term criterion
underestimates the true error by a factor of `~K`. `order_for_tol` instead
estimates the tail via the known power-law decay exponent per quantity
(`Sigma`/`DeltaSigma`: `p=3/2`; `M_2D`: `p=5/2`):

```{math}
R_K \sim \sum_{k>K} u_k \sim u_K \cdot \frac{K}{p - 1}, \qquad
\text{relative error} = \frac{R_K}{|S_K|}
```

Validated against Abel-transform ground truth. Only usable when
`self._series` is True — raises `NotImplementedError` otherwise (there's no
"order" concept for the numerical fallback).

## Exact closed forms: `n = 1/2` and `n = 1`

Two special values of `n` have genuine elementary closed forms and bypass
*both* the Catalan series and the numerical fallback entirely, regardless of
which regime they'd otherwise fall in.

### `n = 1/2`: Gaussian profile

`rho(r) = rho_0 exp(-(r/h)^2)`. Because the projection of a 3D Gaussian is
again Gaussian:

```{math}
\Sigma(R) = \rho_0 h \sqrt{\pi}\, e^{-(R/h)^2}, \qquad
\Delta\Sigma(R) = \frac{\rho_0 h^3 \sqrt{\pi}}{R^2}
\left(1 - e^{-(R/h)^2}\right) - \Sigma(R)
```

```{math}
P(k) = \frac{\rho_0 h^3}{16\sqrt{\pi}}\, e^{-(kh)^2/4}
```

(`power_spectrum`'s own `P = \tilde\rho(k)/(4\pi)^2` convention.) Note: only
`power_spectrum` currently special-cases `n=1/2` in the code; `sigma`/
`deltasigma` at `n=1/2` go through the general numerical fallback below (not
this closed form) - both give equivalent results, since the numerical path
was validated *against* this exact formula (see Validation, below).

### `n = 1`: exponential profile

`rho(r) = rho_0 exp(-r/h)`. Its projection is expressible via modified
Bessel functions of the second kind:

```{math}
\Sigma(R) = 2\rho_0 R\, K_1(R/h)
```

using `K_1(x) = \int_0^\infty e^{-x\cosh t}\cosh(t)\, dt` (the Abel
projection integral, directly matching the Bessel integral representation).
From `M_{\rm 2D}(R) = 4\pi\rho_0 h^3\left[2 - x^2 K_2(x)\right]` and
`d(x^2 K_2(x))/dx = -x^2 K_1(x)`:

```{math}
\Delta\Sigma(R) = \rho_0 h \left[\frac{8}{x^2} - 4 K_2(x) - 2x K_1(x)\right],
\qquad x = R/h
```

```{math}
P(k) = \frac{\rho_0 h^3}{2\pi\left(1 + (kh)^2\right)^2}
```

`sigma`/`deltasigma` check `np.isclose(n_index, 1.0)` *before* checking
`self._series`, so this closed form is used even though `n=1 <= 3/2` would
otherwise route to the numerical fallback.

## Numerical fallback (generic `0 < n <= 3/2`, excluding `n=1/2, 1`)

No elementary closed form and no usable series exist for the rest of this
range. Three independent numerical techniques are used instead, one per
quantity - all built directly from `density(r)`, so they inherit its
exactness rather than any series approximation:

**`Sigma(R)`** — direct Abel (line-of-sight) projection, via
`compute_sigma_quadvec` (adaptive `quad_vec` quadrature):

```{math}
\Sigma(R) = 2\int_R^\infty \rho(r)\, \frac{r\, dr}{\sqrt{r^2 - R^2}}
```

**`DeltaSigma(R)`** — computed from a dense `Sigma(R)` grid (1600
log-spaced points from `1e-4 h` to `40^n h`) via cumulative-trapezoid
enclosed mass (`sigma_to_deltasigma_cumtrapz`), then log-log interpolated
onto the requested `R`. This inherits that function's documented caveat:
the cumulative integral assumes ~0 enclosed mass below the grid's first
point, which is a poor approximation deep in a cored/smooth profile's
center, where `DeltaSigma` itself is small (near-cancellation of
`Sigmabar(<R)` and `Sigma(R)`). See Validation and Known limitations below
for the actual accuracy this achieves as a function of `R/h`.

**`P(k)`** — FFTLog transform (`mcfit.xi2P`) of `density(r)` directly, in
`power_spectrum`'s own `P = \tilde\rho(k)/(4\pi)^2` convention. This replaces
the mathematically-valid-but-numerically-useless small-`k` series (below).

## Power spectrum `P(k)`

`power_spectrum(k, branch="auto")` (the default) dispatches on `n`
independently of the `sigma`/`deltasigma` split above:

- **`n = 1`**: exact closed form (above).
- **`n = 1/2`**: exact closed form (above).
- **`0 < n < 1`, other**: the small-`k` series below is a convergent Cauchy
  series for *all* `k` mathematically, but its finite-precision partial sums
  suffer catastrophic cancellation well before the asymptotic regime (no
  anti-cancellation decomposition like the `n>1` Wright form has been
  derived for it - see Known limitations). Uses the FFTLog numerical
  fallback instead.
- **`n > 1`**: the analytic large-`k` series, which *does* have a working
  anti-cancellation decomposition (below), used via an adaptive per-`k`
  dispatch.

### `n > 1`: large-k analytic series

```{math}
P(k) = \frac{\rho_0 h^3}{4\pi \tilde k^3} \sum_{m \ge 1} A_m^-\, \tilde
k^{-m/n}, \qquad
A_m^- = \frac{(-1)^{m+1}}{m!}\, \Gamma\!\left(2+\frac{m}{n}\right)
\sin\!\left(\frac{\pi m}{2n}\right), \qquad \tilde k \equiv kh
```

This converges for all `k`, but the plateau (small-`\tilde k`) region needs
a different representation numerically. The `"auto"` dispatch, for a given
`n` and `\tilde k`, picks per-point between:

1. **Plateau series** (small `\tilde k`): a related series in
   `(\tilde k^2/4)^m` with coefficients `Gamma(3n+2nm)`, summed only up to
   the smallest truncation `M` where the partial sum's relative contribution
   drops below `tol=1e-2` (an adaptive-order strategy, not a fixed count).
2. **Gauss-Laguerre quadrature** (`_einasto_pk_GL`): a direct numerical
   evaluation of the defining integral,
   `P(k) = \frac{\rho_0 n h^3}{4\pi}\int_0^\infty u^{3n-1} e^{-u}
   \mathrm{sinc}(\tilde k u^n)\, du`, exact at `\tilde k=0` and valid
   for a wide `\tilde k` range with enough nodes (`N_GL`, scaled with `n`).
3. **Wright asymptotic series** (`_einasto_pk_wright_real`, `n>1` only):
   used only where GL's node count would need to be excessive - stable for
   `\tilde k \gtrsim 10^{-4}` and `n>1` specifically (not valid for `n<=1`,
   see below).

For `n <= 1`, the dispatch above skips the Wright series entirely (it's
mathematically invalid there) and falls back to GL quadrature alone for any
point the plateau series doesn't converge on - validated against the exact
`n=1/2` closed form to ~1e-14 relative error out to `P(k)/P(0) ~ 1e-17`
(i.e. across the entire practically relevant range).

## Validation (current test suite: `tests/test_einasto.py`)

`TestSpiralHalo` (native series, `n=4,5`): `Sigma` vs. direct Abel
quadrature, `1e-2` rtol; `order_for_tol` hits its target tolerance within
1.5x; `P(k)` large-k series coefficients match `einasto_power_spectrum.tex`
Table I to `atol=5e-5`.

`TestNumericalFallback` (n≤3/2 path, Fourier→ξ→Σ→ΔΣ end to end):

- `n=1/2` (Gaussian anchor): `P(k)` matches the closed form to `rtol=1e-12`;
  `Sigma` to `rtol=1e-10`; `DeltaSigma` to `rtol=1e-2`.
- `n=1` (exponential/Bessel anchor): `Sigma` and `DeltaSigma` both match
  their closed forms to `rtol=1e-13`.
- `n=0.7` (generic, no exact anchor): `P(k)` matches an independent
  brute-force `scipy.integrate.quad` of the defining Hankel integral to
  `rtol=2e-3`.
- Full pipeline (`n=1/2`): `power_spectrum` → FFTLog `xi(r)` recovers
  `density(r)/(4\pi)^2` to `<1e-3`; Abel-projecting that `xi(r)` into
  `Sigma(R)` matches the class's own `sigma()` to `<1e-3`; cumtrapz-deriving
  `DeltaSigma` from that grid matches the class's own `deltasigma()` to
  `<1e-2` for `R > h` (the innermost points are excluded - see below).

## Known limitations (tracked as GitHub issues for future sessions)

- **[#1](https://github.com/estevesjh/CLensPy/issues/1)** - tracking issue
  for finishing `n <= 3/2` support generally.
- **[#2](https://github.com/estevesjh/CLensPy/issues/2)** - `P(k)` for
  generic (non-anchor) `n <= 3/2` is only cross-validated to `~2e-3`
  (against a limited-precision brute-force reference); target `1e-9`.
- **[#3](https://github.com/estevesjh/CLensPy/issues/3)** - `DeltaSigma(R)`
  accuracy is regime-dependent, not a flat `1e-3` everywhere:
  - The numerical (cumtrapz) path is excellent for `R \gtrsim 0.3-0.8\, h`
    (sub-percent, improving to `<1e-4` by `R \sim h`), but degrades sharply
    approaching `R \to 0` (deep in a cored profile's center, where
    `DeltaSigma` itself is small) - e.g. for the Gaussian anchor, ~1-3% at
    `R \sim 0.03-0.05\, h`, and effectively 0 (100% error) at `R=0.01h`.
  - `EinastoProfileV3` (Retana-Montenegro et al. 2012 case-1 series, a
    *separate* class - see below) is machine-precision accurate at *small*
    `x=R/h` (down to `x=0.01`, no cancellation issue at all), but gives
    wrong answers beyond `x \sim 1-1.2` for small `n` (confirmed: `K=60,
    200, 500, 2000` all give the *same wrong* answer at `x=3` for `n=0.5`).
    A follow-up investigation ({doc}`einasto_series_investigation`) found
    this is actually **two separable problems**: at `n=0.5` specifically it
    is pure under-truncation (the default `J=5` second-track terms are far
    too few - the first track is identically zero there, so the whole
    series collapses onto a bare Taylor series of `e^{-x^2}` needing many
    more terms), while a second, genuine finite-domain-of-validity issue
    (more precision/terms provably does not help) sets in separately around
    `x \sim 7-8` for `n=0.7`. See that document for the full breakdown and
    the approved follow-up plan.
  - These two are complementary (small-`x` vs. large-`x` strengths); a
    hybrid dispatch between them is the likely fix, not yet implemented in
    the main `EinastoProfile` class.

## Related research modules (not part of the primary path)

- **`einasto_v2.py`** (`EinastoProfileV2`) - an "inner-zone dual form"
  re-derivation of the Catalan series for `z=(R/h)^{1/n} < 1`, trading
  per-point special-function calls for a precomputed polynomial evaluation.
  Explicitly marked in its own docstring as a benchmark/research
  implementation - `EinastoProfile` (this document) remains the
  recommended path. See `docs/einasto_proj_density_v2.tex`.
- **`einasto_v3.py`** (`EinastoProfileV3`) - the Retana-Montenegro et al.
  (2012) case-1 elementary gamma-ratio series referenced above, valid (where
  it converges) for any non-integer `n`, 75-90x faster than the native `E_nu`
  series where both apply. Not currently wired into `EinastoProfile` itself
  - see Known limitations. See `docs/einasto_proj_density_v3.tex`.

## References

- `docs/einasto_proj_density.tex` - the native Catalan series (this is the
  main path, `n > 3/2`).
- `docs/einasto_proj_density_v2.tex` - the inner-zone dual form
  (`einasto_v2.py`; also the source of `DeltaSigma`'s small-`z` asymptotic,
  used directly in the main class).
- `docs/einasto_proj_density_v3.tex` - the Retana-Montenegro case-1 series
  (`einasto_v3.py`).
- `docs/einasto_power_spectrum.tex` - the `P(k)` series (both the `n>1`
  large-k form and the `n<1` small-k form) and the exact `n=1/2`, `n=1`
  closed forms.
- `docs/fractional_derivative_einasto.tex` - the fractional-calculus
  interpretation underlying the v3 series's gamma-ratio coefficients.
- Retana-Montenegro, E., Van Hese, E., Gentile, G., Baes, M. & Camps, F.
  (2012), A&A 540, A70; arXiv:1202.5242. Case 1 of Sec. 3.1 (Eqs. 17-18).
