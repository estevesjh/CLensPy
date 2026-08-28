# Einasto Profile: Mathematics and Current Implementation

Reflects the actual state of `src/clenspy/halo/einasto.py` and
`einasto_lown.py`. Full derivations: `docs/einasto_proj_density_v4.tex`.
See {doc}`development` for module layout.

## Definition

The Einasto profile is defined as:

```{math}
\rho(r) = \rho_0 \exp\!\left[-(r/h)^{1/n}\right], \qquad n = 1/\alpha, \quad
h = \frac{r_s}{(2n)^n} \, ,
```

`alpha`, `rho_0`, `r_s` are `EinastoProfile`'s constructor parameters; `n`
(the Einasto index) and `h` (the natural length scale) are derived. The 3D
quantities are exact for any `n > 0`:

```{math}
M_{\rm 3D}(r) = 4\pi \rho_0\, n\, h^3\, \gamma\!\left(3n,\, (r/h)^{1/n}\right),
\qquad
M_{\rm tot} = 4\pi \rho_0\, n\, h^3\, \Gamma(3n)
```

An example of the analytic functions implemented in this repository is presented
below:

```{figure} _static/img/einasto_profiles_overview.png
:alt: Einasto rho(r), Sigma(R), DeltaSigma(R) at n = 4, analytic vs numerical
:width: 100%
:align: center

`rho(r)`, `Sigma(R)`, and `DeltaSigma(R)` at `n = 4` over
`R/r_s = 10^{-2.2}`–`10^{2.2}` (analytic backend in firebrick, numerical
cross-checks dashed black). Note `DeltaSigma -> 0` at the center — a cored
profile has `Sigmabar -> Sigma` there, unlike NFW.
```

## Regime map

`EinastoProfile.__init__` only rejects `n <= 0`; everything else dispatches
internally:

```{list-table}
:header-rows: 1

* - Regime
  - `sigma` / `deltasigma` / `enclosed_mass_2D`
  - `power_spectrum` / `fourier`
* - `n = 1/2` (Gaussian)
  - exact closed forms
  - exact closed form
* - `n = 1` (exponential)
  - exact closed forms (Bessel K + small-x Taylor)
  - exact closed form
* - any other `n > 0`
  - series backend (`einasto_lown.EinastoLowN`)
  - analytic series dispatch (`n > 1` and `n < 1` branches below)
```

Anchor checks use a tight `1e-12` tolerance (`np.isclose`'s default
`rtol=1e-5` would silently misroute e.g. `n = 1 + 1e-7`, which the backend
handles exactly via resonance pairing).

## The series backend (all non-anchor `n`)

`docs/einasto_proj_density_v4.tex` has the derivations; summary:

**Residue series** (`z = (R/h)^{1/n} <= z_sw`). Each quantity has its own
Mellin–Barnes kernel; for `DeltaSigma`, with `x = R/h` and `1 < c < 3`,

```{math}
\Delta\Sigma(x) = 2\sqrt{\pi}\,\rho_0 h\, \frac{n}{2\pi i}
\int_{c-i\infty}^{c+i\infty}
\frac{\Gamma(ns)\,\Gamma\!\big(\tfrac{s+1}{2}\big)}
     {\Gamma\!\big(\tfrac{s}{2}\big)\,(3-s)}\; x^{1-s}\, {\rm d}s .
```

Closing the contour left picks up the two pole strings `s = -k/n` and
`s = 1-2j`, giving:

```{math}
\Sigma(x) = \sqrt{\pi}\rho_0 h \Big[ \sum_{k\ge1} A_k\, x^{k/n+1}
 + \sum_{j\ge0} S_j\, x^{2j} \Big], \qquad
\Delta\Sigma(x) = \sqrt{\pi}\rho_0 h \Big[ \sum_{k\ge1} D_k\, x^{k/n+1}
 + \sum_{j\ge1} T_j\, x^{2j} \Big]
```

```{math}
A_k = \frac{(-1)^k}{k!}
\frac{\Gamma\!\big({-\tfrac12}-\tfrac{k}{2n}\big)}
     {\Gamma\!\big({-\tfrac{k}{2n}}\big)}, \qquad
D_k = -\frac{n+k}{3n+k}\, A_k \, .
```

```{math}
S_j = \frac{2n(-1)^j}{j!}
\frac{\Gamma(n-2nj)}{\Gamma(\tfrac12-j)}, \qquad
T_j = -\frac{j}{j+1}\, S_j \quad (T_0 = 0),
```

```{math}
\bar\Sigma = \Sigma + \Delta\Sigma, \qquad
M_{\rm 2D} = \pi R^2\, \bar\Sigma .
```

The series equals the exact profile at **all** radii, for every `n > 0` —
proved by contour closure and verified by a triangle test (mpmath Abel
quadrature = numerical Mellin–Barnes contour integral = series, to
`1e-40`). An earlier claim of a finite domain of validity
({doc}`einasto_series_investigation`, "Mode B") was refuted; that document
is retained as history with a correction notice.

**Resonance pairing.** The two pole strings collide at `k = n(2j-1)` (any
`n = p/q` with odd `q` — 6/5, 4/3, 7/5 — and *every* `j` for integer `n`).
Writing `\varepsilon \equiv k/n - (2j-1)`, the colliding coefficients of a
pair `(k, j)` — `c_1 \in \{A_k, D_k\}` and `c_2 \in \{S_j, T_j\}` — both
diverge like `1/\varepsilon` (naive fp64: 2510% error at `n = 6/5`, NaNs at
`n = 4/3`). Their joint contribution is evaluated with the exact identity

```{math}
c_1\, x^{k/n+1} + c_2\, x^{2j}
= x^{2j}\big(p \ln x\; \varphi(\varepsilon \ln x) + s\big), \qquad
p \equiv c_1\varepsilon,\quad s \equiv c_1 + c_2,\quad
\varphi(y) \equiv \tfrac{e^y-1}{y},
```

where `p` and `s` are finite (the `1/\varepsilon` parts cancel inside `s`)
and are precomputed in extended precision at build time. For integer `n`
the `\varepsilon \to 0` limits reproduce the case-2/3 logarithmic series
continuously in `n`.

**fp64 budget and switch.** The alternating series loses `~0.5 z` digits to
cancellation for `DeltaSigma` (`~0.87 z` for `Sigma`). The switch point
`z_sw` is *measured* at build time (max-term/result scan) against the
tolerance budget. For `n > 3/2`, `z = x^{1/n}` is so compressed that every
physical radius sits in the series zone.

**`E_nu` branch** (`z > z_sw`). The all-positive Catalan representation
(below) with a Lentz continued fraction for `E_nu` (stable at `z >> nu`,
where upward recurrence explodes), DLMF 8.20(ii) above `nu = 160`,
DLMF 8.19.1 for `nu < 1`, and closed-form integral tail corrections for the
algebraic `k^{-1/2}/(2nk+b)` tail.

**Accuracy** (vs mpmath quadrature): `<= 4e-9` for `n in [0.35, 1.5]`,
`R/h in [0.01, 40]` (incl. resonant 6/5, 4/3, 7/5); `<= 1e-13` for
`n in {2.5, 10/3, 5, 10}`, `R/h in [0.01, 20]`. Cost: ~5–200 ms build per
`n` (mpmath touched only for resonant pairs), ~1–100 ms per 500 radii.

### Accuracy and cost by shape parameter

Max relative error vs mpmath references (dps 40) over `x = R/h` in
`[0.01, 20]`; eval time = `sigma` + `deltasigma` on 500 radii each; build
is once per profile. Default constructor settings.

| `alpha` | `n` | `Sigma` | `DeltaSigma` | build | eval (2×500) | method |
|---|---|---|---|---|---|---|
| 0.1 | 10.0 | 2.2e-16 | 5.8e-15 | 192 ms | 2.5 ms | series (all physical `x`), 21 ε=0 pairs |
| 0.2 | 5.0 | 4.4e-16 | 1.0e-14 | 34 ms | 1.9 ms | series, 24 ε=0 pairs |
| 0.3 | 3.33 | 2.4e-14 | 9.2e-14 | 8 ms | 1.6 ms | series, 27 pairs |
| 0.9 | 1.11 | 1.4e-12 | 2.9e-11 | 11 ms | 25 ms | series (`z<=10`) + `E_nu` |
| 1.0 | 1.0 | 4.4e-16 | 1.6e-11 | ~0 | 0.4 ms | exact Bessel + Taylor |
| 1.1 | 0.91 | 5.9e-11 | 4.3e-12 | 9 ms | 38 ms | series (`z<=10.5`) + `E_nu` |
| 2.0 | 0.5 | 2.2e-16 | 1.3e-12 | ~0 | ~0 | exact Gaussian |
| 5.0 | 0.2 | 2.2e-09 | 5.6e-16 | 22 ms | 119 ms | series (`z<=15.5`) + `E_nu`, 60 pairs |

For comparison, the legacy Catalan path this replaced gave `DeltaSigma`
errors of 1.7e0 / 1.3e0 / 1.3e0 at `alpha = 0.1 / 0.2 / 0.3` (`Sigma`:
8e-8 / 1.5e-3 / 1.4e-2), at comparable or slower eval time.

```{figure} _static/img/einasto_deltasigma_validation.png
:alt: Einasto DeltaSigma, analytic series backend vs Abel+cumtrapz numerical cross-check, n = 4
:width: 85%

`DeltaSigma(R)` at `n = 4`: analytic series backend (firebrick) vs the
retained numerical cross-check (dashed black, Abel projection + cumulative
trapezoid). Inset: fractional difference at ±0.05% limits — the deviations
are the *numerical* method's documented cumtrapz errors; the backend is
exact to ~1e-14 here. Regenerate with `docs/make_einasto_figures.py`.
```

## Exact anchors: `n = 1/2` and `n = 1`

`n = 1/2` (Gaussian; projection of a Gaussian is Gaussian):

```{math}
\Sigma = \rho_0 h \sqrt{\pi}\, e^{-x^2}, \quad
M_{\rm 2D} = \pi^{3/2}\rho_0 h^3 (1 - e^{-x^2}), \quad
\Delta\Sigma = \frac{\rho_0 h\sqrt{\pi}}{x^2}(1-e^{-x^2}) - \Sigma
```

with `1 - e^{-x^2}` evaluated as `-expm1(-x^2)`.

`n = 1` (exponential):

```{math}
\Sigma = 2\rho_0 R\, K_1(x), \quad
M_{\rm 2D} = 4\pi\rho_0 h^3 [2 - x^2 K_2(x)], \quad
\Delta\Sigma = \rho_0 h \Big[\tfrac{8}{x^2} - 4K_2(x) - 2xK_1(x)\Big]
```

The bracketed forms self-cancel as `x -> 0`; below `x = 0.1` verified
Taylor expansions are used (`_expdisk_deltasigma_factor` /
`_expdisk_m2d_factor`; coefficients in v4.tex "Exact anchors").

## Removed: the plain Catalan series as a direct evaluator

`EinastoProfile` used to also carry the native `E_nu` representation
(`docs/einasto_proj_density.tex`) as a direct evaluator, with
`z = (R/h)^{1/n}`, `c_k = \mathrm{Cat}_k/4^k`, `nu_k = 2kn - n + 1`:

```{math}
\Sigma = 2\rho_0 n R \sum_{k \ge 0} (k{+}1) c_k E_{\nu_k}(z), \qquad
\Delta\Sigma = \frac{M_{\rm 3D}}{\pi R^2}
- 2\rho_0 n R \sum_{k \ge 1} k\, c_k E_{\nu_k}(z)
```

as `self._ck`/`self._nu_k`, built by `_build()` and evaluated by `_E_nu()`,
sized by `order_for_tol()`. As a direct evaluator its truncation error was
*absolute* `O(K^{-1/2})` while `DeltaSigma` is small — measured at
30–200% relative error at every radius for `n = 3.3`–`10`, unfixable by
raising the order — and nothing called it once `EinastoLowN` took over
`sigma`/`deltasigma`/`enclosed_mass_2D` for every non-anchor `n`. All four
(`_build`, `self._ck`, `self._nu_k`, `_E_nu`, `order_for_tol`) have been
**deleted** from `einasto.py` rather than left as dead code.

The same terms survive as a live, independent implementation, though:
they are exactly `EinastoLowN`'s own large-`z` branch (its terms are all
positive there, which is the point — no cancellation), reimplemented
locally inside `einasto_lown.py` rather than reusing `EinastoProfile`'s
(now-removed) copy. `power_spectrum`'s own `n > 3/2` branch is unrelated
to either — a separate analytic cascade in `einasto_series.py` — and only
reads the bare integer `self.order` (via its explicit `small_k`/`large_k`
branches, not the default `"auto"`) for its own `A_m^\pm` series.

## Power spectrum `P(k)`

```{note}
`P(k)` here is internal Einasto machinery (mass-dimensioned, `fourier(k)
= (4pi)^2 P(k) -> M_tot` as `k -> 0`) — unrelated to the cosmological
matter power spectrum of {doc}`power_spectrum`. See {doc}`density_profiles`'s
Einasto section for the full normalization note and the `u(k|M)` convention.
```

`power_spectrum(k, branch="auto")` dispatches on `n` independently of the
projected quantities:

All representations descend from one Mellin–Barnes kernel (`\tilde k = kh`,
`0 < c < 1`; derived from the Mellin pair
`\int_0^\infty y^{w-1}\sin y \, {\rm d}y = \Gamma(w)\sin(\pi w/2)`):

```{math}
P(\tilde k) = \frac{\rho_0\, n\, h^3}{4\pi \tilde k}\, \frac{1}{2\pi i}
\int_{c-i\infty}^{c+i\infty}
\Gamma(w)\, \sin\!\big(\tfrac{\pi w}{2}\big)\, \Gamma(2n - nw)\;
\tilde k^{-w}\, {\rm d}w .
```

Its two pole strings cannot collide (`w = -(2j+1) < 0` vs `w = 2 + m/n > 0`),
so `P(k)` has no resonances. Closing left/right gives the two series

```{math}
P = \frac{\rho_0 n h^3}{4\pi} \sum_{m \ge 0} A_m^+
\Big(\frac{\tilde k}{2}\Big)^{2m}, \qquad
A_m^+ = \frac{(-1)^m\, \Gamma(3n+2nm)}{m!\, (3/2)_m},
```

```{math}
P = \frac{\rho_0 h^3}{4\pi \tilde k^3} \sum_{m \ge 1} A_m^- \tilde k^{-m/n},
\qquad
A_m^- = \frac{(-1)^{m+1}}{m!}\, \Gamma\!\big(2+\tfrac mn\big)
\sin\!\big(\tfrac{\pi m}{2n}\big),
```

with mirror-symmetric validity: for `n > 1` the `A_m^-` series converges for
all `\tilde k` and the `A_m^+` series is asymptotic; for `n < 1` the roles
swap — the `A_m^+` series converges everywhere (but self-cancels in fp64),
and the `A_m^-` series is a valid *asymptotic* expansion whose
optimally-truncated error is `\sim e^{-c \tilde k^{1/(1-n)}}`.

Dispatch by `n`:

- **`n = 1`**: `P = \rho_0 h^3 / [2\pi(1+\tilde k^2)^2]`.
- **`n = 1/2`**: `P = \rho_0 h^3 e^{-\tilde k^2/4} / (16\sqrt{\pi})`.
- **`n > 1`**: cost-ordered cascade, each branch with a computable error
  estimate and later (costlier) branches touching only the points earlier
  ones could not certify: (1) plateau `A_m^+` series, optimally truncated;
  (2) direct convergent `A_m^-` series (log-space; estimate covers both
  cancellation and the unsummed tail); (3) crack filler — trapezoidal
  Mellin–Barnes contour quadrature for `n <= 3` (see below), Filon
  quadrature of the master integral for larger `n`, where the integrand
  oscillates `~kt (2n)^n` times against the weight and both Gauss–Laguerre
  and the MB contour undersample. The legacy Wright-rotation branch and the
  1%-tolerance plateau acceptance are retired from the auto path.
- **`0 < n < 1`, other**: per-`k` best-of-three among analytic forms, each
  with a computable error estimate, falling to GL only if none meets
  `1e-9`:
  1. the **Kummer form** — the anti-cancellation decomposition of the
     convergent series,

     ```{math}
     P = \frac{\rho_0 n h^3}{4\pi}\, e^{-\tilde k^2/4}
     \sum_{m\ge0} b_m \Big(\frac{\tilde k}{2}\Big)^{2m},
     \qquad
     b_m = \sum_{i=0}^{m} \frac{A_i^+}{(m-i)!},
     ```

     with `b_m` precomputed once per `n` in mpmath (adaptive precision;
     the alternating cancellation is absorbed at build). Exactly
     `b_m = \delta_{m0}` at `n = 1/2`; cancellation-free at runtime for
     `n \lesssim 0.93` (machine precision at every `\tilde k`, including
     the oscillating `\tilde\rho < 0` regime at `n < 1/2`). The build
     diverges as `n \to 1` (the entire order `1/(2-2n)` of the series
     grows), where it is disabled;
  2. the **plain convergent `A_m^+` series** (log-space terms, measured
     digits lost);
  3. the **optimally-truncated `A_m^-` asymptotic series** (stop at the
     smallest term; estimate = smallest term / sum).

  The fallback (only `n \gtrsim 0.93`, `\tilde k \sim 1.3`–`2`) is the
  MB contour quadrature below. Validated to `<= 1.2e-11` against mpmath
  master-integral quadrature for `n = 0.45, 0.7, 0.9, 0.97` over
  `\tilde k \in [0.1, 60]` (previous FFTLog path: `~2e-3`).

### Contour and Filon quadratures (the crack fillers)

Following the trapezoidal-contour approach of Aceto & Durastante (2022,
M2AN 56) for Wright functions, the kernel itself is integrated along
`w = -\tfrac12 + i\tau`:

```{math}
P(\tilde k) = \frac{\rho_0 n h^3}{4\pi \tilde k}\, \frac{h_\tau}{\pi}
\sum_{j} \mathrm{Re}\Big[\Gamma(w_j)\sin\!\big(\tfrac{\pi w_j}{2}\big)
\Gamma(2n-nw_j)\, \tilde k^{-w_j}\Big], \qquad w_j = -\tfrac12 + i \tau_j .
```

The integrand decays like `e^{-(\pi/2) n |\tau|}` (the sine *grows* like
`e^{+(\pi/2)|\tau|}` — the gamma pair alone decays faster) and is analytic
in `-1 < \mathrm{Re}\, w < 2`, so the trapezoidal rule converges
geometrically; node count is independent of `\tilde k` and the gammas are
reusable across the `k` grid. Validated to `<= 8e-12` for
`n \in [0.45, 2.5]` over `\tilde k \in [10^{-8}, 12]`, at ~1–7 ms per 1000
points — the cheapest evaluator in its window. For `n \gtrsim 3` its phase
gradient `\sim n \ln n` is undersampled; there the crack is covered by
Filon quadrature of the `t = u^n` master integral,

```{math}
P = \frac{\rho_0 h^3}{4\pi \tilde k} \int_0^{t_{\rm hi}}
t\, e^{-t^{1/n}} \sin(\tilde k t)\, {\rm d}t ,
```

with the smooth envelope interpolated piecewise-linearly and the sine
integrated exactly per interval — the node count follows the envelope, not
the oscillation count. Validated to `<= 8.5e-8` at `n = 4, 10` over the
physical `k r_s` range (Gauss–Laguerre, which it replaces there, was wrong
by up to 4% in the `n = 10` turnover).

```{figure} _static/img/einasto_pk_validation.png
:alt: Einasto P(k), analytic dispatch vs FFTLog numerical cross-check, n = 4
:width: 85%
:align: center

`P(k)` at `n = 4` over `k r_s = 10^{-2.2}`–`10^{2.2}`: analytic dispatch
(firebrick) vs the retained FFTLog cross-check (dashed black). Inset:
fractional difference at ±0.05% limits — the residual wiggles are
FFTLog's. Plotting this densely in the plateau exposed (and led to the fix
of) a legacy mis-routing that sent deep-plateau points to the Wright
large-k series. Regenerate with `docs/make_einasto_figures.py`.
```

## Validation (`tests/test_einasto.py`, 55 tests)

- `TestLowNSeries`: `Sigma`/`DeltaSigma` vs hardcoded mpmath references
  (dps 40–50) at `rtol=1e-8` for `n = 0.7, 6/5, 4/3, 1.45, 2.5, 5, 10`
  over `x = 0.01`–`25` (both dispatch zones; resonant and integer-resonant
  indices); `M_2D` consistency; `n = 1{+}10^{-7}` continuity with the
  Bessel anchor (pairing regression); `DeltaSigma -> 0` smoothly as
  `R -> 0`; the `n=1` small-x Taylor branch.
- Anchors: Gaussian `Sigma`/`DeltaSigma`/`P(k)` at `rtol = 1e-14/1e-12/
  1e-12` (down to `R = 0.001 h`); exponential at `1e-13`.
- `TestPowerSpectrumLowN`: `P(k)` vs hardcoded mpmath master-integral
  references at `rtol=1e-8` for `n = 0.45, 0.7, 0.9, 0.97` over
  `kt = 0.1`–`60` (all three analytic branches + the GL bridge; includes
  the oscillating `n < 1/2` regime).
- `n=0.7` `P(k)` vs brute-force Hankel quadrature at `2e-3`; full
  Fourier→ξ→Σ→ΔΣ pipeline consistency at the Gaussian anchor.
- `TestSpiralHalo`: `sigma` vs brute-force Abel at `n = 4, 5`; the `P(k)`
  `A_m^-` coefficient table.

## Open issues

- **[#1](https://github.com/estevesjh/CLensPy/issues/1)** — umbrella for
  `n <= 3/2` support: projected quantities and `P(k)` both done (this
  page).
- **[#2](https://github.com/estevesjh/CLensPy/issues/2)** — **resolved**:
  `P(k)` for generic `n < 1` was `~2e-3` (FFTLog); the analytic dispatch
  above delivers `<= 1.2e-11` (target was `1e-9`).
- **[#3](https://github.com/estevesjh/CLensPy/issues/3)** — **resolved**:
  `DeltaSigma` target was `1e-3` everywhere; delivered `<= 4e-9`
  (`n <= 3/2`) and `~1e-14` (`n > 3/2`).

## Roads not taken

Two earlier series implementations were carried in the package for a while
and have been removed; the derivations they transliterate remain in
`docs/einasto_proj_density_v2.tex` and `_v3.tex`, and the code itself is in
git history.

- **`einasto_v2.py`** — inner-zone dual form; benchmark/research only.
- **`einasto_v3.py`** — original case-1 transcription with fixed `K=60/J=5`
  and no resonance pairing, hence unusable near `n = 6/5, 4/3, 7/5`.
  Superseded by `einasto_lown.py`, whose exact paired residues are what fix
  that failure.

## References

- `docs/einasto_proj_density_v4.tex` — **the production math**
  (`einasto_lown.py`): per-quantity Mellin–Barnes kernels, global validity
  theorem, resonance pairing, fp64 budget, `E_nu` continued fraction + tails.
- `docs/einasto_proj_density.tex` — the Catalan `E_nu` representation (the
  backend's large-`z` branch).
- `docs/einasto_proj_density_v2.tex`, `_v3.tex`,
  `docs/fractional_derivative_einasto.tex` — historical/research notes.
- `docs/einasto_power_spectrum.tex` — the `P(k)` series and closed forms.
- Retana-Montenegro, E., Van Hese, E., Gentile, G., Baes, M. & Camps, F.
  (2012), A&A 540, A70; arXiv:1202.5242 (case 1, Sec. 3.1).
- NIST DLMF §§8.11, 8.19–8.20 (incomplete gamma / `E_p` asymptotics).
