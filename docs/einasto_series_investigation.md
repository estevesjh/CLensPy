# Einasto Series Investigation: Why the Case-1 Series Fails at Large x/k

## Context

{doc}`einasto_math` documents the current implementation: for `0 < n <=
3/2`, `EinastoProfile` falls back to purely numerical methods (Abel
quadrature for Sigma, cumulative-trapezoid for DeltaSigma, FFTLog for P(k))
because the exact analytic series that exists for this regime -
`EinastoProfileV3`'s Retana-Montenegro et al. (2012) "Case 1" residue
expansion - was found to diverge at large `x = R/h` for small `n` (e.g.
2342x relative error at `x=3, n=0.5`), tracked as part of GitHub issue #3.

This was originally assumed to be a **catastrophic cancellation** problem:
an alternating series whose individual terms grow far larger than the final
answer, so double-precision floats lose all their significant digits to
subtraction before reaching the true (small) result. That assumption drove
the question this doc answers: *what would it take to get a full analytic
solution, valid at any x or k, for Sigma/DeltaSigma/M_2D and P(k)?*

Two things happened to investigate this:

1. The actual source paper (Retana-Montenegro, Van Hese, Gentile, Baes &
   Camps 2012, A&A 540, A70; arXiv:1202.5242) was read directly (not just
   the transcribed formulas in `docs/einasto_proj_density_v3.tex`). Section
   3 derives Sigma(x)/M_2D(x) via a Mellin-Barnes/Fox-H-function
   representation, and states the residue expansion ("Case 1", for
   non-integer or even-denominator-rational `n`) is a power series claimed
   to converge for *all* `x` mathematically - not just small `x`.
2. A live diagnostic investigation then tested that claim empirically,
   running `mpmath` arbitrary-precision arithmetic directly against the
   paper's formulas and against `mpmath`-quadrature ground truth. This
   **overturned the catastrophic-cancellation hypothesis** in an important
   way, detailed below.

This doc records what was found, before any of the fix (see the approved
implementation plan for the follow-up session) is executed.

## Finding 1: the large-x/large-k failure is two distinct, separable problems

### Mode A - under-truncation (a cheap bug, not a precision problem)

`EinastoProfileV3`'s series has two "tracks": a first track (`a_k`
coefficients, powers `x^{k/n+1}`) carrying the cusp/core shape, and a
second track (`atilde_k` coefficients, powers `x^{2k}`) carrying the
large-scale geometry. The implementation defaults to `K=60` (first-track)
and `J=5` (second-track) terms, values validated by the paper's authors for
`n ~ 4-4.5` (spiral haloes) - not for `n < 1`.

Empirically, second-track term magnitudes don't peak until roughly
`k_peak ~ x^(1/n) / (2n)`: at `x=10, n=0.7`, individual term magnitudes
(`log10|term_k|`) keep *increasing* up to `k ~ 19-22` (reaching magnitude
~1e11) before finally decaying. The default `J=5` is off by roughly a
factor of 4 in the required order there.

**At exactly `n=0.5`, this is total, not partial**: `Gamma(-k/(2n))` hits an
exact pole for *every* positive integer `k` when `2n=1`, so the entire
first track is identically zero, and the profile collapses onto the second
track alone - which reduces exactly to the plain Taylor series of
`e^{-x^2}` (since `atilde_k(0.5) = (-1)^k/k!` exactly). **This fully
explains** the originally reported "234000% error at x=3 for n=0.5, with
K=60/200/500/2000 all giving the identical wrong answer": increasing `K`
did nothing, because those first-track terms are all exactly zero at
`n=0.5` - `J` (fixed at 5) was simply far too small for a Taylor series
representation of `e^{-9}`.

This is a plain, cheap, precision-independent bug: size `K`/`J` adaptively
per `(n, x)`, not with fixed defaults.

### Mode B - a genuine, previously undocumented finite domain of validity

This is the important, non-obvious discovery, and the reason a "just add
more precision" fix cannot work everywhere.

Decisive test, `Sigma(x)` at `n=0.7, x=8`, using the exact `a_k`/`atilde_k`
coefficients from `docs/einasto_proj_density_v3.tex` evaluated in `mpmath`:

```{list-table}
:header-rows: 1

* - K
  - J
  - dps
  - Result
  - True value (mpmath Abel quadrature)
  - Relative error
* - 200
  - 60
  - 150
  - -4.535017e-7
  - +1.295e-8
  - 3600%, wrong sign
* - 500
  - 150
  - 150
  - -4.535017e-7 (identical)
  - +1.295e-8
  - 3600%, wrong sign
* - 1000
  - 250
  - 300
  - -4.535017e-7 (identical)
  - +1.295e-8
  - 3600%, wrong sign
* - 2000
  - 400
  - 300
  - -4.535017e-7 (identical)
  - +1.295e-8
  - 3600%, wrong sign
* - 3000
  - 600
  - 300
  - -4.535017e-7 (identical)
  - +1.295e-8
  - 3600%, wrong sign
```

Every `(K, J, dps)` combination gives the **identical** wrong value.
Pushing further to `dps=500, K=2000, J=250` at `x=10, x=15` gives *worse*
relative errors (1.7e9 and 5.8e25 respectively) - i.e. more precision and
more terms make the answer worse, not better. This is the signature of
having crossed a genuine domain-of-validity boundary, not a cancellation
problem: a cancellation problem is fixable by adding decimal digits of
precision (the answer converges to the truth as `dps` grows); a
domain-of-validity problem is not (the answer converges to something, just
not the truth, however much precision is added).

By contrast, at `x=3, 4, 5` (same `n=0.7`) the series (with adequately
sized `K, J`) agrees with the reference to `1e-13`-`1e-7`, degrading
smoothly through `x=6` (2e-4) and `x=6.5` (1%) before hitting the wall
around `x ~ 7-8`.

**The identical phenomenon was independently confirmed for P(k)'s small-k
series** (`docs/einasto_power_spectrum.tex`'s `A_m^+` series, valid
"for all k" per its own derivation): at `n=0.7, kt=5`, the series is stable
at `-0.01407` across `dps` 60-500 and `M` up to 10000, versus the true
value `+0.003885` - confirmed by *two independent* references (`mpmath`
quadrature of the Gauss-Laguerre master integral, and `scipy.integrate.quad`
of the original real-space Hankel integral, which agree with each other to
1e-5, ruling out a reference-side bug). The series tracks the truth well up
to `kt ~ 4.2` (2.6e-7), degrades at `kt=4.5` (2e-4), and is qualitatively
broken by `kt=5`.

**Likely mathematical cause** (a hypothesis, not yet confirmed): both
series are derived by formally interchanging an infinite sum (residues of a
Mellin-Barnes contour integral, or a Taylor expansion) with an integral over
an unbounded domain. The ratio test on the resulting coefficients alone can
show "large or infinite radius of convergence" as a bare power series, while
the interchange itself - i.e. the series actually equaling the *original*
target integral, not some other analytic continuation of it - may only be
valid within a bounded region. This is closely analogous to
Watson's-lemma-type asymptotic-versus-convergent subtleties elsewhere in
special function theory. Confirming this against Retana-Montenegro et al.
(2012) Sec. 3.1's own stated validity domain (if any) is part of the
follow-up plan, but does not block it: the valid domain is empirically
characterizable per `n` (via a "doubling test": double the precision/terms
and require the answer not to move), and conveniently overlaps almost
exactly with where the *existing* numerical fallbacks are already strongest
(large `x`/`kt`) - see Finding 3.

**Consequence**: a hybrid dispatch (exact series where it's valid,
numerical methods beyond that point) is mathematically necessary, not an
engineering convenience. No stabilization technique - more precision, more
terms, algebraic resummation - can make the series correct past its true
domain of validity.

## Finding 2: within the valid domain, only modest extra precision is needed

Within the "good" region (e.g. `n=0.7`, `x` up to ~5.5, or `kt` up to
~4.2), a doubling-precision/doubling-term-count scheme converges
monotonically to the reference - e.g. `x=5`: `K=300, J=40` at `dps=60`
already gives `2.6e-7` relative error. This confirms that (a) evaluating
terms in log-space (to prevent individual-term overflow - terms reach
magnitude ~1e11 by `x=10, n=0.7`) and (b) modest, adaptively-chosen `mpmath`
precision are both genuinely useful and cheap *within the valid domain* -
unlike Mode B, this part of the problem responds normally to more
precision.

## Finding 3: the numerical fallbacks are already excellent exactly where the series isn't

- `compute_sigma_quadvec` (Abel quadrature; `src/clenspy/utils/integrate.py`)
  integrates a smooth, positive, rapidly-decaying integrand and is already
  documented (issue #3) as "machine precision everywhere" (1e-12 to 1e-16).
  Its only documented weakness is downstream, in
  `sigma_to_deltasigma_cumtrapz`'s cumulative-trapezoid step near `R -> 0` -
  precisely the region where the series is *strongest* (Mode B doesn't
  occur there; Mode A is a cheap fix). The series and the numerical
  fallback are natural complements, not overlapping competitors.
- `_einasto_pk_GL` (Gauss-Laguerre quadrature of the master integral,
  already implemented in `src/clenspy/halo/einasto.py`) was verified this
  session against independent brute-force `scipy.integrate.quad` for
  `n=0.7`: relative error 1.7e-11 at `kt=0.05`, degrading gracefully to
  1.9e-8 at `kt=3` and 1.1e-5 at `kt=8`. This already covers the entire
  practically relevant `kt` range for `n<1` far better than the `2e-3`
  currently reported in issue #2.

No new numerical method needs to be invented for the "far" regime in
either case - only the dispatch logic to actually reach the numerics that
already work well.

## Finding 4: a real, currently-unflagged dispatch bug explains most of issue #2

`EinastoProfile.power_spectrum(k, branch="auto")` has this structure:

```python
if branch == "auto":
    if np.isclose(n, 1.0): branch = "closed"
    elif np.isclose(n, 0.5): return <exact Gaussian>
    elif n < 1.0:
        return self._power_spectrum_numerical(k)   # <- FFTLog, ALWAYS, for any n<1
    else:
        <elaborate adaptive plateau-series / GL-quadrature / Wright-series dispatch>
        # only ever reached for n>1
```

The much more accurate GL-quadrature/plateau-series dispatch (whose own
docstring claims validation "to ~1e-14 relative error" for n>1) is **never
actually reached for generic `0<n<1`** - confirmed empirically:
`power_spectrum(k)` and `_power_spectrum_numerical(k)` are bit-identical for
`n=0.7`. Routing `n<1` through the same adaptive dispatch already used for
`n>1` (skipping only the Wright large-k series, which the code already
documents as invalid for `n<=1`) is expected to take generic-`n<1` P(k)
accuracy from `2e-3` to roughly `1e-8`-`1e-11`, **with no new stabilization
technique required at all**.

## Finding 5: `sigma`/`deltasigma` don't use the exact n=1/2 closed form

`power_spectrum` already special-cases `n=0.5` with the exact Gaussian
closed form and bypasses everything else. `sigma`/`deltasigma` do not -
they go through the general numerical (or, once implemented, series)
fallback even at exactly `n=0.5`. Mirroring the existing `n=1` special case
with an `n=0.5` one is a free, exact win (watch the
`(1 - e^{-x^2})/x^2` term for numerical safety at small `x`: use
`-np.expm1(-x**2)/x**2`, not `1 - np.exp(-x**2)`, since the two are
mathematically equal but the `expm1` form avoids cancellation).

## What this changes about the original question

The original question - "what does a full analytic solution for any x/k
regime require?" - turns out to have a more precise answer than "derive new
asymptotic expansions" (which would be open-ended mathematical research).
Instead:

- The existing series (Case 1, from the published paper) is exact and
  mathematically valid for all `x`/`k` **only up to an empirically
  characterizable point** per `n`. Beyond that point, no series will do -
  the fix there is dispatch to the already-excellent numerical methods, not
  a better series.
- Within the valid domain, most of the apparent failure (issue #3's
  headline number) was simple under-truncation (Mode A), not a fundamental
  numerical-stability problem - a cheap fix.
- One dispatch bug (Finding 4) likely resolves most of issue #2 with zero
  new mathematics.

See the approved implementation plan (persisted separately, for the
follow-up session) for the phased engineering work this points to: a
high-precision reference solver, empirical characterization of the
per-`n` validity boundary, the two quick fixes above, and a tiered
(fast series / mpmath-stabilized series / existing numerics) dispatch for
`Sigma`/`DeltaSigma`/`M_2D` and `P(k)`.
