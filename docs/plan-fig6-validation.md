# Plan: diagnosing the Fig. 6 (Costanzi et al. 2026) model mismatch

## Context

`validate_fig6_digitized.py` compares CLensPy's `SelBiasEngine`/`SigmaPrj`
model prediction against Costanzi et al. (2026)'s own **model** curve
(digitized from Fig. 6, `validation/data/costanzi2026_fig6.csv` — not a
noisy mock measurement, their theory line). Current result: systematic
overshoot everywhere, richness-dependent —
`lam[20,30)`: +66% to +86% median frac residual;
`lam[60,500)`: +11% to +24% median frac residual.
Sigmoid shape (`k=2.5/theta_lambda`, `theta_0=theta_lambda/2`) and
`boost_slope=0.13` already confirmed exact matches to the paper. A pure
global units bug is disfavored (residual is not constant across bins),
but a units bug entering through a richness-dependent quantity is not
ruled out.

## A. Units / cosmology sanity (cheapest, do first — invalidates everything downstream if wrong)

1. Empirical h-power calibration sweep: multiply the whole `Sigma_prj`
   output by `h^n` for `n in {-2,-1,0,1,2}` at one reference point, check
   if any power lands on the digitized curve.
2. Comoving vs physical `R`: test whether the digitized `R` axis is
   comoving Mpc/h (our assumption) or physical Mpc/h -- a stray
   `(1+z)` would look like this, and is distinguishable since our
   3 z-bins span 0.2-0.65.
3. Confirm the exact cosmology (`Omega_m, h, sigma8, ns`, P(k) source)
   used in Costanzi's own scripts (`SelectionBias/make_mock_lob_sigma_
   catalog.fits` generation + notebooks) against `V.COSMO`.
   `validate_sigma_prj_mock.py`'s own docstring flags this as an
   *assumption* ("Planck 2018... assumed"), not confirmed.
4. Cosmology sensitivity: once (3) is known, if it doesn't match exactly,
   perturb `Omega_m`/`sigma8` within the plausible gap and see how much
   the ratio curve moves.
5. theta units sanity: confirm no stray degrees/radians slip into
   `sigmoid_theta`/`area_overlap`.
6. `R_lambda` convention cross-check: `r_lambda(lam,h)=(lam/100)^0.2/h`
   vs the mock script's own `R_lambda(lam)=(lam/100)^0.2` `[pMpc/h]` --
   hand-derived match already, confirm numerically.

## B. Config/wiring checks

7. `b_eff` really uses Tinker HMF + Tinker bias (same objects `_operators`
   uses, not a stale default).
8. Richness kernel choice: `"y3"` (DES Y3/SDSS-injection EMG) vs `"self"`
   (mock-consistent) for a *synthetic* mock/theory comparison.
9. `min_mass` integration bound (`1e12` vs `1e13`) re-checked under
   `buzzard()` -- only tested under the old MOR.
10. `xi_NL`/`PkGrid` settings: nonlinear halofit vs linear, k-range/
    truncation -- untouched this whole session, sets the correlated
    channel amplitude directly.
11. Photo-z window `sigma_z`: confirm `y3_photoz_window()` matches
    whatever synthetic photo-z scatter Costanzi's model assumes.

## C. Data-driven inversion (strongest diagnostic -- model-agnostic)

12. Fit `(B_small, B_large)` per panel directly from the digitized curve:
    invert `ratio(R)` holding our own geometry/xi_NL/HMF fixed, get the
    implied plateaus, compare to what our closure currently computes.
13. Fit the implied MOR slope `alpha` from (12)'s implied large-R
    plateau; compare to `buzzard()`'s `alpha=0.859`.

## D. More

14. Percolation cap (`lambda<lob`) convergence re-check under `buzzard()`.
15. `n_theta`/`n_M`/`n_z` quadrature convergence under `buzzard()` (tuned
    under a different, flatter MOR).
16. `lob_rep` bias check: does the forward-model representative differ
    systematically between the narrow `[20,30)` bin and the wide, open
    `[60,500)` bin in a way that fakes a richness-dependent residual?

## Order of attack

A (1-6) first -- a units/cosmology bug invalidates every downstream
check. Then C (12-13) -- most direct, model-agnostic read on where the
gap actually is. B and D as needed based on what A/C find.
