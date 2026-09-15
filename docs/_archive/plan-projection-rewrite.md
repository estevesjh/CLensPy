# Plan: projection.py rewrite — readable LOS integration

Goal: same physics, same numbers, same public observables — but the code
reads like `docs/projection_lensing.md`. One geometry object, one
integrator, integrands as plain closures, one class per physical concept.
Status: **implemented**; demo output identical to the pre-rewrite baseline
(bit-level), 43/43 tests green.

## A. cosmology/distances.py — `ComovingDistance`

`ComovingDistance(cosmology)` with `.chi(z)`, `.z_of_chi(chi)`,
`.dchi_dz(z)` (linspace z 1e-4..2, 2000 pts — the existing recipe).
Replaces the `_zs_ref/_chi_ref/_dchi_dz_ref` triple duplicated in
`SigmaPrj` **and** `SelBiasEngine`; both store it as `self.distance`.

## B. utils/los_integrals.py — the quadrature machinery

- `LosGeometry(thetas, chi_o, chi_min, chi_max, r_excl)` — cosh–Abel chord
  and interval limits; exclusion is the boundary between the "inside" and
  "outside" smooth intervals, never a mask.
- `integrate_los(geometry, integrand, n_u, interval)` — GL in u, Jacobian
  `|dchi| = r du`, fg/bg branches summed; returns `(n_theta, ...)`. All
  physics lives in the integrand callable.
- `theta_edges(chi_o, range, n, r_excl)` and `theta_grid(edges)` — the
  θ-shell grid (log-spaced + exclusion tangency edge) and its centres /
  spherical-measure correction.
- `shell_masses` / `tail_masses` — exact per-shell profile masses
  (renamed from kernel_cells/tail_cells; "kernel" is banned, shells are
  the astrophysics nomenclature).
- **Deleted**: `interval_weights` (14-kwarg signature), `los_branches`,
  the `INTEGRATORS` registry and the `SigmaPrjConfig.integrator` field.

Supporting moves: `r_excl(lob, zob, h)` → `selection/geometry.py` (next
to `r_lambda`); `mass_nodes(m_min, m_max, n)` → `utils/integrate.py`.

## C. lensing/projection.py — one class per concept

- `Exclusion(mode, floor_one_plus_bxi)` — the K_exc bookkeeping alone
  (`.channels(n_rnd_in, n_rnd_out, n_lss, b_sel_values) -> (n_rnd, n_cl)`),
  a mode table in its docstring; validates its own mode.
- `MassShells(mis_table, r_trunc)` — mass of the offset neighbour
  profiles in each θ shell (`shell_masses` + optional `tail_masses` +
  spherical measure), with `mean_sigma` (the doc's m̂) and a one-entry
  cache. Pure profile physics, independent of the LOS integral.
- `SigmaPrj.n_los_integral(lob, zob, thetas, b_sel_values)` — the three
  cosh–Abel z integrals, linear:

```python
geometry = self._geometry(thetas, lob, zob)        # LosGeometry
def n_rnd_integrand(r, chi, theta_index):      # background: common/dchidz n(M,z) M dlnM
def n_lss_integrand(r, chi, theta_index):      # correlated: n_rnd * b(M,z) * xi_NL(r)
def n_lss_rnd_integrand(r, chi, theta_index):  # floored full bracket (b_sel inside)
correlated = n_lss_rnd_integrand if cfg.floor_one_plus_bxi else n_lss_integrand
n_rnd_in  = integrate_los(geometry, n_rnd_integrand, cfg.n_u_inside,  "inside")
n_rnd_out = integrate_los(geometry, n_rnd_integrand, cfg.n_u_outside, "outside")
n_lss     = integrate_los(geometry, correlated,      cfg.n_u_outside, "outside")
```

- `sigma_prj` / `deltasigma_prj` are fully inlined and linear: theta
  shells → b_sel values → `n_los_integral` → `Exclusion.channels` →
  `mass_shells` → einsum contraction over (θ, M). No `_assemble`, no
  `_project`, no `_channel_weights`.
- **Deleted from the class**: `chi`, `theta_grid`, `theta_edges`,
  `r_excl`, `_mass_nodes`, `_chi_support`, `_z_of_chi`, `_mhat`,
  `kernel`, `_abel_branches` — relocated per A/B or inlined. `common`
  stays (the doc's common(z)). Kept: `build()` chain, both caches,
  `rnd/cl/components()`, `hmf_model`/`bias_model` alongside the
  evaluators.

## Naming (user-fixed)

integrands `n_rnd_integrand` / `n_lss_integrand` / `n_lss_rnd_integrand`;
integrals `n_rnd_in` / `n_rnd_out` / `n_lss`; channels `(n_rnd, n_cl)`;
`mean_sigma` (a sigma, not a mass); `mass_shells` / `shell_masses` /
`tail_masses` (never "kernel", never "ring", never "pair", never "K").

## Acceptance (met)

- `pytest tests/test_projection.py tests/test_bsel.py
  tests/test_sigma_prj_profile.py` → 43 passed.
- Demo `python -m clenspy.lensing.projection` numerically identical to a
  reconstructed pre-rewrite baseline (only-diff: trailing newline).
- Consumers outside tests (validation, examples, docs, papers scripts)
  use only the public observables and are untouched.
