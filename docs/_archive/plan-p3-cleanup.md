# P3: renames, demo cleanup, timing, yaml configs, docs rebuild

Scope for this branch (`refactor/p0-fixes`), decided 2026-08-27. Six independent chunks,
done in this order so each is independently testable before the next starts.

---

## 1. Rename `observables/abundance.py` -> `observables/number_counts.py`

- `git mv src/clenspy/observables/abundance.py src/clenspy/observables/number_counts.py`
- `ClusterAbundance` -> `ClusterCounts` everywhere (class def, docstrings, `__repr__`).
- Update importers: `observables/__init__.py`, `observables/deltasigma.py` (demo import),
  `covariance/counts.py` docstring reference, `covariance/halo_to_halo.py` docstring
  reference, `tests/test_observables.py`, `tests/test_covariance.py`, `docs/api/index.md`
  (`.. automodule:: clenspy.observables.abundance` -> `.number_counts`).
- `sg -p 'ClusterAbundance' -l python` to catch call sites `sg` finds that grep-only
  passes would miss in string contexts (docstrings render the same either tool).

## 2. Rename `cosmology/mass_function.py` -> `cosmology/halo_mass_function.py`

- Module filename only. `TinkerMassFunction` class name and the `mass_function` callable
  parameter/attribute on `ClusterCounts` are **not** touched by this rename.
- Update importers: `cosmology/__init__.py`, `observables/number_counts.py`'s demo import,
  `docs/api/index.md` (`.. automodule:: clenspy.cosmology.mass_function`), any test file
  importing the module path directly.

## 3. `ClusterCounts.mean_richness()` -- a real third contraction

Currently only `mean_mass()` and `mean_redshift()` exist, both averaging a quantity that
lives on the native `(ln M, z)` grid. Observed richness $\lambda^{\rm ob}$ is not on that
grid -- it is integrated out inside `SelectionFunction.S_ij`, so there is nothing to
average directly yet.

**Design.** Add the first moment of $\lambda^{\rm ob}$ within each bin as a second
contraction the selection layer exposes, analogous to how `S_ij` is the zeroth moment:

$$\langle\lambda^{\rm ob}\rangle_{ij} = \frac{\int d\lambda^{\rm tr}\,
    \lambda^{\rm ob}\,\mathcal S_i(\lambda^{\rm tr}, z)}
    {\int d\lambda^{\rm tr}\,\mathcal S_i(\lambda^{\rm tr}, z)}$$

evaluated the same way `richness_kernel`/`selection_function` already build $\mathcal
S_i$ (EMG closed form in $\lambda^{\rm ob}$, quadrature only over $\lambda^{\rm tr}$), so
this is a new closed-form first moment next to the existing zeroth moment, not a new
integration scheme. Concretely:

- add `first_moment_lambda_ob(...)` to `clenspy.selection.richness_kernel` (or a method
  next to `S_ij` on `SelectionFunction`) returning $\langle\lambda^{\rm ob}\rangle$ per
  $(\lambda^{\rm tr}, z)$ node, then let `ClusterCounts` contract it through the existing
  `average()` machinery like any other per-halo quantity -- it already broadcasts over
  `(n_m, n_z)` and reduces over the weight.
- `ClusterCounts.mean_richness()` wraps that: build the `(n_m, n_z)` grid of $\langle
  \lambda^{\rm ob}\rangle$ from the selection object, call `self.average(...)`.
- Sanity check to assert in tests: `mean_richness()` must land inside each bin's
  $[\lambda_i, \lambda_{i+1})$ edges, the same property `mean_redshift()` already checks
  for $z$.

This is the piece most likely to need a design iteration once `richness_kernel`'s actual
moment machinery is read closely -- flagged here rather than guessed at further.

## 4. Demo: drop the $\Omega(z)$ cancellation block, print mean $\lambda^{\rm ob}$ and mean mass

In `number_counts.py`'s `__main__` (formerly `abundance.py`): remove the "Omega(z)
cancels in `average`..." block at the bottom (the `doubled` `ClusterCounts` and its two
ratio prints). Replace with a table of `mean_richness()` and `mean_mass()` per
$(\lambda, z)$ bin, in the same per-bin print style already used for `mean_mass()` /
`mean_redshift()` above it.

## 5. `time_method` decorator rollout

`clenspy.utils.decorators.time_method` already exists and works (records into
`self.timings`, prints if `self.verbose`). Apply it to:

- **`observables` classes**: `ClusterCounts.weight`, `.counts`, `.average` (the three
  hot paths); `StackedDeltaSigma.profile`.
- **`covariance/` classes**: `CountsCovariance.cov_poisson`, `.cov_sample_variance`;
  `DeltaSigmaGaussianCovariance.cov` (and `._spectra`, `._kernel_matrix` if those turn
  out to dominate); `DeltaSigmaHaloToHaloCovariance.cov`, `.mean_profile`.
- **`TinkerMassFunction`**: `walk` and `outputs` -- `walk` is the one that loops
  `sigma_grid.sigma2`/`dlnsigma2_dlnr` in pure Python over (by default) the 969-point
  production grid, the actual cost center. `multiplicity`/`coefficients` are cheap
  vectorised closed forms and do not need it.

No behavior change for callers who never set `self.verbose`; `self.timings` is opt-in
storage. Existing tests must keep passing unmodified since the decorator is
transparent (`functools.wraps`, passes args/kwargs through, returns the same value).

## 6. Configs: JSON -> YAML, plus a Buzzard config

- Add `pyyaml` to `pyproject.toml` `dependencies` (confirmed importable in the current
  venv already, just not declared).
- Convert `src/clenspy/configs/des_y1.json` and `des_y3.json` to `.yaml`, keeping every
  key and `_provenance`/`_note_*` string verbatim -- these are load-bearing
  documentation, not comments to drop in translation.
- `src/clenspy/survey/survey.py`: `load_config` reads `.yaml` via `yaml.safe_load`;
  `available_configs()` globs `*.yaml`; `CONFIG_DIR / f"{name}.json"` -> `.yaml`.
- `pyproject.toml` `[tool.setuptools.package-data]`: `"configs/*.json"` ->
  `"configs/*.yaml"`.
- Update `tests/test_survey.py` and any other reference to `.json` configs.
- **New**: `src/clenspy/configs/buzzard.yaml`. Buzzard is a simulated light-cone, not a
  real survey with a fitted $\Omega(z)$ footprint or metacal source properties --
  need to decide what `omega_z` and `sources` should be (flat mock footprint? reuse
  Y3's source shape as a placeholder, same pattern `des_y3.json`'s own
  `_note_pz_is_a_placeholder` already uses for its unfinished source n(z)?). This needs
  the Buzzard sky area / mock bin edges the `b_sel`/Buzzard calibration work already
  ported (`selection/bsel.py`, recent commit "Port b_sel") -- reuse those numbers rather
  than inventing new ones, with a `_provenance` note pointing at that module.

## 7. Move `halo/bias.py` -> `cosmology/bias.py`

`BiasModel` already only imports from `cosmology` (`fiducial`, `sigma`) and nothing from
`halo` -- it is a structure-formation fit calibrated on the same peak-height $\nu =
\delta_c/\sigma(M)$ as the Tinker (2008) mass function and the `child18`/`duffy08`
concentration relations, all three already living in `cosmology/`. `halo/` is density
profiles (NFW, Einasto, two-halo term); it does not belong there.

- `git mv src/clenspy/halo/bias.py src/clenspy/cosmology/bias.py`
- `halo/__init__.py`: drop the `BiasModel` import/re-export.
- `cosmology/__init__.py`: add `from .bias import BiasModel`, add to `__all__`.
- Update every importer found by `sg -p 'from ..halo import $$$' -l python` and
  `sg -p 'from .halo import $$$' -l python`, plus docstring/prose mentions:
  `covariance/halo_to_halo.py`, `covariance/counts.py`, `cosmology/mass_function.py`
  (already says "the Tinker (2010) bias in `clenspy.halo.BiasModel`" -- becomes
  `clenspy.cosmology.BiasModel`), `cosmology/concentration.py`, `cosmology/sigma.py`,
  `kernels/limber.py`, `selection/bsel.py`, `selection/__init__.py`,
  `lensing/profile.py`, `examples/demo_basic_usage.py`, `tests/test_bias.py`,
  `tests/test_bsel.py`, `tests/test_covariance.py`, `tests/test_mass_function.py`,
  `docs/api/index.md` (move the `BiasModel` autosummary entry from the
  `clenspy.halo` block to the `clenspy.cosmology` block), `docs/index.md`,
  `docs/notation.md`, `docs/development.md`.
- Keep `BiasModel` importable as `clenspy.halo.BiasModel` for one release? **No** --
  nothing in this codebase is at a public release yet (0.1.0, pre-alpha internal
  refactor branch), so this is a clean move, not a deprecation shim.

## 8. Docs rebuild

Once 1-6 are in and tests pass: regenerate `docs/api/index.md` module paths for the two
renames, then rebuild via the existing sphinx setup (`docs/conf.py`, autosummary ->
`generated/`) so the built docs reflect the new names and the new `buzzard` config /
`mean_richness` method. Last step, so it captures everything above rather than needing a
second pass.

---

## Also found, unrelated to the above but worth fixing now since it was flagged

**Stray `src/data/pk_cache/` (3.6M, untracked, gitignored) duplicates
`src/clenspy/data/`.** Root cause: `cosmology/pkgrid.py::_data_dir()` computes
`_PACKAGE_ROOT = Path(__file__).resolve().parents[2]`, which from
`src/clenspy/cosmology/pkgrid.py` resolves to `src/`, not `src/clenspy/` -- an
off-by-one (`parents[1]` is the package root; `parents[2]` is one level above it). Every
`PkGrid` cache write since whenever this ran from a `src/`-relative cwd landed one
directory too high. Fix: `parents[2]` -> `parents[1]`, then delete the stray
`src/data/` (untracked, safe to remove -- nothing reads from it, only
`src/clenspy/data/` is referenced by `photoz.py`, `richness_kernel.py`,
`miscentering.py`).
