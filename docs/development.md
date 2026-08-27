# Development

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/
```

A few tests validate CLensPy's halo profiles and two-halo term against
independent codes (`pyccl`, `cluster_toolkit`, `CLMM`, `camb`). They skip
automatically if those packages aren't installed; see
{doc}`installation` (the `compare` extra, plus the manual `cluster_toolkit`
install) to run them for real.

## Linting

The project uses [ruff](https://docs.astral.sh/ruff/) for both linting and
formatting (configured in `pyproject.toml`):

```bash
ruff check .
ruff format .
```

## Building the documentation locally

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in a browser. Read the Docs builds
the same way, driven by `.readthedocs.yaml` at the repository root.

## Package layout

- `clenspy.halo`: `NfwProfile`, `EinastoProfile`, `TwoHaloTerm`
- `clenspy.lensing`: `LensingProfile` (a higher-level wrapper; currently only
  wraps the NFW model, and its two-halo term needs the `compare` extra for a
  P(k) backend), plus boost-factor and miscentering corrections
- `clenspy.cosmology`: `PkGrid`, `BiasModel`, `TinkerMassFunction`, critical
  surface density, angular/comoving conversions
- `clenspy.utils`: log-grid interpolation, numerical integration helpers,
  shared decorators
- `clenspy.config`: default cosmology and physical constants

`clenspy.halo.einasto_lown` is the series backend `EinastoProfile` uses for
all non-anchor `n` (see `docs/einasto_proj_density_v4.tex` and
{doc}`einasto_math`); it is internal - construct `EinastoProfile`.
`clenspy.halo.einasto_series` holds the `P(k)` branch evaluators that
`EinastoProfile.power_spectrum` dispatches between, and
`clenspy.utils.special` the generalised `E_nu` and Catalan pieces, which are
not Einasto-specific. None of these are part of the public API - use
`clenspy.halo.EinastoProfile`.
