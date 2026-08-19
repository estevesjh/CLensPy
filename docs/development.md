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

- `clenspy.halo`: `NfwProfile`, `EinastoProfile`, `BiasModel`, `TwoHaloTerm`
- `clenspy.lensing`: `LensingProfile` (a higher-level wrapper; currently only
  wraps the NFW model, and its two-halo term needs the `compare` extra for a
  P(k) backend), plus boost-factor and miscentering corrections
- `clenspy.cosmology`: `PkGrid`, critical surface density, angular/comoving
  conversions
- `clenspy.utils`: log-grid interpolation, numerical integration helpers,
  shared decorators
- `clenspy.config`: default cosmology and physical constants

`clenspy.halo.einasto_v2` and `clenspy.halo.einasto_v3` are research/benchmark
implementations exploring alternative closed-form series for the Einasto
projected density (see `docs/einasto_proj_density_v2.tex` and `_v3.tex`).
They are not part of the public API - use `clenspy.halo.EinastoProfile`.
