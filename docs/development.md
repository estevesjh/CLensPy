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

For what each subpackage computes and why it is laid out the way it is,
see the Theory pages starting at {doc}`cosmology` — one page per physical
effect, prose next to the equation next to a runnable snippet — rather
than a module list here that would drift out of sync with them. The
mechanical {doc}`api/index` lists every public class and function.

`clenspy.halo.einasto_lown` is the series backend `EinastoProfile` uses for
all non-anchor `n` (see `docs/einasto_proj_density_v4.tex` and
{doc}`einasto_math`); it is internal - construct `EinastoProfile`.
`clenspy.halo.einasto_series` holds the `P(k)` branch evaluators that
`EinastoProfile.power_spectrum` dispatches between, and
`clenspy.utils.special` the generalised `E_nu` and Catalan pieces, which are
not Einasto-specific. None of these are part of the public API - use
`clenspy.halo.EinastoProfile`.

## Regenerating the getting-started notebook

`examples/getting_started.ipynb` is jupytext-paired with
`examples/getting_started.py` (percent format), which is the file to edit
— the `.ipynb` is generated. Every docs Theory page's `{literalinclude}`
snippet is pulled from a tagged section of that `.py` file, so an edit
must be synced and re-executed before it can be wired into a page:

```bash
pip install -e ".[docs]"
uv run jupytext --sync examples/getting_started.py
uv run jupyter nbconvert --to notebook --execute \
    --output getting_started.ipynb examples/getting_started.ipynb
```

The whole notebook must execute top to bottom — it is the single source
every Theory page's example is transcribed from, never hand-copied.
