# Development

## Running the tests

```bash
uv sync --extra dev
uv run pytest tests/
```

`camb` (the default `PkGrid` backend) is in the `dev` extra, so the real
halo-model tests run for real, not skipped. A few tests additionally
cross-validate against `pyccl`/`cluster_toolkit`/`CLMM` (the `compare`
extra, plus the manual `cluster_toolkit` install — see {doc}`installation`);
those skip automatically if not installed.

## Linting

The project uses [ruff](https://docs.astral.sh/ruff/) for both linting and
formatting (configured in `pyproject.toml`):

```bash
ruff check .
ruff format .
```

CI's actual build-breaking gate is narrower — `flake8` checking only for
syntax errors and undefined names (`E9,F63,F7,F82`) — which the pre-push
hook below also runs, so a `ruff check` warning won't block a push but a
real syntax/undefined-name error will.

## Before you push

A `pre-push` git hook (`.githooks/pre-push`) mirrors
`.github/workflows/tests.yml` exactly: it builds a clean `git worktree` of
what's about to be pushed (not your working tree, so gitignored-but-present
local files can't hide a missing-from-git bug), runs `uv sync --locked`
(fails if `uv.lock` has drifted from `pyproject.toml`), the same `flake8`
gate CI uses, `ruff check` (advisory), and `pytest`. Enable it once per
clone:

```bash
git config core.hooksPath .githooks
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
