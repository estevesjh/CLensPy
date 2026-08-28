# Installation

CLensPy is not yet published on PyPI; install it from source with `pip`.

## From source

```bash
git clone https://github.com/estevesjh/clenspy.git
cd clenspy
pip install -e .
```

This installs the core runtime dependencies: NumPy, SciPy, Astropy,
Matplotlib, mpmath, and mcfit.

## Optional dependency groups

CLensPy defines a few `pip` extras for optional functionality:

```{list-table}
:header-rows: 1

* - Extra
  - Installs
  - Use case
* - `docs`
  - `sphinx`, `myst-parser`, `sphinx-wagtail-theme`, `sphinx-copybutton`
  - Building this documentation locally
* - `compare`
  - `pyccl`, `clmm`, `camb`
  - Running the tests that validate CLensPy against independent codes
* - `dev`
  - `pytest`, `pytest-cov`, `pytest-benchmark`, `ruff`
  - Running the test suite and linting
```

Install any combination with, e.g.:

```bash
pip install -e ".[dev,docs]"
```

or all of them at once with `pip install -e ".[all]"`.

### A note on `cluster_toolkit`

One comparison test (`tests/test_twohalo.py`) also checks CLensPy's
two-halo term against
[`cluster_toolkit`](https://github.com/tmcclintock/cluster_toolkit), which
is not on PyPI and links against [GSL](https://www.gnu.org/software/gsl/).
It isn't part of the `compare` extra for that reason. To install it:

```bash
# macOS
brew install gsl

# Debian/Ubuntu
sudo apt-get install libgsl-dev

pip install "git+https://github.com/tmcclintock/cluster_toolkit.git"
```

The test skips automatically if `cluster_toolkit` isn't importable.
