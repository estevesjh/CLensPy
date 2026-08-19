# CLensPy

[![Documentation Status](https://readthedocs.org/projects/clenspy/badge/?version=latest)](https://clenspy.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python package for cluster gravitational lensing analysis.

## Overview

CLensPy provides a toolkit for cluster weak-lensing calculations, including:

- **Halo profiles**: NFW and Einasto 3D density, projected surface density
  Sigma(R), and excess surface density (weak-lensing shear proxy)
  DeltaSigma(R)
- **Halo bias**: Linear bias b(M) from the Tinker et al. (2010) fitting
  function
- **Two-halo term**: Correlation function, Sigma(R), and DeltaSigma(R) from a
  gridded linear power spectrum
- **Cosmology utilities**: Critical surface density, angular/comoving
  conversions, and P(k) grids

## Quick Start

### Installation

```bash
git clone https://github.com/estevesjh/clenspy.git
cd clenspy
pip install -e .
```

CLensPy is not yet published on PyPI; install from source as shown above.

### Basic Usage

```python
import numpy as np
from clenspy.halo import NfwProfile, EinastoProfile, BiasModel

# Define halo parameters
M200 = 1e14  # Halo mass [Msun]
c200 = 5.0   # Concentration

# NFW profile
nfw = NfwProfile(m200=M200, c200=c200)
R = np.logspace(-2, 1, 50)  # Projected radius [Mpc]
sigma = nfw.sigma(R)            # Surface density Sigma(R) [Msun/Mpc^2]
deltasigma = nfw.deltasigma(R)  # Excess surface density DeltaSigma(R)

# Einasto profile, for comparison
einasto = EinastoProfile(alpha=0.2, rho_0=nfw.rho_s, r_s=nfw.rs, tol=1e-4)
deltasigma_einasto = einasto.deltasigma(R)

# Linear halo bias, given a matter power spectrum P(k)
k = np.logspace(-3, 1, 200)
Pk = 2e4 * (k / 0.05) ** (-1.5)  # replace with a real P(k), e.g. from CAMB/CLASS
bias = BiasModel(k, Pk).bias(M200)
```

See `examples/demo_basic_usage.py` for the full runnable script, including
plots of density, Sigma(R), and DeltaSigma(R).

## Examples

The `examples/` directory contains detailed demonstrations:

- `demo_basic_usage.py`: NFW vs Einasto profiles, halo bias, quick plots
- `demo_lensing.ipynb`: 1-halo + 2-halo Sigma(R)/DeltaSigma(R) walkthrough
- `einasto_convergence_map.py`: 2D convergence map from an Einasto profile

## Module Structure

- `clenspy.halo`: Halo profiles (`NfwProfile`, `EinastoProfile`), halo bias
  (`BiasModel`), and the two-halo term (`TwoHaloTerm`)
- `clenspy.lensing`: `LensingProfile`, a higher-level wrapper combining a halo
  profile with the two-halo term and boost/miscentering corrections
  (2-halo support currently requires the `compare` extra for a P(k) backend;
  only the NFW model is implemented so far)
- `clenspy.cosmology`: `PkGrid` (linear/nonlinear P(k) grids), critical surface
  density, and angular/comoving coordinate conversions
- `clenspy.utils`: Log-grid interpolation, numerical integration helpers, and
  decorators used across the halo/lensing modules
- `clenspy.config`: Default cosmology and physical constants

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Astropy >= 4.0.0
- Matplotlib >= 3.3.0
- mpmath >= 1.3.0
- mcfit >= 0.0.22

### Optional Dependencies

For MCMC profile fitting:
```bash
pip install -e ".[mcmc]"
```

For building the documentation locally:
```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

For running the tests that validate CLensPy against independent codes
(`pyccl`, `clmm`, `camb` are on PyPI; `cluster_toolkit` needs GSL and is not
on PyPI, see [docs/development.md](docs/development.md)):
```bash
pip install -e ".[compare]"
```

For development (linting, testing):
```bash
pip install -e ".[dev]"
```

## Development

### Installing from Source

```bash
git clone https://github.com/estevesjh/clenspy.git
cd clenspy
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

Tests that compare against `pyccl`/`cluster_toolkit`/`clmm`/`camb` skip
automatically if those packages aren't installed (see the `compare` extra
above).

### Contributing

We welcome contributions! Please see our contributing guidelines for details.

## Documentation

Full documentation, including the API reference, is built with Sphinx and
hosted on Read the Docs: https://clenspy.readthedocs.io

## Citation

If you use CLensPy in your research, please cite:

```bibtex
@software{clenspy2025,
    title={CLensPy: A Python Package for Weak Gravitational Lensing Analysis},
    author={Esteves, J.H.},
    year={2025},
    url={https://github.com/estevesjh/clenspy},
    version={0.1.0}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

CLensPy builds upon decades of research in weak gravitational lensing. We acknowledge the contributions of the broader weak lensing community to the theoretical foundations implemented in this package.
