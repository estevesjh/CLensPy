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
from clenspy.halo import NfwProfile, EinastoProfile
from clenspy.cosmology import BiasModel

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

See `examples/getting_started.ipynb` for the full runnable notebook — one
section per physical effect, from the cosmology through the covariance.

## Using CLensPy from your own project

The canonical consumer workflow — cosmology object in, everything else
computed internally (verified to run as-is):

```python
import numpy as np
from clenspy.cosmology import (BiasModel, PkGrid, TinkerMassFunction,
                               fiducial_cosmology)
from clenspy.halo import TwoHaloTerm
from clenspy.utils.integrate import gl_nodes

# 1. Cosmology: configure this object only -- CLensPy drives CAMB internally.
cosmo = fiducial_cosmology(H0=70.0, Om0=0.286)  # Buzzard-like flat LCDM

# 2. P(k): CAMB runs inside PkGrid (disk-cached); h-free units throughout
#    (k in 1/Mpc, P in Mpc^3, M in Msun, R in Mpc).
pk_nl = PkGrid(cosmo=cosmo, nonlinear=True)      # halofit P(k, z)

# 3. xi_NL(r, z): FFTLog transform of P(k) -- never trapz the sin(kr) kernel.
z = 0.3
two_halo = TwoHaloTerm(pk_nl.k, pk_nl(pk_nl.k, z=z), zvec=z)
r = np.logspace(-0.5, 2.0, 40)                   # Mpc
xi_nl = two_halo.xi(r, z)                        # (40,), vectorized

# 4. sigma(M, z), Tinker bias, Tinker mass function -- all array-in, array-out.
M = np.logspace(13.0, 15.0, 30)                  # Msun
bias_model = BiasModel(cosmo=cosmo)
sigma_M = bias_model.sigma_tophat(M, z=z)        # sigma(M, z), shape (30,)
b_M = bias_model.bias(M, z=z)                    # Tinker (2010) bias
hmf = TinkerMassFunction(cosmo=cosmo)
dndlnm = hmf.dndlnm(M, z=z)                      # Mpc^-3, shape (30,)

# 5. Quadrature from clenspy.utils.integrate: cluster number density
#    n(>1e14) = int dlnM dn/dlnM with cached Gauss-Legendre nodes.
lnM, w = gl_nodes(np.log(1e14), np.log(1e15), 32)
n_cl = np.sum(w * hmf.dndlnm(np.exp(lnM), z=z))  # Mpc^-3
```

### Anti-patterns

1. **Don't `import camb`.** CLensPy wraps CAMB inside `PkGrid` (with the
   h-unit conversion, sigma8 renormalization, and disk caching). Configure
   the astropy cosmology object and let CLensPy call the Boltzmann solver.
2. **Never `np.trapz` these integrals.** The P(k) → ξ(r) and Σ-from-ξ
   transforms are oscillatory/singular; use the FFTLog and quadrature
   machinery in `clenspy.utils.integrate` (`pk_to_xi_fftlog`,
   `compute_sigma_grid`, `gl_nodes`/`mass_nodes`) or the classes that call
   it (`TwoHaloTerm`, `LensingProfile`). Naive trapezoids on linspace grids
   are the failure mode this package exists to prevent.
3. **Don't loop over array inputs.** The API is vectorized:
   `hmf.dndlnm(Mvec, zvec)`, `bias_model.bias(Mvec, zvec)`,
   `nfw.sigma(Rvec)`, `two_halo.xi(Rvec, zvec)` all take whole arrays,
   and the interpolator-backed evaluators (`dndlnm`, `bias`, `xi`,
   `sigma`, `deltasigma`) return the outer `(nx, nz)` grid for vector +
   vector input. Two verified exceptions: `PkGrid.__call__` follows NumPy
   broadcasting instead (`pk(kvec, zvec)` with different lengths raises;
   use `pk(kvec[:, None], zvec)` for the `(nk, nz)` grid), and low-level
   `SigmaGrid.sigma/sigma2` take scalar R — use `sigma2_fftlog` or the
   mass-function/bias wrappers for arrays.

Task-oriented recipes (ΔΣ(R), selection bias, etc.):
[docs/llm_quickstart.md](docs/llm_quickstart.md) or the "LLM/agent
quickstart" page on Read the Docs.

## Examples

The `examples/` directory contains detailed demonstrations:

- `getting_started.ipynb`: one section per physical effect (cosmology,
  power spectrum, mass function, halo bias, concentration, density and
  projected profiles, two-halo term, lensing profile, miscentering, boost
  factor, selection function/bias, survey, lensing kernel, observables,
  covariance) — the source every docs Theory page's code snippet is pulled
  from
- `einasto_convergence_map.py`: 2D convergence map from an Einasto profile

## Module Structure

`clenspy.cosmology`, `clenspy.halo`, `clenspy.lensing`, `clenspy.selection`,
`clenspy.kernels`, `clenspy.survey`, `clenspy.observables`,
`clenspy.covariance`, and `clenspy.utils`. See the
[docs](https://clenspy.readthedocs.io) for the physics behind each layer
and the full API reference.

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Astropy >= 4.0.0
- Matplotlib >= 3.3.0
- mpmath >= 1.3.0
- mcfit >= 0.0.22

### Optional Dependencies

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
