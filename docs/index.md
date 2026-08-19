# CLensPy

Welcome to CLensPy, a Python package for cluster gravitational lensing analysis.

## Overview

CLensPy provides a toolkit for computing weak-lensing observables from dark
matter halo profiles: NFW and Einasto density profiles, projected surface
density Sigma(R), excess surface density DeltaSigma(R), halo bias, and the
two-halo term. It is designed to be:

- **Modular**: each halo profile is a small, self-contained class with the
  same `density` / `sigma` / `deltasigma` interface
- **Fast**: vectorized NumPy/SciPy implementations, with closed-form or
  series expansions used where available (e.g. the Einasto profile's
  projected density)
- **Validated**: cross-checked against independent codes (`pyccl`,
  `cluster_toolkit`, `CLMM`) in the test suite

## Installation

CLensPy is not yet published on PyPI; install it from source:

```bash
git clone https://github.com/estevesjh/clenspy.git
cd clenspy
pip install -e .
```

See {doc}`installation` for optional dependency groups (`mcmc`, `docs`,
`compare`, `dev`).

## Quick Start

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

See `examples/demo_basic_usage.py` in the repository for the full runnable
script, including plots, and `examples/demo_lensing.ipynb` for a 1-halo +
2-halo walkthrough.

## API Reference

For a detailed breakdown of every class and function, see the {doc}`api/index`.

## Additional Resources

- **Development**: see {doc}`development` for running tests, the optional
  comparison-test dependencies, and building these docs locally
- **Notes**: see {doc}`einasto_math` for the Einasto profile's math -
  closed forms, series expansions, and the numerical fallbacks that cover
  the rest
- **Source code**: <https://github.com/estevesjh/clenspy>
- **Issue tracker**: <https://github.com/estevesjh/clenspy/issues>

```{toctree}
:maxdepth: 2
:caption: Contents

installation
api/index
development
Einasto profile math <einasto_math>
Einasto series investigation <einasto_series_investigation>
```
