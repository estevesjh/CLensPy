```{image} _static/img/logo.png
:alt: CLensPy logo
:width: 300px
:align: center
```

# CLensPy

[![Documentation Status](https://readthedocs.org/projects/clenspy/badge/?version=latest)](https://clenspy.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/estevesjh/CLensPy/blob/main/LICENSE)
[![GitHub tag](https://img.shields.io/github/v/tag/estevesjh/CLensPy?label=version&sort=semver)](https://github.com/estevesjh/CLensPy/tags)
[![GitHub issues](https://img.shields.io/github/issues/estevesjh/CLensPy)](https://github.com/estevesjh/CLensPy/issues)
[![GitHub stars](https://img.shields.io/github/stars/estevesjh/CLensPy?style=social)](https://github.com/estevesjh/CLensPy)

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

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"density-profiles\"]"
:end-before: "%% [markdown]"
:language: python
```

```
r200 = 1.4303 Mpc, rs = 0.2861 Mpc, rho_s = 3.547e+14 Msun/Mpc^3
rho_NFW(r)     [Msun/Mpc^3] = [5.57109391e+14 2.68756655e+13 5.02012592e+12 7.94381981e+11]
rho_Einasto(r) [Msun/Mpc^3] = [5.83600877e+14 2.77183496e+13 4.86300049e+12 6.13799586e+11]
rho_tilde_NFW(k)     [Msun] = [9.98998564e+13 9.05346969e+13 8.91993930e+12]
rho_tilde_Einasto(k) [Msun] = [1.83204136e+14 1.06342641e+14 9.27420738e+12]
```

This is one section of the same notebook every Theory page's own example
pulls from — see `examples/getting_started.ipynb` in the repository for
the full runnable notebook, one section per physical effect, from the
cosmology through the covariance.

## API Reference

For a detailed breakdown of every class and function, see the {doc}`api/index`.

## Additional Resources

- **Development**: see {doc}`development` for running tests, the optional
  comparison-test dependencies, and building these docs locally
- **Theory**: the {doc}`cosmology` toctree caption onward is a physics-first
  walkthrough of every quantity CLensPy computes — prose, the governing
  equation, and a runnable snippet, one page per physical effect
- **Notes**: see {doc}`einasto_math` for the Einasto profile's math -
  exact anchors, the stable residue-series backend (any `n > 0`, with
  resonance pairing), and the `P(k)` dispatch
- **Source code**: <https://github.com/estevesjh/clenspy>
- **Issue tracker**: <https://github.com/estevesjh/clenspy/issues>

```{toctree}
:maxdepth: 1
:caption: Getting Started

installation
```

```{toctree}
:maxdepth: 1
:caption: Cosmology

cosmology
power_spectrum
mass_function
halo_bias
concentration
```

```{toctree}
:maxdepth: 1
:caption: Halo profiles

density_profiles
projected_profiles
two_halo_term
```

```{toctree}
:maxdepth: 1
:caption: Cluster lensing

lensing_profile
miscentering
boost_factor
```

```{toctree}
:maxdepth: 1
:caption: Selection effects

selection_function
selection_bias
```

```{toctree}
:maxdepth: 1
:caption: Survey

survey
lensing_kernel
```

```{toctree}
:maxdepth: 1
:caption: Cluster observables

observables
```

```{toctree}
:maxdepth: 1
:caption: Covariance

covariance
covariance_halo_to_halo
```

```{toctree}
:maxdepth: 1
:caption: Reference

api/index
notation
validation
development
```

```{toctree}
:maxdepth: 1
:caption: Notes

Einasto profile math <einasto_math>
Einasto series investigation <einasto_series_investigation>
Miscentering math <miscentering_math>
Covariance FFTLog math <covariance_fftlog_math>
Refactor plan <refactor-plan>
P3 cleanup plan <plan-p3-cleanup>
```
