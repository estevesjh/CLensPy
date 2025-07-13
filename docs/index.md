# CLensPy Documentation

Welcome to CLensPy, a Python package for weak gravitational lensing analysis.

## Overview

CLensPy provides tools for computing weak lensing observables and working with dark matter halo profiles. The package is designed to be:

- **Easy to use**: Simple, intuitive API for common lensing calculations
- **Flexible**: Modular design allows easy extension and customization
- **Fast**: Optimized implementations of key algorithms
- **Well-tested**: Comprehensive test suite ensures reliability

## Features

### Core Lensing Calculations
- Excess surface density (Δσ) profiles
- Tangential shear profiles
- Critical surface density calculations
- Multi-redshift lensing geometry

### Halo Profiles
- NFW (Navarro-Frenk-White) profiles
- 3D density profiles
- Projected surface density profiles
- Mean surface density calculations

### Utilities
- Coordinate transformations (angular ↔ physical)
- Cosmological distance calculations
- Mathematical utilities

## Quick Start

```python
import numpy as np
from clenspy.profiles import NFWProfile
from clenspy.lensing import delta_sigma_nfw

# Create an NFW halo
M200 = 1e14  # Solar masses
c200 = 5.0   # Concentration
z = 0.3      # Redshift

nfw = NFWProfile(M200=M200, c200=c200, z=z)

# Calculate lensing signal
r = np.logspace(-1, 1, 50)  # Mpc
z_source = 1.0
delta_sigma = delta_sigma_nfw(r, M200, c200, z, z_source)
```

## Installation

```bash
pip install clenspy
```

## Examples

See the `examples/` directory for detailed usage examples:

- `demo_basic_usage.py`: Basic usage of all main features
- `demo_profile_fit.ipynb`: Profile fitting with MCMC uncertainty estimation

## API Reference

### Modules

- `clenspy.lensing`: Core lensing calculations
- `clenspy.profiles`: Halo density profiles
- `clenspy.utils`: Utility functions
- `clenspy.config`: Configuration and constants

## Development

CLensPy is actively developed. Contributions are welcome!

### Repository Structure
```
CLensPy/
├── clenspy/          # Main package
├── tests/            # Test suite
├── examples/         # Usage examples
├── docs/             # Documentation
└── .github/          # CI/CD workflows
```

## Citation

If you use CLensPy in your research, please cite:

```bibtex
@software{clenspy,
    title={CLensPy: A Python Package for Weak Gravitational Lensing},
    author={Your Name},
    year={2025},
    url={https://github.com/username/clenspy}
}
```

## License

CLensPy is released under the MIT License.
