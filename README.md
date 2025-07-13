# CLensPy

A Python package for cluster gravitational lensing analysis.

## Overview

CLensPy provides a comprehensive toolkit for cluster gravitational lensing calculations, including:

- **Lensing observables**: Computation of excess surface density (Δ\\Sigma) and shear profiles
- **Halo profiles**: Implementation of NFW and other dark matter halo density profiles  
- **Coordinate utilities**: Conversions between angular and physical coordinates
- **Profile fitting**: Tools for fitting theoretical models to observational data

## Features

- 🔭 **Comprehensive**: Full pipeline from halo profiles to lensing observables
- 🚀 **Fast**: Optimized numerical implementations
- 📊 **Flexible**: Modular design for easy extension and customization
- 🧪 **Well-tested**: Extensive test suite ensures reliability
- 📚 **Documented**: Clear documentation with practical examples

## Quick Start

### Installation

```bash
pip install clenspy
```

### Basic Usage

```python
import numpy as np
from clenspy.profiles import NFWProfile
from clenspy.lensing import delta_sigma_nfw

# Define halo parameters
M200 = 1e14  # Halo mass in solar masses
c200 = 5.0   # Concentration parameter
z_lens = 0.3 # Lens redshift
z_source = 1.0 # Source redshift

# Create NFW profile
nfw = NFWProfile(M200=M200, c200=c200, z=z_lens)

# Calculate excess surface density
r = np.logspace(-1, 1, 50)  # Radii in Mpc
delta_sigma = delta_sigma_nfw(r, M200, c200, z_lens, z_source)

# Plot the results
import matplotlib.pyplot as plt
plt.loglog(r, delta_sigma)
plt.xlabel('Radius [Mpc]')
plt.ylabel('Δσ [M☉/Mpc²]')
plt.show()
```

## Examples

The `examples/` directory contains detailed demonstrations:

- `demo_basic_usage.py`: Introduction to core functionality
- `demo_profile_fit.ipynb`: Profile fitting with MCMC uncertainty estimation

## Module Structure

- `clenspy.lensing`: Core weak lensing calculations
- `clenspy.profiles`: Dark matter halo density profiles
- `clenspy.utils`: Coordinate transformations and utilities
- `clenspy.config`: Configuration settings and physical constants

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.3.0

### Optional Dependencies

For MCMC analysis (required for some examples):
```bash
pip install clenspy[mcmc]
```

For development:
```bash
pip install clenspy[dev]
```

## Development

### Installing from Source

```bash
git clone https://github.com/estevesjh/clenspy.git
cd clenspy
pip install -e .
```

### Running Tests

```bash
pytest tests/
```

### Contributing

We welcome contributions! Please see our contributing guidelines for details.

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

## Features

- Modular architecture for easy extension and benchmarking
- Optimized for large-scale survey datasets
- Built on NumPy, Astropy, and Matplotlib
- Fully tested with `pytest`
- Example workflows and tutorials included

---

## Installation

```bash
pip install clenspy
```

Or clone the repository:

```bash
git clone https://github.com/your-org/clenspy.git
cd clenspy
pip install -e .
```

---

# Documentation

Full documentation and tutorials are available in the [`docs/`](docs/) folder.

---

## License

[MIT](LICENSE)

---

## Credits

CLensPy is inspired by [Cluster Tool-Kit](https://github.com/tmcclintock/cluster_toolkit.git) and developed for the cosmology community.
```
