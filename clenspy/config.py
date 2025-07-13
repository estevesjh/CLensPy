"""
Configuration settings for CLensPy.
"""

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

# Default astropy cosmology
DEFAULT_COSMOLOGY = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)

# Alternative: use simple dictionary for compatibility
DEFAULT_COSMOLOGY_DICT = {
    "H0": 70.0,  # Hubble constant in km/s/Mpc
    "Om0": 0.3,  # Matter density parameter
    "Ode0": 0.7,  # Dark energy density parameter
    "w0": -1.0,  # Dark energy equation of state
}

# Physical constants
G_NEWTON = 4.302e-9  # Gravitational constant in Mpc/Msun (km/s)^2
C_LIGHT = 299792.458  # Speed of light in km/s

# Critical density constant
# RHOCRIT = 1e4*3.*Mpcperkm*Mpcperkm/(8.*PI*G); units are Msun h^2/Mpc^3
RHOCRIT = 2.77533742639e11  # Critical density in Msun h^2/Mpc^3

# Additional useful constants
PI = 3.14159265359
ARCMIN_TO_RAD = PI / (180.0 * 60.0)  # Convert arcmin to radians
ARCSEC_TO_RAD = PI / (180.0 * 3600.0)  # Convert arcsec to radians

# Numerical settings
DEFAULT_INTEGRATION_POINTS = 1000
DEFAULT_TOLERANCE = 1e-6
