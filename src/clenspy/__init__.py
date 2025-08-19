"""
CLensPy: A Python package for cluster weak lensing analysis.

This package provides tools for computing weak lensing observables
from dark matter halo profiles, including NFW profiles and various
corrections for boost factors and miscentering effects.
"""

__version__ = "0.1.0"

# Import main modules for convenience
from . import config, cosmology, halo, lensing, utils
from .config import DEFAULT_COSMOLOGY, RHOCRIT
from .halo import NFWProfile, TwoHaloTerm, biasModel
from .lensing import LensingProfile

__all__ = [
    "lensing",
    "halo",
    "utils",
    "cosmology",
    "config",
    "LensingProfile",
    "NFWProfile",
    "biasModel",
    "TwoHaloTerm",
    "DEFAULT_COSMOLOGY",
    "RHOCRIT",
]
