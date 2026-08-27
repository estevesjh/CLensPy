"""
CLensPy: A Python package for cluster weak lensing analysis.

This package provides tools for computing weak lensing observables
from dark matter halo profiles, including NFW profiles and various
corrections for boost factors and miscentering effects.
"""

__version__ = "0.1.0"

# Import main modules for convenience
from . import cosmology, halo, kernels, lensing, selection, survey, utils
from .cosmology.fiducial import fiducial_cosmology
from .halo import BiasModel, NfwProfile, TwoHaloTerm
from .lensing import LensingProfile

__all__ = [
    "lensing",
    "halo",
    "utils",
    "cosmology",
    "selection",
    "kernels",
    "survey",
    "LensingProfile",
    "NfwProfile",
    "BiasModel",
    "TwoHaloTerm",
    "fiducial_cosmology",
]
