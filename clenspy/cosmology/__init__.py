"""
Cosmology utilities for CLensPy.

This module provides cosmological calculations using astropy.cosmology.
"""

from .utils import comoving_to_theta, sigma_critical

__all__ = ["sigma_critical", "comoving_to_theta"]