"""
Boost factor correction functions for weak lensing profiles.

Boost factors account for the enhancement in the lensing signal due to
correlated satellite galaxies and substructure around the main halo.
"""

import numpy as np
from typing import Union


def boost_factor_nfw(
    R: Union[float, np.ndarray], M200: float, z: float, model: str = "leauthaud11"
) -> Union[float, np.ndarray]:
    """
    Calculate boost factor for NFW profiles.

    The boost factor accounts for the enhancement in the lensing signal
    due to correlated satellites and substructure.

    Parameters
    ----------
    R : float or array-like
        Projected radius in Mpc
    M200 : float
        Halo mass in solar masses
    z : float
        Redshift
    model : str, optional
        Boost model to use (default: "leauthaud11")
        Options: "leauthaud11", "tinker05", "none"

    Returns
    -------
    float or array-like
        Boost factor (dimensionless, typically > 1)

    References
    ----------
    Leauthaud et al. 2011, ApJ, 738, 45
    Tinker et al. 2005, ApJ, 631, 41
    """
    R = np.atleast_1d(R)

    if model.lower() == "none":
        boost = np.ones_like(R)

    elif model.lower() == "leauthaud11":
        # Leauthaud et al. 2011 boost model
        # B(R) = 1 + A * (R/R0)^alpha for R < R_max

        # Model parameters (approximate values)
        A = 0.13  # Amplitude
        R0 = 0.15  # Scale radius in Mpc
        alpha = -0.3  # Power law index
        R_max = 2.0  # Maximum radius for boost in Mpc

        # Mass and redshift dependence (simplified)
        mass_factor = (M200 / 1e14) ** 0.1
        z_factor = (1 + z) ** (-0.3)

        boost = np.ones_like(R)
        mask = R < R_max

        if np.any(mask):
            boost[mask] = 1 + A * mass_factor * z_factor * (R[mask] / R0) ** alpha

    elif model.lower() == "tinker05":
        # Tinker et al. 2005 boost model (simplified)
        # More complex halo occupation distribution based boost

        # Simplified implementation
        boost = 1.0 + 0.1 * (R / 1.0) ** (-0.5) * (M200 / 1e14) ** 0.2
        boost = np.maximum(boost, 1.0)  # Ensure boost >= 1

    else:
        msg = f"Unknown boost model: {model}"
        raise ValueError(msg)

    return boost if R.shape else boost[0]


def boost_factor_satellites(
    R: Union[float, np.ndarray], M200: float, z: float, f_sat: float = 0.15
) -> Union[float, np.ndarray]:
    """
    Calculate boost factor due to satellite galaxies.

    Parameters
    ----------
    R : float or array-like
        Projected radius in Mpc
    M200 : float
        Halo mass in solar masses
    z : float
        Redshift
    f_sat : float, optional
        Satellite fraction (default: 0.15)

    Returns
    -------
    float or array-like
        Satellite boost factor
    """
    R = np.atleast_1d(R)

    # Simplified satellite boost model
    # Satellites contribute more at intermediate scales
    R_sat = 0.5  # Characteristic satellite scale in Mpc
    boost_amplitude = f_sat * (M200 / 1e14) ** 0.3

    boost = 1 + boost_amplitude * np.exp(-R / R_sat)

    return boost if R.shape else boost[0]


def apply_boost_correction(
    delta_sigma: Union[float, np.ndarray],
    R: Union[float, np.ndarray],
    M200: float,
    z: float,
    model: str = "leauthaud11",
) -> Union[float, np.ndarray]:
    """
    Apply boost correction to delta sigma profile.

    Parameters
    ----------
    delta_sigma : float or array-like
        Uncorrected delta sigma profile in Msun/Mpc^2
    R : float or array-like
        Projected radius in Mpc
    M200 : float
        Halo mass in solar masses
    z : float
        Redshift
    model : str, optional
        Boost model to use

    Returns
    -------
    float or array-like
        Boost-corrected delta sigma profile
    """
    boost = boost_factor_nfw(R, M200, z, model)
    return delta_sigma * boost


__all__ = [
    "boost_factor_nfw",
    "boost_factor_satellites",
    "apply_boost_correction",
]
