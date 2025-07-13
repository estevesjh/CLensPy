"""
Cosmology utility functions for weak lensing calculations.

This module provides functions for cosmological calculations using astropy.cosmology,
with explicit units and proper distance calculations.
"""

from typing import Union

import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM


def sigma_critical(
    z_lens: float,
    z_source: float,
    cosmology: FlatLambdaCDM
) -> float:
    """
    Calculate the lensing critical surface density.

    Parameters
    ----------
    z_lens : float
        Lens redshift
    z_source : float
        Source redshift
    cosmology : astropy.cosmology.FlatLambdaCDM
        Astropy cosmology object

    Returns
    -------
    float
        Critical surface density in Msun/Mpc^2

    Raises
    ------
    ValueError
        If z_source <= z_lens

    Notes
    -----
    The critical surface density is given by:
    Σ_crit = c² / (4πG) × D_s / (D_l × D_ls)

    where D_l, D_s, D_ls are angular diameter distances to lens,
    source, and between lens and source respectively.
    """
    if z_source <= z_lens:
        msg = f"Source redshift ({z_source}) must be greater than"
        msg += f" lens redshift ({z_lens})."
        raise ValueError(msg)

    # Angular diameter distances using astropy
    D_l = cosmology.angular_diameter_distance(z_lens)
    D_s = cosmology.angular_diameter_distance(z_source)
    D_ls = cosmology.angular_diameter_distance_z1z2(z_lens, z_source)

    # Physical constants
    c = 299792.458 * u.km / u.s  # Speed of light
    G = 4.302e-9 * u.Mpc / u.Msun * (u.km / u.s) ** 2  # Gravitational constant

    # Critical surface density
    sigma_crit = (c ** 2 / (4 * np.pi * G)) * (D_s / (D_l * D_ls))

    # Convert to Msun/Mpc^2
    sigma_crit = sigma_crit.to(u.Msun / u.Mpc ** 2)

    return sigma_crit.value


def comoving_to_theta(
    D_c: Union[float, np.ndarray],
    z: float,
    cosmology: FlatLambdaCDM,
    unit: str = "arcmin"
) -> Union[float, np.ndarray]:
    """
    Convert comoving distance to angular separation.

    Parameters
    ----------
    D_c : float or array-like
        Comoving distance in Mpc
    z : float
        Redshift at which to evaluate the angular diameter distance
    cosmology : astropy.cosmology.FlatLambdaCDM
        Astropy cosmology object
    unit : str, optional
        Output angular unit: "arcsec", "arcmin", "deg", "rad" (default: "arcmin")

    Returns
    -------
    float or array-like
        Angular separation in specified units

    Notes
    -----
    The angular separation is calculated as:
    θ = D_c / D_A(z)

    where D_A(z) is the angular diameter distance at redshift z.
    """
    # Validate unit
    valid_units = ["arcsec", "arcmin", "deg", "rad"]
    if unit not in valid_units:
        msg = f"Unit '{unit}' not recognized. Valid units: {valid_units}"
        raise ValueError(msg)

    # Angular diameter distance at redshift z
    D_A = cosmology.angular_diameter_distance(z)

    # Angular separation in radians
    theta_rad = np.array(D_c) / D_A.value

    # Convert to requested unit
    if unit == "rad":
        return theta_rad
    elif unit == "deg":
        return np.rad2deg(theta_rad)
    elif unit == "arcmin":
        return np.rad2deg(theta_rad) * 60.0
    elif unit == "arcsec":
        return np.rad2deg(theta_rad) * 3600.0


def theta_to_comoving(
    theta: Union[float, np.ndarray],
    z: float,
    cosmology: FlatLambdaCDM,
    unit: str = "arcmin"
) -> Union[float, np.ndarray]:
    """
    Convert angular separation to comoving distance.

    Parameters
    ----------
    theta : float or array-like
        Angular separation in specified units
    z : float
        Redshift at which to evaluate the angular diameter distance
    cosmology : astropy.cosmology.FlatLambdaCDM
        Astropy cosmology object
    unit : str, optional
        Input angular unit: "arcsec", "arcmin", "deg", "rad" (default: "arcmin")

    Returns
    -------
    float or array-like
        Comoving distance in Mpc

    Notes
    -----
    The comoving distance is calculated as:
    D_c = θ × D_A(z)

    where D_A(z) is the angular diameter distance at redshift z.
    """
    # Validate unit
    valid_units = ["arcsec", "arcmin", "deg", "rad"]
    if unit not in valid_units:
        msg = f"Unit '{unit}' not recognized. Valid units: {valid_units}"
        raise ValueError(msg)

    # Convert to radians
    theta_array = np.array(theta)
    if unit == "rad":
        theta_rad = theta_array
    elif unit == "deg":
        theta_rad = np.deg2rad(theta_array)
    elif unit == "arcmin":
        theta_rad = np.deg2rad(theta_array / 60.0)
    elif unit == "arcsec":
        theta_rad = np.deg2rad(theta_array / 3600.0)

    # Angular diameter distance at redshift z
    D_A = cosmology.angular_diameter_distance(z)

    # Comoving distance
    D_c = theta_rad * D_A.value

    return D_c


def critical_density(z: float, cosmology: FlatLambdaCDM) -> float:
    """
    Calculate the critical density of the universe at redshift z.

    Parameters
    ----------
    z : float
        Redshift
    cosmology : astropy.cosmology.FlatLambdaCDM
        Astropy cosmology object

    Returns
    -------
    float
        Critical density in Msun/Mpc^3
    """
    rho_crit = cosmology.critical_density(z)
    return rho_crit.to(u.Msun / u.Mpc ** 3).value


def hubble_parameter(z: float, cosmology: FlatLambdaCDM) -> float:
    """
    Calculate the Hubble parameter H(z) at redshift z.

    Parameters
    ----------
    z : float
        Redshift
    cosmology : astropy.cosmology.FlatLambdaCDM
        Astropy cosmology object

    Returns
    -------
    float
        Hubble parameter in km/s/Mpc
    """
    H_z = cosmology.H(z)
    return H_z.to(u.km / u.s / u.Mpc).value


__all__ = [
    "sigma_critical",
    "comoving_to_theta",
    "theta_to_comoving",
    "critical_density",
    "hubble_parameter",
]