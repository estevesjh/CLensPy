r"""Distances and angular conversions for the fiducial background.

Thin wrappers that fix the unit convention on top of `astropy.cosmology`, so
that every caller in the package crosses the units boundary in the same
place and only once.

NOTE: units are h-free absolute -- comoving and angular diameter distances
in Mpc, angles in the unit named by the ``unit`` argument.

NOTE: :math:`\Sigma_{\rm crit}` used to live here. It moved to
`clenspy.kernels.sigma_crit`: it depends on the cosmology *and* on two
redshifts, which makes it lens-source geometry rather than a background
quantity.
"""

from typing import Union

import numpy as np
from astropy.cosmology import FlatLambdaCDM


def comoving_to_theta(
    D_c: Union[float, np.ndarray],
    z: float,
    cosmology: FlatLambdaCDM,
    unit: str = "arcmin",
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
    unit: str = "arcmin",
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


__all__ = [
    "comoving_to_theta",
    "theta_to_comoving",
]


if __name__ == "__main__":
    from .fiducial import fiducial_cosmology

    cosmo = fiducial_cosmology()
    z = 0.35
    D_c = np.array([0.1, 1.0, 10.0])
    theta = comoving_to_theta(D_c, z, cosmo, unit="arcmin")
    print(f"z = {z}")
    print("  D_c [Mpc]      ", D_c)
    print("  theta [arcmin] ", theta)
    # the pair must round-trip exactly
    print("  round trip     ", theta_to_comoving(theta, z, cosmo, unit="arcmin"))
