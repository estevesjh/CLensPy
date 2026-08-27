"""The fiducial cosmology, as a factory rather than a shared instance.

This is a function and not a module-level ``DEFAULT_COSMOLOGY`` object on
purpose. A single shared instance used as a default argument is ambient
state: it is constructed once at import, every caller that does not pass a
cosmology silently shares it, and a mutation anywhere is global. Calling
`fiducial_cosmology()` gives each caller its own object and makes the
default visible at the call site.
"""

from astropy.cosmology import FlatLambdaCDM

__all__ = ["fiducial_cosmology"]


def fiducial_cosmology(H0=70.0, Om0=0.3):
    """Flat LambdaCDM with the package's fiducial parameters.

    Parameters
    ----------
    H0 : float
        Hubble constant [km/s/Mpc].
    Om0 : float
        Present-day total matter density parameter (dimensionless).

    Returns
    -------
    astropy.cosmology.FlatLambdaCDM
        A fresh instance on every call.
    """
    return FlatLambdaCDM(H0=H0, Om0=Om0)


if __name__ == "__main__":
    cosmo = fiducial_cosmology()
    print(cosmo)
    print(f"chi(z=0.3) = {cosmo.comoving_distance(0.3).value:.3f} Mpc")
    rho_c0 = cosmo.critical_density0.to_value("Msun/Mpc^3")
    print(f"rho_c(0)   = {rho_c0:.4e} Msun/Mpc^3")
    print("distinct instances:", fiducial_cosmology() is not fiducial_cosmology())
