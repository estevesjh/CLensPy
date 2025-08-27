#!/usr/bin/env python3
"""
Halo bias models for relating halo abundance to matter density.
"""

import mcfit
import numpy as np
from astropy import cosmology

from ..config import DEFAULT_COSMOLOGY


class BiasModel:
    """Compute the Bias Tinker et al. 2010 model
    for a given linear power-spectrum

    The calculation is based on the peak height of a top-hat sphere
    of lagrangian radius R corresponding to a mass M of linear
    power-spectrum.

    Parameters:
    -----------
        k : array, wavenumbers [h/Mpc]
        P : array, linear power-spectrum [(Mpc/h)^3]
        z : array, redshift
        cosmo: astropy.cosmology instance, cosmology to use

    Example:
    --------
    bM = biasModel(k, P, omega_m=0.3)
    bias = bM.bias_at_M(M)
    """

    def __init__(
        self,
        k: np.ndarray,
        P: np.ndarray,
        cosmo: cosmology = DEFAULT_COSMOLOGY,
        odelta: int = 200,
    ):
        self.k = k
        self.P = P
        self.cosmo = cosmo
        self.omega_m = self.cosmo.Om0
        self.odelta = odelta
        self.rhom = self.cosmo.critical_density(0).to_value("Msun/Mpc^3") * self.omega_m

    def bias(self, M):
        """Compute the bias for a given mass M

        Based on Bias Tinker et al. 2010 Eqn 6

        Computes peak height of top hat sphere of lagrangian radius R [Mpc/h comoving]
        corresponding to a mass M [Msun/h] of linear power spectrum.
        """
        if not hasattr(self, "nu"):
            self.nu = self.nu_at_mass(M)

        bias = self.bias_at_nu(self.nu)
        return bias

    def nu_at_mass(self, M, deltac=1.686):
        """Compute peak-height ν = δ_c / σ(M)."""
        sigma = self.sigma_tophat(M)
        return deltac / sigma

    def sigma_tophat(self, M):
        """
        Calculate σ(M) using mcfit.tophat_sigma for the linear power spectrum.

        Parameters
        ----------
        M : float or array, halo mass [Msun/h]

        Returns
        -------
        sigma : float or array, σ(M)
        """
        # Lagrangian radius R (Mpc/h comoving)
        R = (3 * M / (4 * np.pi * self.rhom)) ** (1 / 3)

        # mcfit expects k in [h/Mpc], P in [(Mpc/h)^3], R in [Mpc/h]
        Rvec, var = mcfit.TophatVar(self.k, lowring=True)(self.P, extrap=True)
        sigma_of_R = np.sqrt(np.interp(np.log10(R), np.log10(Rvec), var))
        return sigma_of_R

    def bias_at_nu(self, nu):
        """Bias Tinker et a. 2010 Eqn 6"""
        A, a, B, b, C, c = self.get_tinker_params()
        bias = self._bias_at_nu(nu, A, a, B, b, C, c, deltac=1.686)
        return bias

    def get_tinker_params(self):
        """Get Tinker et al. 2010 parameters for bias model."""
        # Tinker et al. 2010 Eqn 6 parameters
        # These are the best-fit parameters for delta=200
        y = np.log10(self.odelta)
        tinker_best_fit = {
            "A": 1.0 + 0.24 * y * np.exp(-((4 / y) ** 4)),
            "a": 0.44 * y - 0.88,
            "B": 0.183,
            "b": 1.5,
            "C": 0.019 + 0.107 * y + 0.19 * np.exp(-((4 / y) ** 4)),
            "c": 2.4,
        }
        return [tinker_best_fit[col] for col in ["A", "a", "B", "b", "C", "c"]]

    def _bias_at_nu(self, nu, A, a, B, b, C, c, deltac=1.686):
        """Bias Tinker et a. 2010 Eqn 6"""
        res = 1.0 - A * nu**a / (nu**a + deltac**a)
        res += B * nu**b
        res += C * nu**c
        return res
