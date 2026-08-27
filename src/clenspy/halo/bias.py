#!/usr/bin/env python3
"""
Halo bias models for relating halo abundance to matter density.
"""

from __future__ import annotations

import mcfit
import numpy as np
from astropy.cosmology import Cosmology

from ..cosmology.fiducial import fiducial_cosmology


class BiasModel:
    r"""Compute the linear halo bias b(M) from the Tinker et al. (2010) fit,
    for a given linear power spectrum.

    The calculation is based on the peak height of a top-hat sphere
    of lagrangian radius R corresponding to a mass M of linear
    power-spectrum:

    .. math::
        \nu(M) = \frac{\delta_c}{\sigma(M)}, \qquad
        \sigma^2(M) = \int \frac{dk}{2\pi^2} k^2 P(k)\, W^2(kR),

    where :math:`W` is the top-hat window function, :math:`R = (3M /
    4\pi\bar\rho_m)^{1/3}` is the Lagrangian radius, and
    :math:`\delta_c = 1.686`. The bias is then (Tinker et al. 2010, eq. 6)

    .. math::
        b(\nu) = 1 - A \frac{\nu^a}{\nu^a + \delta_c^a} + B \nu^b + C \nu^c,

    with :math:`A, a, B, b, C, c` fit as functions of the spherical
    overdensity :math:`\Delta` (``odelta``).

    Parameters
    ----------
    k : array
        Wavenumbers [1/Mpc], physical (not h-scaled).
    P : array
        Linear power spectrum [Mpc^3], physical (not h-scaled).
    cosmo : astropy.cosmology instance, optional
        Cosmology to use (default: `fiducial_cosmology()`).
    odelta : int, optional
        Spherical overdensity :math:`\Delta` defining the halo mass, e.g.
        200 for :math:`M_{200m}` (default: 200).

    NOTE: units are h-free absolute throughout -- mass in Msun, length in
    Mpc, wavenumbers in 1/Mpc, P(k) in Mpc^3. This class does *not* use the
    "little h" convention (h/Mpc, Msun/h, (Mpc/h)^3) common in the
    literature; matches `~clenspy.halo.NfwProfile` and
    `~clenspy.cosmology.PkGrid`.

    NOTE: :math:`\bar\rho_m` in the Lagrangian radius is the **comoving**
    :math:`\Omega_{m,0}\rho_{c,0}`, so R(M) carries no redshift
    dependence and matches the P(k) it is integrated against.

    NOTE: the Tinker et al. (2010) fit is calibrated for
    :math:`\Delta = 200`-:math:`1600` and :math:`\nu \lesssim 4`; b(M)
    outside that is an extrapolation.

    Examples
    --------
    >>> bias_model = BiasModel(k, P)
    >>> bias = bias_model.bias(M)
    """

    def __init__(
        self,
        k: np.ndarray,
        P: np.ndarray,
        cosmo: Cosmology | None = None,
        odelta: int = 200,
    ):
        self.k = k
        self.P = P
        self.cosmo = fiducial_cosmology() if cosmo is None else cosmo
        self.omega_m = self.cosmo.Om0
        self.odelta = odelta
        self.rhom = self.cosmo.critical_density(0).to_value("Msun/Mpc^3") * self.omega_m

    def bias(self, M):
        """
        Compute the linear bias b(M) for a given halo mass.

        Caches the peak height ν(M) on first call (as ``self.nu``); calling
        `bias` again reuses it even for a different M, so construct a new
        `BiasModel` (or call `nu_at_mass` directly) if you need bias at more
        than one mass.

        Parameters
        ----------
        M : float or array
            Halo mass [Msun].

        Returns
        -------
        float or array
            Linear bias b(M), same shape as M.
        """
        if not hasattr(self, "nu"):
            self.nu = self.nu_at_mass(M)

        bias = self.bias_at_nu(self.nu)
        return bias

    def nu_at_mass(self, M, deltac=1.686):
        r"""
        Compute peak-height :math:`\nu(M) = \delta_c / \sigma(M)`.

        Parameters
        ----------
        M : float or array
            Halo mass [Msun].
        deltac : float, optional
            Critical linear overdensity for collapse (default: 1.686).

        Returns
        -------
        float or array
            Peak height ν(M), same shape as M.
        """
        sigma = self.sigma_tophat(M)
        return deltac / sigma

    def sigma_tophat(self, M):
        r"""
        Calculate σ(M) using mcfit.tophat_sigma for the linear power spectrum.

        .. math::
            \sigma^2(M) = \int \frac{dk}{2\pi^2} k^2 P(k)\, W^2(kR),
            \qquad R = \left(\frac{3M}{4\pi\bar\rho_m}\right)^{1/3}

        where :math:`W` is the Fourier transform of the real-space top-hat
        window and :math:`\bar\rho_m` is the mean matter density today.

        Parameters
        ----------
        M : float or array
            Halo mass [Msun].

        Returns
        -------
        sigma : float or array
            σ(M), same shape as M.
        """
        # Lagrangian radius R [Mpc], physical
        R = (3 * M / (4 * np.pi * self.rhom)) ** (1 / 3)

        Rvec, var = mcfit.TophatVar(self.k, lowring=True)(self.P, extrap=True)
        sigma_of_R = np.sqrt(np.interp(np.log10(R), np.log10(Rvec), var))
        return sigma_of_R

    def bias_at_nu(self, nu):
        """
        Evaluate the Tinker et al. (2010) bias function at peak height ν.

        Parameters
        ----------
        nu : float or array
            Peak height ν, e.g. from `nu_at_mass`.

        Returns
        -------
        float or array
            Linear bias b(ν), same shape as nu.
        """
        A, a, B, b, C, c = self.get_tinker_params()
        bias = self._bias_at_nu(nu, A, a, B, b, C, c, deltac=1.686)
        return bias

    def get_tinker_params(self):
        r"""
        Get the Tinker et al. (2010) bias fit parameters for ``self.odelta``.

        .. math::
            A = 1 + 0.24\, y\, e^{-(4/y)^4}, \qquad a = 0.44 y - 0.88, \qquad
            B = 0.183, \qquad b = 1.5,

        .. math::
            C = 0.019 + 0.107 y + 0.19\, e^{-(4/y)^4}, \qquad c = 2.4,
            \qquad y = \log_{10}\Delta

        with :math:`\Delta` = ``self.odelta`` the spherical overdensity.

        Returns
        -------
        list of float
            ``[A, a, B, b, C, c]``.
        """
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
        r"""
        Tinker et al. (2010) eq. 6:
        :math:`b(\nu) = 1 - A \nu^a / (\nu^a + \delta_c^a) + B \nu^b + C \nu^c`.
        """
        res = 1.0 - A * nu**a / (nu**a + deltac**a)
        res += B * nu**b
        res += C * nu**c
        return res
