#!/usr/bin/env python3
"""
Halo bias models for relating halo abundance to matter density.
"""

from __future__ import annotations

import numpy as np
from astropy.cosmology import Cosmology

from .fiducial import fiducial_cosmology
from .sigma import LinearPk, SigmaGrid


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

    NOTE: units are h-free absolute throughout -- mass in Msun, length in
    Mpc, wavenumbers in 1/Mpc, P(k) in Mpc^3. This class does *not* use the
    "little h" convention (h/Mpc, Msun/h, (Mpc/h)^3) common in the
    literature; matches `~clenspy.halo.NfwProfile` and
    `~clenspy.cosmology.PkGrid`. Lives in `clenspy.cosmology`, not
    `clenspy.halo`, because it is a structure-formation fit calibrated on
    the same peak height as `~clenspy.cosmology.TinkerMassFunction` and
    `~clenspy.cosmology.concentration`, not a density profile.

    NOTE: :math:`\bar\rho_m` in the Lagrangian radius is the **comoving**
    :math:`\Omega_{m,0}\rho_{c,0}`, so R(M) carries no redshift
    dependence and matches the P(k) it is integrated against.

    NOTE: the Tinker et al. (2010) fit is calibrated for
    :math:`\Delta = 200`-:math:`1600` and :math:`\nu \lesssim 4`; b(M)
    outside that is an extrapolation.

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
        r"""
        Compute the linear bias b(M) for a given halo mass.

        NOTE: this used to cache :math:`\nu` on ``self`` on the first call
        and reuse it for *every later* M, returning the first mass's bias
        for the second mass's argument. It no longer caches. The expensive
        part -- the FFTLog -- is cached where it belongs, on the shared
        `~clenspy.cosmology.SigmaGrid`, so recomputing :math:`\nu` per call
        is cheap and correct.

        Parameters
        ----------
        M : float or array
            Halo mass [Msun].

        Returns
        -------
        float or array
            Linear bias b(M), same shape as M.
        """
        return self.bias_at_nu(self.nu_at_mass(M))

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

    @property
    def sigma_grid(self):
        r"""The shared :math:`\sigma^2` evaluator, built once on first use.

        NOTE: the Tinker (2010) bias and the Tinker (2008) mass function are
        two fits to the **same** peak height, so they must read the same
        :math:`\sigma(M)`. This property is the shared object; see
        `clenspy.cosmology.sigma`.

        NOTE: `~clenspy.cosmology.SigmaGrid` is unit-agnostic -- the
        integral needs only :math:`kR` dimensionless and :math:`P` in units
        of :math:`k^{-3}` -- which is why this h-free class can share it
        with the h-scaled mass function. Two caveats, and **both** limits
        are dimensionful:

        - :math:`20/R` is in h/Mpc, so this class never truncates, and
          `sigma_tophat` passes ``truncate=False``;
        - the *fixed lower* limit :math:`10^{-4}` is in h/Mpc too. It only
          binds when the tabulated k range extends below it. `PkGrid`
          defaults to :math:`k_{\min} = 10^{-4}\,\mathrm{Mpc}^{-1}`, i.e.
          right at it, so in the h-free convention the cut is at
          :math:`10^{-4}` Mpc^-1 rather than at :math:`10^{-4}` h/Mpc --
          a 1/h shift in where the integral starts. Harmless here, because
          :math:`k^3 P W^2 \to 0` there and the large-scale contribution to
          :math:`\sigma(M)` at cluster masses is negligible; recorded
          because it is not zero and it is not obvious.
        """
        if getattr(self, "_sigma_grid", None) is None:
            self._sigma_grid = SigmaGrid(LinearPk(self.k, self.P))
        return self._sigma_grid

    def sigma_tophat(self, M):
        r"""
        Calculate σ(M), the top-hat variance amplitude at the Lagrangian
        radius of mass M.

        .. math::
            \sigma^2(M) = \int \frac{dk}{2\pi^2} k^2 P(k)\, W^2(kR),
            \qquad R = \left(\frac{3M}{4\pi\bar\rho_m}\right)^{1/3}

        where :math:`W` is the Fourier transform of the real-space top-hat
        window and :math:`\bar\rho_m` is the comoving mean matter density.

        NOTE: delegates to `sigma_grid`, which fixed three defects in the
        previous inline version: it splined the **variance** linearly in
        :math:`\log_{10} R` and then square-rooted it (percent-level error
        where :math:`\sigma^2` is curved), it let ``np.interp`` silently
        **clamp** outside the FFTLog range instead of refusing, and it
        rebuilt the FFTLog on every single call.

        NOTE: untruncated, i.e. the full tabulated :math:`k` range. The
        :math:`k \le 20/R` convention belongs to the production mass
        function, not to the bias -- see `sigma_grid`.

        Parameters
        ----------
        M : float or array
            Halo mass [Msun].

        Returns
        -------
        sigma : float or array
            σ(M), same shape as M.
        """
        # Lagrangian radius R [Mpc], comoving
        R = (3 * np.asarray(M, dtype=float)
             / (4 * np.pi * self.rhom)) ** (1 / 3)
        ln_sigma2, _ = self.sigma_grid.sigma2_fftlog(np.log(R))
        return np.exp(0.5 * ln_sigma2)

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


if __name__ == "__main__":
    import numpy as np

    # a smooth power-law P(k), so no Boltzmann solver is needed
    k = np.logspace(-4, 3, 800)
    P = 2e4 * k**-1.5 / (1.0 + (k / 0.2) ** 2)

    model = BiasModel(k, P)
    M = np.array([1e13, 1e14, 5e14, 1e15])
    print("Tinker et al. (2010) linear halo bias, Delta = 200m")
    print(f"{'M [Msun]':>11s}  {'sigma(M)':>9s}  {'nu':>7s}  {'b(M)':>7s}")
    for m in M:
        s = float(np.ravel(model.sigma_tophat(m))[0])
        nu = float(np.ravel(model.nu_at_mass(m))[0])
        b = float(np.ravel(model.bias(m))[0])
        print(f"{m:11.2e}  {s:9.4f}  {nu:7.4f}  {b:7.4f}")

    print("\nb rises with M and nu, as it must: rarer haloes are more biased.")
    print("NOTE: the Tinker fit is calibrated for nu <~ 4; beyond that b(M)")
    print("      is an extrapolation. Units are h-free absolute throughout.")
