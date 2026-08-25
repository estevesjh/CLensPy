"""Halo mass function and halo bias on a :class:`~clenspy.cosmology.PkGrid`.

Physical units throughout (no little-h): masses in Msun, distances in
comoving Mpc, number densities in comoving Mpc^-3.

Contents
--------
``SigmaGrid``
    :math:`\\sigma(M, z)` tabulated on the PkGrid nodes and spline-interpolated.
``Tinker08MassFunction``
    Tinker et al. (2008) :math:`dn/d\\ln M` for :math:`\\Delta = 200m`,
    with the Table-4 redshift evolution of the fit parameters.
``Tinker10Bias``
    Tinker et al. (2010) peak-height bias :math:`b(M, z)` evaluated through
    ``SigmaGrid`` (no state caching between calls).
``ConstantBias``
    :math:`b(M, z) = b_0` — for validation runs, precomputed effective
    biases, and covariance evaluations with externally supplied bias.

All bias models share the protocol ``__call__(M, z)`` / ``at_lnM(lnM, z)``.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import RectBivariateSpline

from ..config import DEFAULT_COSMOLOGY

__all__ = [
    "SigmaGrid",
    "Tinker08MassFunction",
    "Tinker10Bias",
    "ConstantBias",
    "DELTA_C",
]

DELTA_C = 1.686

TINKER08_DELTA200M = dict(A=0.186, a=1.47, b=2.57, c=1.19)


def _top_hat_window(x: np.ndarray) -> np.ndarray:
    r"""Fourier top-hat :math:`W(x) = 3(\sin x - x\cos x)/x^3`, Taylor near 0."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-3
    xs = x[small]
    out[small] = 1.0 - xs * xs / 10.0 + xs**4 / 280.0
    xl = x[~small]
    out[~small] = 3.0 * (np.sin(xl) - xl * np.cos(xl)) / xl**3
    return out


def _alpha_tinker(delta: float) -> float:
    """Tinker08 Table-4 exponent for the ``b(z)`` scaling."""
    return 10.0 ** (-((0.75 / np.log10(delta / 75.0)) ** 1.2))


class SigmaGrid:
    r"""R.m.s. linear density fluctuation :math:`\sigma(M, z)`.

    .. math::

        \sigma^2(R, z) = \frac{1}{2\pi^2} \int_0^\infty dk\, k^2\,
            P_{\rm lin}(k, z)\, W_{\rm th}^2(kR),
        \qquad R(M) = \left(\frac{3M}{4\pi\bar\rho_{m,0}}\right)^{1/3}

    with :math:`\bar\rho_{m,0}` the comoving mean matter density
    (Msun/Mpc^3, physical units) and :math:`R` in comoving Mpc.

    Tabulated once on the PkGrid ``(z, k)`` nodes by Simpson integration in
    :math:`\ln k`, then evaluated by a ``RectBivariateSpline`` over
    ``(z, lnM)``.

    Parameters
    ----------
    pkgrid : clenspy.cosmology.PkGrid
        Linear power spectrum grid, physical units (k in 1/Mpc, P in Mpc^3).
    cosmo : astropy.cosmology.Cosmology, optional
        Must match the cosmology used to build ``pkgrid``.
    lnM_range : tuple of float
        Range of ln(M/Msun) for the table.
    n_lnM : int
        Number of table nodes in lnM.
    """

    def __init__(
        self,
        pkgrid,
        cosmo=DEFAULT_COSMOLOGY,
        lnM_range: tuple[float, float] = (np.log(1e11), np.log(5e16)),
        n_lnM: int = 200,
    ) -> None:
        self.pkgrid = pkgrid
        self.cosmo = cosmo
        rhoc0 = cosmo.critical_density(0).to_value("Msun/Mpc^3")
        self.rho_m0 = rhoc0 * cosmo.Om0  # comoving mean matter density

        k = np.asarray(pkgrid.k, dtype=float)
        lnk = np.log(k)
        z_grid = np.asarray(pkgrid.z, dtype=float)
        pk_tab = np.asarray(pkgrid.pk, dtype=float)  # (nz, nk)

        self._lnM = np.linspace(lnM_range[0], lnM_range[1], n_lnM)
        M_arr = np.exp(self._lnM)
        R_arr = self.radius_of_mass(M_arr)  # (n_lnM,)

        # integrand in dlnk: k^3 P(k, z) W(kR)^2 / (2 pi^2)
        kR = k[None, :] * R_arr[:, None]  # (n_lnM, nk)
        W_sq = _top_hat_window(kR) ** 2
        sigma = np.empty((z_grid.size, n_lnM))
        for iz in range(z_grid.size):
            integrand = k[None, :] ** 3 * pk_tab[iz][None, :] * W_sq
            sigma2 = simpson(integrand, x=lnk, axis=1) / (2.0 * np.pi**2)
            sigma[iz] = np.sqrt(sigma2)
        self._sigma_tab = sigma
        self._z_grid = z_grid
        self._spl = RectBivariateSpline(z_grid, self._lnM, sigma, kx=3, ky=3)

    def radius_of_mass(self, M):
        r"""Lagrangian radius :math:`R(M)` in comoving Mpc."""
        M = np.asarray(M, dtype=float)
        return (3.0 * M / (4.0 * np.pi * self.rho_m0)) ** (1.0 / 3.0)

    def __call__(self, M, z):
        r""":math:`\sigma(M, z)` for broadcast-compatible ``M`` [Msun], ``z``."""
        M_b, z_b = np.broadcast_arrays(
            np.asarray(M, dtype=float), np.asarray(z, dtype=float)
        )
        shape = M_b.shape
        out = self._spl.ev(z_b.ravel(), np.log(M_b).ravel())
        return out.reshape(shape) if shape else float(out[0])

    def dlnsigma_dlnM(self, M, z, h: float = 1e-3):
        r""":math:`d\ln\sigma/d\ln M` by centred finite difference."""
        M = np.asarray(M, dtype=float)
        s_up = self(M * np.exp(h), z)
        s_dn = self(M * np.exp(-h), z)
        return (np.log(s_up) - np.log(s_dn)) / (2.0 * h)

    def nu(self, M, z, deltac: float = DELTA_C):
        r"""Peak height :math:`\nu = \delta_c / \sigma(M, z)`."""
        return deltac / self(M, z)


class Tinker08MassFunction:
    r"""Tinker et al. (2008) halo mass function, :math:`\Delta = 200m`.

    .. math::

        \frac{dn}{d\ln M} = \frac{\bar\rho_{m,0}}{M}\, f(\sigma)\,
            \left| \frac{d\ln\sigma}{d\ln M} \right|,
        \qquad
        f(\sigma) = A \left[ \left(\frac{\sigma}{b}\right)^{-a} + 1 \right]
            e^{-c/\sigma^2}

    Table-2 parameters at z=0 (Delta=200m): ``A=0.186, a=1.47, b=2.57,
    c=1.19``.  With ``z_evolution=True`` (default) the Table-4 scalings
    are applied: :math:`A(z) = A_0 (1+z)^{-0.14}`,
    :math:`a(z) = a_0 (1+z)^{-0.06}`,
    :math:`b(z) = b_0 (1+z)^{-\alpha}` with
    :math:`\alpha = 10^{-(0.75/\log_{10}(\Delta/75))^{1.2}}`.

    Number densities are comoving Mpc^-3 (physical units, no little-h).
    """

    def __init__(
        self, sigma_grid: SigmaGrid, odelta: int = 200, z_evolution: bool = True
    ) -> None:
        if odelta != 200:
            raise NotImplementedError("Only Delta = 200m is supported")
        self.sigma_grid = sigma_grid
        self.rho_m0 = sigma_grid.rho_m0
        self._params = dict(TINKER08_DELTA200M)
        self._z_evolution = bool(z_evolution)
        self._alpha = _alpha_tinker(float(odelta))

    def _params_at_z(self, z):
        A0 = self._params["A"]
        a0 = self._params["a"]
        b0 = self._params["b"]
        c0 = self._params["c"]
        if not self._z_evolution:
            return A0, a0, b0, c0
        opz = 1.0 + np.asarray(z, dtype=float)
        return (
            A0 * opz ** (-0.14),
            a0 * opz ** (-0.06),
            b0 * opz ** (-self._alpha),
            c0,
        )

    def f_sigma(self, sigma, z=0.0):
        r"""Multiplicity function :math:`f(\sigma)` at redshift ``z``."""
        A, a, b, c = self._params_at_z(z)
        sigma = np.asarray(sigma, dtype=float)
        return A * ((sigma / b) ** (-a) + 1.0) * np.exp(-c / sigma**2)

    def dn_dlnM(self, M, z):
        r""":math:`dn/d\ln M` in comoving Mpc^-3."""
        M = np.asarray(M, dtype=float)
        sig = self.sigma_grid(M, z)
        dls = self.sigma_grid.dlnsigma_dlnM(M, z)
        return (self.rho_m0 / M) * self.f_sigma(sig, z) * np.abs(dls)

    def at_lnM(self, lnM, z):
        r""":math:`dn/d\ln M` evaluated at ``lnM = ln(M/Msun)``."""
        return self.dn_dlnM(np.exp(np.asarray(lnM, dtype=float)), z)

    def __call__(self, M, z):
        r""":math:`dn/dM` in Msun^-1 Mpc^-3."""
        M = np.asarray(M, dtype=float)
        return self.dn_dlnM(M, z) / M


def _tinker10_params(delta: float = 200.0):
    y = np.log10(delta)
    A = 1.0 + 0.24 * y * np.exp(-((4.0 / y) ** 4))
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-((4.0 / y) ** 4))
    c = 2.4
    return A, a, B, b, C, c


class Tinker10Bias:
    r"""Tinker et al. (2010) peak-height halo bias, Eq. 6 / Table 2.

    .. math::

        b(\nu) = 1 - A \frac{\nu^a}{\nu^a + \delta_c^a} + B \nu^b + C \nu^c,
        \qquad \nu = \delta_c / \sigma(M, z)

    Evaluated through :class:`SigmaGrid` on every call — stateless in
    ``(M, z)``, unlike :class:`clenspy.halo.BiasModel` whose ``bias()``
    caches :math:`\nu` from the first call.
    """

    def __init__(self, sigma_grid: SigmaGrid, odelta: float = 200.0) -> None:
        self.sigma_grid = sigma_grid
        self._pars = _tinker10_params(float(odelta))

    def bias_at_nu(self, nu):
        A, a, B, b, C, c = self._pars
        nu = np.asarray(nu, dtype=float)
        return 1.0 - A * nu**a / (nu**a + DELTA_C**a) + B * nu**b + C * nu**c

    def __call__(self, M, z):
        return self.bias_at_nu(self.sigma_grid.nu(M, z))

    def at_lnM(self, lnM, z):
        return self(np.exp(np.asarray(lnM, dtype=float)), z)


class ConstantBias:
    r"""Mass- and redshift-independent bias :math:`b(M, z) = b_0`.

    Same protocol as :class:`Tinker10Bias`.  Useful for validation runs,
    injecting a precomputed effective bias (e.g. an S_ij-weighted
    :math:`\langle b \rangle` or a selection-bias plateau), and covariance
    evaluations with externally supplied bias.
    """

    def __init__(self, b0: float) -> None:
        self.b0 = float(b0)

    def bias_at_nu(self, nu):
        return np.full_like(np.asarray(nu, dtype=float), self.b0)

    def __call__(self, M, z):
        M_b, z_b = np.broadcast_arrays(
            np.asarray(M, dtype=float), np.asarray(z, dtype=float)
        )
        out = np.full(M_b.shape, self.b0)
        return out if M_b.shape else self.b0

    def at_lnM(self, lnM, z):
        lnM_b, z_b = np.broadcast_arrays(
            np.asarray(lnM, dtype=float), np.asarray(z, dtype=float)
        )
        out = np.full(lnM_b.shape, self.b0)
        return out if lnM_b.shape else self.b0
