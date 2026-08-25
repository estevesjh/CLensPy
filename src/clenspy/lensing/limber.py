"""Limber angular power spectra for cluster-lensing covariances.

Computes, on a shared log-ell grid, the spectra entering the Gaussian
:math:`\\Delta\\Sigma` covariance (Wu et al. 2019 eq. 22 structure):

- :math:`C_\\ell^{\\Sigma\\Sigma}` — :math:`\\Sigma_{\\rm crit}`-weighted
  matter spectrum,
  :math:`\\bar\\rho_m^2 \\sum \\Delta\\chi\\, q_\\Sigma^2/\\chi^2\\, P(k, z)`
  with :math:`k = (\\ell + 1/2)/\\chi`;
- :math:`C_\\ell^{hh}` — halo slab spectrum
  :math:`\\sum \\Delta\\chi\\, \\chi^2 P_{hh}(k, z) / V^2` plus shot noise
  :math:`1/\\bar n_h[{\\rm sr}]`;
- :math:`C_\\ell^{h\\Sigma}` — halo-matter cross,
  :math:`\\bar\\rho_m \\sum \\Delta\\chi\\, q_\\Sigma P_{h\\Sigma}(k, z)/V`;
- the shape-noise level
  :math:`\\sigma_\\gamma^2 \\langle\\Sigma_{\\rm crit}\\rangle^2 /
  n_{\\rm src}^{\\rm eff}`.

Halo spectra default to linear bias (:math:`b^2 P_{\\rm lin}`,
:math:`b P_{\\rm lin}`) and accept full halo-model overrides — e.g.
:class:`clenspy.clusters.BinHaloModelSpectra` — via ``pk_hh`` / ``pk_hm``.

All inputs are plain callables so any provider (live clenspy objects or
frozen snapshot tables) can drive it:

- ``chi(z)``: comoving distance [Mpc];
- ``pk_lin(k, z)``: linear matter power [Mpc^3], k in 1/Mpc;
- ``q_sigma(z_l, z_h)``: the Sigma-weighted lensing kernel;
- ``mean_sigma_crit(z_h)`` [Msun/Mpc^2], ``f_src_behind(z_h)``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["LimberProjector", "ARCMIN_TO_RAD"]

ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)


class LimberProjector:
    """Angular power spectra on a shared log-ell grid (default: 1000
    points/decade over [1e-1, 2e7], the Wu et al. 2019 convention)."""

    def __init__(
        self,
        *,
        chi: Callable,
        pk_lin: Callable,
        rho_mean0: float,
        q_sigma: Callable,
        mean_sigma_crit: Callable,
        f_src_behind: Callable,
        sigma_gamma: float,
        n_src_arcmin2: float,
        n_ell: int = 8000,
        ell_range: tuple[float, float] = (1e-1, 2e7),
    ) -> None:
        self.chi = chi
        self.pk_lin = pk_lin
        self.rho_mean0 = float(rho_mean0)
        self.q_sigma = q_sigma
        self.mean_sigma_crit = mean_sigma_crit
        self.f_src_behind = f_src_behind
        self.sigma_gamma = float(sigma_gamma)
        self.n_src_arcmin2 = float(n_src_arcmin2)
        self.ell = np.exp(
            np.linspace(np.log(ell_range[0]), np.log(ell_range[1]), n_ell)
        )

    # -- Sigma-Sigma ---------------------------------------------------
    def c_ell_sigma(self, zl_min: float, zl_max: float, z_h: float):
        """Sigma_crit-weighted matter C_ell (dz = 0.1 slab summation)."""
        nzl = max(int((zl_max - zl_min) / 0.1), 1)
        edges = np.linspace(zl_min, zl_max, nzl + 1)
        C = np.zeros_like(self.ell)
        for lo, hi in zip(edges[:-1], edges[1:]):
            z_mid = 0.5 * (lo + hi)
            chi_l = float(self.chi(z_mid))
            dchi = float(self.chi(hi) - self.chi(lo))
            kern = float(self.q_sigma(z_mid, z_h))
            k = (self.ell + 0.5) / chi_l
            C += dchi * kern**2 / chi_l**2 * self.pk_lin(k, z_mid)
        return C * self.rho_mean0**2

    def shape_noise_sigma(self, z_h: float) -> float:
        """sigma_gamma^2 <Sigma_crit>^2 / n_src^eff [sr]."""
        n_src_sr = (
            self.n_src_arcmin2 * self.f_src_behind(z_h) / ARCMIN_TO_RAD**2
        )
        return (
            self.sigma_gamma**2 / n_src_sr * self.mean_sigma_crit(z_h) ** 2
        )

    # -- halo-halo -------------------------------------------------------
    def c_ell_h(
        self, z_min: float, z_max: float, bias: float,
        counts: float, area_sr: float, pk_hh: Callable | None = None,
    ):
        """Halo slab C_ell and shot noise (= area_sr / counts).

        ``pk_hh(k, z)`` overrides the linear-bias ``bias^2 P_lin``.
        """
        nzh = max(int((z_max - z_min) / 0.1), 1)
        edges = np.linspace(z_min, z_max, nzh + 1)
        C = np.zeros_like(self.ell)
        vol = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            z_mid = 0.5 * (lo + hi)
            chi_h = float(self.chi(z_mid))
            dchi = float(self.chi(hi) - self.chi(lo))
            vol += dchi * chi_h**2
            k = (self.ell + 0.5) / chi_h
            if pk_hh is not None:
                p_term = pk_hh(k, z_mid)
            else:
                p_term = bias**2 * self.pk_lin(k, z_mid)
            C += dchi * chi_h**2 * p_term
        C /= vol**2
        return C, area_sr / counts

    # -- halo-Sigma cross ---------------------------------------------------
    def c_ell_h_sigma(
        self, z_min: float, z_max: float, bias: float, z_h: float,
        pk_hm: Callable | None = None,
    ):
        """Halo-Sigma cross C_ell.

        ``pk_hm(k, z)`` overrides the linear-bias ``bias * P_lin`` (e.g.
        the 2h + NFW-1h spectrum of the bin)."""
        nzh = max(int((z_max - z_min) / 0.1), 1)
        edges = np.linspace(z_min, z_max, nzh + 1)
        C = np.zeros_like(self.ell)
        vol = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            z_mid = 0.5 * (lo + hi)
            chi_h = float(self.chi(z_mid))
            dchi = float(self.chi(hi) - self.chi(lo))
            vol += dchi * chi_h**2
            k = (self.ell + 0.5) / chi_h
            kern = float(self.q_sigma(z_mid, z_h))
            if pk_hm is not None:
                p_term = pk_hm(k, z_mid)
            else:
                p_term = bias * self.pk_lin(k, z_mid)
            C += dchi * kern * p_term
        return C * self.rho_mean0 / vol
