"""Per-bin halo-model power spectra for the cluster covariance.

Builds, for one selection bin, the S_ij-weighted spectra consumed by the
Limber projection of the Gaussian :math:`\\Delta\\Sigma` covariance:

.. math::

    P_{hh}(k, z) = \\langle b \\rangle_S^2(z)\\, P_{\\rm lin}(k, z)
    \\qquad \\text{(2-halo; the hh 1-halo term is the shot noise)}

.. math::

    P_{h\\Sigma}(k, z) = \\langle b \\rangle_S(z)\\, P_{\\rm lin}(k, z)
        + \\Big\\langle \\frac{M}{\\bar\\rho_m}\\, \\tilde u(k \\mid M)
          \\Big\\rangle_S(z)

where :math:`\\tilde u(k|M)` is the mass-normalized (truncated) NFW
Fourier profile and :math:`\\langle\\cdot\\rangle_S(z)` the selection-
weighted population mean at each redshift node of the
:class:`~clenspy.clusters.weights.ZResolvedWeights` tensor.  The second
term is the halo-matter 1-halo contribution that upgrades
:math:`C_\\ell^{h\\Sigma}` beyond linear bias at small scales.

In the :math:`k \\to 0` limit :math:`\\tilde u \\to 1`, so the 1-halo term
tends to :math:`\\langle M \\rangle_S / \\bar\\rho_m` (a white shot-noise-
like plateau), and :math:`P_{h\\Sigma} \\to \\langle b\\rangle P_{\\rm lin}`
wherever the 2-halo term dominates.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline

from ..halo.nfw import NfwProfile
from .weights import ZResolvedWeights

__all__ = ["BinHaloModelSpectra"]


class BinHaloModelSpectra:
    """S_ij-weighted halo-model spectra for the bins of a weight tensor.

    Parameters
    ----------
    weights : ZResolvedWeights
        z-resolved selection weights (defines mass nodes, z nodes, bins).
    bias : object with ``at_lnM(lnM, z)``
        Halo bias model (Tinker10Bias, ConstantBias, ...).
    pk_lin : callable ``P(k, z)`` [Mpc^3]
        Linear matter power (e.g. a PkGrid).
    cosmology : astropy cosmology
        For the NFW profiles and the mean matter density.
    concentration : float or callable ``c(M, z_eff)``
    k_grid : ndarray, optional
        Table wavenumbers [1/Mpc] (default logspace 1e-4 .. 2e4).
    cross_model : {"additive", "max"}
        Composition of the halo-matter spectrum ``pk_hm``:

        - ``"additive"``: :math:`b P_{\\rm lin} + P_{1h}` (standard halo
          model);
        - ``"max"``: the Hayashi & White (2008) pointwise maximum in
          configuration space,
          :math:`\\xi_{hm}(r) = \\max[\\langle\\rho_{\\rm NFW}\\rangle_S/
          \\bar\\rho_m,\\; \\langle b\\rangle_S\\, \\xi_{\\rm lin}(r)]`,
          transformed back to :math:`P_{hm}(k)` with FFTLog — the same
          1h/2h composition used for the mean
          :math:`\\Delta\\Sigma_{\\rm 1h2hMax}` signal.
    """

    def __init__(
        self,
        weights: ZResolvedWeights,
        bias,
        pk_lin,
        cosmology,
        concentration: float | callable = 4.0,
        k_grid: np.ndarray | None = None,
        cross_model: str = "max",
    ) -> None:
        if cross_model not in ("additive", "max"):
            raise ValueError(f"unknown cross_model: {cross_model}")
        self.cross_model = cross_model
        self.w = weights
        self.pk_lin = pk_lin
        if k_grid is None:
            k_grid = np.logspace(-4, np.log10(2e4), 400)
        self.k_grid = np.asarray(k_grid, dtype=float)

        lnm_x, z_x = weights.lnm_x, weights.z_x
        M = np.exp(lnm_x)
        z_eff = float(np.median(z_x))
        if callable(concentration):
            c200 = concentration(M, z_eff)
        else:
            c200 = np.full_like(M, float(concentration))
        nfw = NfwProfile(m200=M, c200=c200, cosmo=cosmology)
        self.rho_m0 = float(nfw.rhom)

        # normalized per-bin, per-z weights over the mass nodes
        wkq = weights.W2d * weights.lnm_w[None, :, None]  # (nb, nk, nq)
        norm = wkq.sum(axis=1)  # (nb, nq)
        self._wnorm = wkq / np.clip(norm[:, None, :], 1e-300, None)
        # z nodes outside the bin's photo-z window have zero weight (0/0):
        # fill population means there from the nearest populated node so
        # the (k, z) interpolants stay sane near the window edges
        self._valid = norm > 1e-12 * np.max(norm, axis=1, keepdims=True)

        # <b>_S(z) per bin
        bkq = bias.at_lnM(lnm_x[:, None], z_x[None, :])  # (nk, nq)
        self.b_eff = np.einsum("bkq,kq->bq", self._wnorm, bkq)  # (nb, nq)
        for b in range(weights.n_bins):
            good = self._valid[b]
            if good.any() and not good.all():
                self.b_eff[b] = np.interp(
                    z_x, z_x[good], self.b_eff[b, good]
                )
                good_idx = np.where(good)[0]
                for qb in np.where(~good)[0]:
                    qn = good_idx[np.argmin(np.abs(z_x[good_idx] - z_x[qb]))]
                    self._wnorm[b, :, qb] = self._wnorm[b, :, qn]

        # <(M/rho_m) u(k|M)>_S(z) per bin: (nb, n_k, nq).
        # NfwProfile.fourier follows the pyccl convention rho~(k) = M u(k),
        # so dividing by rho_m0 gives (M/rho_m) u directly [Mpc^3].
        rho_tilde = nfw.fourier(self.k_grid)  # (n_halo, n_k) = M u(k)
        one_h = np.einsum("bkq,km->bmq", self._wnorm, rho_tilde / self.rho_m0)
        self._one_halo_tab = one_h  # (nb, n_k, nq)
        self._z_x = z_x
        self._spl_b = [None] * weights.n_bins
        self._spl_1h = [None] * weights.n_bins
        self._spl_max = [None] * weights.n_bins

        if self.cross_model == "max":
            self._build_max_tables(nfw)

    def _build_max_tables(self, nfw) -> None:
        """P_hm(k, z) from the Hayashi & White max composition in xi-space.

        xi_1h(r) = <rho_NFW(r|M)>_S / rho_m0 (the 1-halo halo-matter
        correlation), xi_2h(r, z) = <b>_S xi_lin(r, z); the pointwise max
        is FFTLog-transformed back to P_hm per z node.
        """
        import mcfit

        from ..utils.integrate import pk_to_xi_fftlog

        r_grid = np.logspace(-3, np.log10(500.0), 400)
        z_x = self._z_x
        # <rho_NFW(r)>_S(z) per bin: (nb, n_r, nq)
        rho_r = nfw.density(r_grid)  # (n_halo, n_r)
        xi_1h = np.einsum("bkq,km->bmq", self._wnorm, rho_r) / self.rho_m0

        # xi_lin(r, z) on the nodes
        k_int = self.k_grid
        xi_lin = np.stack(
            [
                pk_to_xi_fftlog(k_int, self.pk_lin(k_int, float(zq)), r_grid)
                for zq in z_x
            ],
            axis=1,
        )  # (n_r, nq)

        xi2p = mcfit.xi2P(r_grid, lowring=True)
        tabs = np.empty((self.n_bins, self.k_grid.size, z_x.size))
        for b in range(self.n_bins):
            for q in range(z_x.size):
                xi_max = np.maximum(xi_1h[b, :, q],
                                    self.b_eff[b, q] * xi_lin[:, q])
                k_out, p_out = xi2p(xi_max, extrap=True)
                tabs[b, :, q] = np.exp(
                    np.interp(np.log(self.k_grid), np.log(k_out),
                              np.log(np.clip(p_out, 1e-300, None)))
                )
        self._max_tab = tabs

    @property
    def n_bins(self) -> int:
        return self.w.n_bins

    def _b_of_z(self, b: int):
        z = self._z_x
        return lambda zz: np.interp(zz, z, self.b_eff[b])

    def _one_halo(self, b: int, k, z):
        if self._spl_1h[b] is None:
            self._spl_1h[b] = RectBivariateSpline(
                np.log(self.k_grid), self._z_x,
                np.log(np.clip(self._one_halo_tab[b], 1e-300, None)),
                kx=1, ky=1,
            )
        k_b, z_b = np.broadcast_arrays(
            np.asarray(k, dtype=float), np.asarray(z, dtype=float)
        )
        out = np.exp(
            self._spl_1h[b].ev(np.log(k_b).ravel(), z_b.ravel())
        ).reshape(k_b.shape)
        return out

    def pk_hh(self, b: int):
        r"""Callable ``P_hh(k, z)`` = :math:`\langle b\rangle_S^2 P_{\rm lin}`."""
        b_of_z = self._b_of_z(b)

        def pk(k, z):
            return b_of_z(z) ** 2 * self.pk_lin(k, z)

        return pk

    def _max_pk(self, b: int, k, z):
        if self._spl_max[b] is None:
            self._spl_max[b] = RectBivariateSpline(
                np.log(self.k_grid), self._z_x,
                np.log(np.clip(self._max_tab[b], 1e-300, None)),
                kx=1, ky=1,
            )
        k_b, z_b = np.broadcast_arrays(
            np.asarray(k, dtype=float), np.asarray(z, dtype=float)
        )
        return np.exp(
            self._spl_max[b].ev(np.log(k_b).ravel(), z_b.ravel())
        ).reshape(k_b.shape)

    def pk_hm(self, b: int):
        r"""Callable ``P_hSigma(k, z)``: additive 2h + NFW-1h, or the
        Hayashi & White max composition (``cross_model``)."""
        if self.cross_model == "max":
            return lambda k, z: self._max_pk(b, k, z)

        b_of_z = self._b_of_z(b)

        def pk(k, z):
            return b_of_z(z) * self.pk_lin(k, z) + self._one_halo(b, k, z)

        return pk
