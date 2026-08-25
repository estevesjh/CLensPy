"""Richness selection function S_ij(lnM, z) as a 2D interpolation table.

Implements the analytic reduction of
``RichnessSelection/docs/richness_selection_function.tex``: the 5-D
integral over :math:`(M, z^{\\rm tr}, \\lambda^{\\rm tr}, \\lambda^{\\rm ob},
z^{\\rm ob})` collapses to a 2-D table because the observed-space integrals
are closed forms (:mod:`clenspy.clusters.kernels`) and the
:math:`\\lambda^{\\rm tr}` integral is contracted once by fixed
Gauss-Legendre quadrature on feature-placed nodes:

.. math::

    S_{ij}(\\ln M, z) = \\mathcal{K}_j(z)\\, S_i(\\ln M, z), \\qquad
    S_i = \\int_0^\\infty d\\lambda^{\\rm tr}\\,
        \\mathcal{K}_i(\\lambda^{\\rm tr}, z)\\,
        P(\\lambda^{\\rm tr} \\mid M, z)

Algorithm ported from ``y3_cluster_cpp`` ``sel_function.py``: per-cell GL
bracket :math:`[\\max(0, \\mu_{\\rm eff} - L\\sigma_{\\rm eff}),\\,
\\mu_{\\rm eff} + L\\sigma_{\\rm eff}]`, kernel CDFs evaluated at the few
*unique* bin edges then differenced.

.. warning::

    ``n_lnm = 64`` hits a documented Gauss-Legendre / feature-placement
    resonance (~4.5% drift in the contracted counts).  Use 96/128/192.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.interpolate import RectBivariateSpline

from ..utils.gl import _leggauss_cached
from .kernels import K_j, RichnessKernel
from .mor import MassObservableRelation

__all__ = ["BinDefinition", "SelectionTable", "SelectionFunctionBuilder"]


@dataclass(frozen=True)
class BinDefinition:
    """One observed (richness, photo-z) selection bin.

    ``sigma_z`` is the photo-z scatter of the redshift kernel (may be
    richness-bin dependent); ``sigma_z = 0`` gives a top-hat in z.
    """

    lam_min: float
    lam_max: float
    zob_min: float
    zob_max: float
    sigma_z: float = 0.0


@dataclass
class SelectionTable:
    r"""Tabulated :math:`S_{ij}(\ln M, z)`, one plane per bin.

    Attributes
    ----------
    lnM : ndarray, shape (n_lnm,)
        Linear grid in ln(M/Msun) (physical mass).
    z : ndarray, shape (n_z,)
        Linear grid in true redshift.
    S : ndarray, shape (n_bins, n_lnm, n_z)
        Selection probability per bin.
    bins : tuple of BinDefinition
    """

    lnM: np.ndarray
    z: np.ndarray
    S: np.ndarray
    bins: tuple[BinDefinition, ...]
    _spl: list = field(default_factory=list, repr=False)

    @property
    def n_bins(self) -> int:
        return self.S.shape[0]

    def interpolator(self, b: int) -> RectBivariateSpline:
        """Lazy per-bin bilinear spline over ``(lnM, z)``."""
        if not self._spl:
            self._spl = [None] * self.n_bins
        if self._spl[b] is None:
            self._spl[b] = RectBivariateSpline(
                self.lnM, self.z, self.S[b], kx=1, ky=1
            )
        return self._spl[b]

    def __call__(self, b: int, lnM, z):
        """Interpolated S_ij at arbitrary broadcastable ``(lnM, z)``."""
        lnM_b, z_b = np.broadcast_arrays(
            np.asarray(lnM, dtype=float), np.asarray(z, dtype=float)
        )
        shape = lnM_b.shape
        out = self.interpolator(b).ev(lnM_b.ravel(), z_b.ravel())
        out = np.clip(out, 0.0, 1.0)
        return out.reshape(shape) if shape else float(out[0])


class SelectionFunctionBuilder:
    r"""Contract :math:`P(\lambda^{\rm tr} \mid M, z)` against the richness
    kernel into a :class:`SelectionTable`.

    Parameters
    ----------
    mor : MassObservableRelation
    kernel : RichnessKernel
    n_lnm, n_z : int
        Table grid (avoid 64 in lnM — GL resonance).
    n_q : int
        Gauss-Legendre order of the per-cell lambda_tr quadrature.
    L : float
        Bracket half-width in units of the MOR effective sigma.
    lnM_range, z_range : tuple of float
        Table envelope; ln(M/Msun) physical, true redshift.
    """

    def __init__(
        self,
        mor: MassObservableRelation,
        kernel: RichnessKernel,
        *,
        n_lnm: int = 128,
        n_z: int = 64,
        n_q: int = 32,
        L: float = 6.0,
        lnM_range: tuple[float, float] = (np.log(1e13 / 0.7), np.log(9e15 / 0.7)),
        z_range: tuple[float, float] = (0.05, 0.80),
    ) -> None:
        if n_lnm == 64:
            raise ValueError(
                "n_lnm=64 hits a documented GL/feature-placement resonance "
                "(~4.5% drift); use 96, 128 or 192"
            )
        self.mor = mor
        self.kernel = kernel
        self.n_lnm = int(n_lnm)
        self.n_z = int(n_z)
        self.n_q = int(n_q)
        self.L = float(L)
        self.lnM_range = lnM_range
        self.z_range = z_range

    # ------------------------------------------------------------------
    def s_i_pointwise(self, M, z, lam_min: float, lam_max: float):
        r"""Direct :math:`S_i(M, z)` for one bin (slow reference / testing).

        Same quadrature as the table builder, evaluated at scalar or
        1-D ``M`` with scalar ``z``.
        """
        M_arr = np.atleast_1d(np.asarray(M, dtype=float))
        t, w = _leggauss_cached(self.n_q)
        a, b = self.mor.bracket(M_arr, z, L=self.L)
        if getattr(self.kernel, "hard_edges", False):
            # step kernel: clip the bracket to the bin, integrate pdf alone
            a = np.clip(a, lam_min, lam_max)
            b = np.clip(b, lam_min, lam_max)
        half = 0.5 * (b - a)
        mid = 0.5 * (b + a)
        lam = mid[:, None] + half[:, None] * t[None, :]  # (NM, n_q)
        wq = half[:, None] * w[None, :]
        p = self.mor.pdf(lam, M_arr[:, None], z)
        if getattr(self.kernel, "hard_edges", False):
            out = np.sum(wq * p, axis=1)
        else:
            ki = self.kernel.K_i(lam, z, lam_min, lam_max)
            out = np.sum(wq * p * ki, axis=1)
        return float(out[0]) if np.ndim(M) == 0 else out

    # ------------------------------------------------------------------
    def build(self, bins: Sequence[BinDefinition]) -> SelectionTable:
        """Build the S_ij table for all bins on the shared grid."""
        bins = tuple(bins)
        lnM = np.linspace(self.lnM_range[0], self.lnM_range[1], self.n_lnm)
        z = np.linspace(self.z_range[0], self.z_range[1], self.n_z)
        M = np.exp(lnM)

        t, w = _leggauss_cached(self.n_q)
        a0, b0 = self.mor.bracket(M[:, None], z[None, :], L=self.L)  # (nM, nz)
        S = np.empty((len(bins), self.n_lnm, self.n_z))

        if getattr(self.kernel, "hard_edges", False):
            # step kernel: per-bin clipped brackets, integrate pdf alone
            for ib, bd in enumerate(bins):
                a = np.clip(a0, bd.lam_min, bd.lam_max)
                b = np.clip(b0, bd.lam_min, bd.lam_max)
                half = 0.5 * (b - a)
                mid = 0.5 * (b + a)
                lam = mid[..., None] + half[..., None] * t
                wq = half[..., None] * w
                p = self.mor.pdf(lam, M[:, None, None], z[None, :, None])
                s_i = np.sum(wq * p, axis=2)
                kj = K_j(z, bd.zob_min, bd.zob_max, bd.sigma_z)
                S[ib] = np.clip(s_i, 0.0, 1.0) * kj[None, :]
            return SelectionTable(lnM=lnM, z=z, S=S, bins=bins)

        # smooth kernel: shared per-cell GL nodes on the MOR support ----
        half = 0.5 * (b0 - a0)
        mid = 0.5 * (b0 + a0)
        lam = mid[..., None] + half[..., None] * t  # (nM, nz, nq)
        wq = half[..., None] * w  # (nM, nz, nq)

        p_hod = self.mor.pdf(lam, M[:, None, None], z[None, :, None])
        weighted = wq * p_hod  # (nM, nz, nq)

        # kernel CDFs at the unique richness edges, then difference ----
        edges = np.unique(
            np.concatenate([[bd.lam_min, bd.lam_max] for bd in bins])
        )
        z_bcast = z[None, :, None]
        E = np.empty((edges.size, self.n_lnm, self.n_z))
        for ie, edge in enumerate(edges):
            cdf = self.kernel.cdf(edge, lam, z_bcast)
            E[ie] = np.sum(weighted * cdf, axis=2)

        edge_index = {float(e): i for i, e in enumerate(edges)}
        for ib, bd in enumerate(bins):
            s_i = E[edge_index[float(bd.lam_max)]] - E[edge_index[float(bd.lam_min)]]
            kj = K_j(z, bd.zob_min, bd.zob_max, bd.sigma_z)  # (nz,)
            S[ib] = np.clip(s_i, 0.0, 1.0) * kj[None, :]

        return SelectionTable(lnM=lnM, z=z, S=S, bins=bins)
