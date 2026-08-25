"""Fixed Gauss-Legendre weight engines for binned cluster observables.

Pythonic port of ``y3_cluster_cpp``'s ``SelGlWeights``
(``src/pipelines/shared/sel_gl_weights.hh``): the (lnM, z) population
integral against the halo mass function and the selection table is
performed once per parameter sample into reusable weight tensors, and
every observable becomes a cheap contraction of an integrand
:math:`f` against them:

.. math::

    N_i[f] = \\int d\\ln M \\int dz\\; \\Omega(z)\\,
        \\frac{dV}{dz\\,d\\Omega}\\, n(M, z)\\, S_{ij}(\\ln M, z)\\,
        f(\\ln M, z)

- ``MassZWeights`` — the z-contracted weight :math:`W_{ij}(\\ln M)`
  (valid whenever :math:`f` does not couple lnM to z), plus the free
  moments (norm = counts, mean lnM, second central moment).
- ``ZResolvedWeights`` — the z-resolved tensor ``W2d[b, k, q]`` needed
  by non-separable integrands such as the 1h/2h pointwise-max lensing
  model (:class:`clenspy.clusters.observables.DeltaSigmaMaxOperator`).

Units: physical (Msun, comoving Mpc); the weight carries
Mpc^3/sr x sr x Mpc^-3 = dimensionless, so ``norm`` is a raw count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..cosmology.utils import comoving_volume_element
from ..utils.gl import gl_nodes
from .selection import SelectionTable

__all__ = ["MassZWeights", "ZResolvedWeights", "build_mass_weights",
           "build_zresolved_weights"]


@dataclass
class MassZWeights:
    r"""z-contracted GL weights :math:`W_{ij}(\ln M)` and free moments.

    Attributes
    ----------
    lnm_x, lnm_w : ndarray, shape (n_lnm,)
        GL nodes/weights in ln(M/Msun).
    z_x, z_w : ndarray, shape (n_z,)
        GL nodes/weights in true redshift.
    W : ndarray, shape (n_bins, n_lnm)
        :math:`W[b, k] = \sum_q \mathrm{zfac}_q\, n(\ln M_k, z_q)\,
        S_b(\ln M_k, z_q)` (lnM weight NOT folded in).
    norm : ndarray, shape (n_bins,)
        :math:`\sum_k w_k W[b, k]` — the binned counts :math:`N_{ij}`
        when no :math:`\Sigma_{\rm crit}^{-1}` factor was folded in.
    lnm_eff, mu2 : ndarray, shape (n_bins,)
        Mean and second central moment of lnM under the weight.
    """

    lnm_x: np.ndarray
    lnm_w: np.ndarray
    z_x: np.ndarray
    z_w: np.ndarray
    W: np.ndarray
    norm: np.ndarray
    lnm_eff: np.ndarray
    mu2: np.ndarray

    @property
    def n_bins(self) -> int:
        return self.W.shape[0]

    def contract(self, f_vals: np.ndarray) -> np.ndarray:
        r""":math:`N_i[f] = \sum_k w_k\, W[b, k]\, f_{(b,)k}`.

        ``f_vals`` has shape ``(n_lnm,)``, ``(n_bins, n_lnm)`` or
        ``(..., n_lnm)`` (e.g. ``(n_R, n_lnm)``); returns
        ``(n_bins, ...)``.
        """
        f_vals = np.asarray(f_vals, dtype=float)
        wW = self.W * self.lnm_w[None, :]  # (n_bins, n_lnm)
        if f_vals.ndim == 1:
            return wW @ f_vals
        if f_vals.shape[0] == self.n_bins and f_vals.ndim == 2:
            return np.sum(wW * f_vals, axis=1)
        # generic trailing-lnM axis: (..., n_lnm) -> (n_bins, ...)
        return np.tensordot(wW, f_vals, axes=([1], [-1]))

    def expectation(self, f_vals: np.ndarray) -> np.ndarray:
        r"""Population mean :math:`N_i[f] / N_i[1]` per bin."""
        out = self.contract(f_vals)
        return out / self.norm.reshape((self.n_bins,) + (1,) * (out.ndim - 1))


@dataclass
class ZResolvedWeights:
    r"""z-resolved GL weights ``W2d[b, k, q]`` (invariant:
    ``W2d.sum(axis=2) == MassZWeights.W`` for the same inputs)."""

    lnm_x: np.ndarray
    lnm_w: np.ndarray
    z_x: np.ndarray
    z_w: np.ndarray
    W2d: np.ndarray  # (n_bins, n_lnm, n_z)

    @property
    def n_bins(self) -> int:
        return self.W2d.shape[0]


def _zfac(
    z_x: np.ndarray,
    z_w: np.ndarray,
    cosmology,
    omega_z: Callable,
    sigma_crit_inv: Callable | None,
) -> np.ndarray:
    """z-only factor: w_z * dV/dzdOmega [Mpc^3/sr] * Omega [sr]
    (* Sigma_crit^-1 [Mpc^2/Msun] for lensing weights)."""
    zfac = z_w * comoving_volume_element(z_x, cosmology) * np.asarray(
        omega_z(z_x), dtype=float
    )
    if sigma_crit_inv is not None:
        zfac = zfac * np.asarray(sigma_crit_inv(z_x), dtype=float)
    return zfac


def _hmf_sel_blocks(sel, hmf, lnm_x, z_x):
    """Evaluate hmf and selection on the (n_lnm, n_z) node grid."""
    lnm_grid = lnm_x[:, None]
    z_grid = z_x[None, :]
    n_block = hmf.at_lnM(lnm_grid, z_grid)  # (n_lnm, n_z)
    S_blocks = np.stack(
        [sel(b, lnm_grid, z_grid) for b in range(sel.n_bins)]
    )  # (n_bins, n_lnm, n_z)
    return n_block, S_blocks


def _resolve_ranges(sel, lnm_range, z_range):
    if lnm_range is None:
        lnm_range = (float(sel.lnM[0]), float(sel.lnM[-1]))
    if z_range is None:
        z_range = (float(sel.z[0]), float(sel.z[-1]))
    return lnm_range, z_range


def build_mass_weights(
    sel: SelectionTable,
    hmf,
    cosmology,
    omega_z: Callable,
    *,
    n_lnm: int = 96,
    n_z: int = 64,
    lnm_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
    sigma_crit_inv: Callable | None = None,
) -> MassZWeights:
    r"""Build the z-contracted weights (SelGlWeights ``build_weights``).

    Parameters
    ----------
    sel : SelectionTable
        S_ij(lnM, z) table (defines the default integration envelope).
    hmf : object with ``at_lnM(lnM, z)``
        Halo mass function dn/dlnM [comoving Mpc^-3].
    cosmology : astropy cosmology
    omega_z : callable
        Survey solid angle Omega(z) [sr].
    n_lnm, n_z : int
        GL orders (never 64 in lnM — resonance; enforced by the
        SelectionTable builder grid, re-checked here).
    sigma_crit_inv : callable, optional
        :math:`\langle\Sigma_{\rm crit}^{-1}\rangle(z)` [Mpc^2/Msun] to fold
        into the z weight (lensing numerator); omit for counts.
    """
    if n_lnm == 64:
        raise ValueError("n_lnm=64 hits the documented GL resonance; use 96+")
    lnm_range, z_range = _resolve_ranges(sel, lnm_range, z_range)
    lnm_x, lnm_w = gl_nodes(lnm_range[0], lnm_range[1], n_lnm)
    z_x, z_w = gl_nodes(z_range[0], z_range[1], n_z)

    zfac = _zfac(z_x, z_w, cosmology, omega_z, sigma_crit_inv)  # (n_z,)
    n_block, S_blocks = _hmf_sel_blocks(sel, hmf, lnm_x, z_x)

    W = np.einsum("q,kq,bkq->bk", zfac, n_block, S_blocks)
    norm = W @ lnm_w
    lnm_eff = (W * lnm_x[None, :]) @ lnm_w / norm
    mu2 = (
        (W * (lnm_x[None, :] - lnm_eff[:, None]) ** 2) @ lnm_w / norm
    )
    return MassZWeights(
        lnm_x=lnm_x, lnm_w=lnm_w, z_x=z_x, z_w=z_w,
        W=W, norm=norm, lnm_eff=lnm_eff, mu2=mu2,
    )


def build_zresolved_weights(
    sel: SelectionTable,
    hmf,
    cosmology,
    omega_z: Callable,
    *,
    n_lnm: int = 96,
    n_z: int = 64,
    lnm_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
    sigma_crit_inv: Callable | None = None,
) -> ZResolvedWeights:
    """Build the z-resolved weight tensor (for non-separable integrands)."""
    if n_lnm == 64:
        raise ValueError("n_lnm=64 hits the documented GL resonance; use 96+")
    lnm_range, z_range = _resolve_ranges(sel, lnm_range, z_range)
    lnm_x, lnm_w = gl_nodes(lnm_range[0], lnm_range[1], n_lnm)
    z_x, z_w = gl_nodes(z_range[0], z_range[1], n_z)

    zfac = _zfac(z_x, z_w, cosmology, omega_z, sigma_crit_inv)
    n_block, S_blocks = _hmf_sel_blocks(sel, hmf, lnm_x, z_x)

    W2d = zfac[None, None, :] * n_block[None, :, :] * S_blocks
    return ZResolvedWeights(
        lnm_x=lnm_x, lnm_w=lnm_w, z_x=z_x, z_w=z_w, W2d=W2d
    )
