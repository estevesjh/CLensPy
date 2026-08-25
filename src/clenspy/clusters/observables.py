"""Binned cluster observables: counts N_ij and stacked lensing profiles.

The operator pattern (pythonic ``NOperatorSel`` from y3_cluster_cpp): one
weight engine (:mod:`clenspy.clusters.weights`), many integrands.

.. math::

    N_i[f](R) = \\int d\\ln M \\int dz\\; \\Omega(z)\\,
        \\frac{dV}{dz\\,d\\Omega}\\, n(M, z)\\, S_{ij}(\\ln M, z)\\,
        f(R, \\ln M, z)

with :math:`f = 1` giving the counts, :math:`f = \\ln M` the mass moments,
:math:`f = b(M, z)` the effective bias, :math:`f = \\Delta\\Sigma_{\\rm 1h}`
the one-halo stack and
:math:`f = \\max[\\Delta\\Sigma_{\\rm 1h},\\, b\\,\\rho_{m,0}\\,
\\Delta\\Sigma_{\\rm 2h}]` the Hayashi & White (2008) max model
(``DeltaSigma_1h2hMax``).  The max couples lnM to z nonlinearly, so its
z-integral cannot be contracted — it consumes
:class:`~clenspy.clusters.weights.ZResolvedWeights`.

Stacked profiles are ratios of two contractions:
``numerator`` built with :math:`\\Sigma_{\\rm crit}^{-1}(z)` folded into the
z-weight (when provided), ``denominator`` the plain counts.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from ..config import DEFAULT_COSMOLOGY
from ..cosmology.pkgrid import PkGrid
from ..halo.mass_function import SigmaGrid, Tinker08MassFunction, Tinker10Bias
from ..halo.nfw import NfwProfile
from ..halo.twohalo import TwoHaloTerm
from .kernels import RichnessKernel
from .mor import MassObservableRelation
from .selection import BinDefinition, SelectionFunctionBuilder, SelectionTable
from .weights import (
    MassZWeights,
    ZResolvedWeights,
    build_mass_weights,
    build_zresolved_weights,
)

__all__ = [
    "duffy08_concentration",
    "DeltaSigma1hOperator",
    "DeltaSigmaMaxOperator",
    "BinnedClusterModel",
]


def duffy08_concentration(M, z, h: float):
    r"""Duffy et al. (2008) mean-density c(M, z) relation (M200m, full sample).

    .. math::

        c_{200m} = 10.14\, (M / 2\times 10^{12}\, h^{-1} M_\odot)^{-0.081}
                   (1 + z)^{-1.01}

    ``M`` in physical Msun.
    """
    M = np.asarray(M, dtype=float)
    return 10.14 * (M / (2e12 / h)) ** (-0.081) * (1.0 + np.asarray(z)) ** (-1.01)


class DeltaSigma1hOperator:
    r"""One-halo :math:`\Delta\Sigma_{\rm 1h}(R, \ln M)` on the GL mass nodes.

    Wraps the mass-vectorized :class:`~clenspy.halo.NfwProfile`; the matrix
    is built once per (nodes, concentration model) and cached.

    Parameters
    ----------
    lnm_x : ndarray, shape (n_lnm,)
        GL nodes in ln(M/Msun).
    z_eff : float
        Effective redshift for the concentration relation.
    cosmology : astropy cosmology
    concentration : float or callable ``c(M, z)``
        Fixed value or a relation such as :func:`duffy08_concentration`.
    """

    def __init__(
        self,
        lnm_x: np.ndarray,
        z_eff: float,
        cosmology=DEFAULT_COSMOLOGY,
        concentration: float | Callable = 4.0,
    ) -> None:
        self.lnm_x = np.asarray(lnm_x, dtype=float)
        M = np.exp(self.lnm_x)
        if callable(concentration):
            c200 = concentration(M, z_eff)
        else:
            c200 = np.full_like(M, float(concentration))
        self.nfw = NfwProfile(m200=M, c200=c200, cosmo=cosmology)
        self._cache: dict[bytes, np.ndarray] = {}

    def matrix(self, R) -> np.ndarray:
        r""":math:`\Delta\Sigma_{\rm 1h}` [Msun/Mpc^2], shape (n_lnm, n_R)."""
        R = np.atleast_1d(np.asarray(R, dtype=float))
        key = R.tobytes()
        if key not in self._cache:
            self._cache[key] = self.nfw.deltasigma(R)
        return self._cache[key]


class DeltaSigmaMaxOperator:
    r"""Hayashi & White (2008) pointwise-max 1h/2h combination.

    .. math::

        \Delta\Sigma_{\max}(R, M, z) = \max\!\left[
            \Delta\Sigma_{\rm 1h}(R, M),\;
            b(M, z)\, \rho_{m,0}\, \Delta\Sigma_{\rm hh}(R, z) \right]

    The max is nonlinear and the 2h term is z-dependent, so nothing
    commutes: the stack is a double GL sum over the z-resolved weights.

    Parameters
    ----------
    one_halo : DeltaSigma1hOperator
    twohalo : clenspy.halo.TwoHaloTerm
        Matter :math:`\Delta\Sigma_{\rm hh}` engine.  NOTE: its outputs are
        *not* premultiplied by the mean matter density — ``rho_m0`` is
        applied here.
    bias : object with ``at_lnM(lnM, z)``
        Halo bias model (Tinker10Bias, ConstantBias, ...).
    rho_m0 : float
        Comoving mean matter density [Msun/Mpc^3].
    """

    def __init__(
        self,
        one_halo: DeltaSigma1hOperator,
        twohalo: TwoHaloTerm,
        bias,
        rho_m0: float,
    ) -> None:
        self.one_halo = one_halo
        self.twohalo = twohalo
        self.bias = bias
        self.rho_m0 = float(rho_m0)
        self._two_cache: dict[bytes, np.ndarray] = {}
        self._bias_cache: dict[bytes, np.ndarray] = {}

    def _two_matrix(self, R: np.ndarray, z_x: np.ndarray) -> np.ndarray:
        """rho_m0 * DeltaSigma_hh(R, z) on the nodes, shape (n_R, n_z)."""
        key = R.tobytes() + z_x.tobytes()
        if key not in self._two_cache:
            two = np.empty((R.size, z_x.size))
            for q, zq in enumerate(z_x):
                two[:, q] = self.twohalo.deltasigma(R, float(zq))
            self._two_cache[key] = self.rho_m0 * two
        return self._two_cache[key]

    def _bias_matrix(self, lnm_x: np.ndarray, z_x: np.ndarray) -> np.ndarray:
        """b(lnM, z) on the node grid, shape (n_lnm, n_z)."""
        key = lnm_x.tobytes() + z_x.tobytes()
        if key not in self._bias_cache:
            self._bias_cache[key] = self.bias.at_lnM(
                lnm_x[:, None], z_x[None, :]
            )
        return self._bias_cache[key]

    def stack(self, R, w: ZResolvedWeights) -> np.ndarray:
        r"""Weighted :math:`N_i[\Delta\Sigma_{\max}]`, shape (n_bins, n_R).

        Divide by the counts norm for the stacked mean profile.
        """
        R = np.atleast_1d(np.asarray(R, dtype=float))
        one = self.one_halo.matrix(R)  # (n_lnm, n_R)
        two = self._two_matrix(R, w.z_x)  # (n_R, n_z)
        bkq = self._bias_matrix(w.lnm_x, w.z_x)  # (n_lnm, n_z)

        phi = np.maximum(
            one.T[:, :, None],  # (n_R, n_lnm, 1)
            bkq[None, :, :] * two[:, None, :],  # (n_R, n_lnm, n_z)
        )
        wkq = w.W2d * w.lnm_w[None, :, None]  # (n_bins, n_lnm, n_z)
        return np.einsum("rkq,bkq->br", phi, wkq)


class BinnedClusterModel:
    r"""Facade wiring cosmology, MOR, selection and weights into binned
    observables (lazy compute-and-cache-on-self).

    Parameters
    ----------
    pkgrid : PkGrid
        Linear P(k, z) grid; must cover the selection z-envelope.
    mor : MassObservableRelation
    kernel : RichnessKernel
    bins : sequence of BinDefinition
    cosmology : astropy cosmology
    omega_z : callable
        Survey solid angle Omega(z) [sr].
    sigma_crit_inv : callable, optional
        :math:`\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)` [Mpc^2/Msun];
        folded into the lensing z-weight when given.
    concentration : float or callable
        Concentration for the 1h profile.
    hmf, bias : optional overrides
        Any objects with ``at_lnM(lnM, z)``; default Tinker08 / Tinker10
        built on ``pkgrid``.
    """

    def __init__(
        self,
        *,
        pkgrid: PkGrid,
        mor: MassObservableRelation,
        kernel: RichnessKernel,
        bins: Sequence[BinDefinition],
        cosmology=DEFAULT_COSMOLOGY,
        omega_z: Callable,
        sigma_crit_inv: Callable | None = None,
        concentration: float | Callable = 4.0,
        hmf=None,
        bias=None,
        n_lnm: int = 96,
        n_z: int = 64,
        n_q: int = 32,
        sel_n_lnm: int = 128,
        sel_n_z: int = 64,
        lnM_range: tuple[float, float] | None = None,
        z_range: tuple[float, float] | None = None,
    ) -> None:
        self.pkgrid = pkgrid
        self.cosmo = cosmology
        self.mor = mor
        self.kernel = kernel
        self.bins = tuple(bins)
        self.omega_z = omega_z
        self.sigma_crit_inv = sigma_crit_inv
        self.concentration = concentration
        self.n_lnm, self.n_z, self.n_q = n_lnm, n_z, n_q
        self.sel_n_lnm, self.sel_n_z = sel_n_lnm, sel_n_z

        z_pad = max(3.0 * max((bd.sigma_z for bd in self.bins), default=0.0), 0.0)
        z_lo = max(min(bd.zob_min for bd in self.bins) - z_pad, 0.01)
        z_hi = max(bd.zob_max for bd in self.bins) + z_pad
        self._z_range = z_range if z_range is not None else (z_lo, z_hi)
        if self._z_range[1] > float(np.max(pkgrid.z)):
            raise ValueError(
                f"PkGrid z-range {np.max(pkgrid.z):.2f} does not cover the "
                f"selection envelope {self._z_range[1]:.2f}"
            )
        self._lnM_range = lnM_range
        self._hmf = hmf
        self._bias = bias

    # -- lazy ingredients ------------------------------------------------
    @property
    def sigma_grid(self) -> SigmaGrid:
        if not hasattr(self, "_sigma_grid"):
            self._sigma_grid = SigmaGrid(self.pkgrid, cosmo=self.cosmo)
        return self._sigma_grid

    @property
    def hmf(self):
        if self._hmf is None:
            self._hmf = Tinker08MassFunction(self.sigma_grid)
        return self._hmf

    @property
    def bias(self):
        if self._bias is None:
            self._bias = Tinker10Bias(self.sigma_grid)
        return self._bias

    @property
    def selection(self) -> SelectionTable:
        if not hasattr(self, "_selection"):
            kwargs = {}
            if self._lnM_range is not None:
                kwargs["lnM_range"] = self._lnM_range
            self._selection = SelectionFunctionBuilder(
                self.mor,
                self.kernel,
                n_lnm=self.sel_n_lnm,
                n_z=self.sel_n_z,
                n_q=self.n_q,
                z_range=self._z_range,
                **kwargs,
            ).build(self.bins)
        return self._selection

    @property
    def weights(self) -> MassZWeights:
        if not hasattr(self, "_weights"):
            self._weights = build_mass_weights(
                self.selection, self.hmf, self.cosmo, self.omega_z,
                n_lnm=self.n_lnm, n_z=self.n_z,
            )
        return self._weights

    @property
    def zweights(self) -> ZResolvedWeights:
        if not hasattr(self, "_zweights"):
            self._zweights = build_zresolved_weights(
                self.selection, self.hmf, self.cosmo, self.omega_z,
                n_lnm=self.n_lnm, n_z=self.n_z,
            )
        return self._zweights

    @property
    def lensing_zweights(self) -> ZResolvedWeights:
        """z-resolved weights with Sigma_crit^-1 folded in (if provided)."""
        if not hasattr(self, "_lensing_zweights"):
            self._lensing_zweights = build_zresolved_weights(
                self.selection, self.hmf, self.cosmo, self.omega_z,
                n_lnm=self.n_lnm, n_z=self.n_z,
                sigma_crit_inv=self.sigma_crit_inv,
            )
        return self._lensing_zweights

    @property
    def z_eff(self) -> float:
        """Midpoint of the integration z-envelope (concentration pivot)."""
        return 0.5 * (self._z_range[0] + self._z_range[1])

    # -- counts side -----------------------------------------------------
    def counts(self) -> np.ndarray:
        """N_ij per bin."""
        return self.weights.norm

    def mean_lnM(self) -> np.ndarray:
        """Population mean ln(M/Msun) per bin."""
        return self.weights.lnm_eff

    def mean_bias(self) -> np.ndarray:
        r"""S_ij-weighted effective bias :math:`\langle b \rangle_{ij}`."""
        w = self.zweights
        bkq = self.bias.at_lnM(w.lnm_x[:, None], w.z_x[None, :])
        num = np.einsum("bkq,kq,k->b", w.W2d, bkq, w.lnm_w)
        return num / self.counts()

    # -- lensing side ----------------------------------------------------
    @property
    def one_halo(self) -> DeltaSigma1hOperator:
        if not hasattr(self, "_one_halo"):
            self._one_halo = DeltaSigma1hOperator(
                self.weights.lnm_x, self.z_eff, self.cosmo, self.concentration
            )
        return self._one_halo

    @property
    def twohalo(self) -> TwoHaloTerm:
        if not hasattr(self, "_twohalo"):
            self._twohalo = TwoHaloTerm(
                self.pkgrid.k, self.pkgrid.pk, zvec=self.pkgrid.z
            )
        return self._twohalo

    def _stack_weights(self) -> tuple[ZResolvedWeights, np.ndarray]:
        """Weight tensor and its own norm — stacked means are weighted
        averages, so numerator and denominator share one tensor
        (Sigma_crit^-1-weighted when provided)."""
        w = self.lensing_zweights if self.sigma_crit_inv else self.zweights
        norm = np.einsum("bkq,k->b", w.W2d, w.lnm_w)
        return w, norm

    def stacked_deltasigma_1h(self, R) -> np.ndarray:
        r"""Stacked one-halo :math:`\langle\Delta\Sigma\rangle_{ij}(R)`
        [Msun/Mpc^2], shape (n_bins, n_R)."""
        w, norm = self._stack_weights()
        one = self.one_halo.matrix(R)  # (n_lnm, n_R)
        wk = np.einsum("bkq->bk", w.W2d) * w.lnm_w[None, :]
        num = wk @ one
        return num / norm[:, None]

    def stacked_deltasigma_max(self, R) -> np.ndarray:
        r"""Stacked :math:`\Delta\Sigma_{\rm 1h2hMax}` [Msun/Mpc^2],
        shape (n_bins, n_R)."""
        w, norm = self._stack_weights()
        op = DeltaSigmaMaxOperator(
            self.one_halo, self.twohalo, self.bias, self.sigma_grid.rho_m0
        )
        num = op.stack(R, w)
        return num / norm[:, None]
