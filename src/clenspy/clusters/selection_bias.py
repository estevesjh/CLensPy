"""Selection bias b_sel(theta | lambda_ob, z_ob) — Costanzi 2026 eqs. 3-9.

Full port of the RichnessSelection ``SelBias`` engine to clenspy
(physical units, astropy cosmology, PkGrid-backed nonlinear xi).

Model summary (``RichnessSelection/docs/richness_selection.tex``):

.. math::

    \\mathcal{P}[X] = \\int dz \\frac{dV}{dz\\,d\\Omega} \\int dM\\, n(M, z)
        \\int d\\lambda\\, P(\\lambda \\mid M, z)\\;
        2\\pi \\int d\\theta\\, \\sin\\theta\\;
        w_z(z, z^{\\rm ob})\\, f_A(\\theta, \\lambda, z)\\, \\lambda\\;
        X(M, z, \\theta)

with the three specializations :math:`P_1 = \\mathcal{P}[1]`,
:math:`I_2 = \\mathcal{P}[b\\,\\xi_{\\rm NL}]`,
:math:`I_1 = \\mathcal{P}[b\\,\\xi_{\\rm NL}\\,\\sigma(\\theta)]`; the closure

.. math::

    \\Delta^{\\rm prj}_{\\rm RND} = P_1 + b_{\\rm eff} I_2, \\qquad
    b_{\\rm large} = b_{\\rm eff}\\,[1 + 0.13\\,\\delta^{\\rm prj}], \\qquad
    b_{\\rm small} = \\frac{(\\lambda^{\\rm ob} - \\lambda^{\\rm tr})
        - P_1 - b_{\\rm large} I_1}{I_2 - I_1}

and the sigmoid profile with fixed shape constants
:math:`k = 2.5/\\theta_\\lambda`, :math:`\\theta_0 = \\theta_\\lambda/2`.
The :math:`\\lambda^{\\rm tr}` marginalisation commutes with the sigmoid
(exact algebra), so the deliverable is two scalars per
:math:`(\\lambda^{\\rm ob}, z^{\\rm ob})` row: :class:`SigmoidBias` /
:class:`SelectionBiasTable`.

Numerics ported verbatim: z-axis ring + outer fg/bg split (GL in
:math:`\\ln|\\Delta\\chi|`), theta-axis split at the exclusion boundary
:math:`\\theta_{\\rm excl}(z)`, :math:`\\xi_{\\rm NL}` clipped at zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..config import DEFAULT_COSMOLOGY
from ..cosmology.utils import comoving_volume_element
from ..utils.gl import gl_nodes
from ..utils.integrate import pk_to_xi_fftlog
from .geometry import area_overlap, r_lambda, sigmoid_theta, theta_lambda
from .kernels import EmgRichnessKernel
from .photoz import w_z, z_support
from .selection import BinDefinition

__all__ = ["SigmoidBias", "SelectionBiasTable", "XiNL", "SelBiasEngine"]


@dataclass(frozen=True)
class SigmoidBias:
    r"""Marginalised selection-bias profile at one
    :math:`(\lambda^{\rm ob}, z^{\rm ob})`:

    .. math::

        b_{\rm sel}(\theta) = b_{\rm small}\,[1 - \sigma(\theta)]
            + b_{\rm large}\,\sigma(\theta)
    """

    lob: float
    zob: float
    theta_lambda: float
    b_small: float
    b_large: float
    damping: float = 2.5
    theta0_frac: float = 0.5

    def __call__(self, theta):
        s = sigmoid_theta(theta, self.theta_lambda, self.damping,
                          self.theta0_frac)
        return self.b_small + (self.b_large - self.b_small) * s


@dataclass
class SelectionBiasTable:
    """b_sel plateaus per (richness bin, z_ob) row — 2 scalars per row,
    matching the y3_cluster_cpp wall contract (no theta grid stored)."""

    lam_min: np.ndarray
    lam_max: np.ndarray
    zo_low: np.ndarray
    zo_high: np.ndarray
    lob: np.ndarray
    zob: np.ndarray
    theta_lambda: np.ndarray
    b_small: np.ndarray
    b_large: np.ndarray

    @property
    def n_rows(self) -> int:
        return self.lob.size

    def row(self, i: int) -> SigmoidBias:
        return SigmoidBias(
            lob=float(self.lob[i]),
            zob=float(self.zob[i]),
            theta_lambda=float(self.theta_lambda[i]),
            b_small=float(self.b_small[i]),
            b_large=float(self.b_large[i]),
        )

    def to_file(self, path: str | Path) -> None:
        np.savez(
            path,
            **{k: getattr(self, k) for k in (
                "lam_min", "lam_max", "zo_low", "zo_high",
                "lob", "zob", "theta_lambda", "b_small", "b_large",
            )},
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "SelectionBiasTable":
        with np.load(path) as f:
            return cls(**{k: f[k] for k in f.files})


class XiNL:
    r"""Nonlinear matter correlation :math:`\xi_{\rm NL}(r, z^{\rm ob})`
    from a (nonlinear) PkGrid via FFTLog, cached per requested redshift.

    ``r`` in comoving Mpc; negative BAO-trough values are clipped at zero
    (the engine's convention — the effect is O(1e-4) in a w_z-suppressed
    region).
    """

    def __init__(self, pkgrid, r_range=(1e-2, 800.0), n_r: int = 600) -> None:
        self.pkgrid = pkgrid
        self.rvals = np.logspace(
            np.log10(r_range[0]), np.log10(r_range[1]), n_r
        )
        self._cache: dict[float, np.ndarray] = {}

    def __call__(self, r, zob: float):
        key = round(float(zob), 8)
        if key not in self._cache:
            pk = self.pkgrid(self.pkgrid.k, key)
            self._cache[key] = pk_to_xi_fftlog(
                self.pkgrid.k, pk, self.rvals
            )
        xi_tab = self._cache[key]
        r = np.asarray(r, dtype=float)
        out = np.interp(r, self.rvals, xi_tab, left=xi_tab[0], right=0.0)
        return np.maximum(out, 0.0)


class SelBiasEngine:
    """Costanzi 2026 selection-bias engine (see module docstring).

    Parameters
    ----------
    cosmology : astropy cosmology
    xi_nl : callable ``xi(r, zob)`` or XiNL
        Nonlinear matter correlation function, r in comoving Mpc.
    hmf : callable ``n(M, z)``
        dn/dM [Msun^-1 Mpc^-3 comoving], physical masses.
    bias : callable ``b(M, z)``
    mor : MassObservableRelation
        P(lambda_tr | M, z) of the projected (secondary) halos.
    plob_kernel : EmgRichnessKernel, optional
        P(lambda_ob | lambda_tr, z) for the lambda_tr marginalisation
        (default: vendored DES Y3 fit).
    n_z, n_M, n_theta, n_ltr, ltr_grid_size : int
        Quadrature orders (ported defaults: 48/24/10/60/16).
    min_mass, log10_M_max : float, optional
        Mass integration range in physical Msun; defaults are the
        RichnessSelection 1e13 - 10^15.5 Msun/h converted with h.
    """

    damping: float = 2.5
    theta0_frac: float = 0.5
    boost_slope: float = 0.13

    def __init__(
        self,
        *,
        cosmology=DEFAULT_COSMOLOGY,
        xi_nl: Callable,
        hmf: Callable,
        bias: Callable,
        mor,
        plob_kernel: EmgRichnessKernel | None = None,
        n_z: int = 48,
        n_M: int = 24,
        n_theta: int = 10,
        n_ltr: int = 60,
        ltr_grid_size: int = 16,
        min_mass: float | None = None,
        log10_M_max: float | None = None,
    ) -> None:
        self.cosmo = cosmology
        self.h = cosmology.h
        self.xi_nl = xi_nl
        self.hmf = hmf
        self.bias = bias
        self.mor = mor
        self.plob = plob_kernel if plob_kernel is not None else EmgRichnessKernel()
        self.n_z, self.n_M, self.n_theta = n_z, n_M, n_theta
        self.n_ltr, self.ltr_grid_size = n_ltr, ltr_grid_size
        self.min_mass = min_mass if min_mass is not None else 1.0e13 / self.h
        self.log10_M_max = (
            log10_M_max if log10_M_max is not None
            else np.log10(10.0**15.5 / self.h)
        )
        # fast comoving-distance interpolant [Mpc]
        self._zs_ref = np.linspace(1e-4, 2.0, 2000)
        self._chi_ref = cosmology.comoving_distance(self._zs_ref).to_value("Mpc")
        self._dchi_dz_ref = np.gradient(self._chi_ref, self._zs_ref)
        self._cache: dict = {}

    # -- helpers ---------------------------------------------------------
    def chi(self, z):
        return np.interp(np.asarray(z, dtype=float), self._zs_ref, self._chi_ref)

    def _dv(self, z):
        return comoving_volume_element(z, self.cosmo)

    def _theta_lob(self, lob, zob) -> float:
        return float(theta_lambda(lob, zob, self.chi, self.h))

    def _mass_nodes(self):
        lnMs, wM = gl_nodes(
            np.log(self.min_mass), np.log(10.0**self.log10_M_max), self.n_M
        )
        Ms = np.exp(lnMs)
        return Ms, wM * Ms  # nodes, dlnM-weight x M (so hmf = dn/dM)

    # -- z grid: ring + outer fg/bg split --------------------------------
    def _z_grid(self, lob, zob, z_fg_lo, z_bg_hi):
        chi_o = float(self.chi(zob))
        R_excl = r_lambda(lob, self.h) * (1.0 + zob)
        dchi_dz_at_zob = float(
            np.interp(zob, self._zs_ref, self._dchi_dz_ref)
        )
        dz_excl = R_excl / dchi_dz_at_zob

        n_ring = max(9, self.n_z // 4)
        z_ring, w_ring = gl_nodes(
            max(zob - dz_excl, z_fg_lo), min(zob + dz_excl, z_bg_hi), n_ring
        )

        n_outer = max(15, (self.n_z - n_ring) // 2)
        dis_fg_max = chi_o - float(self.chi(z_fg_lo))
        dis_bg_max = float(self.chi(z_bg_hi)) - chi_o

        def _outer(dis_max, sign):
            if R_excl >= dis_max:
                return np.array([]), np.array([])
            u, w_u = gl_nodes(np.log(R_excl), np.log(dis_max), n_outer)
            dis = np.exp(u)
            z_out = np.interp(chi_o + sign * dis, self._chi_ref, self._zs_ref)
            dchi_dz = np.interp(z_out, self._zs_ref, self._dchi_dz_ref)
            return z_out, w_u * dis / dchi_dz

        z_fg, w_fg = _outer(dis_fg_max, -1.0)
        z_bg, w_bg = _outer(dis_bg_max, +1.0)
        zs = np.concatenate([z_fg[::-1], z_ring, z_bg])
        wzs = np.concatenate([w_fg[::-1], w_ring, w_bg])
        return zs, wzs

    # -- the P operator ---------------------------------------------------
    def operators(self, lob: float, zob: float) -> tuple[float, float, float]:
        """(P1, I1, I2) at one (lambda_ob, z_ob)."""
        key = ("ops", float(lob), float(zob))
        if key in self._cache:
            return self._cache[key]

        theta_lob = self._theta_lob(lob, zob)
        theta_max = 2.0 * theta_lob
        eps_theta = 1e-6
        chi_o = float(self.chi(zob))
        R_excl = r_lambda(lob, self.h) * (1.0 + zob)

        z_fg_lo, z_bg_hi = z_support(zob)
        zs, wzs = self._z_grid(lob, zob, z_fg_lo, z_bg_hi)
        chi_z = self.chi(zs)
        dV = self._dv(zs)
        wz_kern = w_z(zs, zob)

        Ms, M_weight = self._mass_nodes()
        lam_grid, wlam = gl_nodes(1e-6, float(lob), self.n_ltr)

        cos_excl = np.clip(
            (chi_z**2 + chi_o**2 - R_excl**2) / (2.0 * chi_z * chi_o + 1e-30),
            -1.0, 1.0,
        )
        theta_excl_z = np.where(
            cos_excl >= 1.0 - 1e-12, eps_theta, np.arccos(cos_excl)
        )

        b_mz = self.bias(Ms[:, None], zs[None, :])
        n_mz = self.hmf(Ms[:, None], zs[None, :])
        p_lmz = self.mor.pdf(
            lam_grid[:, None, None], Ms[None, :, None], zs[None, None, :]
        )

        P1_z = np.zeros(zs.size)
        I1_z = np.zeros(zs.size)
        I2_z = np.zeros(zs.size)
        for iz in range(zs.size):
            th_lo = max(theta_excl_z[iz], eps_theta)
            if th_lo >= theta_max or wz_kern[iz] == 0.0:
                continue
            ths, wth = gl_nodes(th_lo, theta_max, self.n_theta)
            th_weight = wth * 2.0 * np.pi * np.sin(ths)
            sig = sigmoid_theta(ths, theta_lob, self.damping, self.theta0_frac)
            dchi = np.sqrt(np.maximum(
                chi_z[iz] ** 2 + chi_o**2
                - 2.0 * chi_z[iz] * chi_o * np.cos(ths), 0.0,
            ))
            xi = np.maximum(self.xi_nl(dchi, zob), 0.0)
            theta_lam = (
                r_lambda(lam_grid, self.h) * (1.0 + zs[iz]) / chi_z[iz]
            )
            fA = area_overlap(ths, theta_lob, theta_lam)  # (Nth, Nlam)

            ang_P1 = np.einsum("t,tL->L", th_weight, fA)
            ang_I2 = np.einsum("t,tL,t->L", th_weight, fA, xi)
            ang_I1 = np.einsum("t,t,tL,t->L", th_weight, sig, fA, xi)

            rho_pref = wz_kern[iz] * lam_grid
            p_lm = p_lmz[:, :, iz]
            lam_P1 = np.einsum("L,LM,L->M", wlam, p_lm, rho_pref * ang_P1)
            lam_I2 = np.einsum("L,LM,L->M", wlam, p_lm, rho_pref * ang_I2)
            lam_I1 = np.einsum("L,LM,L->M", wlam, p_lm, rho_pref * ang_I1)

            P1_z[iz] = np.sum(M_weight * n_mz[:, iz] * lam_P1)
            I2_z[iz] = np.sum(M_weight * n_mz[:, iz] * b_mz[:, iz] * lam_I2)
            I1_z[iz] = np.sum(M_weight * n_mz[:, iz] * b_mz[:, iz] * lam_I1)

        out = (
            float(np.sum(wzs * dV * P1_z)),
            float(np.sum(wzs * dV * I1_z)),
            float(np.sum(wzs * dV * I2_z)),
        )
        self._cache[key] = out
        return out

    # -- b_eff ------------------------------------------------------------
    def b_eff(self, lob: float, zob: float) -> float:
        r""":math:`b_{\rm eff} = \langle b(M, z^{\rm ob})\rangle_{P(M \mid
        \lambda^{\rm ob})}` with weight :math:`n(M) P(\lambda^{\rm ob} \mid
        M) M` in dlnM."""
        key = ("b_eff", float(lob), float(zob))
        if key in self._cache:
            return self._cache[key]
        m_grid = np.logspace(
            np.log10(self.min_mass), self.log10_M_max, 100
        )
        n_m = self.hmf(m_grid, zob)
        b_m = self.bias(m_grid, zob)
        P = self.mor.pdf(
            np.array([float(lob)])[:, None], m_grid[None, :], zob
        ).ravel()
        wt = n_m * P * m_grid
        num = np.trapezoid(wt * b_m, np.log(m_grid))
        den = np.trapezoid(wt, np.log(m_grid))
        val = float(num / den) if den > 0 else float("nan")
        self._cache[key] = val
        return val

    # -- closure + marginalisation ----------------------------------------
    def _closure(self, lob, P1, I1, I2, b_eff, ltr_vec):
        """(delta_prj, b_small, b_large) on an ltr array."""
        ltr_vec = np.asarray(ltr_vec, dtype=float)
        D_RND = P1 + b_eff * I2
        denom = I2 - I1
        delta = (lob - ltr_vec) / D_RND - 1.0
        b_large = b_eff * (1.0 + self.boost_slope * delta)
        if abs(denom) < 1e-12 * (abs(I1) + abs(I2) + 1e-30):
            b_small = b_large.copy()
        else:
            b_small = ((lob - ltr_vec) - P1 - b_large * I1) / denom
        return delta, b_small, b_large

    def _ltr_weights(self, lob, zob, use_plob_ltr: bool = True):
        """(ltr nodes, normalised GL x P(ltr | lob, zob)) weights."""
        t_nodes, t_wts = gl_nodes(1.0, 3.0 * float(lob), self.ltr_grid_size * 2)
        m_grid = np.logspace(np.log10(self.min_mass), self.log10_M_max, 50)
        hmf_m = self.hmf(m_grid, zob)
        p_ltr_M = self.mor.pdf(t_nodes[:, None], m_grid[None, :], zob)
        prior = np.trapezoid(
            p_ltr_M * (hmf_m * m_grid)[None, :], np.log(m_grid), axis=1
        )
        if use_plob_ltr:
            p_lob_ltr = self.plob.pdf_lob(float(lob), t_nodes, zob)
            p_ltr = np.asarray(p_lob_ltr, dtype=float) * prior
        else:
            p_ltr = prior
        weight = t_wts * p_ltr
        den = float(np.sum(weight))
        return t_nodes, (weight / den if den > 0 else np.full_like(weight, np.nan))

    def plateaus(
        self, lob: float, zob: float, use_plob_ltr: bool = True
    ) -> tuple[float, float]:
        """lambda_tr-marginalised (b_small, b_large) at one (lob, zob)."""
        P1, I1, I2 = self.operators(lob, zob)
        beff = self.b_eff(lob, zob)
        ltr, w_ltr = self._ltr_weights(lob, zob, use_plob_ltr)
        _, b_small_vec, b_large_vec = self._closure(lob, P1, I1, I2, beff, ltr)
        return float(np.sum(w_ltr * b_small_vec)), float(
            np.sum(w_ltr * b_large_vec)
        )

    def marginalised_bias(self, lob: float, zob: float,
                          use_plob_ltr: bool = True) -> SigmoidBias:
        """The theta-callable b_sel(theta | lob, zob)."""
        b_small, b_large = self.plateaus(lob, zob, use_plob_ltr)
        return SigmoidBias(
            lob=float(lob),
            zob=float(zob),
            theta_lambda=self._theta_lob(lob, zob),
            b_small=b_small,
            b_large=b_large,
            damping=self.damping,
            theta0_frac=self.theta0_frac,
        )

    # -- table builder -----------------------------------------------------
    def build_table(
        self,
        bins: Sequence[BinDefinition],
        lob_per_bin: Sequence[float] | None = None,
    ) -> SelectionBiasTable:
        """One SigmoidBias row per bin, evaluated at ``lob_per_bin``
        (default: geometric mean of the richness edges) and the z-bin
        midpoint."""
        bins = tuple(bins)
        if lob_per_bin is None:
            lob_per_bin = [
                float(np.sqrt(bd.lam_min * min(bd.lam_max, 3.0 * bd.lam_min)))
                for bd in bins
            ]
        rows = [
            self.marginalised_bias(
                lob, 0.5 * (bd.zob_min + bd.zob_max)
            )
            for bd, lob in zip(bins, lob_per_bin)
        ]
        return SelectionBiasTable(
            lam_min=np.array([bd.lam_min for bd in bins]),
            lam_max=np.array([bd.lam_max for bd in bins]),
            zo_low=np.array([bd.zob_min for bd in bins]),
            zo_high=np.array([bd.zob_max for bd in bins]),
            lob=np.array([r.lob for r in rows]),
            zob=np.array([r.zob for r in rows]),
            theta_lambda=np.array([r.theta_lambda for r in rows]),
            b_small=np.array([r.b_small for r in rows]),
            b_large=np.array([r.b_large for r in rows]),
        )
