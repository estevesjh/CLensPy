r"""The selection-affected halo bias :math:`b_{\rm sel}(\theta)`.

The paper's Section 4.1, and the piece that motivates the whole package: a
closed-form alternative to calibrating the redMaPPer selection effect on
Buzzard light-cones.

A cluster selected at observed richness :math:`\lambda^{\rm ob}`
preferentially sits behind extra line-of-sight structure, so the effective
bias of its two-halo term is not :math:`b(M,z)` but a
:math:`\theta`-dependent :math:`b_{\rm sel}` interpolating between two
plateaus:

.. math::
    b_{\rm sel}(\theta) = b_{\rm small}\left[1 - \sigma(\theta)\right]
                        + b_{\rm large}\,\sigma(\theta)

The plateaus come from a **closure**: averaging any quantity :math:`X`
against the projection kernel defines

.. math::
    \mathcal P[X] = \int dz\,\frac{dV}{dz\,d\Omega}\int dM\,n(M,z)
        \int d\lambda\,P(\lambda\mid M,z)\;
        2\pi\!\int d\theta\,\sin\theta\;
        w_z(z,z^{\rm ob})\,f_A(\theta,\lambda,z)\,\lambda\;X

with the three specialisations
:math:`P_1 = \mathcal P[1]`,
:math:`I_2 = \mathcal P[b\,\xi_{\rm NL}]`,
:math:`I_1 = \mathcal P[b\,\xi_{\rm NL}\,\sigma(\theta)]`, and then

.. math::
    \Delta^{\rm prj}_{\rm RND} = P_1 + b_{\rm eff}I_2,
    \qquad
    \delta^{\rm prj} = \frac{\lambda^{\rm ob}-\lambda^{\rm tr}}
        {\Delta^{\rm prj}_{\rm RND}} - 1

.. math::
    b_{\rm large} = b_{\rm eff}\left[1 + 0.13\,\delta^{\rm prj}\right],
    \qquad
    b_{\rm small} = \frac{(\lambda^{\rm ob}-\lambda^{\rm tr}) - P_1
        - b_{\rm large}I_1}{I_2 - I_1}

**The deliverable is two scalars per bin.** The
:math:`\lambda^{\rm tr}` marginalisation commutes with the sigmoid --
:math:`\sigma(\theta)` does not depend on :math:`\lambda^{\rm tr}` -- so
averaging the plateaus is exact rather than an approximation, and a whole
:math:`\theta` grid never has to be stored. `SelectionBiasTable` is
therefore two columns wide, matching the ``y3_cluster_cpp`` wall contract.

NOTE: **the 0.13 is empirical and Buzzard-calibrated.** It is the one
number in this closed-form model that is *not* closed-form -- an amplitude
fitted to simulations, exposed as `SelBiasEngine.boost_slope` so it can be
varied. The paper is explicit that this is the residual simulation
dependence; it is not hidden.

NOTE: **units.** Masses are **physical** :math:`M_\odot` here, not
:math:`h^{-1}M_\odot` -- the engine's ``hmf``, ``bias`` and ``mor``
callables are all in that convention, and the default mass range is the
RichnessSelection :math:`10^{13}` to :math:`10^{15.5}\,h^{-1}M_\odot`
converted once with ``h`` at construction. :math:`R_\lambda` is physical
Mpc, :math:`\chi` and :math:`\xi_{\rm NL}` are comoving Mpc, angles are
radians, :math:`b_{\rm sel}` is dimensionless. This differs from
`clenspy.selection.scaling_relation`, which is h-scaled; convert at the
boundary and see `hod_pdf_physical`.

NOTE: the photo-z weight is the **exact tabulated window**
(`clenspy.kernels.photoz.y3_photoz_window`), passed with
``n_sigma=1.0`` because that table already *is* the
:math:`n_\sigma\sigma_z` half-width. Its support is asymmetric by 17% at
:math:`z^{\rm ob} = 0.4`, so the line-of-sight bounds come from
`photoz_projection_support` and not from a symmetric
:math:`z^{\rm ob}\pm` width.

NOTE: three named approximations in the quadrature, all inherited from the
production engine and all reproduced deliberately:

- :math:`\theta` is integrated from the **exclusion boundary**
  :math:`\theta_{\rm excl}(z)` to :math:`2\theta_\lambda`, not from zero:
  neighbours closer than :math:`R_\lambda` along the line of sight are the
  cluster itself, and the paper excludes a **slab**, not a ball, because
  redMaPPer ranks membership in projected separation only.
- the :math:`z` grid is a **ring plus an outer split**: Gauss--Legendre
  across :math:`z^{\rm ob}\pm\Delta z_{\rm excl}`, then log-spaced in
  :math:`|\Delta\chi|` outwards on each side. A single grid either
  under-resolves the ring or wastes nodes in the tails.
- :math:`\xi_{\rm NL}` is **clipped at zero**, discarding the BAO trough.
  Measured at :math:`O(10^{-4})` in a :math:`w_z`-suppressed region, and
  it keeps :math:`I_1, I_2` positive so the closure cannot divide by a
  sign-indefinite denominator.

NOTE: :math:`b_{\rm small}` is obtained by **linear inversion**, so it is
the one output that can go unstable: when :math:`I_2 \to I_1` the
denominator vanishes. `_closure` falls back to :math:`b_{\rm large}` there
rather than returning an arbitrarily large number, which is a named
degradation and not a silent one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np

from ..cosmology.distances import comoving_volume_element
from ..cosmology.fiducial import fiducial_cosmology
from ..kernels.photoz import (
    photoz_projection,
    photoz_projection_support,
    y3_photoz_window,
)
from ..utils.integrate import gl_nodes, pk_to_xi_fftlog
from .geometry import area_overlap, r_lambda, sigmoid_theta, theta_lambda
from .richness_kernel import EmgParams, richness_pdf

__all__ = [
    "SigmoidBias",
    "SelectionBiasTable",
    "XiNL",
    "SelBiasEngine",
    "PhysicalMassMor",
]


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

    ``r`` in comoving Mpc. With ``clip=True`` (the engine's convention)
    negative BAO-trough values are clipped at zero — the effect is O(1e-4)
    in a w_z-suppressed region, and it keeps the closure's :math:`I_1, I_2`
    positive. `clenspy.lensing.projection.SigmaPrj` wants the signed
    :math:`\xi` instead (``clip=False``): the trough at
    :math:`r \simeq 100\!-\!150` comoving Mpc sits inside its
    line-of-sight window.
    """

    def __init__(self, pkgrid, r_range=(1e-2, 800.0), n_r: int = 600,
                 clip: bool = True) -> None:
        self.pkgrid = pkgrid
        self.rvals = np.logspace(
            np.log10(r_range[0]), np.log10(r_range[1]), n_r
        )
        self.clip = bool(clip)
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
        return np.maximum(out, 0.0) if self.clip else out


class PhysicalMassMor:
    r"""Adapter: an h-scaled MOR seen as a function of **physical** mass.

    `clenspy.selection.scaling_relation` works in
    :math:`\ln M` with :math:`M` in :math:`h^{-1}M_\odot`;
    `SelBiasEngine` works in physical :math:`M_\odot`. This wraps the
    former for the latter.

    NOTE: **the whole point of this class is that the conversion happens
    once, here, with the direction written down.**
    :math:`M[h^{-1}M_\odot] = M[M_\odot]\times h`, so the wrapped call is
    ``pdf(lam, ln(M * h), z)``. Getting it backwards scales every mass by
    :math:`h^2 \approx 2`, which shifts :math:`\langle b\rangle` by ~10%
    and would be very hard to spot in :math:`b_{\rm sel}` alone.

    Parameters
    ----------
    mor : LogNormalMor or HodMor
        Stored verbatim.
    h : float
        Hubble parameter.
    """

    def __init__(self, mor, h):
        self.mor = mor
        self.h = float(h)

    def pdf(self, lambda_true, mass_physical, z):
        r""":math:`P(\lambda^{\rm tr}\mid M, z)` at physical mass."""
        ln_mass_hinv = np.log(np.asarray(mass_physical, dtype=float) * self.h)
        return self.mor.pdf(lambda_true, ln_mass_hinv, z)

    def __repr__(self):
        return f"PhysicalMassMor({self.mor!r}, h={self.h:g})"


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
    plob_params : EmgParams, optional
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
        cosmology=None,
        xi_nl: Callable,
        hmf: Callable,
        bias: Callable,
        mor,
        plob_params: EmgParams | None = None,
        n_z: int = 48,
        n_M: int = 24,
        n_theta: int = 10,
        n_ltr: int = 60,
        ltr_grid_size: int = 16,
        min_mass: float | None = None,
        log10_M_max: float | None = None,
    ) -> None:
        cosmology = fiducial_cosmology() if cosmology is None else cosmology
        self.cosmo = cosmology
        self.h = cosmology.h
        self.xi_nl = xi_nl
        self.hmf = hmf
        self.bias = bias
        self.mor = mor
        self.plob = (plob_params if plob_params is not None
                     else EmgParams.from_y3_table())
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
        # the exact tabulated window; its table IS the n_sigma*sigma_z
        # half-width, hence n_sigma = 1.0 at every call site below
        self._window = y3_photoz_window()
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

        z_fg_lo, z_bg_hi = photoz_projection_support(
            zob, self._window, n_sigma=1.0
        )
        zs, wzs = self._z_grid(lob, zob, z_fg_lo, z_bg_hi)
        chi_z = self.chi(zs)
        dV = self._dv(zs)
        wz_kern = photoz_projection(
            zs, zob, self._window, n_sigma=1.0
        )

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

    def operators_var(self, lob: float, zob: float) -> tuple[float, float, float]:
        r"""The **variance** operators: (P1, I1, I2) with the squared
        weights :math:`\lambda^2`, :math:`w_z^2`, :math:`f_A^2` — the
        second moment of the projected-richness boost
        :math:`{\rm Var}[\Delta^{\rm prj}]_{\rm RND} \approx P_1^{(2)} +
        b_{\rm eff} I_2^{(2)}` (Costanzi notebook
        ``var_delta_prj_Beff``). Everything else mirrors `operators`."""
        key = ("ops_var", float(lob), float(zob))
        if key in self._cache:
            return self._cache[key]

        theta_lob = self._theta_lob(lob, zob)
        theta_max = 2.0 * theta_lob
        eps_theta = 1e-6
        chi_o = float(self.chi(zob))
        R_excl = r_lambda(lob, self.h) * (1.0 + zob)

        z_fg_lo, z_bg_hi = photoz_projection_support(
            zob, self._window, n_sigma=1.0
        )
        zs, wzs = self._z_grid(lob, zob, z_fg_lo, z_bg_hi)
        chi_z = self.chi(zs)
        dV = self._dv(zs)
        wz_kern = photoz_projection(zs, zob, self._window, n_sigma=1.0) ** 2

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
            fA = area_overlap(ths, theta_lob, theta_lam) ** 2  # squared

            ang_P1 = np.einsum("t,tL->L", th_weight, fA)
            ang_I2 = np.einsum("t,tL,t->L", th_weight, fA, xi)
            ang_I1 = np.einsum("t,t,tL,t->L", th_weight, sig, fA, xi)

            rho_pref = wz_kern[iz] * lam_grid**2               # lambda^2
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

    def delta_stats(self, lob: float, zob: float,
                    b_eff: float | None = None) -> tuple[float, float]:
        r"""(mean, variance) of the random-line-of-sight richness boost:
        :math:`\bar\Delta = P_1 + b_{\rm eff} I_2` and its second-moment
        analog from `operators_var`."""
        beff = self.b_eff(lob, zob) if b_eff is None else float(b_eff)
        P1, _, I2 = self.operators(lob, zob)
        P1v, _, I2v = self.operators_var(lob, zob)
        return P1 + beff * I2, P1v + beff * I2v

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

    def _ltr_weights(self, lob, zob, use_plob_ltr: bool = True,
                     plob_mode: str = "y3", b_eff: float | None = None):
        r"""(ltr nodes, normalised GL x P(ltr | lob, zob)) weights.

        ``plob_mode``:

        - ``"y3"`` — the vendored DES Y3 EMG kernel (`richness_pdf`).
        - ``"self"`` — the **self-consistent** exponential kernel of the
          Costanzi notebook (``plob_ltr``): the model's own boost
          statistics from `delta_stats` set
          :math:`\tau = 2\bar\Delta/(\bar\Delta^2 + {\rm Var}\Delta)`,
          :math:`f^{\rm prj} = \min(1, 2\bar\Delta^2/(\bar\Delta^2 +
          {\rm Var}\Delta))`, and
          :math:`P(\lambda^{\rm ob}\mid\lambda^{\rm tr}) = f^{\rm prj}
          \tau e^{-\tau(\lambda^{\rm ob}-\lambda^{\rm tr})}
          \Theta(\lambda^{\rm ob}-\lambda^{\rm tr}) +
          (1-f^{\rm prj})\,\delta_D(\lambda^{\rm ob}-\lambda^{\rm tr})`.
          Marginalising the closure under a kernel whose mean boost IS
          the closure's own :math:`\bar\Delta` keeps
          :math:`b_{\rm small}` stable; the Y3 kernel's longer tail
          (calibrated on SDSS injections, not on this halo model)
          overstates :math:`\langle\lambda^{\rm ob}-\lambda^{\rm tr}
          \rangle` against a mock and inflates the inner plateau.
        """
        t_nodes, t_wts = gl_nodes(1.0, 3.0 * float(lob), self.ltr_grid_size * 2)
        m_grid = np.logspace(np.log10(self.min_mass), self.log10_M_max, 50)
        hmf_m = self.hmf(m_grid, zob)

        def _prior(nodes):
            p_ltr_M = self.mor.pdf(nodes[:, None], m_grid[None, :], zob)
            return np.trapezoid(
                p_ltr_M * (hmf_m * m_grid)[None, :], np.log(m_grid), axis=1
            )

        prior = _prior(t_nodes)
        if not use_plob_ltr:
            weight = t_wts * prior
        elif plob_mode == "y3":
            p_lob_ltr = richness_pdf(float(lob), t_nodes, zob, self.plob)
            weight = t_wts * np.asarray(p_lob_ltr, dtype=float) * prior
        elif plob_mode == "self":
            mean_d, var_d = self.delta_stats(lob, zob, b_eff)
            tau = 2.0 * mean_d / (mean_d**2 + var_d)
            f_prj = min(1.0, 2.0 * mean_d**2 / (mean_d**2 + var_d))
            gap = float(lob) - t_nodes
            pdf = np.where(gap >= 0.0,
                           f_prj * tau * np.exp(-tau * gap), 0.0)
            weight = t_wts * pdf * prior
            # the (1 - f_prj) delta at lambda_tr = lambda_ob: one extra
            # node carrying that probability mass times the prior there
            t_nodes = np.append(t_nodes, float(lob))
            weight = np.append(
                weight, (1.0 - f_prj) * float(_prior(np.array([float(lob)]))[0])
            )
        else:
            raise ValueError(f"plob_mode must be 'y3' or 'self', got {plob_mode!r}")
        den = float(np.sum(weight))
        return t_nodes, (weight / den if den > 0 else np.full_like(weight, np.nan))

    def plateaus(
        self, lob: float, zob: float, use_plob_ltr: bool = True,
        b_eff: float | None = None, plob_mode: str = "y3",
    ) -> tuple[float, float]:
        """lambda_tr-marginalised (b_small, b_large) at one (lob, zob).

        ``b_eff=None`` uses the engine's own fixed-lambda_ob average
        (`b_eff`); pass a float to use an externally computed value, e.g.
        the bin-averaged N[b]/N[1] from
        `clenspy.observables.ClusterCounts.average`. ``plob_mode`` as in
        `_ltr_weights`: ``"y3"`` (EMG table) or ``"self"`` (the model's
        own boost statistics — the mock-consistent choice).
        """
        P1, I1, I2 = self.operators(lob, zob)
        beff = self.b_eff(lob, zob) if b_eff is None else float(b_eff)
        ltr, w_ltr = self._ltr_weights(lob, zob, use_plob_ltr,
                                       plob_mode=plob_mode, b_eff=beff)
        _, b_small_vec, b_large_vec = self._closure(lob, P1, I1, I2, beff, ltr)
        return float(np.sum(w_ltr * b_small_vec)), float(
            np.sum(w_ltr * b_large_vec)
        )

    def marginalised_bias(self, lob: float, zob: float,
                          use_plob_ltr: bool = True,
                          b_eff: float | None = None,
                          plob_mode: str = "y3") -> SigmoidBias:
        """The theta-callable b_sel(theta | lob, zob); ``b_eff`` and
        ``plob_mode`` as in `plateaus`."""
        b_small, b_large = self.plateaus(lob, zob, use_plob_ltr, b_eff,
                                         plob_mode)
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
        bins: Sequence,
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

if __name__ == "__main__":
    from ..cosmology.fiducial import fiducial_cosmology
    from .scaling_relation import HodMor

    cosmo = fiducial_cosmology()
    h = cosmo.h

    # analytic stand-ins for the halo model, in PHYSICAL Msun, so the demo
    # needs neither CAMB nor a sigma grid
    def hmf(mass, z):
        """dn/dM [Msun^-1 Mpc^-3], a smooth exponential-cutoff form."""
        m, zz = np.broadcast_arrays(np.asarray(mass, float),
                                    np.asarray(z, float))
        return 1e-19 * (m / 1e14) ** -2.0 * np.exp(-m / 5e14) / (1.0 + zz)

    def bias(mass, z):
        """b(M, z), rising with mass as it must."""
        m, zz = np.broadcast_arrays(np.asarray(mass, float),
                                    np.asarray(z, float))
        return 1.0 + 0.9 * (m / 3e14) ** 0.3 * (1.0 + zz) ** 0.5

    def xi_nl(r, zob):
        """A power-law correlation function, clipped at zero."""
        r = np.asarray(r, dtype=float)
        return np.maximum((np.maximum(r, 1e-3) / 5.0) ** -1.8, 0.0)

    engine = SelBiasEngine(
        cosmology=cosmo, xi_nl=xi_nl, hmf=hmf, bias=bias,
        mor=PhysicalMassMor(HodMor.des_y1(), h),
        n_z=32, n_M=16, n_theta=8, n_ltr=40, ltr_grid_size=10,
    )
    print(f"SelBiasEngine: boost_slope = {engine.boost_slope} "
          f"(the one Buzzard-calibrated number)")
    print(f"mass range {engine.min_mass:.3e} to "
          f"{10 ** engine.log10_M_max:.3e} physical Msun\n")

    lob, zob = 40.0, 0.4
    theta_lam = engine._theta_lob(lob, zob)
    print(f"lambda_ob = {lob}, z_ob = {zob}")
    print(f"  R_lambda    = {r_lambda(lob, h).item():.4f} Mpc physical")
    print(f"  theta_lambda= {theta_lam:.6f} rad "
          f"({np.degrees(theta_lam) * 60:.3f} arcmin)")

    p1, i1, i2 = engine.operators(lob, zob)
    print("\nthe three operators:")
    print(f"  P1 = P[1]              = {p1:.6e}")
    print(f"  I1 = P[b xi sigma]     = {i1:.6e}")
    print(f"  I2 = P[b xi]           = {i2:.6e}")
    print(f"  I2 - I1                = {i2 - i1:.6e}   <- the b_small")
    print("     denominator; b_small is a linear inversion and this is")
    print("     what can make it blow up.")

    b_eff = engine.b_eff(lob, zob)
    print(f"\nb_eff = {b_eff:.5f}   (the unselected aggregate)")

    b_small, b_large = engine.plateaus(lob, zob)
    ltr_nodes, ltr_w = engine._ltr_weights(lob, zob)
    delta, _, _ = engine._closure(lob, p1, i1, i2, b_eff, ltr_nodes)
    delta_mean = float(np.sum(ltr_w * delta))
    d_rnd = p1 + b_eff * i2
    print(f"b_small = {b_small:.5f}")
    print(f"b_large = {b_large:.5f}")
    print(f"  b_large/b_eff - 1 = {b_large / b_eff - 1:+.4%}  "
          "<- the 0.13 delta_prj boost")

    print("\n*** b_small here is NOT physical, and the reason is worth")
    print("    seeing, because it is the instability the module NOTE")
    print("    warns about. ***")
    print(f"  Delta_RND = P1 + b_eff I2 = {d_rnd:.5f}")
    print(f"  mean (lob - ltr)          = "
          f"{float(np.sum(ltr_w * (lob - ltr_nodes))):.5f}")
    print(f"  so delta_prj = excess/Delta_RND - 1 = {delta_mean:+.4f}")
    print("  A self-consistent halo model gives Delta_RND ~ the observed")
    print("  richness excess, hence delta_prj ~ 0 and b_small ~ b_large ~")
    print("  b_eff. These toy hmf / xi_nl are NOT mutually calibrated, so")
    print(f"  Delta_RND under-predicts the excess ~{delta_mean + 1:.1f}x and the")
    print(f"  linear inversion divides that by I2 - I1 = {i2 - i1:.2e},")
    print("  which is what inflates b_small. The shape of b_sel(theta)")
    print("  below is still correct; the amplitude needs a real P(k).")

    profile = engine.marginalised_bias(lob, zob)
    print(f"\nb_sel(theta), from {profile!r}:")
    print(f"{'theta/theta_lam':>16s}  {'b_sel':>9s}  {'sigmoid':>8s}")
    for frac in (0.0, 0.25, 0.5, 1.0, 2.0, 5.0):
        th = frac * theta_lam
        print(f"{frac:16.2f}  {profile(th):9.5f}  "
              f"{sigmoid_theta(th, theta_lam):8.5f}")
    print("  <- 0.5 of the way at theta = theta_lambda/2 by construction,")
    print("     and it tends to b_small inside the aperture and b_large")
    print("     well outside it.")

    # the identity that makes the two-scalar table exact
    print("\nthe lambda_tr marginalisation commutes with the sigmoid,")
    print("because sigma(theta) carries no lambda_tr. So averaging the")
    print("plateaus and then building the sigmoid equals building per-ltr")
    print("sigmoids and averaging them:")
    ltr, w = engine._ltr_weights(lob, zob)
    _, bs_vec, bl_vec = engine._closure(
        lob, p1, i1, i2, b_eff, ltr
    )
    for frac in (0.3, 1.0, 3.0):
        th = frac * theta_lam
        sig = sigmoid_theta(th, theta_lam)
        averaged_then_built = profile(th)
        built_then_averaged = float(np.sum(
            w * (bs_vec + (bl_vec - bs_vec) * sig)
        ))
        print(f"  theta/theta_lam = {frac:4.1f}: "
              f"{averaged_then_built:.10f} vs {built_then_averaged:.10f}  "
              f"rel {abs(averaged_then_built / built_then_averaged - 1):.1e}")

    print("\nthe photo-z window is the exact table, and asymmetric:")
    z_lo, z_hi = photoz_projection_support(zob, engine._window, n_sigma=1.0)
    print(f"  support ({z_lo:.5f}, {z_hi:.5f}) about z_ob = {zob}")
    print(f"  half-widths {zob - z_lo:.5f} below, {z_hi - zob:.5f} above "
          f"-> {abs((z_hi - zob) - (zob - z_lo)) / ((z_hi - z_lo) / 2):.1%} "
          "asymmetric")

    print("\na two-column table, one row per bin:")
    bins = [
        SimpleNamespace(lam_min=20.0, lam_max=30.0, zob_min=0.2, zob_max=0.35),
        SimpleNamespace(lam_min=30.0, lam_max=45.0, zob_min=0.2, zob_max=0.35),
        SimpleNamespace(lam_min=45.0, lam_max=60.0, zob_min=0.2, zob_max=0.35),
    ]
    table = engine.build_table(bins)
    print(f"{'lam bin':>12s}  {'lob':>7s}  {'b_small':>9s}  {'b_large':>9s}")
    for i in range(table.n_rows):
        print(f"{f'[{table.lam_min[i]:.0f},{table.lam_max[i]:.0f})':>12s}  "
              f"{table.lob[i]:7.2f}  {table.b_small[i]:9.5f}  "
              f"{table.b_large[i]:9.5f}")
    print("  <- two scalars per row, no theta grid: that is the whole")
    print("     point of the commuting marginalisation.")
