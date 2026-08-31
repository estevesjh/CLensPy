r"""The selection-affected halo bias :math:`b_{\rm sel}(\theta)`.

The paper's Section 4.1: a closed-form alternative to calibrating the
redMaPPer selection effect on Buzzard light-cones. A cluster selected at
observed richness :math:`\lambda^{\rm ob}` sits behind extra correlated
structure, so its two-halo term carries a :math:`\theta`-dependent
:math:`b_{\rm sel}` -- two plateaus joined by a sigmoid, from a closure
of the projection kernel :math:`\mathcal P[X]`. Shares its line-of-sight
machinery with `clenspy.lensing.projection.SigmaPrj` (`_geometry`,
`clenspy.utils.los_integrals`).

NOTE: physical :math:`M_\odot`, comoving Mpc, h-free -- differs from
`clenspy.selection.scaling_relation` (h-scaled); `PhysicalMassMor`
converts at the boundary. Full derivation, units, and named
approximations: ``docs/selection_bias.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np

from ..cosmology.distances import ComovingDistance, comoving_volume_element
from ..cosmology.fiducial import fiducial_cosmology
from ..kernels.photoz import (
    photoz_chi_bounds,
    photoz_projection,
    photoz_projection_support,
    y3_photoz_window,
)
from ..utils.integrate import gl_nodes, mass_nodes, pk_to_xi_fftlog
from ..utils.los_integrals import LosGeometry, field_integrand, integrate_los
from .geometry import area_overlap, r_excl, r_lambda, sigmoid_theta, theta_lambda
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

    Shares its halo model with `clenspy.lensing.projection.SigmaPrj`
    rather than rebuilding it: pass an already-``build()``-ed `SigmaPrj`
    (or let this engine build a default one) and its ``hmf``/``bias``/
    ``xi_nl`` are reused as-is -- one cosmo -> Pk -> xi -> HMF/Bias chain,
    not two. `SigmaPrj` has no mass-observable-relation concept, so
    ``mor`` is still supplied here directly; the h-scaled
    `clenspy.selection.scaling_relation` convention is wrapped into the
    physical-mass one internally (`PhysicalMassMor`), so the caller never
    constructs that adapter by hand.

    Parameters
    ----------
    sigma_prj : SigmaPrj, optional
        Supplies ``cosmo``, ``h``, ``distance``, ``hmf``, ``bias``,
        ``xi_nl``. Built with its own defaults if omitted; built in place
        (mutating it) if given but not yet built.
    mor : MassObservableRelation
        The h-scaled P(lambda_tr | ln M, z) of `clenspy.selection.
        scaling_relation` (e.g. `HodMor`/`LogNormalMor`) -- wrapped in
        `PhysicalMassMor` here, once, with the direction written down.
    plob_params : EmgParams, optional
        P(lambda_ob | lambda_tr, z) for the lambda_tr marginalisation
        (default: vendored DES Y3 fit).
    n_z, n_M, n_theta, n_ltr, ltr_grid_size : int
        Quadrature orders (defaults: 96/48/96/90/24, all with margin over
        their measured convergence floor). ``n_z`` is the Gauss-Legendre
        order of the cosh-Abel ``u``-integral outside the exclusion
        sphere (`_operators`), not a redshift grid size.
        ``n_theta`` (not the paper's ``10``): with ``theta`` as the
        *outer* grid, each node aggregates the ``f_A(theta,lambda,z)``
        structure across the whole cosh-Abel ``z``-integral rather than
        resolving it per-``z``, so it needs more nodes for the same
        accuracy than the paper's per-``z`` theta split did. Checked
        against `validation/validate_bsel_quad.py`'s ``scipy.quad``
        reference at fixed ``n_z=48, n_M=24, n_ltr=60``:
        ``n_theta=10`` is 1.2-1.3% off on ``I1``/``I2``; ``n_theta=64``
        is sub-0.1%, flat through ``n_theta=120``. ``n_z`` and ``n_M``
        are already flat at their old (48/24) values (<0.01% out to
        3x); ``n_ltr`` is GL and converges *past* the quad reference's
        own dense-trapz floor (N=150) rather than matching it exactly,
        which is expected -- GL(60) already out-resolves trapz(150).
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
        sigma_prj=None,
        mor,
        plob_params: EmgParams | None = None,
        n_z: int = 96,
        n_M: int = 48,
        n_theta: int = 96,
        n_ltr: int = 90,
        ltr_grid_size: int = 24,
        min_mass: float | None = None,
        log10_M_max: float | None = None,
    ) -> None:
        from ..lensing.projection import SigmaPrj

        if sigma_prj is None:
            sigma_prj = SigmaPrj().build()
        elif not sigma_prj.is_built:
            sigma_prj.build()
        self.sigma_prj = sigma_prj
        self.cosmo = sigma_prj.cosmo
        self.h = sigma_prj.h
        # SigmaPrj's own hmf/bias/xi_nl, unmodified -- one built chain
        self.hmf = sigma_prj.hmf
        self.bias = sigma_prj.bias
        self.xi_nl = sigma_prj.xi_nl
        # M[h^-1 Msun] = M[Msun] x h, once, with the direction written down
        self.mor = PhysicalMassMor(mor, self.h)
        self.plob = (plob_params if plob_params is not None
                     else EmgParams.from_y3_table())
        self.n_z, self.n_M, self.n_theta = n_z, n_M, n_theta
        self.n_ltr, self.ltr_grid_size = n_ltr, ltr_grid_size
        self.min_mass = min_mass if min_mass is not None else 1.0e13 / self.h
        self.log10_M_max = (
            log10_M_max if log10_M_max is not None
            else np.log10(10.0**15.5 / self.h)
        )
        # fast comoving-distance interpolant [Mpc], shared with SigmaPrj
        self.distance = sigma_prj.distance
        # the exact tabulated window; its table IS the n_sigma*sigma_z
        # half-width, hence n_sigma = 1.0 at every call site below
        self._window = y3_photoz_window()
        self._cache: dict = {}
        self._d_cache: dict = {}  # I2 - I1, computed directly (stable)

    # -- helpers ---------------------------------------------------------
    def chi(self, z):
        return self.distance.chi(z)

    def _dv(self, z):
        return comoving_volume_element(z, self.cosmo)

    def _theta_lob(self, lob, zob) -> float:
        return float(theta_lambda(lob, zob, self.chi, self.h))

    def _geometry(self, thetas, lob: float, zob: float) -> LosGeometry:
        r"""Cosh-Abel LOS geometry for one (lob, zob) -- mirrors
        `clenspy.lensing.projection.SigmaPrj._geometry` exactly (same
        exclusion radius, same photo-z window); the one extracted LOS
        helper, matching that class's own style (everything else in
        `_operators` below is inline)."""
        chi_o = float(self.chi(zob))
        chi_min, chi_max = photoz_chi_bounds(zob, self._window, self.distance)
        return LosGeometry(thetas, chi_o, chi_min, chi_max,
                           r_excl=r_excl(lob, zob, self.h))

    # -- the P operator ---------------------------------------------------
    def _operators(self, lob: float, zob: float, squared: bool = False):
        r"""(P1, I1, I2, D) at one (lambda_ob, z_ob); ``D = I2 - I1 =
        P[b xi (1-sigma)]``, computed as its own direct quadrature rather
        than a subtraction of ``I2`` and ``I1`` -- see the module NOTE on
        why that subtraction is the one place ``b_small`` goes unstable.
        ``squared=True`` gives the variance operators of `operators_var`
        (weights :math:`\lambda^2`, :math:`w_z^2`, :math:`f_A^2`; ``D``
        is not needed there and is not returned meaningfully).

        Same recipe as `clenspy.lensing.projection.SigmaPrj.n_los_integral`:
        one `_geometry` call, everything else -- the mass/theta/lambda
        grids, the integrand closures -- built inline, exactly as that
        method does. The two integrands here are `n(M,z)` (bare) and
        `n(M,z) b(M,z) xi_NL` (`utils.los_integrals.field_integrand` plus
        bias/xi -- identical to `SigmaPrj`'s own `n_rnd_integrand`/
        `n_lss_integrand`), each additionally carrying the
        line-of-sight-coupled :math:`f_A(\theta,\lambda,z)` marginalised
        over :math:`\lambda^{\rm tr}` -- the one factor with no
        `SigmaPrj` analogue, because a projected halo's own
        :math:`\theta_\lambda(z)` genuinely depends on its redshift (no
        thin-window shortcut, unlike the mass-shell profile).
        """
        theta_lob = self._theta_lob(lob, zob)
        theta_max = 2.0 * theta_lob
        eps_theta = 1e-6

        thetas, w_theta = gl_nodes(eps_theta, theta_max, self.n_theta)
        w_theta = w_theta * 2.0 * np.pi * np.sin(thetas)
        sig_theta = sigmoid_theta(thetas, theta_lob, self.damping,
                                  self.theta0_frac)

        geometry = self._geometry(thetas, lob, zob)
        Ms, M_weight = mass_nodes(self.min_mass, 10.0**self.log10_M_max,
                                  self.n_M)
        lam, w_lam = gl_nodes(1e-6, float(lob), self.n_ltr)

        wz_power = 2.0 if squared else 1.0
        lam_power = 2.0 if squared else 1.0
        fA_power = 2.0 if squared else 1.0

        def common(z):
            return (self._dv(z)
                    * photoz_projection(z, zob, self._window, n_sigma=1.0)
                    ** wz_power)

        n_field = field_integrand(self.distance, self.hmf, common,
                                  Ms, M_weight)

        def weighted(with_bias_xi):
            def integrand(r, chi, theta_index):
                ## n(M,z), optionally x b(M,z) xi_NL(r) -- the two bare
                ## LOS/mass building blocks, shared with SigmaPrj
                base = n_field(r, chi, theta_index)         # (M, branch, u)
                if with_bias_xi:
                    z_flat = self.distance.z_of_chi(chi).ravel()
                    xi = np.maximum(self.xi_nl(r.ravel(), zob), 0.0
                                    ).reshape(r.shape)
                    b_M = self.bias(Ms, z_flat).reshape(Ms.size, *r.shape)
                    base = base * b_M * xi[None, :, :]

                ## the f_A(theta,lambda,z) piece, lambda-tr marginalised
                z = self.distance.z_of_chi(chi)              # (branch, u)
                theta_lam = (r_lambda(lam, self.h)[:, None, None]
                            * (1.0 + z)[None, :, :] / chi[None, :, :])
                theta_b = np.broadcast_to(
                    thetas[theta_index][None, :, None], theta_lam.shape)
                fA = area_overlap(theta_b, theta_lob, theta_lam) ** fA_power
                p_lmz = self.mor.pdf(lam[:, None, None, None],
                                     Ms[None, :, None, None],
                                     z[None, None, :, :])
                lam_weight = (lam[:, None, None] ** lam_power) * fA

                return base[None, :, :, :] * p_lmz * lam_weight[:, None, :, :]
            return integrand

        p1_los = integrate_los(geometry, weighted(False), self.n_z, "outside")
        bxi_los = integrate_los(geometry, weighted(True), self.n_z, "outside")

        def contract(los):
            ## sum lambda (w_lam) and M (already M-weighted inside n_field)
            return np.einsum("tLM,L->t", los, w_lam)

        per_theta_P1 = contract(p1_los)
        per_theta_bxi = contract(bxi_los)

        P1 = float(np.sum(w_theta * per_theta_P1))
        I2 = float(np.sum(w_theta * per_theta_bxi))
        I1 = float(np.sum(w_theta * sig_theta * per_theta_bxi))
        D = float(np.sum(w_theta * (1.0 - sig_theta) * per_theta_bxi))
        return P1, I1, I2, D

    def operators(self, lob: float, zob: float) -> tuple[float, float, float]:
        """(P1, I1, I2) at one (lambda_ob, z_ob)."""
        key = ("ops", float(lob), float(zob))
        if key in self._cache:
            return self._cache[key]
        P1, I1, I2, D = self._operators(lob, zob, squared=False)
        self._cache[key] = (P1, I1, I2)
        self._d_cache[key] = D
        return self._cache[key]

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
        P1, I1, I2, _ = self._operators(lob, zob, squared=True)
        self._cache[key] = (P1, I1, I2)
        return self._cache[key]

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
        Ms, M_weight = mass_nodes(self.min_mass, 10.0**self.log10_M_max,
                                  self.n_M)
        n_m = self.hmf(Ms, zob)
        b_m = self.bias(Ms, zob)
        P = self.mor.pdf(
            np.array([float(lob)])[:, None], Ms[None, :], zob
        ).ravel()
        wt = M_weight * n_m * P
        num = float(np.sum(wt * b_m))
        den = float(np.sum(wt))
        val = num / den if den > 0 else float("nan")
        self._cache[key] = val
        return val

    # -- closure + marginalisation ----------------------------------------
    def _closure(self, lob, P1, I1, I2, b_eff, ltr_vec, D=None):
        r"""(delta_prj, b_small, b_large) on an ltr array.

        ``D``, when given, is the directly-quadratured :math:`I_2-I_1 =
        P[b\,\xi_{\rm NL}(1-\sigma)]` (see `_operators`) used in place of
        the float subtraction ``I2 - I1`` -- the one place this closure
        can go unstable (module NOTE). ``None`` (the default, for direct
        callers with hand-picked numbers) falls back to the subtraction.
        """
        ltr_vec = np.asarray(ltr_vec, dtype=float)
        D_RND = P1 + b_eff * I2
        denom = (I2 - I1) if D is None else D
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

    def b_small_large(
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
        D = self._d_cache[("ops", float(lob), float(zob))]
        beff = self.b_eff(lob, zob) if b_eff is None else float(b_eff)
        ltr, w_ltr = self._ltr_weights(lob, zob, use_plob_ltr,
                                       plob_mode=plob_mode, b_eff=beff)
        _, b_small_vec, b_large_vec = self._closure(
            lob, P1, I1, I2, beff, ltr, D=D)
        return float(np.sum(w_ltr * b_small_vec)), float(
            np.sum(w_ltr * b_large_vec)
        )

    def marginalised_bias(self, lob: float, zob: float,
                          use_plob_ltr: bool = True,
                          b_eff: float | None = None,
                          plob_mode: str = "y3") -> SigmoidBias:
        """The theta-callable b_sel(theta | lob, zob); ``b_eff`` and
        ``plob_mode`` as in `b_small_large`."""
        b_small, b_large = self.b_small_large(lob, zob, use_plob_ltr, b_eff,
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
    from ..lensing.projection import SigmaPrj
    from .scaling_relation import HodMor

    cosmo = fiducial_cosmology()
    h = cosmo.h

    # the real halo model, once, shared with SigmaPrj -- Tinker (2008)
    # mass function, Tinker (2010) bias, CAMB halofit xi_NL. PkGrid
    # disk-caches the CAMB call, so this costs seconds once and
    # milliseconds after.
    prj = SigmaPrj(cosmology=cosmo).build()
    engine = SelBiasEngine(
        sigma_prj=prj, mor=HodMor.des_y1(),
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

    b_small, b_large = engine.b_small_large(lob, zob)
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
    print("  b_eff. HodMor.des_y1() is not calibrated against Tinker(2008)")
    print("  + Tinker(2010) at this (lob, zob), so")
    print(f"  Delta_RND under-predicts the excess ~{delta_mean + 1:.1f}x and the")
    print(f"  linear inversion divides that by I2 - I1 = {i2 - i1:.2e},")
    print("  which is what inflates b_small. The shape of b_sel(theta)")
    print("  below is still correct; the amplitude needs a mutually")
    print("  calibrated MOR (e.g. HodMor.from_lognormal()/.buzzard()).")

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
