r"""Projection lensing: :math:`\Sigma_{\rm prj}` and :math:`\Delta\Sigma_{\rm prj}`.

The two-halo projected surface density around a richness-selected cluster
(Costanzi 2026 eq. 13; the excess-density formulation of
``y3_cluster_cpp/review/08282026/sigma_prj_excess_equations.md``, whose
equation numbers are cited below; ``docs/refactor-plan.md`` errata E.3,
and ``docs/notation.md`` section 5). **The two-halo term is the
correlated excess above the mean matter column — there is no background
in it** (cluster_toolkit's :math:`\Sigma_{2h}` convention, and what a
random-point-subtracted measurement contains). Halo exclusion acts on
the **complete pair distribution**: with the ball indicator
:math:`E = \mathbb 1[|d\chi| > R_{\rm excl}]` on the exact chord, the
pair weight of the excess observable is (note eqs. 7–8)

.. math::
    \mathcal K_{\rm exc} = \big(1 + b\,b_{\rm sel}\,\xi_{\rm NL}\big)E - 1
    = \begin{cases}
        b\,b_{\rm sel}\,\xi_{\rm NL} & \text{outside the ball,}\\
        -1 & \text{inside,}
      \end{cases}

and the master equation (note eq. 14) is

.. math::
    \Sigma_{\rm prj}(R) = \int d\theta\, 2\pi\sin\theta
        \int dz\;{\rm common}(z) \int dM\; n(M, z)\,
        \mathcal K_{\rm exc}\,
        \Sigma_{\rm mis}(R, \theta\chi_o \mid M),
    \qquad
    {\rm common}(z) = \frac{dV}{d\Omega\,dz}(z)\; w_{pz}(z; z^{\rm ob}).

In channel-weight language: **outside the ball the correlated weight is
w_cl; inside the ball it is −w_rnd** — minus the background integrand,
carrying no bias, no b_sel, no xi (certainty of absence, not
clustering). :math:`\Delta\Sigma_{\rm prj}` is the same assembly — same
:math:`\mathcal K_{\rm exc}`, counterterm included — with the kernel
swap :math:`\Sigma_{\rm mis} \to \Delta\Sigma_{\rm mis}` (note eq. 22):
the excess functional acts only on the radial argument and commutes with
the outer :math:`(\theta, z, M)` integrals.

A raw projected *mass map* (e.g. the Costanzi mock's per-halo columns)
additionally contains the mean background column — the ``1`` of the
halo-model bracket :math:`[1 + b\,b_{\rm sel}\,\xi_{\rm NL}]`. That piece
is kept available as the separate ``rnd`` channel,

.. math::
    \Sigma_{\rm bkg}(R) = \int d\theta\, 2\pi\sin\theta
        \sum_M w_{\rm rnd}(\theta, M)\,
        \Sigma_{\rm mis}(R, \theta\chi_o \mid M),
    \qquad
    w_{\rm rnd} = \int dz\;{\rm common}(z)\, n(M, z)\, m_{\rm rnd},

returned only on request (``channel="sum"``/``"rnd"``); it is
near-uniform in :math:`R`, cancels against the random-point subtraction
in :math:`\Sigma`, and is annihilated exactly by the excess functional in
:math:`\Delta\Sigma`.

The traps, all named in E.3 and all honoured here:

- the measure is :math:`2\pi\sin\theta\,d\theta` — solid angle on the
  sphere. **No Limber approximation, no Bessel transform.**
- :math:`|d\chi|` and :math:`\theta_{\rm excl}(z)` are **law-of-cosines**
  (:math:`d\chi^2 = \chi_z^2 + \chi_o^2 - 2\chi_z\chi_o\cos\theta`), never
  :math:`|\chi_z - \chi_o|`: near the exclusion ring the transverse leg
  dominates.
- the photo-z weight is the **parabolic** projection kernel
  (`clenspy.kernels.photoz.photoz_projection`), not the Gaussian counts
  kernel.
- :math:`b_{\rm sel}(\theta)` multiplies the **correlated channel only**.
- the ``rnd`` and ``cl`` channels are stored separately on ``self`` and
  summed at the end — the scientific argument is about which dominates
  where.

Halo exclusion comes in three named semantics (plus ``"none"``):

- ``"counter"`` (default — the :math:`\mathcal K_{\rm exc}` pair weight
  above; the Costanzi notebook convention
  ``bM_bsel_xi[dis < R_excl] = -1``): inside the 3-D chord ball the
  correlated weight is :math:`-w_{\rm rnd}` — minus the background
  integrand — cancelling the background's :math:`+1` exactly. The total
  vanishes in the ball, the background stays **strictly uniform**, and
  the exclusion hole lives in the cl channel — where a
  random-point-subtracted measurement keeps it.
- ``"cl"`` (E.3 / legacy slab): :math:`m_{\rm cl} =
  \mathbb 1[|d\chi| > R_{\rm excl}]`, :math:`m_{\rm rnd} = 1` — the
  correlated term is zeroed. **No counterterm: not halo exclusion in
  the pair distribution** (note eq. 16 with the eq. 17 term dropped);
  differs from ``"counter"`` by :math:`\lesssim 0.6\%` of the summed
  profile at :math:`R \to 0`, gone by :math:`R \approx 2` cMpc.
- ``"ball"``: both channels removed inside the chord ball — the same
  total as ``"counter"``, but the hole is booked in the background.

with :math:`R_{\rm excl} = R_\lambda(\lambda^{\rm ob})(1 + z^{\rm ob})`
comoving. Two named approximations in the exclusion kernel, both the
Costanzi effective prescription: :math:`R_{\rm excl}` is the richness
aperture, **independent of the neighbour's mass** (physical hard-sphere
exclusion would use :math:`R_\Delta(M_{\rm cls}) + R_\Delta(M)`, García
et al. 2021), and it is evaluated at the single :math:`(\lambda^{\rm
ob}, z^{\rm ob})` point rather than averaged over
:math:`p(M_{\rm cls} \mid \lambda^{\rm ob})` (excess-equations note
eq. 24). The exclusion boundary itself is a curve in the
:math:`(\theta, z)` plane; `theta_edges` aligns the cell edges with
every z node's exact crossing angle, so no kernel cell straddles the
indicator (see its docstring).

NOTE: **the rnd channel is the selected-halo background column, not the
cosmological mean-matter column.** It is the mean column of the modelled
halo population — mass-restricted to ``min_mass``..``log10_M_max`` and
dressed with *untruncated* NFW wings unless ``r_trunc`` is set — so it
depends on the transverse aperture and carries only the halo-budget
share of :math:`\bar\rho_m\times 2\,{\rm depth}` (≈ 0.2–0.4 for the
default mass cut). That is exactly the mock's background (mock matter
*is* those halos); it is **not** :math:`\bar\rho_m` — a full-matter
closure is a separate, not-yet-implemented mode.

NOTE: **units.** Physical :math:`M_\odot`, comoving Mpc, h-free — the
`clenspy.selection.bsel.SelBiasEngine` convention. ``hmf`` is
:math:`dn/dM` in :math:`M_\odot^{-1}\,{\rm Mpc}^{-3}` comoving at physical
mass; :math:`\Sigma_{\rm prj}` comes out in :math:`M_\odot\,{\rm Mpc}^{-2}`
**comoving**. Mock catalogues in :math:`h`-scaled units convert at the
caller's boundary, not here.

NOTE: **the comoving/physical** :math:`(1+z)^2` **enters through the
miscentering table's density scale** :math:`\rho_{\rm def}`. The table
itself is dimensionless; its runtime prefactor is
:math:`\Sigma_0 = 2 r_s \rho_s`, and :math:`\rho_s \propto \rho_{\rm
def}`. This module passes the comoving :math:`\bar\rho_m =
\Omega_m\rho_{c,0}` (no :math:`(1+z)^3`), so :math:`r_s` and
:math:`\Sigma` are comoving — matching the Costanzi mock's columns. The
``y3_cluster_cpp`` ``nfw_off_center`` tables are instead rescaled with
the **physical** :math:`\rho_{\rm crit}(z)`, so their
:math:`\Sigma_{\rm mis}` is physical:
:math:`\Sigma_{\rm phys} = (1+z)^2\,\Sigma_{\rm com}` — a factor 2.25 at
:math:`z = 0.5`. Compare across the two conventions with that one
visible multiplication, never by re-deriving either side.

NOTE: two named approximations that make the E.3 factorisation possible:

- the neighbour kernel is evaluated at the **cluster's** distance,
  :math:`R_\theta = \theta\,\chi(z^{\rm ob})`, and pulled out of the
  :math:`z` integral (thin line-of-sight window: :math:`\chi` varies by
  a few per cent across the support);
- the neighbour concentration is evaluated at :math:`z^{\rm ob}`
  (:math:`c \propto (1+z)^{-1.01}` varies by :math:`\lesssim 2\%` across
  a hard :math:`\pm 50\,h^{-1}` cMpc window).

NOTE: :math:`b_{\rm sel}(\theta)` is applied at the neighbour's polar
angle about the cluster centre (the E.3 factorised form). The Costanzi
notebook instead evaluates it inside the azimuthal average around the
point :math:`R`; the two are the same double integral in exchanged polar
coordinates and agree in the continuum limit.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..cosmology.concentration import duffy08
from ..cosmology.distances import comoving_volume_element
from ..cosmology.fiducial import fiducial_cosmology, mean_matter_density
from ..halo.nfw import NfwProfile
from ..kernels.photoz import (
    photoz_projection,
    photoz_projection_support,
    y3_photoz_window,
)
from ..selection.geometry import r_lambda
from ..selection.miscentering import load_nfw_miscentering_table
from ..utils.integrate import gl_nodes

__all__ = ["SigmaPrj"]


class SigmaPrj:
    r"""Projected two-halo surface density around a selected cluster.

    Parameters
    ----------
    cosmology : astropy cosmology, optional
        Default `clenspy.cosmology.fiducial_cosmology`.
    xi_nl : callable ``xi(r, zob)``
        Nonlinear matter correlation, r in comoving Mpc. Use
        `clenspy.selection.bsel.XiNL` with ``clip=False`` (the BAO trough
        is inside the line-of-sight window).
    hmf : callable ``n(M, z)``
        dn/dM [Msun^-1 Mpc^-3 comoving], physical mass.
    bias : callable ``b(M, z)``
        Halo bias, physical mass.
    concentration : callable ``c(M, z)``, optional
        Physical mass. Default: Duffy08 "200m" — the notebook's
        10.14 (M / 2e12 h^-1 Msun)^-0.081 (1+z)^-1.01, with the one
        visible M -> M h conversion here.
    mis_table : NfwMiscenteringTable, optional
        Default: the packaged table.
    n_theta, n_z_side, n_M : int
        Quadrature sizes. theta: log cells integrated exactly by the
        kernel (plus edges aligned to every z node's exclusion crossing,
        see `theta_edges`); z: Gauss-Legendre in log |dchi| per side,
        split at R_excl; mass: Gauss-Legendre in ln M. Defaults measured
        on the Buzzard configuration (2026-08-29 convergence scan,
        doubling each axis at lob=20, zob=0.5): n_M=40 converged to
        <0.1%; n_theta=96 puts Sigma within 0.3% (1.4% at R=25 for 64);
        n_z_side=160 is set by DeltaSigma near R_excl — the exclusion
        step in the z direction converges only ~first order (R=2 cMpc:
        +9% at 40, +0.14% at 160 against the n=640 reference) while
        Sigma is already <0.2% at 40. z cost is weights-only (linear,
        no kernel evaluations), so the high default is nearly free.
    theta_perp_range : tuple of float
        Transverse comoving span (Mpc) of the theta grid at z_ob;
        theta = span / chi(z_ob). Lower edge 1e-3: the 2 pi sin(theta)
        measure kills the integrand faster than Sigma_mis grows. Upper
        edge 90: the kernel at the outermost mock radius (~43 cMpc) still
        draws on neighbours a full NFW tail beyond it; at 90 cMpc the
        remaining contribution is in the kernel's 1/R_theta^2 tail with
        xi_NL(90) already small.
    min_mass, log10_M_max : float, optional
        Mass range, physical Msun; defaults 1e13 and 10^15.5 h^-1 Msun
        converted once with h (the RichnessSelection range).
    los_window : {"wpz", "hard"}
        "wpz": parabolic photo-z weight with the exact Y3 table
        (production, E.3). "hard": top-hat in comoving distance,
        chi(z_ob) +/- los_depth, w_pz = 1 (the Costanzi mock).
    los_depth : float, optional
        Half-depth in comoving Mpc for ``los_window="hard"``
        (mock: 50 h^-1 cMpc).
    exclusion : {"counter", "cl", "ball", "none"}
        See the module docstring. ``"counter"`` (default, the Costanzi
        convention) keeps the background uniform and sets the correlated
        integrand to -1 inside the chord ball, so the exclusion hole
        lives in the cl channel; ``"ball"`` removes both channels there
        (same sum, different split); ``"cl"`` is the E.3 slab on the
        correlated channel only.
    r_trunc : float, optional
        Halo-centric truncation of the neighbour profile [comoving Mpc]:
        each halo contributes surface density only within ``r_trunc`` of
        its own centre. ``None`` (default) is the untruncated NFW. The
        Costanzi mock samples its particles out to 30 cMpc/h from each
        halo, so its validation passes ``30/h``. Applied as the azimuthal
        fraction of the evaluation ring inside the truncation disk —
        exact up to the ring's :math:`\sim r_s` width against the
        truncation scale, i.e. to :math:`O(r_s/r_{\rm trunc})`.
    floor_one_plus_bxi : bool
        Floor the bracket 1 + b_sel b xi at zero pointwise in
        (theta, z, M) — density positivity, the notebook's convention.
        Couples the channels: the floored excess is reported in ``cl``.
    """

    def __init__(
        self,
        *,
        cosmology=None,
        xi_nl: Callable,
        hmf: Callable,
        bias: Callable,
        concentration: Callable | None = None,
        mis_table=None,
        n_theta: int = 96,
        n_z_side: int = 160,
        n_M: int = 40,
        theta_perp_range: tuple[float, float] = (1e-3, 90.0),
        min_mass: float | None = None,
        log10_M_max: float | None = None,
        los_window: str = "wpz",
        los_depth: float | None = None,
        exclusion: str = "counter",
        r_trunc: float | None = None,
        floor_one_plus_bxi: bool = False,
    ) -> None:
        cosmology = fiducial_cosmology() if cosmology is None else cosmology
        self.cosmo = cosmology
        self.h = cosmology.h
        self.xi_nl = xi_nl
        self.hmf = hmf
        self.bias = bias
        self.concentration = (
            concentration if concentration is not None
            # physical Msun -> h^-1 Msun once, visibly
            else (lambda m, z: duffy08(np.asarray(m, float) * self.h, z,
                                       mass_def="200m"))
        )
        self.mis_table = (mis_table if mis_table is not None
                          else load_nfw_miscentering_table())
        self.n_theta, self.n_z_side, self.n_M = n_theta, n_z_side, n_M
        self.theta_perp_range = tuple(theta_perp_range)
        self.min_mass = min_mass if min_mass is not None else 1.0e13 / self.h
        self.log10_M_max = (
            log10_M_max if log10_M_max is not None
            else np.log10(10.0**15.5 / self.h)
        )
        if los_window not in ("wpz", "hard"):
            raise ValueError(f"los_window must be 'wpz' or 'hard', got {los_window!r}")
        if los_window == "hard" and los_depth is None:
            raise ValueError("los_window='hard' needs los_depth [comoving Mpc]")
        if exclusion not in ("counter", "cl", "ball", "none"):
            raise ValueError(
                f"exclusion must be 'counter', 'cl', 'ball' or 'none', "
                f"got {exclusion!r}")
        self.los_window = los_window
        self.los_depth = None if los_depth is None else float(los_depth)
        self.exclusion = exclusion
        self.r_trunc = None if r_trunc is None else float(r_trunc)
        self.floor_one_plus_bxi = bool(floor_one_plus_bxi)
        # fast comoving-distance interpolant [Mpc], SelBiasEngine pattern
        self._zs_ref = np.linspace(1e-4, 2.0, 2000)
        self._chi_ref = cosmology.comoving_distance(self._zs_ref).to_value("Mpc")
        self._window = y3_photoz_window()
        self.rho_m = mean_matter_density(cosmology)
        self._cache: dict = {}
        # channels of the last assembly (rule 6: keep the decomposition)
        self.rnd: np.ndarray | None = None
        self.cl: np.ndarray | None = None

    # -- geometry ---------------------------------------------------------
    def chi(self, z):
        r"""Comoving distance :math:`\chi(z)` [Mpc]."""
        return np.interp(np.asarray(z, dtype=float), self._zs_ref, self._chi_ref)

    def _z_of_chi(self, chi_val):
        return np.interp(np.asarray(chi_val, dtype=float),
                         self._chi_ref, self._zs_ref)

    def theta_grid(self, zob: float, lob: float | None = None):
        r"""Cell centres of the :math:`\theta` grid, and the
        spherical-measure correction :math:`\sin\bar\theta/\bar\theta` per
        cell. The measure itself lives in the **exact per-cell kernel
        masses** of `kernel` — a pointwise :math:`2\pi\sin\theta\,d\theta`
        rule cannot resolve the :math:`\Sigma_{\rm mis}` ring of width
        :math:`\sim r_s` at :math:`\theta\chi_o \approx R`."""
        edges = self.theta_edges(zob, lob)
        centres = np.sqrt(edges[:-1] * edges[1:])
        return centres, np.sin(centres) / centres

    def theta_edges(self, zob: float, lob: float | None = None):
        r""":math:`\theta` cell edges [rad] — log-spaced, n_theta + 1
        without exclusion.

        With ``lob`` given and exclusion active, the exact per-node
        crossing angles of the chord ball,
        :math:`\cos\theta_{\rm ex}(z_k) = [\chi_k^2 + \chi_o^2 -
        R_{\rm excl}^2]/[2\chi_k\chi_o]`, are inserted as extra edges.
        The exclusion boundary is a *curve* in the :math:`(\theta, z)`
        plane — :math:`\theta` and :math:`z` are coupled through the
        chord — so classifying whole log cells by their centre against a
        single radius misclassifies every boundary-straddling cell.
        Aligning the edges with every z node's own crossing makes the
        indicator exact at cell resolution for **each** node: no kernel
        cell ever straddles the boundary of any quadrature node. The
        cell count grows by up to one per in-ball z node; all downstream
        shapes are dynamic."""
        key = ("tedges", round(zob, 8),
               None if lob is None else round(lob, 8), self.exclusion)
        if key in self._cache:
            return self._cache[key]
        chi_o = float(self.chi(zob))
        edges = np.geomspace(*self.theta_perp_range, self.n_theta + 1) / chi_o
        if lob is not None and self.exclusion != "none":
            r_ex = self.r_excl(lob, zob)
            zs, _ = self._z_grid(zob, split_at=r_ex)
            chi_z = self.chi(zs)
            in_ball = np.abs(chi_z - chi_o) < r_ex
            cos_t = ((chi_z[in_ball] ** 2 + chi_o**2 - r_ex**2)
                     / (2.0 * chi_z[in_ball] * chi_o))
            th_ex = np.arccos(np.clip(cos_t, -1.0, 1.0))
            th_ex = th_ex[(th_ex > edges[0]) & (th_ex < edges[-1])]
            edges = np.unique(np.concatenate([edges, th_ex]))
            # merge near-degenerate edges (GL nodes cluster near the pole)
            keep = np.concatenate([[True], np.diff(edges) / edges[1:] > 1e-9])
            edges = edges[keep]
        self._cache[key] = edges
        return edges

    def _z_grid(self, zob: float, split_at: float | None = None):
        r"""Gauss-Legendre in :math:`\ln|\Delta\chi|` on each side of the
        cluster; returns z nodes and dz-equivalent weights.

        The z-integral is the line-of-sight parametrization of a
        truncated Abel transform of :math:`\xi` — i.e. already the
        edge-safe :math:`r = R\cosh u` substitution, with no hidden
        :math:`1/\sqrt{r^2 - R^2}` singularity (see
        ``review/08282026/abel_projection_note.md``). The one genuinely
        non-smooth feature is the exclusion indicator: at fixed
        :math:`\theta` it switches at
        :math:`|\Delta\chi| = \sqrt{R_{\rm excl}^2 - R_\theta^2} \le
        R_{\rm excl}`, so each side is integrated as two GL segments
        split at ``split_at`` (:math:`= R_{\rm excl}`) — the crossing
        then always falls in the densely resolved inner segment instead
        of jittering between nodes, which is what made the assembled
        channels non-monotone in ``n_z_side``."""
        key = ("zgrid", round(zob, 8),
               None if split_at is None else round(split_at, 8))
        if key in self._cache:
            return self._cache[key]
        chi_o = float(self.chi(zob))
        if self.los_window == "hard":
            dis_fg = dis_bg = self.los_depth
        else:
            z_lo, z_hi = photoz_projection_support(
                zob, self._window, n_sigma=1.0
            )
            dis_fg = chi_o - float(self.chi(z_lo))
            dis_bg = float(self.chi(z_hi)) - chi_o
        dchi_dz_ref = np.gradient(self._chi_ref, self._zs_ref)

        def _segment(dis_lo, dis_hi, n, sign):
            u, w_u = gl_nodes(np.log(dis_lo), np.log(dis_hi), n)
            dis = np.exp(u)
            z_out = self._z_of_chi(chi_o + sign * dis)
            dchi_dz = np.interp(z_out, self._zs_ref, dchi_dz_ref)
            # dz = d(dis) / (dchi/dz); d(dis) = dis du
            return z_out, w_u * dis / dchi_dz

        def _side(dis_max, sign):
            if split_at is not None and 1e-3 < split_at < dis_max:
                n_in = max(8, self.n_z_side // 2)
                n_out = max(8, self.n_z_side - n_in)
                z1, w1 = _segment(1e-3, split_at, n_in, sign)
                z2, w2 = _segment(split_at, dis_max, n_out, sign)
                return (np.concatenate([z1, z2]),
                        np.concatenate([w1, w2]))
            return _segment(1e-3, dis_max, self.n_z_side, sign)

        z_fg, w_fg = _side(dis_fg, -1.0)
        z_bg, w_bg = _side(dis_bg, +1.0)
        zs = np.concatenate([z_fg[::-1], z_bg])
        wzs = np.concatenate([w_fg[::-1], w_bg])
        out = (zs, wzs)
        self._cache[key] = out
        return out

    def common(self, zs, zob: float):
        r"""E.3: :math:`{\rm common}(z) = \frac{dV}{d\Omega dz} w_{pz}` —
        **no** :math:`\Omega(z)` (it cancels in the surface density)."""
        dV = comoving_volume_element(zs, self.cosmo)
        if self.los_window == "hard":
            w_pz = np.ones_like(np.asarray(zs, dtype=float))
        else:
            w_pz = photoz_projection(zs, zob, self._window, n_sigma=1.0)
        return dV * w_pz

    def dchi(self, thetas, zs, zob: float):
        r"""Law-of-cosines 3-D chord (E.3 trap 2), shape (n_theta, n_z):
        :math:`d\chi^2 = \chi_z^2 + \chi_o^2 - 2\chi_z\chi_o\cos\theta`."""
        chi_o = float(self.chi(zob))
        chi_z = self.chi(zs)
        return np.sqrt(np.maximum(
            chi_z[None, :] ** 2 + chi_o**2
            - 2.0 * chi_z[None, :] * chi_o * np.cos(thetas)[:, None], 0.0,
        ))

    def r_excl(self, lob: float, zob: float) -> float:
        r""":math:`R_{\rm excl} = R_\lambda(\lambda^{\rm ob})(1+z^{\rm ob})`
        [comoving Mpc]."""
        return float(r_lambda(lob, self.h) * (1.0 + zob))

    # -- mass nodes ---------------------------------------------------------
    def _mass_nodes(self):
        lnMs, wM = gl_nodes(
            np.log(self.min_mass), np.log(10.0**self.log10_M_max), self.n_M
        )
        Ms = np.exp(lnMs)
        return Ms, wM * Ms  # dlnM-weight x M, so hmf stays dn/dM

    # -- channel weights ----------------------------------------------------
    def _channel_weights(self, lob: float, zob: float, b_sel: Callable):
        r"""(W_rnd, W_cl), each (n_theta, n_M): the z integrals of E.3.

        With ``floor_one_plus_bxi`` the floored bracket is integrated as a
        whole and the excess over the rnd channel is reported as cl."""
        thetas, _ = self.theta_grid(zob, lob)
        zs, wzs = self._z_grid(
            zob,
            split_at=(self.r_excl(lob, zob)
                      if self.exclusion != "none" else None),
        )
        cmn = self.common(zs, zob) * wzs                     # (n_z,)
        Ms, M_weight = self._mass_nodes()
        n_mz = self.hmf(Ms[:, None], zs[None, :]) * M_weight[:, None]
        b_mz = self.bias(Ms[:, None], zs[None, :])

        dchi = self.dchi(thetas, zs, zob)                    # (n_th, n_z)
        xi = self.xi_nl(dchi.ravel(), zob).reshape(dchi.shape)
        outside = dchi > self.r_excl(lob, zob)               # (n_th, n_z)
        ones = np.ones_like(dchi)
        m_cl = outside if self.exclusion in ("cl", "ball", "counter") else ones
        m_rnd = outside if self.exclusion == "ball" else ones

        bsel_t = np.broadcast_to(
            np.asarray(b_sel(thetas), dtype=float), thetas.shape
        )                                                     # (n_th,)

        m_rnd = np.broadcast_to(np.asarray(m_rnd, dtype=float), dchi.shape)
        m_cl = np.broadcast_to(np.asarray(m_cl, dtype=float), dchi.shape)

        W_rnd = np.einsum("z,tz,mz->tm", cmn, m_rnd, n_mz)
        if not self.floor_one_plus_bxi:
            W_cl = np.einsum("z,tz,mz->tm", cmn, xi * m_cl, n_mz * b_mz)
            W_cl *= bsel_t[:, None]
            if self.exclusion == "counter":
                # K_exc = (1 + b b_sel xi) E - 1 (excess-equations note
                # eq. 7): exclusion acts on the complete pair
                # distribution. Outside the ball the weight is w_cl
                # (built above); inside it is -w_rnd -- minus the
                # background integrand, no bias, no b_sel, no xi. The
                # total vanishes in the ball, the background stays
                # strictly uniform, and the exclusion hole lives in the
                # cl channel -- where a random-point-subtracted
                # measurement keeps it.
                W_cl -= np.einsum("z,tz,mz->tm", cmn, 1.0 - m_cl, n_mz)
        else:
            # floored bracket: integrate the whole and report the excess
            # over the (uniform or masked) rnd channel as cl. For
            # "counter" the total is ball-masked while rnd stays uniform,
            # so the counter term emerges in the difference.
            bxi = (bsel_t[:, None, None] * (xi * m_cl)[:, :, None]
                   * b_mz.T[None, :, :])                     # (t, z, m)
            m_tot = m_cl if self.exclusion == "counter" else m_rnd
            brac = np.maximum(1.0 + bxi, 0.0) * m_tot[:, :, None]
            W_tot = np.einsum("z,tzm,mz->tm", cmn, brac, n_mz)
            W_cl = W_tot - W_rnd
        return W_rnd, W_cl

    def _profiles(self, zob: float):
        r"""(r_s, Sigma_0 = 2 r_s rho_s) of the neighbour NFW population at
        :math:`c = c(M, z^{\rm ob})` — one of the two named thin-window
        approximations."""
        Ms, _ = self._mass_nodes()
        prof = NfwProfile(m200=Ms, c200=self.concentration(Ms, zob),
                          rho_ref=self.rho_m)
        rs = np.asarray(prof.rs, dtype=float)
        sigma0 = 2.0 * rs * np.asarray(prof.rho_s, dtype=float)
        return rs, sigma0

    def _mhat(self, x, x_mis):
        r""":math:`\hat m(x, x_{\rm mis}) = \bar\Sigma_{\rm mis}(<x r_s \mid
        x_{\rm mis} r_s)/\Sigma_0 = \hat\Sigma_{\rm mis} +
        \widehat{\Delta\Sigma}_{\rm mis}` — the aperture mean, so
        :math:`\pi s^2 \Sigma_0 \hat m(s/r_s, \cdot)` is an enclosed
        projected mass."""
        return (self.mis_table.sigma_hat(x, x_mis)
                + self.mis_table.ds_hat(x, x_mis))

    # -- kernel ---------------------------------------------------------------
    def kernel(self, R, lob: float, zob: float, which: str = "sigma"):
        r"""Per-cell kernel masses
        :math:`\int_{\rm cell} 2\pi\sin\theta\,d\theta\;
        K(R, \theta\chi_o \mid M)`, shape (n_theta, n_M, n_R), with
        :math:`K = \Sigma_{\rm mis}` (``"sigma"``) or the signed
        :math:`\Delta\Sigma_{\rm mis}` (``"ds"``).

        :math:`\Sigma_{\rm mis}(R, s)` is a ring of width :math:`\sim r_s`
        at :math:`s \approx R` that no affordable pointwise :math:`\theta`
        rule resolves. Both integrals are therefore done by parts:

        - **sigma** — the azimuthal average is symmetric,
          :math:`\Sigma_{\rm mis}(R, s) = \Sigma_{\rm mis}(s, R)`, so the
          cell integral is an exact annulus-mass difference of the halo
          *offset by R*:
          :math:`\int_{s_1}^{s_2} 2\pi s\,\Sigma_{\rm mis}(R,s)\,ds
          = \pi\Sigma_0\big[s^2 \hat m(s/r_s,\, R/r_s)\big]_{s_1}^{s_2}`.
        - **ds** — split :math:`\Delta\Sigma_{\rm mis}(R, s) =
          \bar\Sigma_{\rm mis}(<R \mid s) - \Sigma_{\rm mis}(R, s)`: the
          first term is smooth in :math:`s` (an aperture mean — the ring
          is integrated over) and takes a per-cell trapezoid on the
          edges; the second is the exact annulus mass above. This keeps
          the signed lobe and never reconstructs
          :math:`\Delta\Sigma` from :math:`\Sigma`.

        The flat-sky :math:`2\pi s\,ds = \chi_o^2\, 2\pi\theta\,d\theta`
        is corrected to the spherical measure by
        :math:`\sin\bar\theta/\bar\theta` per cell (:math:`< 0.4\%` at
        :math:`\theta_{\max} = 0.15`). :math:`R_\theta = \theta\chi(z^{\rm
        ob})` — the other thin-window approximation."""
        R = np.atleast_1d(np.asarray(R, dtype=float))
        chi_o = float(self.chi(zob))
        s_edges = self.theta_edges(zob, lob) * chi_o
        _, sin_corr = self.theta_grid(zob, lob)
        rs, sigma0 = self._profiles(zob)
        n_t, n_m = s_edges.size - 1, rs.size

        K = np.empty((n_t, n_m, R.size))
        for im in range(n_m):
            # exact annulus masses of the halo offset by R (symmetry)
            m_edges = np.empty((n_t + 1, R.size))
            for ir in range(R.size):
                m_edges[:, ir] = (s_edges**2) * self._mhat(
                    s_edges / rs[im], R[ir] / rs[im]
                )
            ring = np.pi * sigma0[im] * np.diff(m_edges, axis=0)
            if which == "sigma":
                K[:, im, :] = ring
            else:
                # smooth term: 2 pi s SigmaBar_mis(<R | s). Smooth in s,
                # but curved on the r_s scale near s ~ R (the halo cusp
                # crossing the aperture edge), which an edge trapezoid
                # under-resolves at default cell widths (-5% in
                # DeltaSigma at R = 8, test resolution). Per-cell
                # Gauss-Legendre nodes resolve it at fixed cell count.
                x_gl, w_gl = gl_nodes(0.0, 1.0, 4)
                width = np.diff(s_edges)
                smooth = np.zeros((n_t, R.size))
                for j in range(x_gl.size):
                    s_j = s_edges[:-1] + width * x_gl[j]
                    for it in range(n_t):
                        smooth[it] += (w_gl[j] * width[it]
                                       * 2.0 * np.pi * s_j[it] * sigma0[im]
                                       * self._mhat(R / rs[im],
                                                    s_j[it] / rs[im]))
                K[:, im, :] = smooth - ring
        if self.r_trunc is not None:
            if which != "sigma":
                raise NotImplementedError(
                    "r_trunc is a mock-matching device for Sigma_prj; "
                    "DeltaSigma_prj uses the untruncated kernel"
                )
            K -= self._tail_cells(R, s_edges, rs, sigma0)
        # to the spherical measure, per cell
        K *= sin_corr[:, None, None] / chi_o**2
        return K

    def _tail_cells(self, R, s_edges, rs, sigma0):
        r"""Per-cell mass of the neighbour profile **beyond** the
        halo-centric truncation radius, to subtract from the untruncated
        exact cells.

        The removed tail :math:`\Sigma(u)\,\mathbb 1[u > r_{\rm trunc}]`
        carries no cusp (:math:`r_{\rm trunc} \gg r_s`), so its azimuthal
        average is smooth in the offset and ordinary quadrature converges:
        with :math:`u^2 = R^2 + s^2 - 2Rs\cos\varphi` monotone in
        :math:`\varphi`, the tail sits at :math:`\varphi > \varphi_t`,
        :math:`\cos\varphi_t = (R^2 + s^2 - r_{\rm trunc}^2)/(2Rs)`.
        Weighting the *arc length* instead (a naive
        :math:`\varphi_t/\pi` mask on the cell mass) discards the
        cusp-concentrated ring mass and over-truncates by tens of percent
        at the outermost radii."""
        r_t = self.r_trunc
        n_phi = 16
        x_gl, w_gl = gl_nodes(0.0, 1.0, n_phi)
        # phi_t at every (cell edge, R): 0 = all tail, pi = no tail
        cos_t = ((R[None, :] ** 2 + s_edges[:, None] ** 2 - r_t**2)
                 / (2.0 * R[None, :] * s_edges[:, None]))
        phi_t = np.arccos(np.clip(cos_t, -1.0, 1.0))           # (n_e, n_R)
        span = np.pi - phi_t
        phi = phi_t[..., None] + span[..., None] * x_gl        # (n_e,n_R,n_phi)
        u = np.sqrt(np.maximum(
            R[None, :, None] ** 2 + s_edges[:, None, None] ** 2
            - 2.0 * R[None, :, None] * s_edges[:, None, None]
            * np.cos(phi), 0.0,
        ))
        tail = np.zeros((s_edges.size - 1, rs.size, R.size))
        for im in range(rs.size):
            fhat = NfwProfile._fNfw(u / rs[im])
            # sigma of the tail, azimuth-averaged: (1/pi) int_{phi_t}^pi
            sig_tail = (sigma0[im] / np.pi) * np.einsum(
                "erp,p,er->er", fhat, w_gl, span)
            g = 2.0 * np.pi * s_edges[:, None] * sig_tail      # (n_e, n_R)
            tail[:, im, :] = (0.5 * (g[1:] + g[:-1])
                              * np.diff(s_edges)[:, None])
        return tail

    # -- assembly ---------------------------------------------------------------
    def _assemble(self, R, lob: float, zob: float, b_sel: Callable,
                  which: str, channel: str):
        if channel not in ("cl", "sum", "rnd"):
            raise ValueError(
                f"channel must be 'cl', 'sum' or 'rnd', got {channel!r}")
        W_rnd, W_cl = self._channel_weights(lob, zob, b_sel)
        K = self.kernel(R, lob, zob, which)
        # b_sel already inside W_cl; the theta measure inside K
        self.rnd = np.einsum("tm,tmr->r", W_rnd, K)
        self.cl = np.einsum("tm,tmr->r", W_cl, K)
        return {"cl": self.cl, "rnd": self.rnd,
                "sum": self.rnd + self.cl}[channel]

    def sigma_prj(self, R, lob: float, zob: float, b_sel: Callable,
                  channel: str = "cl"):
        r""":math:`\Sigma_{\rm prj}(R \mid \lambda^{\rm ob}, z^{\rm ob})`
        [comoving Msun/Mpc^2]. ``b_sel``: any callable of theta [rad] —
        a `SigmoidBias`, or ``lambda th: b_eff`` for the random-stack
        (unselected) model.

        NOTE: **the default is the correlated channel only** — the
        two-halo term is the excess above the mean background, which is
        what a random-point-subtracted measurement (and cluster_toolkit's
        :math:`\Sigma_{2h}`) contains; the production pipeline returns
        the cl piece for exactly this reason (E.3). Pass
        ``channel="sum"`` to add the mean-background rnd channel — a raw
        projected mass map (e.g. the Costanzi mock's per-halo columns)
        includes it. Both channels are stored on ``self`` either way."""
        return self._assemble(R, lob, zob, b_sel, "sigma", channel)

    def deltasigma_prj(self, R, lob: float, zob: float, b_sel: Callable,
                       channel: str = "cl"):
        r""":math:`\Delta\Sigma_{\rm prj}(R)` [comoving Msun/Mpc^2] —
        the kernel swap :math:`\Sigma_{\rm mis} \to \Delta\Sigma_{\rm mis}`
        (signed, never clamped). Default ``channel="cl"``: the excess
        functional annihilates the uniform background exactly, so the rnd
        channel is a :math:`\theta_{\max}`-truncation boundary term, not
        physics — dropping it is the model form of the random-point
        subtraction."""
        return self._assemble(R, lob, zob, b_sel, "ds", channel)

    def components(self) -> dict:
        """The channels of the last assembly (Estimator contract)."""
        return {"rnd": self.rnd, "cl": self.cl,
                "sum": None if self.rnd is None else self.rnd + self.cl}


if __name__ == "__main__":
    # the real halo model — Tinker (2008) mass function, Tinker (2010)
    # bias, CAMB halofit xi_NL. PkGrid disk-caches the CAMB call, so this
    # costs seconds once and milliseconds after.
    from ..cosmology.bias import BiasModel
    from ..cosmology.halo_mass_function import TinkerMassFunction
    from ..cosmology.pkgrid import PkGrid
    from ..selection.bsel import SigmoidBias, XiNL

    cosmo = fiducial_cosmology()
    h, omega_m = cosmo.h, cosmo.Om0
    tmf = TinkerMassFunction(cosmo=cosmo, zvec=np.linspace(0.0, 1.0, 21))
    bias_model = BiasModel(cosmo=cosmo)
    xi_nl = XiNL(PkGrid(cosmo=cosmo, nonlinear=True), clip=False)

    def hmf(mass, z):
        """dn/dM [Msun^-1 Mpc^-3] at physical Msun: the one visible unit
        boundary, physical Msun -> the Tinker grid's Omega_m h^-1 Msun."""
        m, zz = np.broadcast_arrays(np.asarray(mass, float),
                                    np.asarray(z, float))
        vals = tmf.dndlnm(m.ravel() * h / omega_m, zz.ravel())
        return vals.reshape(m.shape) * h**3 / m

    def bias(mass, z):
        m, zz = np.broadcast_arrays(np.asarray(mass, float),
                                    np.asarray(z, float))
        return np.asarray(
            bias_model.bias(m.ravel(), zz.ravel())
        ).reshape(m.shape)

    # default exclusion="counter": the K_exc pair weight -- the one mode
    # whose cl channel is the mode-invariant random-subtracted excess
    # (under "ball" the exclusion hole is booked in rnd instead, and the
    # default channel="cl" of deltasigma_prj would silently omit it)
    prj = SigmaPrj(cosmology=cosmo, xi_nl=xi_nl, hmf=hmf, bias=bias,
                   los_window="hard", los_depth=71.4)
    lob, zob = 20.0, 0.5
    bsel = SigmoidBias(lob=lob, zob=zob,
                       theta_lambda=prj.r_excl(lob, zob) / prj.chi(zob),
                       b_small=4.0, b_large=3.0)
    R = np.array([0.5, 1.0, 3.0, 10.0, 20.0, 40.0])  # comoving Mpc

    sig = prj.sigma_prj(R, lob, zob, bsel, channel="sum")
    parts = prj.components()
    print(f"Sigma_prj at (lob={lob}, zob={zob}), hard +/-{prj.los_depth} cMpc:")
    print(f"{'R [cMpc]':>9s} {'rnd':>12s} {'cl':>12s} {'sum':>12s}")
    for i, r in enumerate(R):
        print(f"{r:9.2f} {parts['rnd'][i]:12.4e} {parts['cl'][i]:12.4e} "
              f"{sig[i]:12.4e}")

    dsig = prj.deltasigma_prj(R, lob, zob, bsel)
    print("\nDeltaSigma_prj (signed; rnd channel cancels to a boundary "
          "term):")
    print("  " + "  ".join(f"{v:+.4e}" for v in dsig))

    # rnd channel sanity: with no selection and no exclusion it is the
    # mean projected mass column of the halo population -- compare
    # rho_halos x 2 los_depth, i.e. the halo mass fraction times the
    # uniform-universe column rho_m x 2 los_depth
    prj0 = SigmaPrj(cosmology=cosmo, xi_nl=xi_nl, hmf=hmf, bias=bias,
                    los_window="hard", los_depth=71.4, exclusion="none")
    prj0.sigma_prj(R, lob, zob, lambda th: 0.0)
    column = prj0.rho_m * 2.0 * prj0.los_depth
    print(f"\nno-exclusion rnd plateau: {prj0.rnd[0]:.4e} Msun/Mpc^2")
    print(f"rho_m x 2 depth          : {column:.4e} Msun/Mpc^2 "
          f"(ratio {prj0.rnd[0] / column:.3f} = halo-budget fraction; the "
          "untruncated NFW wings push it above the naive mass fraction)")
