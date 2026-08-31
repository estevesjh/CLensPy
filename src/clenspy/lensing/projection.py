r"""Projection lensing: :math:`\Sigma_{\rm prj}` and :math:`\Delta\Sigma_{\rm prj}`.

The two-halo projected surface density around a richness-selected cluster
(Costanzi 2026 eq. 13): a sum over neighbour mass :math:`M` and offset
angle :math:`\theta` of the correlated weight :math:`n_{\rm cl}(\theta, M)`
against the mass shell :math:`M_\theta(R \mid M)` — not an aperture mass
"inside :math:`R`", but the neighbour halo's own offset profile,
:math:`R_\theta = \theta\chi_o`, converted from a shell mass into a
density (see `MassShells`):

.. math::
    \Sigma_{\rm prj}(R) = \int d\theta\, 2\pi\sin\theta \int dM\;
        n_{\rm cl}(\theta, M)\, M_\theta(R \mid M)

:math:`\Delta\Sigma_{\rm prj}` is the same sum with the mass shell
swapped for its signed excess, :math:`M_\theta \to \Delta M_\theta`
(never a numerical reconstruction of a tabulated :math:`\Sigma_{\rm prj}`).

NOTE: physical :math:`M_\odot`, comoving Mpc, h-free; :math:`\Sigma` in
comoving :math:`M_\odot\,{\rm Mpc}^{-2}`. Full derivation, channel split
(:math:`n_{\rm rnd}`/:math:`n_{\rm cl}`), exclusion semantics, named
approximations, and the comoving/:math:`(1+z)^2` table convention:
``docs/projection_lensing.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..cosmology.bias import BiasModel
from ..cosmology.concentration import duffy08
from ..cosmology.distances import ComovingDistance, comoving_volume_element
from ..cosmology.fiducial import fiducial_cosmology, mean_matter_density
from ..cosmology.halo_mass_function import TinkerMassFunction
from ..cosmology.pkgrid import PkGrid
from ..cosmology.sigma import SigmaGrid
from ..halo.nfw import NfwProfile
from ..halo.twohalo import TwoHaloTerm
from ..kernels.photoz import (
    photoz_projection,
    photoz_projection_support,
    y3_photoz_window,
)
from ..selection.geometry import r_excl
from ..selection.miscentering import load_nfw_miscentering_table
from ..utils.integrate import mass_nodes
from ..utils.interpolate import LogGridInterpolator
from ..utils.los_integrals import (
    LosGeometry,
    integrate_los,
    shell_masses,
    tail_masses,
    theta_edges,
    theta_grid,
)

__all__ = ["SigmaPrj", "SigmaPrjConfig"]


@dataclass(frozen=True)
class SigmaPrjConfig:
    r"""Numerical, line-of-sight, and profile settings for ``SigmaPrj``.

    Parameters
    ----------
    mis_table : NfwMiscenteringTable, optional
        Default: the packaged table.
    n_theta, n_M : int
        Quadrature sizes: log theta cells integrated exactly by the
        profile kernel; Gauss-Legendre in ln M. n_theta=96 from a
        measured convergence scan (0.2% vs 192 at the smallest R).
    n_u_inside, n_u_outside : int
        Cosh-Abel Gauss-Legendre orders inside and outside the exclusion
        sphere; the discontinuity is an interval boundary, never a mask.
        Compare 6/16 against 8/24 when changing the projection support.
    theta_perp_range : tuple of float
        Transverse comoving span (Mpc) of the theta grid at z_ob. Lower
        edge 1e-3: the 2 pi sin(theta) measure kills the integrand faster
        than Sigma_mis grows; upper edge 90: the kernel tail at the
        outermost mock radius with xi_NL already small.
    min_mass, log10_M_max : float, optional
        Mass range, physical Msun; defaults 1e13 and 10^15.5 h^-1 Msun
        converted once with h (the RichnessSelection range).
    los_window : {"wpz", "hard"}
        "wpz": parabolic photo-z weight, exact Y3 table (production).
        "hard": top-hat in comoving distance, needs ``los_depth``.
    los_depth : float, optional
        Half-depth [comoving Mpc] for ``los_window="hard"``.
    exclusion : {"counter", "cl", "ball", "none"}
        Halo-exclusion semantics; see ``docs/projection_lensing.md``.
        "counter" (default): the neighbour count inside the chord ball
        is zeroed by subtracting the background weight from the
        correlated channel, background strictly uniform, hole booked
        in cl.
    r_trunc : float, optional
        Halo-centric truncation of the neighbour profile [comoving Mpc];
        None = untruncated NFW (the mock passes 30/h).
    floor_one_plus_bxi : bool
        Floor 1 + b_sel b xi at zero pointwise (density positivity);
        couples the channels, floored excess reported in cl.
    """

    mis_table: object | None = None
    n_theta: int = 96
    n_M: int = 40
    n_u_inside: int = 6
    n_u_outside: int = 16
    theta_perp_range: tuple[float, float] = (1e-3, 90.0)
    min_mass: float | None = None
    log10_M_max: float | None = None
    los_window: str = "wpz"
    los_depth: float | None = None
    exclusion: str = "counter"
    r_trunc: float | None = None
    floor_one_plus_bxi: bool = False


@dataclass(frozen=True)
class Exclusion:
    r"""Halo-exclusion bookkeeping of ``docs/projection_lensing.md``,
    "Exclusion".

    A neighbour within the exclusion ball *is* the cluster, so its
    count there must vanish entirely rather than merely lack
    clustering. Books the three line-of-sight weights into the (rnd,
    cl) channels — pure algebra, independent of how the weights were
    integrated. One return per mode:

    =========  ====================  ==================================
    mode       rnd channel           cl channel
    =========  ====================  ==================================
    counter    n_rnd_in + n_rnd_out  n_lss - n_rnd_in
    ball       n_rnd_out             n_lss
    cl         n_rnd_in + n_rnd_out  n_lss
    none       (n_rnd_in = 0)        n_lss
    =========  ====================  ==================================

    ``b_sel`` is already inside ``n_lss`` (`SigmaPrj.n_los_integral`
    applies it). Every mode has the same channel *sum*; they differ only
    in where the exclusion hole is booked. Under ``floor_one_plus_bxi``
    the floored n_lss is the full bracket — background included — so cl
    subtracts the background column.
    """

    mode: str = "counter"
    floor_one_plus_bxi: bool = False

    def __post_init__(self):
        if self.mode not in ("counter", "cl", "ball", "none"):
            raise ValueError(
                f"exclusion must be 'counter', 'cl', 'ball' or 'none', "
                f"got {self.mode!r}")

    def channels(self, n_rnd_in, n_rnd_out, n_lss):
        """(n_rnd, n_cl) from the three weights, per the table above."""
        n_rnd = n_rnd_in + n_rnd_out           # the uniform background column

        if self.floor_one_plus_bxi:
            if self.mode == "ball":
                return n_rnd_out, n_lss - n_rnd_out
            if self.mode == "cl":
                return n_rnd, n_lss - n_rnd_out
            return n_rnd, n_lss - n_rnd        # "counter" and "none"

        if self.mode == "counter":
            return n_rnd, n_lss - n_rnd_in     # -1 counterterm inside the ball
        if self.mode == "ball":
            return n_rnd_out, n_lss            # hole booked in the background
        return n_rnd, n_lss                    # "cl" slab and "none"


class MassShells:
    r"""The mass shell :math:`M_\theta(R \mid M)`: mean projected mass of
    a neighbour halo of mass :math:`M`, offset by
    :math:`R_\theta = \theta\chi_o`, inside radius :math:`R`.

    .. math::
        M_\theta(R \mid M) = \frac{\sin\bar\theta/\bar\theta}{\chi_o^2}
            \int_{\rm shell} 2\pi s\, \Sigma_{\rm mis}(R, s \mid M)\, ds,

    with the signed :math:`\Delta\Sigma_{\rm mis}` in place of
    :math:`\Sigma_{\rm mis}` under ``which="ds"``, integrated exactly by parts
    (`clenspy.utils.los_integrals.shell_masses`) and corrected to the
    spherical measure. Pure profile physics — independent of the
    line-of-sight weights. Self-contained: owns the theta-shell grid and
    the neighbour NFW profiles, so one call is ``shells(R, lob, zob)``.
    One-entry cache: the shells are b_sel-independent, so repeated channel
    evaluations at the same (R, lob, zob) reuse them.

    Parameters
    ----------
    mis_table : NfwMiscenteringTable
        The dimensionless offset-profile table.
    distance : ComovingDistance
    concentration : callable ``c(M, z)``
        Physical mass.
    rho_m : float
        Comoving mean matter density [Msun/Mpc^3].
    h : float
    config : SigmaPrjConfig
        Theta grid, n_M, exclusion mode, and the optional halo-centric
        truncation ``r_trunc`` (the removed tail is subtracted per shell,
        `tail_masses`; ``"sigma"`` only).
    min_mass, log10_M_max : float
        Mass range, physical Msun.
    """

    def __init__(self, *, mis_table, distance, concentration, rho_m, h,
                 config, min_mass, log10_M_max):
        self.mis_table = mis_table
        self.distance = distance
        self.concentration = concentration
        self.rho_m = rho_m
        self.h = h
        self.config = config
        self.min_mass = min_mass
        self.log10_M_max = log10_M_max
        self._cache = None

    def theta_shells(self, lob: float, zob: float):
        r"""Shell edges [rad], centres, and the spherical-measure
        correction :math:`\sin\bar\theta/\bar\theta` of the
        :math:`\theta` grid at one cluster."""
        r_ex = (r_excl(lob, zob, self.h)
                if self.config.exclusion != "none" else 0.0)
        edges = theta_edges(float(self.distance.chi(zob)),
                            self.config.theta_perp_range,
                            self.config.n_theta, r_excl=r_ex)
        thetas, sin_corr = theta_grid(edges)
        return edges, thetas, sin_corr

    def profiles(self, zob: float):
        r"""(r_s, Sigma_0 = 2 r_s rho_s) of the neighbour NFW population
        at :math:`c = c(M, z^{\rm ob})` — one of the two named thin-window
        approximations."""
        Ms, _ = mass_nodes(self.min_mass, 10.0**self.log10_M_max,
                           self.config.n_M)
        prof = NfwProfile(m200=Ms, c200=self.concentration(Ms, zob),
                          rho_ref=self.rho_m)
        rs = np.asarray(prof.rs, dtype=float)
        sigma0 = 2.0 * rs * np.asarray(prof.rho_s, dtype=float)
        return rs, sigma0

    def mean_sigma(self, x, x_mis):
        r"""Mean enclosed surface density of the offset profile, per
        :math:`\Sigma_0`: :math:`\bar\Sigma_{\rm mis}(<x r_s \mid
        x_{\rm mis} r_s)/\Sigma_0 = \hat\Sigma_{\rm mis} +
        \widehat{\Delta\Sigma}_{\rm mis}` (the doc's :math:`\hat m`).
        A sigma, not a mass: :math:`\pi s^2 \Sigma_0` times it is the
        enclosed projected mass, applied in `shell_masses`."""
        return (self.mis_table.sigma_hat(x, x_mis)
                + self.mis_table.ds_hat(x, x_mis))

    def __call__(self, R, lob: float, zob: float, which: str = "sigma"):
        r"""The mass shell :math:`M_\theta(R \mid M)`, shape (n_theta,
        n_M, n_R); see the class docstring."""
        R = np.atleast_1d(np.asarray(R, dtype=float))
        key = (which, R.tobytes(), round(lob, 8), round(zob, 8))
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]

        chi_o = float(self.distance.chi(zob))
        edges, _, sin_corr = self.theta_shells(lob, zob)
        rs, sigma0 = self.profiles(zob)
        masses = shell_masses(R, edges * chi_o, rs, sigma0,
                              self.mean_sigma, which)
        if self.config.r_trunc is not None:
            if which != "sigma":
                raise NotImplementedError(
                    "r_trunc is a mock-matching device for Sigma_prj; "
                    "DeltaSigma_prj uses the untruncated profile"
                )
            masses -= tail_masses(R, edges * chi_o, rs, sigma0,
                                  self.config.r_trunc, NfwProfile._fNfw)
        # to the spherical measure, per shell
        masses *= sin_corr[:, None, None] / chi_o**2
        self._cache = (key, masses)
        return masses


class SigmaPrj:
    r"""Projected two-halo surface density around a selected cluster.

    Orchestrates the physics ingredients — ``hmf``, ``bias``, ``xi_nl``,
    the miscentering table, and the selection bias ``b_sel`` passed per
    call — and hands every integral to `clenspy.utils.los_integrals`.
    The constructor stores; `build` runs on first use.

    Parameters
    ----------
    cosmology : astropy cosmology, optional
        Default `clenspy.cosmology.fiducial_cosmology`.
    pk : PkGrid, optional
        Nonlinear P(k, z); built from ``cosmology`` when the TwoHaloTerm
        step runs.
    hmf : TinkerMassFunction or callable ``n(M, z)``, optional
        Grid model (anything with ``build``) or dn/dM
        [Msun^-1 Mpc^-3 comoving] at physical mass.
    two_halo : TwoHaloTerm, optional
        Supplies ``xi_nl`` when no callable is injected.
    bias : BiasModel or callable ``b(M, z)``, optional
        Grid model or halo bias at physical mass.
    xi_nl : callable ``xi(r, zob)``, optional
        Nonlinear matter correlation, r in comoving Mpc; replaces the
        TwoHaloTerm step (e.g. `clenspy.selection.bsel.XiNL` with
        ``clip=False`` — the BAO trough is inside the window).
    concentration : callable ``c(M, z)``, optional
        Physical mass. Default: Duffy08 "200m", with the one visible
        M -> M h conversion here.
    config : SigmaPrjConfig, optional
        Numerical, line-of-sight, and profile settings.

    Examples
    --------
    Cosmology triggers all, with the mock's hard window::

        cfg = SigmaPrjConfig(los_window="hard", los_depth=71.4)
        prj = SigmaPrj(cosmology=cosmo, config=cfg)
        sig = prj.sigma_prj(R, lob=20.0, zob=0.5, b_sel=bsel)

    Injected ingredients on one shared (M, z) grid::

        hmf = TinkerMassFunction(cosmo=cosmo, mvec=M, zvec=z)
        bias = BiasModel(cosmo=cosmo, mvec=M, zvec=z)
        prj = SigmaPrj(cosmology=cosmo, hmf=hmf, bias=bias,
                       xi_nl=XiNL(pk, clip=False), config=cfg)

    Both channels of the last call stay on ``self``::

        prj.deltasigma_prj(R, lob, zob, bsel)
        parts = prj.components()          # {"rnd", "cl", "sum"}
    """

    def __init__(
        self,
        *,
        cosmology=None,
        pk=None,
        hmf=None,
        two_halo=None,
        bias=None,
        xi_nl: Callable | None = None,
        concentration: Callable | None = None,
        config: SigmaPrjConfig | None = None,
    ) -> None:
        self.cosmo = fiducial_cosmology() if cosmology is None else cosmology
        self.h = self.cosmo.h
        cfg = self.config = SigmaPrjConfig() if config is None else config
        self._validate(cfg)
        self.k_exc = Exclusion(cfg.exclusion, cfg.floor_one_plus_bxi)
        self.concentration = (
            concentration if concentration is not None
            # physical Msun -> h^-1 Msun once, visibly
            else (lambda m, z: duffy08(np.asarray(m, float) * self.h, z,
                                       mass_def="200m"))
        )
        self.mis_table = (cfg.mis_table if cfg.mis_table is not None
                          else load_nfw_miscentering_table())
        # mass range default: the RichnessSelection range, h applied once
        self.min_mass = (cfg.min_mass if cfg.min_mass is not None
                         else 1.0e13 / self.h)
        self.log10_M_max = (
            cfg.log10_M_max if cfg.log10_M_max is not None
            else np.log10(10.0**15.5 / self.h)
        )
        # fast comoving-distance interpolant [Mpc], shared with SelBiasEngine
        self.distance = ComovingDistance(self.cosmo)
        self._window = y3_photoz_window()
        self.rho_m = mean_matter_density(self.cosmo)
        self.shells = MassShells(
            mis_table=self.mis_table, distance=self.distance,
            concentration=self.concentration, rho_m=self.rho_m, h=self.h,
            config=cfg, min_mass=self.min_mass,
            log10_M_max=self.log10_M_max)
        # the constructor stores; build() runs on first use (or explicitly)
        self._models = {"pk": pk, "hmf": hmf, "two_halo": two_halo,
                        "bias": bias, "xi_nl": xi_nl}
        self._built = False
        self._los_weight_cache: dict = {}  # b_sel-independent LOS integrals
        # channels of the last projection (rule 6: keep the decomposition)
        self.rnd: np.ndarray | None = None
        self.cl: np.ndarray | None = None

    @staticmethod
    def _validate(cfg) -> None:
        if not isinstance(cfg, SigmaPrjConfig):
            raise TypeError("config must be a SigmaPrjConfig")
        if cfg.los_window not in ("wpz", "hard"):
            raise ValueError(
                f"los_window must be 'wpz' or 'hard', got {cfg.los_window!r}")
        if cfg.los_window == "hard" and cfg.los_depth is None:
            raise ValueError("los_window='hard' needs los_depth [comoving Mpc]")
        # exclusion mode is validated by Exclusion itself

    def build(self, *, pk=None, hmf=None, two_halo=None, bias=None,
              xi_nl=None):
        """Build the projection products once and expose grid interpolators.

        Injected actors are kept; missing ones are constructed from the
        chain cosmo -> Pk -> xi -> HMF/Bias on one shared (M, z) grid.
        ``hmf``/``bias`` accept a grid model (anything with ``build``)
        or a plain callable ``n(M, z)`` [Msun^-1 Mpc^-3] / ``b(M, z)``;
        ``xi_nl`` a callable ``xi(r, zob)`` replacing the TwoHaloTerm step.
        Arguments override those stored at construction.
        """
        m = dict(self._models)
        m.update({k: v for k, v in dict(pk=pk, hmf=hmf, two_halo=two_halo,
                                        bias=bias, xi_nl=xi_nl).items()
                  if v is not None})
        # the chain, in dependency order (rule 7): each row is
        # (actor, needed, factory); a factory sees the actors before it
        for name, needed, make in (
            ("pk", m["xi_nl"] is None and m["two_halo"] is None,
             lambda: PkGrid(cosmo=self.cosmo, nonlinear=True)),
            ("two_halo", m["xi_nl"] is None,
             lambda: TwoHaloTerm(m["pk"].k, m["pk"].pk, zvec=m["pk"].z,
                                 n_grid=600, r_min=1.0e-2, r_max=800.0)),
            ("xi_nl", True, lambda: self._xi_of(m["two_halo"])),
            ("hmf", True, lambda: TinkerMassFunction(
                cosmo=self.cosmo, **self._default_grid())),
            ("bias", True, lambda: BiasModel(
                cosmo=self.cosmo, **self._default_grid(P=True))),
        ):
            if needed and m[name] is None:
                m[name] = make()
        self.pk, self.two_halo, self.xi_nl = m["pk"], m["two_halo"], m["xi_nl"]
        # the models as handed in (or built); self.hmf/self.bias below are
        # their (M, z) evaluators
        self.hmf_model, self.bias_model = m["hmf"], m["bias"]
        self.hmf = self._field(m["hmf"], lambda g: (
            np.asarray(g.dndlnm_grid)
            / np.asarray(g.mval, float)[:, None]))     # dn/dM from dn/dlnM
        self.bias = self._field(m["bias"], lambda g: np.asarray(g.bias_grid))
        self._built = True
        return self

    @staticmethod
    def _xi_of(two_halo):
        """xi(r, z) from a TwoHaloTerm — xi only, not the Sigma grids."""
        if not getattr(two_halo, "is_built", False):
            two_halo.xi()
        return two_halo.xi

    def _default_grid(self, P=False):
        """One linear P(k), one sigma(M), one (M, z) grid for hmf and bias."""
        if getattr(self, "_lin", None) is None:
            lin = self._lin = PkGrid(cosmo=self.cosmo, nonlinear=False)
            pk0 = np.asarray(lin(lin.k, z=0.0), dtype=float)
            # one sigma(M) table: the two Tinker fits share one peak height
            self._shared = {"pk0": pk0, "sigma_grid": SigmaGrid(lin.k, pk0)}
        lin, s = self._lin, self._shared
        mgrid = np.geomspace(self.min_mass, 10.0**self.log10_M_max, 256)
        return {"k": lin.k, ("P" if P else "pk"): s["pk0"],
                "mvec": mgrid, "zvec": lin.z,
                "sigma_grid": s["sigma_grid"]}

    @property
    def is_built(self) -> bool:
        """Whether `build` has run (it runs itself on the first query)."""
        return self._built

    @staticmethod
    def _field(model, grid_of):
        """(M, z) evaluator from a grid model or a plain callable."""
        if callable(model) and not hasattr(model, "build"):
            return model
        model.build()
        return LogGridInterpolator(np.asarray(model.mval, dtype=float),
                                   np.asarray(model.zvec, dtype=float),
                                   grid_of(model))

    def common(self, zs, zob: float):
        r"""E.3: :math:`{\rm common}(z) = \frac{dV}{d\Omega dz} w_{pz}` —
        **no** :math:`\Omega(z)` (it cancels in the surface density)."""
        dV = comoving_volume_element(zs, self.cosmo)
        if self.config.los_window == "hard":
            w_pz = np.ones_like(np.asarray(zs, dtype=float))
        else:
            w_pz = photoz_projection(zs, zob, self._window, n_sigma=1.0)
        return dV * w_pz

    def _geometry(self, thetas, lob: float, zob: float) -> LosGeometry:
        r"""Exact chord and exclusion limits on both cosh--Abel branches,
        over the projection support of the photo-z (or hard) window."""
        chi_o = float(self.distance.chi(zob))
        if self.config.los_window == "hard":
            chi_min = chi_o - self.config.los_depth
            chi_max = chi_o + self.config.los_depth
        else:
            z_min, z_max = photoz_projection_support(
                zob, self._window, n_sigma=1.0)
            chi_min = float(self.distance.chi(z_min))
            chi_max = float(self.distance.chi(z_max))
        r_ex = (r_excl(lob, zob, self.h)
                if self.config.exclusion != "none" else 0.0)
        return LosGeometry(thetas, chi_o, chi_min, chi_max, r_excl=r_ex)

    def n_los_integral(self, lob: float, zob: float, b_sel: Callable):
        r"""The three cosh--Abel z integrals of the master equation, each
        (n_theta, n_M): the background weight :math:`n_{\rm rnd}` inside
        and outside the exclusion sphere, and the correlated weight
        :math:`n_{\rm lss} = b_{\rm sel}(\theta)\int dz\,{\rm common}\,
        n\,b\,\xi_{\rm NL}` outside it — ``b_sel`` is applied here.

        The b_sel-independent products are cached per (lob, zob); the
        positivity floor couples b_sel into the integrand nonlinearly and
        skips the cache (its n_lss is then the floored full bracket).
        """
        cfg = self.config
        if not self._built:
            self.build()
        _, thetas, _ = self.shells.theta_shells(lob, zob)
        b_sel_values = np.broadcast_to(
            np.asarray(b_sel(thetas), dtype=float), thetas.shape)

        key = (round(lob, 8), round(zob, 8))
        if not cfg.floor_one_plus_bxi and key in self._los_weight_cache:
            n_rnd_in, n_rnd_out, n_lss = self._los_weight_cache[key]
            return n_rnd_in, n_rnd_out, n_lss * b_sel_values[:, None]

        Ms, M_weight = mass_nodes(self.min_mass, 10.0**self.log10_M_max,
                                  cfg.n_M)
        geometry = self._geometry(thetas, lob, zob)

        def n_rnd_integrand(r, chi, theta_index):
            ## master equation, background: common(z)/(dchi/dz) n(M,z) M dlnM
            z = self.distance.z_of_chi(chi)
            common = (self.common(z.ravel(), zob)
                      / self.distance.dchi_dz(z.ravel())).reshape(z.shape)
            n_M = self.hmf(Ms, z.ravel()).reshape(Ms.size, *z.shape)
            return common * n_M * M_weight[:, None, None]

        def n_lss_integrand(r, chi, theta_index):
            ## master equation, correlated excess: n_rnd x b(M,z) xi_NL(r)
            z_flat = self.distance.z_of_chi(chi).ravel()
            xi = self.xi_nl(r.ravel(), zob).reshape(r.shape)
            b = self.bias(Ms, z_flat).reshape(Ms.size, *r.shape)
            return n_rnd_integrand(r, chi, theta_index) * b * xi

        def n_lss_rnd_integrand(r, chi, theta_index):
            ## floored full bracket n_rnd x max(1 + b_sel b xi, 0) —
            ## background included, nonlinear in b_sel
            z_flat = self.distance.z_of_chi(chi).ravel()
            xi = self.xi_nl(r.ravel(), zob).reshape(r.shape)
            b = self.bias(Ms, z_flat).reshape(Ms.size, *r.shape)
            q = b_sel_values[theta_index][None, :, None] * xi * b
            return (n_rnd_integrand(r, chi, theta_index)
                    * np.maximum(1.0 + q, 0.0))

        correlated = (n_lss_rnd_integrand if cfg.floor_one_plus_bxi
                      else n_lss_integrand)
        n_rnd_in = integrate_los(geometry, n_rnd_integrand,
                                 cfg.n_u_inside, "inside")
        n_rnd_out = integrate_los(geometry, n_rnd_integrand,
                                  cfg.n_u_outside, "outside")
        n_lss = integrate_los(geometry, correlated,
                              cfg.n_u_outside, "outside")

        if not cfg.floor_one_plus_bxi:
            self._los_weight_cache[key] = (n_rnd_in, n_rnd_out, n_lss)
            n_lss = n_lss * b_sel_values[:, None]
        return n_rnd_in, n_rnd_out, n_lss

    def sigma_prj(self, R, lob: float, zob: float, b_sel: Callable,
                  channel: str = "cl"):
        r""":math:`\Sigma_{\rm prj}(R \mid \lambda^{\rm ob}, z^{\rm ob})`
        [comoving Msun/Mpc^2]. ``b_sel``: any callable of theta [rad].
        Default ``channel="cl"`` is the correlated excess — what a
        random-point-subtracted measurement contains; ``"sum"`` adds the
        mean-background rnd channel (a raw mass map includes it)."""
        if channel not in ("cl", "sum", "rnd"):
            raise ValueError(
                f"channel must be 'cl', 'sum' or 'rnd', got {channel!r}")

        n_rnd_in, n_rnd_out, n_lss = self.n_los_integral(lob, zob, b_sel)
        n_rnd, n_cl = self.k_exc.channels(n_rnd_in, n_rnd_out, n_lss)
        masses = self.shells(R, lob, zob, "sigma")
        # master equation: sum over theta shells (t) and halo masses (m)
        self.rnd = (n_rnd[:, :, None] * masses).sum(axis=(0, 1))
        self.cl = (n_cl[:, :, None] * masses).sum(axis=(0, 1))
        return {"cl": self.cl, "rnd": self.rnd,
                "sum": self.rnd + self.cl}[channel]

    def deltasigma_prj(self, R, lob: float, zob: float, b_sel: Callable,
                       channel: str = "cl"):
        r""":math:`\Delta\Sigma_{\rm prj}(R)` [comoving Msun/Mpc^2] — its
        own integral with the swap :math:`\Sigma_{\rm mis} \to
        \Delta\Sigma_{\rm mis}` (signed, never clamped); the excess
        functional annihilates the uniform rnd channel, so the default
        ``"cl"`` is the physics."""
        if channel not in ("cl", "sum", "rnd"):
            raise ValueError(
                f"channel must be 'cl', 'sum' or 'rnd', got {channel!r}")

        n_rnd_in, n_rnd_out, n_lss = self.n_los_integral(lob, zob, b_sel)
        n_rnd, n_cl = self.k_exc.channels(n_rnd_in, n_rnd_out, n_lss)
        masses = self.shells(R, lob, zob, "ds")
        self.rnd = (n_rnd[:, :, None] * masses).sum(axis=(0, 1))
        self.cl = (n_cl[:, :, None] * masses).sum(axis=(0, 1))
        return {"cl": self.cl, "rnd": self.rnd,
                "sum": self.rnd + self.cl}[channel]

    def components(self) -> dict:
        """The channels of the last projection (Estimator contract)."""
        return {"rnd": self.rnd, "cl": self.cl,
                "sum": None if self.rnd is None else self.rnd + self.cl}


if __name__ == "__main__":
    # the real halo model — Tinker (2008) mass function, Tinker (2010)
    # bias, CAMB halofit xi_NL. PkGrid disk-caches the CAMB call, so this
    # costs seconds once and milliseconds after.
    from ..selection.bsel import SigmoidBias, XiNL

    cosmo = fiducial_cosmology()
    # one shared (M, z) grid for the mass function and the bias
    mgrid = np.geomspace(1.0e12, 1.0e16, 256)
    zgrid = np.linspace(0.0, 1.0, 21)
    tmf = TinkerMassFunction(cosmo=cosmo, mvec=mgrid, zvec=zgrid)
    bias_model = BiasModel(cosmo=cosmo, mvec=mgrid, zvec=zgrid)
    xi_nl = XiNL(PkGrid(cosmo=cosmo, nonlinear=True), clip=False)

    # default exclusion="counter": zeroes the neighbour count inside the
    # ball -- the one mode whose cl channel is the mode-invariant
    # random-subtracted excess
    # (under "ball" the exclusion hole is booked in rnd instead, and the
    # default channel="cl" of deltasigma_prj would silently omit it)
    prj = SigmaPrj(cosmology=cosmo, xi_nl=xi_nl, hmf=tmf, bias=bias_model,
                   config=SigmaPrjConfig(los_window="hard", los_depth=71.4))
    lob, zob = 20.0, 0.5
    bsel = SigmoidBias(lob=lob, zob=zob,
                       theta_lambda=(r_excl(lob, zob, prj.h)
                                     / prj.distance.chi(zob)),
                       b_small=4.0, b_large=3.0)
    R = np.array([0.5, 1.0, 3.0, 10.0, 20.0, 40.0])  # comoving Mpc

    sig = prj.sigma_prj(R, lob, zob, bsel, channel="sum")
    parts = prj.components()
    print(f"Sigma_prj at (lob={lob}, zob={zob}), hard +/-{prj.config.los_depth} cMpc:")
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
    prj0 = SigmaPrj(cosmology=cosmo, xi_nl=xi_nl, hmf=tmf, bias=bias_model,
                    config=SigmaPrjConfig(los_window="hard", los_depth=71.4,
                                          exclusion="none"))
    prj0.sigma_prj(R, lob, zob, lambda th: 0.0)
    column = prj0.rho_m * 2.0 * prj0.config.los_depth
    print(f"\nno-exclusion rnd plateau: {prj0.rnd[0]:.4e} Msun/Mpc^2")
    print(f"rho_m x 2 depth          : {column:.4e} Msun/Mpc^2 "
          f"(ratio {prj0.rnd[0] / column:.3f} = halo-budget fraction; the "
          "untruncated NFW wings push it above the naive mass fraction)")
