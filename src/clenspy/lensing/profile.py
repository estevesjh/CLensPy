"""
LensingProfile class for weak lensing calculations.

This module provides a unified interface for computing weak lensing observables
from dark matter halo profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Union

import numpy as np
from astropy.cosmology import Cosmology

from ..cosmology import BiasModel, PkGrid
from ..cosmology.fiducial import fiducial_cosmology, mean_matter_density
from ..halo import NfwProfile, TwoHaloTerm
from ..kernels import sigma_critical

__all__ = ["LensingProfile", "LensingProfileInfo"]

#: Halo profile models `LensingProfile` can build, as normalised (lower-case)
#: names. This tuple and the `halo_profile` property are the only authority
#: on what is supported; `_validate_inputs` reads it so an unsupported name
#: is rejected in the constructor, before anything is built.
#:
#: "einasto" is deliberately absent: `EinastoProfile` is parameterised by
#: (alpha, rho_0, r_s) and has no M_200m-based constructor, so wiring it here
#: needs a mass-to-(rho_0, r_s) inversion that does not exist yet.
SUPPORTED_MODELS = ("nfw",)

#: Default wavenumber grid for the 2-halo term [1/Mpc].
#:
#: NOTE: the extent is set by the radii the 2-halo term is asked for, not by
#: taste. `TwoHaloTerm` FFTLogs P(k) to xi(r) and then projects, so the grid
#: must bracket 1/R over the useful range R ~ 0.1-100 Mpc with a decade of
#: margin at each end: k_min = 1e-3, k_max = 10 /Mpc. 100 points is ~25 per
#: decade, which the FFTLog of a smooth linear P(k) converges at. Pass
#: ``k_grid=`` to override.
K_GRID_2HALO = np.logspace(-3, 1, 100)

@dataclass
class LensingProfileInfo:
    """Summary of a `LensingProfile`'s parameters, returned by `LensingProfile.info`.

    NOTE: units follow `LensingProfile` -- ``m200`` in Msun (M_200m, w.r.t.
    200x the comoving mean matter density), ``r200`` and ``rs`` in Mpc,
    ``sigma_crit`` in Msun/Mpc^2, ``H0`` in km/s/Mpc, redshifts and
    ``concentration`` and ``Om0`` dimensionless.
    """

    model: str
    z_cluster: float
    z_source: float
    m200: float
    concentration: float
    r200: float
    rs: float
    sigma_crit: float
    include_2halo: bool
    H0: float
    Om0: float


class LensingProfile:
    r"""
    A unified class for weak lensing calculations, combining a 1-halo term
    (`NfwProfile`) with an optional linear-bias 2-halo term (`TwoHaloTerm`):

    NOTE: masses are **M_200m** (200x the comoving mean matter density at
    z=0), inherited from `NfwProfile` -- not M_200c.

    NOTE: all quantities are h-free absolute units -- mass in Msun, lengths
    in Mpc, Sigma and DeltaSigma in Msun/Mpc^2, wavenumbers in 1/Mpc.

    .. math::
        \Sigma(R) = \Sigma_{\rm 1h}(R) + b(M)\, \rho_m\, \Sigma_{\rm 2h}(R)

    where :math:`b(M)` is the linear halo bias (`BiasModel`) and
    :math:`\rho_m = \Omega_{m,0}\,\rho_{c,0}` is the present-day
    (comoving) mean matter density -- no redshift dependence.

    Attributes
    ----------
    z_cluster : float
        Redshift of the cluster.
    m200 : float
        Halo mass M_200m [Msun], w.r.t. 200x the comoving mean matter density.
    cosmology : astropy.cosmology.Cosmology
        Cosmology used for all calculations.
    concentration : float
        Halo concentration c_200.
    model : str
        Halo profile model, normalised to lower case. See `SUPPORTED_MODELS`;
        currently only "nfw" is implemented.
    include_2halo : bool
        Whether to add the 2-halo term to `sigma`/`deltasigma`/
        `fourier_profile`.
    backend_2halo : str
        `PkGrid` backend used for the 2-halo term's P(k) ("camb" or
        "pyccl").
    z_source : float
        Source redshift used for `sigma_crit`.

    Notes
    -----
    **The constructor stores; it does not compute.** `Pkvec`,
    `two_halo_profile`, `bias_model`, `bias`, `halo_profile` and
    `sigma_crit` are `functools.cached_property`, so
    ``LensingProfile(z_cluster=0.3, m200=1e14)`` costs nothing and only
    `Pkvec` starts a Boltzmann solver -- on the first call that needs it.
    Pass ``two_halo=``, ``bias=`` or ``halo_profile=`` to supply a
    collaborator instead, which is how a driver reuses one P(k) across many
    halos. Only ``_validate_inputs`` runs eagerly, so a bad redshift or mass
    still raises at construction rather than pages later.

    `deltasigma`'s 2-halo term previously multiplied by a hardcoded
    ``1e12`` instead of ``self.rho_m`` (the factor `sigma`'s 2-halo term
    correctly uses - `TwoHaloTerm.sigma`/`deltasigma` are both derived from
    the same un-normalized xi(r) grid, so they need the same normalization).
    Confirmed this was wrong empirically (`deltasigma` came out ~1e10x
    larger than `sigma`, giving unphysical shear >> 1) and against
    `tests/test_twohalo.py`'s own cluster_toolkit-validated reference
    pattern, which uses ``TwoHaloTerm.deltasigma(...) * rho_m`` with no
    extra factor. Fixed to match.
    """

    def __init__(
        self,
        z_cluster: float,
        m200: float,
        cosmology: Cosmology | None = None,
        concentration: float = 4.0,
        model: str = "NFW",
        include_2halo: bool = True,
        backend_2halo: str = "camb",
        z_source: float = 1.0,
        halo_profile=None,
        two_halo=None,
        bias: float | None = None,
        k_grid=None,
    ) -> None:
        """
        Parameters
        ----------
        z_cluster : float
            Cluster (lens) redshift.
        m200 : float
            Halo mass M_200m [Msun], w.r.t. 200x the comoving mean matter density.
        cosmology : astropy.cosmology.Cosmology, optional
            Cosmology to use (default: `fiducial_cosmology()`).
        concentration : float, optional
            Halo concentration c_200 (default: 4.0).
        model : str, optional
            Halo profile model, case-insensitive and normalised to lower case.
            See `SUPPORTED_MODELS`; only "nfw" is implemented (default: "NFW").
        include_2halo : bool, optional
            Whether to add the 2-halo term (default: True). Nothing is built
            until an observable is evaluated -- see the class Notes.
        backend_2halo : {"camb", "pyccl"}, optional
            `PkGrid` backend for the 2-halo term's P(k) (default: "camb").
        z_source : float, optional
            Source redshift, must exceed ``z_cluster`` (default: 1.0).
        halo_profile : optional
            A pre-built 1-halo profile, e.g. `NfwProfile`. Built from
            ``model`` if omitted.
        two_halo : optional
            A pre-built `TwoHaloTerm`. Built from ``k_grid`` and a fresh
            `~clenspy.cosmology.PkGrid` if omitted -- which is what runs the
            Boltzmann solver, so pass one to reuse a P(k).
        bias : float, optional
            Linear halo bias :math:`b(M)`. Computed from the same P(k) as
            ``two_halo`` if omitted. Pass a number to fix it to a published
            value and skip that computation.
        k_grid : array-like, optional
            Wavenumbers for the 2-halo term [1/Mpc]. Defaults to
            `K_GRID_2HALO`; see the note there for why that extent.
        """
        self.cosmo = fiducial_cosmology() if cosmology is None else cosmology
        self.z_cluster = z_cluster
        self.m200 = m200
        self.concentration = concentration
        self.model = model.lower()
        self.include_2halo = include_2halo
        self.backend_2halo = backend_2halo
        self.z_source = z_source
        self.omega_m = self.cosmo.Om0
        self.kvec = K_GRID_2HALO if k_grid is None else np.asarray(k_grid, float)

        # Collaborators the caller supplied, stored verbatim. Each has a
        # cached_property below that builds it on first use if it is None.
        self._halo_profile = halo_profile
        self._two_halo = two_halo
        self._bias = bias

        # Comoving, so no redshift dependence: the 2h tables and NfwProfile
        # are both comoving, and rho_c(z) here would mix in E^2(z).
        self.rho_m = mean_matter_density(self.cosmo)

        # Cheap, and it must fail before anything expensive is attempted.
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.z_cluster < 0:
            raise ValueError("Cluster redshift must be non-negative")
        if self.z_source <= self.z_cluster:
            raise ValueError("Source redshift must be greater than cluster redshift")
        if self.m200 <= 0:
            raise ValueError("Mass must be positive")
        if self.concentration <= 0:
            raise ValueError("Concentration must be positive")
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{self.model}' not supported. "
                f"Available: {', '.join(SUPPORTED_MODELS)}"
            )

    # -- collaborators, built on first use --------------------------------

    @cached_property
    def halo_profile(self):
        """The 1-halo profile: as supplied, or built from ``model``."""
        if self._halo_profile is not None:
            return self._halo_profile
        if self.model == "nfw":
            # A density, not a cosmology. Passing rho_m here is what makes
            # this class's m200 mean M_200m; we already hold that number.
            return NfwProfile(
                m200=self.m200, c200=self.concentration, rho_ref=self.rho_m
            )
        # pragma: no cover - _validate_inputs rejects anything else first
        raise NotImplementedError(f"Model '{self.model}' not implemented")

    @cached_property
    def Pkvec(self):
        """Linear P(k) on `kvec` at ``z_cluster`` [Mpc^3]. Runs the solver."""
        return PkGrid(cosmo=self.cosmo, backend=self.backend_2halo)(
            self.kvec, self.z_cluster
        )

    @cached_property
    def two_halo_profile(self):
        """The 2-halo term: as supplied, or built from `Pkvec`."""
        if self._two_halo is not None:
            return self._two_halo
        return TwoHaloTerm(self.kvec, self.Pkvec, zvec=self.z_cluster)

    @cached_property
    def bias_model(self) -> BiasModel:
        """`BiasModel` on the same P(k) the 2-halo term uses."""
        return BiasModel(self.kvec, self.Pkvec, cosmo=self.cosmo, odelta=200)

    @cached_property
    def bias(self) -> float:
        r"""Linear halo bias :math:`b(M_{200m})`, as supplied or computed."""
        if self._bias is not None:
            return self._bias
        return self.bias_model.bias(self.m200)

    @cached_property
    def sigma_crit(self) -> float:
        r""":math:`\Sigma_{\rm crit}(z_l, z_s)` [Msun/Mpc^2]."""
        return sigma_critical(self.z_cluster, self.z_source, self.cosmo)

    def deltasigma(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Excess surface density, in Msun/Mpc^2.

        .. math::
            \Delta\Sigma(R) = \Delta\Sigma_{\rm 1h}(R)
            + b(M)\, \Delta\Sigma_{\rm 2h}(R)

        See the class Notes for a units caveat on the 2-halo term here.
        """
        deltasigma = self.halo_profile.deltasigma(R)
        if self.include_2halo:
            deltasigma2h = self.rho_m * self.two_halo_profile.deltasigma(R, self.z_cluster)
            deltasigma += self.bias * deltasigma2h
        return deltasigma

    def sigma(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Surface density, in Msun/Mpc^2.

        .. math::
            \Sigma(R) = \Sigma_{\rm 1h}(R) + b(M)\, \rho_m\, \Sigma_{\rm 2h}(R)
        """
        sigma = self.halo_profile.sigma(R)
        if self.include_2halo:
            sigma2h = self.rho_m * self.two_halo_profile.sigma(R, self.z_cluster)
            sigma += self.bias * sigma2h
        return sigma

    def density(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        3D density, in Msun/Mpc^3.

        .. math::
            \rho(r) = \rho_{\rm 1h}(r) + \rho_m \left[1 + b(M)\, \xi_{\rm 2h}(r)\right]

        where :math:`\xi_{\rm 2h}` is the matter correlation function
        (`TwoHaloTerm.xi`).
        """
        density = self.halo_profile.density(r)

        if self.include_2halo:
            xi = self.two_halo_profile.xi(r, self.z_cluster)
            density += self.rho_m * (1 + self.bias * xi)

        return density

    def shear(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Tangential shear (dimensionless).

        .. math::
            \gamma_t(R) = \frac{\Delta\Sigma(R)}{\Sigma_{\rm crit}}
        """
        _deltasigma = self.deltasigma(R)
        return _deltasigma / self.sigma_crit

    def convergence(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Convergence (dimensionless).

        .. math::
            \kappa(R) = \frac{\Sigma(R)}{\Sigma_{\rm crit}}
        """
        _sigma = self.sigma(R)
        return _sigma / self.sigma_crit

    def reduced_shear(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Reduced shear.

        .. math::
            g_t(R) = \frac{\gamma_t(R)}{1 - \kappa(R)}

        Raises
        ------
        ValueError
            If :math:`\kappa(R) \geq 1` anywhere (the reduced-shear
            expansion is invalid in the strong-lensing regime).
        """
        _kappa = self.convergence(R)
        if np.any(_kappa >= 1.0):
            raise ValueError(
                "Convergence must be less than 1 for reduced shear calculation"
            )
        _gamma = self.shear(R)
        return _gamma / (1.0 - _kappa)

    def fourier_profile(self, k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Intended: the mass-normalized Fourier transform
        :math:`u(k) \equiv \tilde\rho(k)/M`,

        .. math::
            u(k) = u_{\rm 1h}(k) + \frac{b(M)}{M}\, P_{\rm 2h}(k).

        NOTE: **not what this currently returns.**
        ``self.halo_profile.fourier(k)`` is :math:`\tilde\rho_{\rm 1h}(k)`
        [Msun], *not* divided by ``m200`` (see `NfwProfile.fourier`'s own
        NOTE) -- while the 2-halo term here already is. The two terms are
        off by a factor of order :math:`M_{200}` in scale, so the sum below
        is :math:`\tilde\rho_{\rm 1h}(k) + b(M)P_{\rm 2h}(k)/M`, not
        :math:`u(k)`: the 1-halo term dominates at every :math:`k`, and the
        2-halo term is numerically swamped rather than combined. Divide
        ``self.halo_profile.fourier(k)`` by ``self.m200`` to get the
        :math:`u(k)` documented above; not fixed here since it changes this
        method's public output.

        Parameters
        ----------
        k : float or np.ndarray
            Wavenumber [1/Mpc].

        Returns
        -------
        float or np.ndarray
            Scalar if ``k`` was scalar -- see the NOTE for what it means.
        """
        k = np.atleast_1d(k)
        result = self.halo_profile.fourier(k)
        if self.include_2halo:
            result += self.bias * self.two_halo_profile.p_kz(k, self.z_cluster) / self.m200

        return result if np.ndim(k) > 0 else np.asarray(result).item()

    @property
    def info(self) -> LensingProfileInfo:
        """Return a summary dictionary of the profile parameters."""
        return LensingProfileInfo(
            model=self.model,
            z_cluster=self.z_cluster,
            z_source=self.z_source,
            m200=self.m200,
            concentration=self.concentration,
            r200=self.halo_profile.r200,
            rs=self.halo_profile.rs,
            sigma_crit=self.sigma_crit,
            include_2halo=self.include_2halo,
            H0=self.cosmo.H0.to_value("km / (s Mpc)"),
            Om0=self.cosmo.Om0,
        )
    def __repr__(self) -> str:
        return (
            f"LensingProfile(model={self.model}, z_cluster={self.z_cluster:0.3f}, "
            f"m200={self.m200:.2e}, c={self.concentration:0.2f}), include_2halo={self.include_2halo})"
        )


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    lp = LensingProfile(z_cluster=0.3, m200=1e14, concentration=4.0)
    print(f"construct: {(time.perf_counter() - t0) * 1e3:.1f} ms  (nothing built)")
    print(lp)

    R = np.array([0.1, 0.5, 1.0, 5.0])
    t0 = time.perf_counter()
    ds = lp.deltasigma(R)
    print(f"first deltasigma: {time.perf_counter() - t0:.2f} s  (P(k) built here)")
    print("  R [Mpc]                  ", R)
    print("  DeltaSigma [Msun/Mpc^2]  ", ds)
    print(f"  b(M) = {lp.bias:.3f}   Sigma_crit = {lp.sigma_crit:.3e} Msun/Mpc^2")

    ds_1h = LensingProfile(
        z_cluster=0.3, m200=1e14, include_2halo=False
    ).deltasigma(R)
    print("  1-halo only              ", ds_1h)
    print("  2-halo fraction          ", 1.0 - ds_1h / ds)
