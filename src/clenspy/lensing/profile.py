"""
LensingProfile class for weak lensing calculations.

This module provides a unified interface for computing weak lensing observables
from dark matter halo profiles.
"""

from dataclasses import dataclass
from typing import Union

import numpy as np
from astropy.cosmology import Cosmology

from ..config import DEFAULT_COSMOLOGY
from ..cosmology import PkGrid, sigma_critical
from ..halo import NfwProfile, TwoHaloTerm, BiasModel

__all__ = ["LensingProfile", "LensingProfileInfo"]

@dataclass
class LensingProfileInfo:
    """Summary of a `LensingProfile`'s parameters, returned by `LensingProfile.info`."""

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

    .. math::
        \Sigma(R) = \Sigma_{\rm 1h}(R) + b(M)\, \rho_m\, \Sigma_{\rm 2h}(R)

    where :math:`b(M)` is the linear halo bias (`BiasModel`) and
    :math:`\rho_m` is the mean matter density at ``z_cluster``.

    Attributes
    ----------
    z_cluster : float
        Redshift of the cluster.
    m200 : float
        Halo mass M_200 [Msun].
    cosmology : astropy.cosmology.Cosmology
        Cosmology used for all calculations.
    concentration : float
        Halo concentration c_200.
    model : str
        Halo profile model; currently only "NFW" is implemented.
    include_2halo : bool
        Whether to add the 2-halo term to `sigma`/`deltasigma`/
        `fourier_profile`.
    backend_2halo : str
        `PkGrid` backend used for the 2-halo term's P(k) ("camb" or
        "pyccl").
    z_source : float
        Source redshift used for `_sigma_crit`.

    Notes
    -----
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
        cosmology: Cosmology = DEFAULT_COSMOLOGY,
        concentration: float = 4.0,
        model: str = "NFW",
        include_2halo: bool = True,
        backend_2halo: str = "camb",
        z_source: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        z_cluster : float
            Cluster (lens) redshift.
        m200 : float
            Halo mass M_200 [Msun].
        cosmology : astropy.cosmology.Cosmology, optional
            Cosmology to use (default: `~clenspy.config.DEFAULT_COSMOLOGY`).
        concentration : float, optional
            Halo concentration c_200 (default: 4.0).
        model : str, optional
            Halo profile model; only "NFW" is implemented (default: "NFW").
        include_2halo : bool, optional
            Whether to add the 2-halo term (default: True). If True, builds
            a `~clenspy.cosmology.PkGrid` (via ``backend_2halo``) and a
            `TwoHaloTerm` at construction time.
        backend_2halo : {"camb", "pyccl"}, optional
            `PkGrid` backend for the 2-halo term's P(k) (default: "camb").
        z_source : float, optional
            Source redshift, must exceed ``z_cluster`` (default: 1.0).
        """
        self.cosmo = cosmology
        self.z_cluster = z_cluster
        self.m200 = m200
        self.concentration = concentration
        self.model = model.upper()
        self.include_2halo = include_2halo
        self.z_source = z_source
        self.omega_m = self.cosmo.Om0

        rhocrit = self.cosmo.critical_density(z_cluster).to_value("Msun/Mpc^3")
        self.rho_m = rhocrit * self.omega_m

        # Validate inputs
        self._validate_inputs()

        # Initialize matter power spectrum grid
        if self.include_2halo:
            self.kvec = np.logspace(-3, 1, 100)
            bPk = PkGrid(cosmo=self.cosmo, backend=backend_2halo)
            self.Pkvec = bPk(self.kvec, self.z_cluster)

        # Initialize halo profile
        self._setup_halo_profile()

        # Critical surface density using cosmology utils
        self._sigma_crit = sigma_critical(self.z_cluster, self.z_source, self.cosmo)

        # Halo bias if needed
        if self.include_2halo:
            self.bias_model = BiasModel(
                self.kvec, self.Pkvec, cosmo=self.cosmo, odelta=200
            )
            self.bias = self.bias_model.bias(self.m200)

    def _validate_inputs(self) -> None:
        if self.z_cluster < 0:
            raise ValueError("Cluster redshift must be non-negative")
        if self.z_source <= self.z_cluster:
            raise ValueError("Source redshift must be greater than cluster redshift")
        if self.m200 <= 0:
            raise ValueError("Mass must be positive")
        if self.concentration <= 0:
            raise ValueError("Concentration must be positive")
        if self.model not in ["NFW", "Einasto"]:
            raise ValueError(f"Model '{self.model}' not supported. Available: NFW, Einasto")

    def _setup_halo_profile(self) -> None:
        if self.model == "NFW":
            self.halo_profile = NfwProfile(
                m200=self.m200, c200=self.concentration, cosmo=self.cosmo
            )
        else:
            raise NotImplementedError(f"Model {self.model} not implemented")

        if self.include_2halo:
            self.two_halo_profile = TwoHaloTerm(
                self.kvec, self.Pkvec, zvec=self.z_cluster
            )

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
        return _deltasigma / self._sigma_crit

    def convergence(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Convergence (dimensionless).

        .. math::
            \kappa(R) = \frac{\Sigma(R)}{\Sigma_{\rm crit}}
        """
        _sigma = self.sigma(R)
        return _sigma / self._sigma_crit

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
        Mass-normalized Fourier transform :math:`u(k) \equiv \tilde\rho(k)/M`.

        .. math::
            u(k) = u_{\rm 1h}(k) + \frac{b(M)}{M}\, P_{\rm 2h}(k)

        Parameters
        ----------
        k : float or np.ndarray
            Wavenumber [1/Mpc].

        Returns
        -------
        float or np.ndarray
            :math:`u(k)`, scalar if ``k`` was scalar.

        Notes
        -----
        Previously called a nonexistent ``self.two_halo_profile.pk(...)``
        method (`TwoHaloTerm` has no `pk`) - would raise `AttributeError`
        whenever ``include_2halo=True`` (the default). Fixed to use
        `TwoHaloTerm`'s actual P(k, z) interpolator, ``p_kz``.
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
            sigma_crit=self._sigma_crit,
            include_2halo=self.include_2halo,
            H0=self.cosmo.H0.to_value("km/s/Mpc"),
            Om0=self.cosmo.Om0,
        )
    def __repr__(self) -> str:
        return (
            f"LensingProfile(model={self.model}, z_cluster={self.z_cluster:0.3f}, "
            f"m200={self.m200:.2e}, c={self.concentration:0.2f}), include_2halo={self.include_2halo})"
        )

