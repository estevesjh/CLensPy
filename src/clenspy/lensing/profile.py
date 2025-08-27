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
    """
    A unified class for weak lensing calculations.

    This class computes weak-lensing observables like ΔΣ(R), Σ(R), shear, etc.

    Attributes:
        zCluster (float): Redshift of the cluster.
        m200 (float): Mass of the halo in Msun.
        cosmology (Cosmology): Cosmology object for calculations.
        concentration (float): Concentration parameter of the halo.
        model (str): Halo profile model, currently only supports "NFW".
        include_2halo (bool): Whether to include the two-halo term in calculations.
        backend_2halo (str): Backend for two-halo term calculations, default is "camb".
        z_source (float): Redshift of the source galaxy for lensing calculations.

    Methods:
        deltasigma(R): Computes the excess surface density ΔΣ(R).
        sigma(R): Computes the surface density Σ(R).
        density(r): Computes the 3D density profile ρ(r).
        shear(R): Computes the tangential shear γ_t(R).
        convergence(R): Computes the convergence κ(R).
        reduced_shear(R): Computes the reduced shear g_t(R).
        fourier_profile(k): Computes the Fourier transform of the halo profile.
        info(): Returns a summary of the profile parameters.
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
        """Excess surface density ΔΣ(R) in Msun/Mpc^2."""
        deltasigma = self.halo_profile.deltasigma(R)
        if self.include_2halo:
            deltasigma2h = self.two_halo_profile.deltasigma(R, self.z_cluster)
            deltasigma2h *= 1e12  # keep scaling from earlier version
            deltasigma += self.bias * deltasigma2h
        return deltasigma

    def sigma(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Surface density Σ(R) in Msun/Mpc^2."""
        sigma = self.halo_profile.sigma(R)
        if self.include_2halo:
            sigma2h = self.rho_m * self.two_halo_profile.sigma(R, self.z_cluster)
            sigma += self.bias * sigma2h
        return sigma

    def density(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """3D density ρ(r) in Msun/Mpc^3."""
        density = self.halo_profile.density(r)
        
        if self.include_2halo:
            xi = self.two_halo_profile.xi(r, self.z_cluster)
            density += self.rho_m * (1 + self.bias * xi)

        return density

    def shear(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Tangential shear γ_t(R) (dimensionless)."""
        _deltasigma = self.deltasigma(R)
        return _deltasigma / self._sigma_crit

    def convergence(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convergence κ(R) (dimensionless)."""
        _sigma = self.sigma(R)
        return _sigma / self._sigma_crit

    def reduced_shear(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Reduced shear g_t(R) = γ_t(R) / (1 − κ(R))."""
        _kappa = self.convergence(R)
        if np.any(_kappa >= 1.0):
            raise ValueError(
                "Convergence must be less than 1 for reduced shear calculation"
            )
        _gamma = self.shear(R)
        return _gamma / (1.0 - _kappa)

    def fourier_profile(self, k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Fourier transform u(k)."""
        k = np.atleast_1d(k)
        result = self.halo_profile.fourier(k)
        if self.include_2halo:
            result += self.bias * self.two_halo_profile.pk(k, self.z_cluster) / self.m200

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

