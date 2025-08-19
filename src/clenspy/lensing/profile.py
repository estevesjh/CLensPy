"""
LensingProfile class for weak lensing calculations.

This module provides a unified interface for computing weak lensing observables
from dark matter halo profiles.
"""

from typing import Any, Dict, Union

import numpy as np
from astropy.cosmology import Cosmology

from ..config import DEFAULT_COSMOLOGY
from ..cosmology import PkGrid, sigma_critical
from ..halo import NFWProfile, TwoHaloTerm, biasModel


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
        include2Halo (bool): Whether to include the two-halo term in calculations.
        backend2Halo (str): Backend for two-halo term calculations, default is "camb".
        zSource (float): Redshift of the source galaxy for lensing calculations.

    Methods:
        deltaSigmaR(R): Computes the excess surface density ΔΣ(R).
        sigmaR(R): Computes the surface density Σ(R).
        density3D(r): Computes the 3D density profile ρ(r).
        shear(R): Computes the tangential shear γ_t(R).
        convergence(R): Computes the convergence κ(R).
        reducedShear(R): Computes the reduced shear g_t(R).
        fourierProfile(k): Computes the Fourier transform of the halo profile.
        info(): Returns a summary of the profile parameters.
    """

    def __init__(
        self,
        zCluster: float,
        m200: float,
        cosmology: Cosmology = DEFAULT_COSMOLOGY,
        concentration: float = 4.0,
        model: str = "NFW",
        include2Halo: bool = True,
        backend2Halo: str = "camb",
        zSource: float = 1.0,
    ) -> None:
        self.cosmo = cosmology
        self.zCluster = zCluster
        self.m200 = m200
        self.concentration = concentration
        self.model = model.upper()
        self.include2Halo = include2Halo
        self.zSource = zSource
        self.omega_m = self.cosmo.Om0

        rhocrit = self.cosmo.critical_density(zCluster).to_value("Msun/Mpc^3")
        self.rho_m = rhocrit * self.omega_m

        # Validate inputs
        self._validate_inputs()

        # Initialize matter power spectrum grid
        if self.include2Halo:
            self.kvec = np.logspace(-3, 1, 100)
            bPk = PkGrid(cosmo=self.cosmo, backend=backend2Halo)
            self.Pkvec = bPk(self.kvec, self.zCluster)

        # Initialize halo profile
        self._setupHaloProfile()

        # Critical surface density using cosmology utils
        self._sigmaCritical = sigma_critical(self.zCluster, self.zSource, self.cosmo)

        # Halo bias if needed
        if self.include2Halo:
            self.bias_model = biasModel(
                self.kvec, self.Pkvec, cosmo=self.cosmo, odelta=200
            )
            self.bias = self.bias_model.biasAtM(self.m200)

    def _validate_inputs(self) -> None:
        if self.zCluster < 0:
            raise ValueError("Cluster redshift must be non-negative")
        if self.zSource <= self.zCluster:
            raise ValueError("Source redshift must be greater than cluster redshift")
        if self.m200 <= 0:
            raise ValueError("Mass must be positive")
        if self.concentration <= 0:
            raise ValueError("Concentration must be positive")
        if self.model not in ["NFW"]:
            raise ValueError(f"Model '{self.model}' not supported. Available: NFW")

    def _setupHaloProfile(self) -> None:
        if self.model == "NFW":
            self.halo_profile = NFWProfile(
                m200=self.m200, c200=self.concentration, cosmo=self.cosmo
            )
        else:
            raise NotImplementedError(f"Model {self.model} not implemented")

        if self.include2Halo:
            self.two_halo_profile = TwoHaloTerm(
                self.kvec, self.Pkvec, zvec=self.zCluster
            )

    def deltaSigmaR(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Excess surface density ΔΣ(R) in Msun/Mpc^2."""
        deltasigma = self.halo_profile.deltaSigma(R)
        if self.include2Halo:
            deltasigma2h = self.rho_m * self.two_halo_profile.deltaSigma(R)
            deltasigma2h *= 1e12  # keep scaling from earlier version
            deltasigma += self.bias * deltasigma2h
        return deltasigma

    def sigmaR(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Surface density Σ(R) in Msun/Mpc^2."""
        sigma = self.halo_profile.sigma(R)
        if self.include2Halo:
            sigma2h = self.rho_m * self.two_halo_profile.sigma(R)
            sigma += self.bias * sigma2h
        return sigma

    def density3D(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """3D density ρ(r) in Msun/Mpc^3."""
        return self.halo_profile.density(r)

    def shear(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Tangential shear γ_t(R) (dimensionless)."""
        delta_sigma = self.deltaSigmaR(R)
        return delta_sigma / self._sigmaCritical

    def convergence(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convergence κ(R) (dimensionless)."""
        sigma = self.sigmaR(R)
        return sigma / self._sigmaCritical

    def reducedShear(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Reduced shear g_t(R) = γ_t(R) / (1 − κ(R))."""
        kappa = self.convergence(R)
        if np.any(kappa >= 1.0):
            raise ValueError(
                "Convergence must be less than 1 for reduced shear calculation"
            )
        gamma = self.shear(R)
        return gamma / (1.0 - kappa)

    def fourierProfile(self, k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Fourier transform u(k)."""
        k = np.atleast_1d(k)
        result = self.halo_profile.fourier(k)
        return result if np.ndim(k) > 0 else np.asarray(result).item()

    def info(self) -> Dict[str, Any]:
        """Return a summary dictionary of the profile parameters."""
        return {
            "model": self.model,
            "z_cluster": self.zCluster,
            "zSource": self.zSource,
            "m200": self.m200,
            "concentration": self.concentration,
            "r200": self.halo_profile.r200,
            "rs": self.halo_profile.rs,
            "sigma_crit": self._sigmaCritical,
            "include_2halo": self.include_2halo,
            "H0": self.cosmo.H0.to_value("km/s/Mpc"),
            "Om0": self.cosmo.Om0,
        }

    def __repr__(self) -> str:
        return (
            f"LensingProfile(model={self.model}, z_cluster={self.zCluster}, "
            f"m200={self.m200:.2e}, c={self.concentration})"
        )

__all__ = ["LensingProfile"]
