"""
LensingProfile class for weak lensing calculations.

This module provides a unified interface for computing weak lensing observables
from dark matter halo profiles.
"""

from typing import Any, Dict, Union

import numpy as np
from astropy import cosmology

from ..config import DEFAULT_COSMOLOGY
from ..cosmology import PkGrid, sigma_critical
from ..halo import NFWProfile, TwoHaloTerm, biasModel


class LensingProfile:
    """
    A unified class for weak lensing calculations.

    This class handles the computation of various weak lensing observables
    including excess surface density, surface density, and shear profiles.

    Parameters
    ----------
    cosmology : astropy.cosmology object
        Cosmological model (e.g., FlatLambdaCDM)
    z_cluster : float
        Cluster redshift
    m200 : float
        Cluster mass M200 in solar masses
    concentration : float, optional
        Concentration parameter (default: 4.0)
    model : str, optional
        Halo model type (default: "NFW")
    include_2halo : bool, optional
        Whether to include 2-halo term (default: True)
    backend_2halo : str, optional
        Backend for 2-halo term calculations (default: "camb")
    z_source : float, optional
        Source redshift for lensing calculations (default: 1.0)

    Examples
    --------
    >>> from clenspy.lensing import LensingProfile
    >>> from clenspy.config import DEFAULT_COSMOLOGY
    >>> import numpy as np
    >>>
    >>> profile = LensingProfile(
    ...     cosmology=DEFAULT_COSMOLOGY,
    ...     z_cluster=0.3,
    ...     m200=1e15,
    ...     concentration=4.0
    ... )
    >>>
    >>> R = np.logspace(-2, 2, 50)  # Mpc
    >>> dsig = profile.delta_sigma(R)
    """

    def __init__(
        self,
        z_cluster: float,
        m200: float,
        cosmo: cosmology = DEFAULT_COSMOLOGY,
        concentration: float = 4.0,
        model: str = "NFW",
        include_2halo: bool = True,
        backend_2halo: str = "camb",
        z_source: float = 1.0,
    ) -> None:
        self.cosmo = cosmo
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

        # Initialize halo profile
        self._setup_halo_profile()

        # Initialize CAMB Power Spectrum
        if self.include_2halo:
            self.kvec = np.logspace(-3, 1, 100)  # Example k-vector
            bPk = PkGrid(cosmo=self.cosmo, backend=self.backend_2halo)
            self.Pkvec = bPk(self.kvec, self.z_cluster)

        # Calculate critical surface density using new cosmology utils
        self._sigma_crit = sigma_critical(self.z_cluster, self.z_source, self.cosmology)

        # Calculate halo bias if needed
        if self.include_2halo:
            self.bias_model = biasModel(
                self.kvec, self.Pkvec, omega_m=self.cosmo.Om0, odelta=200
            )
            self.bias = self.bias_model.bias_at_M(self.m200)

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if self.z_cluster < 0:
            msg = "Cluster redshift must be non-negative"
            raise ValueError(msg)
        if self.z_source <= self.z_cluster:
            msg = "Source redshift must be greater than cluster redshift"
            raise ValueError(msg)
        if self.m200 <= 0:
            msg = "Mass must be positive"
            raise ValueError(msg)
        if self.concentration <= 0:
            msg = "Concentration must be positive"
            raise ValueError(msg)
        if self.model not in ["NFW"]:
            msg = f"Model '{self.model}' not supported. Available: NFW"
            raise ValueError(msg)

    def _setup_halo_profile(self) -> None:
        """Initialize the halo density profile."""
        if self.model == "NFW":
            self.halo_profile = NFWProfile(
                M200=self.m200,
                c200=self.concentration,
                z=self.z_cluster,
                cosmo=self.cosmo,  # Pass astropy cosmology directly
            )
        elif self.include_2halo:
            # Use TwoHaloTerm for 2-halo term calculations
            self.two_halo_profile = TwoHaloTerm(
                self.kvec,
                self.Pkvec,
                z=self.z_cluster,
                cosmo=self.cosmo,  # Pass astropy cosmology directly
            )
        else:
            msg = f"Model {self.model} not implemented"
            raise NotImplementedError(msg)

    def deltasigmaR(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate excess surface density profile.

        Parameters
        ----------
        R : float or array-like
            Projected radius in Mpc

        Returns
        -------
        float or array-like
            Excess surface density in Msun/Mpc^2
        """
        # Get surface density and mean surface density
        deltasigma = self.halo_profile.deltasigmaR(R)

        # Add 2-halo term if requested
        if self.include_2halo:
            deltasigma2h = self.rho_m * self.two_halo_profile.deltasigmaR(R)
            deltasigma += self.bias*deltasigma2h

        return deltasigma

    def sigmaR(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate surface density profile.

        Parameters
        ----------
        R : float or array-like
            Projected radius in Mpc

        Returns
        -------
        float or array-like
            Surface density in Msun/Mpc^2
        """
        sigma = self.halo_profile.sigmaR(R)
        # If 2-halo term is included, add it to the surface density
        if self.include_2halo:
            sigma2h = self.rho_m * self.two_halo_profile.sigmaR(R)
            sigma += self.bias * sigma2h  # Apply bias to 2-halo term
        return sigma

    def density_3d(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate 3D density profile.

        Parameters
        ----------
        r : float or array-like
            3D radius in Mpc

        Returns
        -------
        float or array-like
            3D density in Msun/Mpc^3
        """
        return self.halo_profile.density_3d(r)

    def shear(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate tangential shear profile.

        Parameters
        ----------
        R : float or array-like
            Projected radius in Mpc

        Returns
        -------
        float or array-like
            Tangential shear (dimensionless)
        """
        delta_sigma = self.delta_sigma(R)
        return delta_sigma / self._sigma_crit

    def convergence(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate convergence profile.

        Parameters
        ----------
        R : float or array-like
            Projected radius in Mpc

        Returns
        -------
        float or array-like
            Convergence (dimensionless)
        """
        sigma = self.sigmaR(R)
        return sigma / self._sigma_crit

    def reduced_shear(
        self, R: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        reduced_shear(R)
        Returns the reduced shear as a function of cosmology,
        radius, halo mass and the scale factors of the
        source and the lens.
        
        .. math::
           g_t (R) = \\frac{\\gamma(R)}{(1 - \\kappa(R))},

        where :math:`\\gamma(R)` is the shear and :math:`\\kappa(R)` is the
        convergence.

        """
        convergence = self.convergence(R)
        if np.any(convergence >= 1.0):
            raise ValueError("Convergence must be less than 1 for reduced shear calculation")   
                        
        shear = self.shear(R)
        return shear / (1.0 - convergence)

    def fourier_profile(self, k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate Fourier transform of the density profile.

        Parameters
        ----------
        k : float or array-like
            Wavenumber in h/Mpc

        Returns
        -------
        float or array-like
            Fourier profile u(k)
        """
        k = np.atleast_1d(k)
        result = self.halo_profile.fourier(k)

        if self.include_2halo:
            # Add 2-halo term contribution
            two_halo_result = self.Pkec
            result += self.bias * two_halo_result

        return result if k.shape else result[0]

    def info(self) -> Dict[str, Any]:
        """
        Return summary information about the lensing profile.

        Returns
        -------
        dict
            Dictionary containing profile parameters and derived quantities
        """
        return {
            "model": self.model,
            "z_cluster": self.z_cluster,
            "z_source": self.z_source,
            "m200": self.m200,
            "concentration": self.concentration,
            "r200": self.halo_profile.r200,
            "rs": self.halo_profile.rs,
            "sigma_crit": self._sigma_crit,
            "include_2halo": self.include_2halo,
            "H0": self.cosmo.H0.to_value("km/s/Mpc"),
            "Om0": self.cosmo.Om0,
        }

    def __repr__(self) -> str:
        """String representation of the LensingProfile."""
        return (
            f"LensingProfile(model={self.model}, "
            f"z_cluster={self.z_cluster}, "
            f"m200={self.m200:.2e}, c={self.concentration})"
        )

__all__ = ["LensingProfile"]
