"""
LensingProfile class for weak lensing calculations.

This module provides a unified interface for computing weak lensing observables
from dark matter halo profiles.
"""

from typing import Any, Dict, Union

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from ..cosmology import sigma_critical
from ..halo import NFWProfile


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
        cosmology: FlatLambdaCDM,
        z_cluster: float,
        m200: float,
        concentration: float = 4.0,
        model: str = "NFW",
        include_2halo: bool = True,
        z_source: float = 1.0
    ) -> None:
        self.cosmology = cosmology
        self.z_cluster = z_cluster
        self.m200 = m200
        self.concentration = concentration
        self.model = model.upper()
        self.include_2halo = include_2halo
        self.z_source = z_source

        # Validate inputs
        self._validate_inputs()

        # Initialize halo profile
        self._setup_halo_profile()

        # Calculate critical surface density using new cosmology utils
        self._sigma_crit = sigma_critical(
            self.z_cluster, self.z_source, self.cosmology
        )

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
                cosmology=self.cosmology  # Pass astropy cosmology directly
            )
        else:
            msg = f"Model {self.model} not implemented"
            raise NotImplementedError(msg)

    def delta_sigma(
        self, R: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
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
        sigma = self.halo_profile.surface_density(R)
        sigma_mean = self.halo_profile.mean_surface_density(R)

        # Delta Sigma = <Sigma>(<R) - Sigma(R)
        delta_sigma = sigma_mean - sigma

        # Add 2-halo term if requested
        if self.include_2halo:
            delta_sigma += self._delta_sigma_2halo(R)

        return delta_sigma

    def surface_density(
        self, R: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
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
        return self.halo_profile.surface_density(R)

    def mean_surface_density(
        self, R: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Calculate mean surface density within radius R.

        Parameters
        ----------
        R : float or array-like
            Projected radius in Mpc

        Returns
        -------
        float or array-like
            Mean surface density in Msun/Mpc^2
        """
        return self.halo_profile.mean_surface_density(R)

    def density_3d(
        self, r: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
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

    def shear(
        self, R: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
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

    def convergence(
        self, R: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
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
        sigma = self.surface_density(R)
        return sigma / self._sigma_crit

    def fourier_profile(
        self, k: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
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
        # This is a placeholder for the Fourier profile calculation
        # In practice, this would involve the Fourier transform of the NFW
        k = np.atleast_1d(k)

        # Simplified NFW Fourier profile (placeholder)
        # Real implementation would use proper NFW Fourier transform
        rs = self.halo_profile.rs
        result = np.exp(-k * rs)  # Simplified exponential cutoff

        return result if k.shape else result[0]

    def _delta_sigma_2halo(
        self, R: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Calculate 2-halo term contribution to delta sigma.

        This is a placeholder for the 2-halo term calculation.
        In practice, this would involve halo bias and matter power spectrum.

        Parameters
        ----------
        R : float or array-like
            Projected radius in Mpc

        Returns
        -------
        float or array-like
            2-halo term contribution in Msun/Mpc^2
        """
        # Placeholder: very simplified 2-halo term
        R = np.atleast_1d(R)

        # 2-halo term typically becomes important at large scales (R > few Mpc)
        # and has a power-law dependence
        two_halo = np.zeros_like(R)
        large_scale_mask = R > 2.0  # Mpc

        if np.any(large_scale_mask):
            # Very simplified power law for large scales
            two_halo[large_scale_mask] = (
                1e12 * (R[large_scale_mask] / 5.0) ** (-1.5)
            )

        return two_halo if R.shape else two_halo[0]

    def info(self) -> Dict[str, Any]:
        """
        Return summary information about the lensing profile.

        Returns
        -------
        dict
            Dictionary containing profile parameters and derived quantities
        """
        return {
            'model': self.model,
            'z_cluster': self.z_cluster,
            'z_source': self.z_source,
            'm200': self.m200,
            'concentration': self.concentration,
            'r200': self.halo_profile.r200,
            'rs': self.halo_profile.rs,
            'sigma_crit': self._sigma_crit,
            'include_2halo': self.include_2halo,
            'H0': self.cosmology.H0.value,
            'Om0': self.cosmology.Om0
        }

    def __repr__(self) -> str:
        """String representation of the LensingProfile."""
        return (
            f"LensingProfile(model={self.model}, "
            f"z_cluster={self.z_cluster}, "
            f"m200={self.m200:.2e}, c={self.concentration})"
        )


__all__ = ["LensingProfile"]