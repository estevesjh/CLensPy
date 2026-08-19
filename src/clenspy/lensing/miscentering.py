"""
Miscentering correction functions for cluster lensing profiles.
"""

# Make a miscentring class
from typing import Union

import numpy as np
from astropy.cosmology import Cosmology

from ..config import DEFAULT_COSMOLOGY
from ..lensing.profile import LensingProfile


class MiscenteringProfile(LensingProfile):
    """
    A `LensingProfile` with a placeholder miscentering correction.

    Inherits all of `LensingProfile`'s behavior unchanged (sigma/deltasigma
    are NOT overridden here); `apply_miscentering` is a standalone utility,
    not currently wired into the profile calculations.

    Warning
    -------
    `apply_miscentering` is a placeholder linear rescaling of R, not a
    physical miscentering deprojection. A real miscentering correction
    convolves the profile with an offset (e.g. Rayleigh/Gamma-distributed
    R_mis) distribution and integrates over the azimuthal angle - see e.g.
    Johnston et al. (2007) or Simet et al. (2017) for the standard
    formalism. This has not been implemented yet.

    Attributes
    ----------
    miscentering_factor : float
        Linear scale factor applied to R by `apply_miscentering`.
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
        miscentering_factor: float = 1.0,
    ) -> None:
        super().__init__(
            zCluster, m200, cosmology, concentration, model, include2Halo, backend2Halo, zSource
        )
        self.miscentering_factor = miscentering_factor

    def apply_miscentering(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Rescale R by `miscentering_factor` (placeholder - see class Warning).

        Parameters
        ----------
        R : float or np.ndarray
            Radius or radii [Mpc].

        Returns
        -------
        float or np.ndarray
            ``R * self.miscentering_factor``.
        """
        return R * self.miscentering_factor
