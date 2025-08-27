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
    A class for handling miscentering corrections in weak lensing profiles.

    Inherits from LensingProfile and adds functionality for miscentering corrections.

    Attributes:
        miscentering_factor (float): Factor to apply for miscentering correction.
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
    
    def apply_miscentering(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Apply miscentering correction to the radius R.

        Args:
            R (Union[float, np.ndarray]): Radius or radii to apply the correction.

        Returns:
            Union[float, np.ndarray]: Corrected radius or radii.
        """
        return R * self.miscentering_factor
