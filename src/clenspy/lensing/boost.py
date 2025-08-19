"""
Boost factor correction functions for weak lensing profiles.

Boost factors account for the enhancement in the lensing signal due to
correlated satellite galaxies and substructure around the main halo.
"""

import numpy as np
from typing import Union
from dataclasses import dataclass


def boost_factor_nfw(
    R: Union[float, np.ndarray], B0: float, rs: float) -> Union[float, np.ndarray]:
    """
    Calculate boost factor for NFW profiles.

    The boost factor accounts for the enhancement in the lensing signal
    due to correlated satellites and substructure.

    Parameters
    ----------
    R : float or array-like
        Projected radius in Mpc
    B0 : float
        Boost factor amplitude (dimensionless)
    rs : float
        Scale radius of the NFW profile in Mpc    

    Returns
    -------
    float or array-like
        Boost factor (dimensionless, typically > 1)

    """
    R = np.atleast_1d(R)
    
    x = R / rs
    fx = np.zeros_like(x)
    fx[x > 1] = np.arctan(np.sqrt(x[x > 1]**2 - 1)) / np.sqrt(x[x > 1]**2 - 1)
    fx[x == 1] = 1
    fx[x < 1] = np.arctanh(np.sqrt(1 - x[x < 1]**2)) / np.sqrt(1 - x[x < 1]**2)
    #fix the warning error
    denominator = x**2 - 1
    denominator[denominator == 0] = 1e-10  # or some small value
    B = 1 + B0 * (1 - fx) / denominator
    B[np.isnan(B)] = (B0 + 3) / 3
    return B

@dataclass
class BoostFactorData:
    R: np.ndarray
    data_vector: np.ndarray
    sigma_B: np.ndarray
    covariance: np.ndarray
    inv_cov: np.ndarray
    l: int
    z: int

@dataclass
class BoostFactorCollection:
    lbins: list
    zbins: list
    datasets: dict[str, BoostFactorData]

def load_boost_factor_data(path: str, lbin: int, zbin: int, scale_cut: tuple[float, float]) -> BoostFactorData:
    config = BoostFactorData(None, None, None, None, None, lbin, zbin)
    data_file = f"{path}/full-unblind-v2-mcal-zmix_y1clust_l{lbin}_z{zbin}_zpdf_boost.dat"
    cov_file  = f"{path}/full-unblind-v2-mcal-zmix_y1clust_l{lbin}_z{zbin}_zpdf_boost_cov.dat"
    
    # load the data
    config.R, config.data_vector, config.sigma_B = np.genfromtxt(data_file, unpack=True)
    config.covariance = np.genfromtxt(cov_file)

    # Apply scale cuts
    # r_max <5 makes the same as R[:8]
    r_min, r_max = scale_cut
    config = scale_cuts(config, r_min, r_max)

    # Invert covariance matrix 
    # np.linalg.pinv is more stable than np.linalg.inv
    config.inv_cov = np.linalg.pinv(config.covariance)

    return config

def load_boost_factor_collection(
        path: str, 
        l0: int = 0, 
        le: int = 10, 
        z0: int = 0, 
        ze: int = 3, 
        scale_cut: tuple[float, float] = (0.1, 5.0)
        ) -> BoostFactorCollection:
    lambda_bins = range(l0, le)  # Richness bins from l0 to le
    z_bins = range(z0, ze)        # Redshift bins from z0 to ze
    
    configCollection = BoostFactorCollection(lambda_bins, z_bins, {})
    for l in lambda_bins:
        for z in z_bins:
            configCollection.datasets[f'{l}l_{z}z'] = load_boost_factor_data(path, l, z, scale_cut)
    return configCollection

def scale_cuts(config: BoostFactorData, r_min: float =0.1, r_max: float =5.0) -> BoostFactorData:
    mask = (config.R >= r_min) & (config.R <= r_max)
    config.R = config.R[mask]
    config.data_vector = config.data_vector[mask]
    config.sigma_B = config.sigma_B[mask]
    config.covariance = config.covariance[np.ix_(mask, mask)]
    return config

__all__ = [
    "boost_factor_nfw"
    "load_boost_factor_collection",
    "load_boost_factor_data",
]
