r"""The boost factor: member dilution of the source sample.

Cluster members and correlated structure scattered into the source
catalogue are not lensed by the cluster, so they dilute the measured
signal. :math:`\mathcal{B}(R)` is the multiplicative correction that
removes that dilution (McClintock et al. 2019, eq. 27).

NOTE: :math:`\mathcal{B}(R)` is **dimensionless** and > 1, and the radii it
is evaluated at are in Mpc, matching the rest of the package's h-free
absolute convention. The loaders below read DES Y1 data files whose radii
are also in Mpc; nothing here converts units.

NOTE: this is a *measured* correction read from disk, not a model. The
files are the DES Y1 unblinded boost measurements and are not distributed
with `clenspy` -- `load_boost_factor_data` needs a path to them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np


def boost_factor_nfw(
    R: Union[float, np.ndarray], B0: float, rs: float) -> Union[float, np.ndarray]:
    r"""
    Calculate boost factor for NFW profiles.

    The boost factor :math:`B(R) = \Sigma_{\rm crit}/\Sigma_{\rm crit}^{\rm eff}`
    accounts for the enhancement in the lensing signal due to correlated
    satellites and substructure diluting the background source sample
    (McClintock et al. 2019, eq. 27):

    .. math::
        B(R) = 1 + B_0\, \frac{1 - f(x)}{x^2 - 1}, \qquad x = \frac{R}{r_s}

    .. math::
        f(x) =
        \begin{cases}
        \dfrac{\mathrm{arctanh}\sqrt{1 - x^2}}{\sqrt{1 - x^2}}, & x < 1 \\
        1, & x = 1 \\
        \dfrac{\arctan\sqrt{x^2 - 1}}{\sqrt{x^2 - 1}}, & x > 1
        \end{cases}

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
    r"""
    Measured boost factor B(R) for one richness/redshift bin.

    NOTE: ``R`` is in Mpc; ``data_vector``, ``sigma_B``, ``covariance`` and
    ``inv_cov`` are all in units of :math:`\mathcal{B}` itself, which is
    dimensionless. ``i_lam`` and ``i_z`` are bin *indices*, not physical
    values -- see `~clenspy.utils.RichnessBin` for the edges they name.

    Attributes
    ----------
    R : np.ndarray
        Projected radius [Mpc], after scale cuts.
    data_vector : np.ndarray
        Measured boost factor B(R) at each R.
    sigma_B : np.ndarray
        Per-point uncertainty on ``data_vector``.
    covariance : np.ndarray
        Covariance matrix of ``data_vector``, after scale cuts.
    inv_cov : np.ndarray
        Pseudo-inverse of ``covariance`` (set by `load_boost_factor_data`).
    i_lam : int
        Richness bin index -- the :math:`i` of `~clenspy.utils.RichnessBin`.
    i_z : int
        Redshift bin index -- the :math:`j` of `~clenspy.utils.RichnessBin`.
    """

    R: np.ndarray
    data_vector: np.ndarray
    sigma_B: np.ndarray
    covariance: np.ndarray
    inv_cov: np.ndarray
    i_lam: int
    i_z: int

@dataclass
class BoostFactorCollection:
    """
    A collection of `BoostFactorData` spanning a grid of richness/redshift bins.

    NOTE: units are those of `BoostFactorData`. ``lbins`` and ``zbins``
    hold bin indices, and ``datasets`` is keyed ``"{l}l_{z}z"`` on those
    same indices.

    Attributes
    ----------
    lbins : list
        Richness bin indices in the collection.
    zbins : list
        Redshift bin indices in the collection.
    datasets : dict[str, BoostFactorData]
        Maps ``"{l}l_{z}z"`` to the corresponding `BoostFactorData`.
    """

    lbins: list
    zbins: list
    datasets: dict[str, BoostFactorData]

def load_boost_factor_data(
    path: str, lbin: int, zbin: int, scale_cut: tuple[float, float]
) -> BoostFactorData:
    """
    Load a measured boost factor for one richness/redshift bin from disk.

    Reads ``full-unblind-v2-mcal-zmix_y1clust_l{lbin}_z{zbin}_zpdf_boost.dat``
    (radius, boost factor, uncertainty) and the matching ``_cov.dat``
    covariance file from ``path``, applies ``scale_cut`` via `scale_cuts`,
    and inverts the resulting covariance with `numpy.linalg.pinv`.

    Parameters
    ----------
    path : str
        Directory containing the boost factor data and covariance files.
    lbin : int
        Richness bin index.
    zbin : int
        Redshift bin index.
    scale_cut : tuple[float, float]
        ``(r_min, r_max)`` in Mpc; points outside this range are dropped.

    Returns
    -------
    BoostFactorData
        The loaded, scale-cut data with its inverse covariance.
    """
    config = BoostFactorData(None, None, None, None, None, lbin, zbin)
    stem = f"{path}/full-unblind-v2-mcal-zmix_y1clust_l{lbin}_z{zbin}_zpdf_boost"
    data_file = f"{stem}.dat"
    cov_file = f"{stem}_cov.dat"
    
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
    """
    Load `BoostFactorData` for every richness/redshift bin in a grid.

    Calls `load_boost_factor_data` for each ``(l, z)`` pair with
    ``l in range(l0, le)`` and ``z in range(z0, ze)``.

    Parameters
    ----------
    path : str
        Directory containing the boost factor data and covariance files.
    l0, le : int, optional
        Richness bin range ``[l0, le)`` (default: 0 to 10).
    z0, ze : int, optional
        Redshift bin range ``[z0, ze)`` (default: 0 to 3).
    scale_cut : tuple[float, float], optional
        ``(r_min, r_max)`` in Mpc passed to `load_boost_factor_data`
        (default: ``(0.1, 5.0)``).

    Returns
    -------
    BoostFactorCollection
        One `BoostFactorData` per ``(l, z)`` bin, keyed ``"{l}l_{z}z"``.
    """
    lambda_bins = range(l0, le)
    z_bins = range(z0, ze)

    collection = BoostFactorCollection(lambda_bins, z_bins, {})
    for i_lam in lambda_bins:
        for i_z in z_bins:
            collection.datasets[f"{i_lam}l_{i_z}z"] = load_boost_factor_data(
                path, i_lam, i_z, scale_cut
            )
    return collection

def scale_cuts(
    config: BoostFactorData, r_min: float = 0.1, r_max: float = 5.0
) -> BoostFactorData:
    """
    Restrict a `BoostFactorData` to radii in ``[r_min, r_max]``.

    Filters ``R``, ``data_vector``, and ``sigma_B`` elementwise, and
    ``covariance`` to the matching rows/columns. Does not touch ``inv_cov``;
    callers should re-invert the covariance after cutting (as
    `load_boost_factor_data` does).

    Parameters
    ----------
    config : BoostFactorData
        Data to cut. Modified in place and returned.
    r_min, r_max : float, optional
        Radius range to keep, in Mpc (default: 0.1 to 5.0).

    Returns
    -------
    BoostFactorData
        ``config``, with all fields cut to the scale range.
    """
    mask = (config.R >= r_min) & (config.R <= r_max)
    config.R = config.R[mask]
    config.data_vector = config.data_vector[mask]
    config.sigma_B = config.sigma_B[mask]
    config.covariance = config.covariance[np.ix_(mask, mask)]
    return config

__all__ = [
    "boost_factor_nfw",
    "load_boost_factor_collection",
    "load_boost_factor_data",
]
