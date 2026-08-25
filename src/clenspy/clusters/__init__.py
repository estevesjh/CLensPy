"""Cluster-cosmology model layer: MOR, selection function, binned observables.

Modules
-------
``mor``            Mass-observable relations (HOD, lognormal).
``kernels``        Closed-form richness (EMG) and photo-z kernels.
``selection``      S_ij(lnM, z) 2D selection tables.
``weights``        Fixed Gauss-Legendre weight engines.
``observables``    Binned N_ij and stacked DeltaSigma_ij operators.
``selection_bias`` Selection bias b_sel(theta): sigmoid plateaus + engine.
``geometry``       redMaPPer aperture geometry helpers.
``survey``         Survey solid-angle functions Omega(z).
"""

from .halomodel import BinHaloModelSpectra
from .intrinsic_variance import IntrinsicProfileVariance
from .kernels import (
    AnalyticLogNormalKernel,
    EmgRichnessKernel,
    K_j,
    PlobLtrParams,
    RichnessKernel,
    emg_cdf,
)
from .mor import (
    HodMOR,
    HodParams,
    LogNormalMOR,
    LogNormalParams,
    MassObservableRelation,
)
from .observables import (
    BinnedClusterModel,
    DeltaSigma1hOperator,
    DeltaSigmaMaxOperator,
    duffy08_concentration,
)
from .selection import BinDefinition, SelectionFunctionBuilder, SelectionTable
from .selection_bias import SelBiasEngine, SelectionBiasTable, SigmoidBias, XiNL
from .survey import omega_z_const_factory, omega_z_des, omega_z_sdss
from .weights import (
    MassZWeights,
    ZResolvedWeights,
    build_mass_weights,
    build_zresolved_weights,
)

__all__ = [
    "BinHaloModelSpectra",
    "IntrinsicProfileVariance",
    "SigmoidBias",
    "SelectionBiasTable",
    "SelBiasEngine",
    "XiNL",
    "MassZWeights",
    "ZResolvedWeights",
    "build_mass_weights",
    "build_zresolved_weights",
    "BinnedClusterModel",
    "DeltaSigma1hOperator",
    "DeltaSigmaMaxOperator",
    "duffy08_concentration",
    "omega_z_des",
    "omega_z_sdss",
    "omega_z_const_factory",
    "BinDefinition",
    "SelectionTable",
    "SelectionFunctionBuilder",
    "MassObservableRelation",
    "HodMOR",
    "HodParams",
    "LogNormalMOR",
    "LogNormalParams",
    "RichnessKernel",
    "EmgRichnessKernel",
    "AnalyticLogNormalKernel",
    "PlobLtrParams",
    "emg_cdf",
    "K_j",
]
