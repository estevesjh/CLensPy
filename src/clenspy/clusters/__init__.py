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
from .selection import BinDefinition, SelectionFunctionBuilder, SelectionTable

__all__ = [
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
