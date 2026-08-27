r"""What the dataset is: footprint, source population, bin definitions.

This layer holds everything that changes when you change the *survey* while
holding the universe and the nuisance model fixed. It may import
`clenspy.cosmology` and `clenspy.utils`; nothing here knows about a halo
profile or a lens-source geometry.

NOTE: units are h-free absolute except for :math:`n_{\rm src}`, which is in
arcmin^-2 because that is how shape catalogues are quoted; the identifier
carries the unit (``n_src_arcmin``). :math:`\Omega(z)` is in steradians.

Three separate concerns, deliberately not one object
----------------------------------------------------
`area`
    :math:`\Omega(z)`, the effective survey area. Appears in
    :math:`\langle N_{ij}\rangle` and **cancels** in the shear projection,
    so it must be passed per-observable and never applied to both
    (errata E.2).
`sources`
    `SourcePopulation`: :math:`p(z_s)`, :math:`\sigma_\gamma`,
    :math:`n_{\rm src}`. Consumed by the lensing weights, irrelevant to
    the counts.
`bins`
    The :math:`(\Delta\lambda_i, \Delta z_j)` grid the analysis used --
    the labels results are addressed by.

Coverage is uneven, on purpose: `clenspy` transcribes survey definitions
rather than estimating them, so a survey has only the pieces that exist in
a source.

======== =============== ================== ==============
survey   :math:`\Omega(z)` source population  bins
======== =============== ================== ==============
DES Y1   3-piece fit     yes                4 x 3
DES Y3   flat 4143 deg^2 yes (Y1 p(z))      as Y1
SDSS     degree-11 fit   **raises**         **raises**
Y3xSPT   flat 2500 deg^2 --                 --
======== =============== ================== ==============
"""

from .area import (
    DES_Y1_Z_RANGE,
    SDSS_Z_RANGE,
    deg2,
    omega_des_y1,
    omega_des_y3,
    omega_sdss,
    omega_y3xspt,
    survey_area,
)
from .bins import des_y1_bins, des_y3_bins, sdss_bins, survey_bins
from .sources import SourcePopulation

__all__ = [
    # area
    "omega_des_y1",
    "omega_des_y3",
    "omega_sdss",
    "omega_y3xspt",
    "survey_area",
    "deg2",
    "DES_Y1_Z_RANGE",
    "SDSS_Z_RANGE",
    # sources
    "SourcePopulation",
    # bins
    "des_y1_bins",
    "des_y3_bins",
    "sdss_bins",
    "survey_bins",
]
