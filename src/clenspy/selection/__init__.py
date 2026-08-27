r"""Systematics and the mass--observable relation: what sits between a halo
and what a survey records.

NOTE: this layer is **above** `clenspy.halo` and **below**
`clenspy.lensing`. A systematic is defined relative to a profile, so
importing `clenspy.halo` here is the intended direction; nothing in
`clenspy.halo` may import this package.

Contents
--------
`miscentering`
    Runtime lookup of the offset NFW profiles: :math:`\Sigma_{\rm mis}` and
    the signed :math:`\Delta\Sigma_{\rm mis}`.
`miscentering_kernel`
    The offline quadrature that generated that table. Not a runtime path.
`boost`
    :math:`\mathcal{B}(R)`, member dilution of the source sample.
`richness_kernel`
    :math:`P(\lambda^{\rm ob}\mid\lambda^{\rm tr})`, the EMG projection
    kernel, and its bin integral in closed form.
`scaling_relation`
    The two mass--observable relations: log-normal and the HOD
    shifted-Poisson.
`selection_function`
    :math:`\mathcal S_{ij}(M, z^{\rm tr}) = S_i\,\mathcal S_j`, the
    factor that turns a mass function into a catalogue prediction.
`geometry`
    The redMaPPer aperture: :math:`R_\lambda`, its angle
    :math:`\theta_\lambda`, the disk-overlap fraction :math:`f_A`, and the
    :math:`\sigma(\theta)` sigmoid -- the geometric ingredients of
    :math:`b_{\rm sel}`.
"""

from . import (
    boost,
    geometry,
    miscentering,
    miscentering_kernel,
    richness_kernel,
    scaling_relation,
    selection_function,
)
from .boost import boost_factor_nfw
from .geometry import (
    area_overlap,
    r_lambda,
    sigmoid_theta,
    theta_lambda,
)
from .miscentering import (
    MiscenteringTableError,
    NfwMiscenteringTable,
    load_nfw_miscentering_table,
    require_tabulated_profile,
)
from .richness_kernel import EmgParams, emg_cdf, richness_bin_probability
from .scaling_relation import HodMor, LogNormalMor
from .selection_function import SelectionFunction

__all__ = [
    "boost",
    "geometry",
    "miscentering",
    "miscentering_kernel",
    "richness_kernel",
    "scaling_relation",
    "selection_function",
    "boost_factor_nfw",
    "NfwMiscenteringTable",
    "MiscenteringTableError",
    "load_nfw_miscentering_table",
    "require_tabulated_profile",
    "EmgParams",
    "emg_cdf",
    "richness_bin_probability",
    "LogNormalMor",
    "HodMor",
    "SelectionFunction",
    "r_lambda",
    "theta_lambda",
    "area_overlap",
    "sigmoid_theta",
]
