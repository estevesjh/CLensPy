"""Deprecated location for the Limber projection.

The projection is machinery -- line-of-sight windows and a Hankel-style
integral -- so it belongs to `clenspy.kernels`, one layer below the
probe. It moved to `clenspy.kernels.limber`.

NOTE: this shim exists because ``cluster-lensing-cov``'s
``clens/covariance/limber.py`` imports ``clenspy.lensing.limber``. Update
that import to ``clenspy.kernels.limber`` and this file goes away.

NOTE: the method names also changed, to follow Wu et al. (2019):
``c_ell_sigma`` -> `C_ell_SS`, ``c_ell_h`` -> `C_ell_hh` plus
`shot_noise_h`, ``c_ell_h_sigma`` -> `C_ell_hS`, ``shape_noise_sigma`` ->
`shape_noise_Sigma`. The old names remain as aliases on the class for one
release, so this shim is enough for the downstream to keep working
unchanged.
"""

import warnings

from ..kernels.limber import ARCMIN_TO_RAD, LimberProjector, limber

__all__ = ["ARCMIN_TO_RAD", "LimberProjector", "limber"]

warnings.warn(
    "clenspy.lensing.limber moved to clenspy.kernels.limber; the alias will "
    "be removed in the next release. Method names now follow Wu et al. "
    "(2019) -- see the module docstring.",
    DeprecationWarning,
    stacklevel=2,
)
