r"""The world model: background, distances, and the linear power spectrum.

The top layer -- it imports only `clenspy.utils`, and everything else in the
package imports it.

NOTE: :math:`\Sigma_{\rm crit}` is no longer here. It moved to
`clenspy.kernels.sigma_crit`, because it depends on the cosmology *and* on
two redshifts, which makes it lens-source geometry rather than a property of
the universe.
"""

from .concentration import child18, child18_powerlaw, duffy08, m_star_hinv
from .distances import comoving_to_theta, theta_to_comoving
from .fiducial import fiducial_cosmology, mean_matter_density
from .growth import growth_factor, growth_unnormalised
from .mass_function import TinkerMassFunction, consumed_mask
from .pkgrid import PkGrid
from .sigma import LinearPk, SigmaGrid, lnr_grid

__all__ = [
    "comoving_to_theta",
    "theta_to_comoving",
    "PkGrid",
    "fiducial_cosmology",
    "mean_matter_density",
    "child18",
    "child18_powerlaw",
    "duffy08",
    "m_star_hinv",
    "growth_factor",
    "growth_unnormalised",
    "LinearPk",
    "SigmaGrid",
    "lnr_grid",
    "TinkerMassFunction",
    "consumed_mask",
]


# -- deprecated alias, one release --------------------------------------
#
# Imported lazily: `clenspy.cosmology` must not depend on
# `clenspy.kernels`, which sits below it.


def __getattr__(name):
    if name != "sigma_critical":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import warnings

    from ..kernels.sigma_crit import sigma_critical

    warnings.warn(
        "clenspy.cosmology.sigma_critical moved to "
        "clenspy.kernels.sigma_critical; the alias will be removed in the "
        "next release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return sigma_critical
