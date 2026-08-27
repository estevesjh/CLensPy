"""
Core weak lensing algorithms and observables.
"""

from .miscentering import MiscenteringProfile
from .profile import LensingProfile

__all__ = ["LensingProfile", "MiscenteringProfile"]


# -- deprecated alias, one release --------------------------------------
#
# `boost` moved to `clenspy.selection.boost`: the boost factor is a
# correction to the source sample, not a lensing observable.


def __getattr__(name):
    if name != "boost":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    import warnings

    warnings.warn(
        "clenspy.lensing.boost moved to clenspy.selection.boost; the alias "
        "will be removed in the next release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return importlib.import_module("clenspy.selection.boost")
