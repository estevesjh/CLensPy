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
"""

from . import boost, miscentering, miscentering_kernel
from .boost import boost_factor_nfw
from .miscentering import (
    MiscenteringTableError,
    NfwMiscenteringTable,
    load_nfw_miscentering_table,
    require_tabulated_profile,
)

__all__ = [
    "boost",
    "miscentering",
    "miscentering_kernel",
    "boost_factor_nfw",
    "NfwMiscenteringTable",
    "MiscenteringTableError",
    "load_nfw_miscentering_table",
    "require_tabulated_profile",
]
