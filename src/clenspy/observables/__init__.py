r"""The binned observables: what an analysis actually compares to data.

This layer contracts the selection function and the mass function into the
two quantities a cluster cosmology likelihood consumes:

`abundance`
    :math:`W_{ij}(M,z)`, the weight, and its two contractions --
    :math:`\langle N_{ij}\rangle` and the weight-normalised stack
    :math:`\langle X\rangle_{ij}`.
`deltasigma`
    :math:`\Delta\Sigma_{ij}(R)`, the second contraction, with
    :math:`X = \Delta\Sigma(R\mid M,z)`.

NOTE: this layer sits **above** `clenspy.selection`, `clenspy.halo` and
`clenspy.cosmology`, and imports from all three. Nothing in those packages
may import this one.

NOTE: units are inherited, not redeclared -- h-scaled masses from the mass
function and the mass--observable relations, h-free lengths from the
cosmology. `abundance.ClusterAbundance._volume_per_dz` is the single place
the two conventions meet.
"""

from . import abundance, deltasigma
from .abundance import ClusterAbundance
from .deltasigma import StackedDeltaSigma

__all__ = [
    "abundance",
    "deltasigma",
    "ClusterAbundance",
    "StackedDeltaSigma",
]
