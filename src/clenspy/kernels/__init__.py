r"""Line-of-sight windows and the geometry that weights them.

This layer is machinery: it holds the quantities that turn a 3D or radial
model into something projected on the sky, and nothing about any particular
halo or dataset. It may import `clenspy.cosmology` and `clenspy.utils`;
`clenspy.lensing` imports it.

NOTE: units are h-free absolute throughout -- Mpc, Msun, Msun/Mpc^2.

Contents
--------
`sigma_crit`
    :math:`\Sigma_{\rm crit}(z_l, z_s)` for one lens-source pair.

To come (``docs/refactor-plan.md`` A.3): the source-averaged inverse
:math:`\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)`, the three callables the
covariance consumes, the two photo-z kernels, and the Limber projection
written once with windows passed in.
"""

from .sigma_crit import sigma_critical

__all__ = ["sigma_critical"]
