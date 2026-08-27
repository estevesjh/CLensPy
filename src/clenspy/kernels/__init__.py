r"""Line-of-sight windows and the geometry that weights them.

This layer is machinery: it holds the quantities that turn a 3D or radial
model into something projected on the sky, and nothing about any particular
halo or dataset. It may import `clenspy.cosmology` and `clenspy.utils`;
`clenspy.lensing` imports it.

NOTE: units are h-free absolute throughout -- Mpc, Msun, Msun/Mpc^2.

Contents
--------
`sigma_crit`
    :math:`\Sigma_{\rm crit}(z_l, z_s)` for one lens-source pair,
    **physical**.
`lensing_kernel`
    `LensingKernel`: the source-averaged weights, all **comoving** --
    :math:`\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)` and the three
    callables the covariance consumes.

NOTE: the two modules use **different** :math:`\Sigma_{\rm crit}`
conventions and the difference is exactly :math:`(1+z_l)^2`.
`sigma_critical` is physical, built from angular diameter distances;
`sigma_crit_comoving` is comoving, which is what `clenspy`'s comoving
:math:`\Delta\Sigma` needs for :math:`\gamma_t` to come out
dimensionless. Both are named for what they return.

To come (``docs/refactor-plan.md`` A.3): the two photo-z kernels, and the
Limber projection written once with windows passed in.
"""

from .lensing_kernel import LensingKernel, sigma_crit_comoving
from .sigma_crit import sigma_critical

__all__ = ["sigma_critical", "sigma_crit_comoving", "LensingKernel"]
