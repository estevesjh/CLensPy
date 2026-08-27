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
`photoz`
    The **two** photo-z kernels, which are different functions: a Gaussian
    CDF difference for the counts and a compactly-supported parabola for
    the projection. Substituting one for the other is a silent bias.
`limber`
    `LimberProjector`: the Wu et al. (2019) angular power spectra, written
    as **one** projection with :math:`F_{\rm h}` / :math:`F_\Sigma` passed
    in, which is how the paper writes them.
`bessel`
    :math:`\hat J_2`, the annulus-averaged Bessel kernel. One copy, shared
    by the direct quadrature and the FFTLog engine.
`fftlog_cov`
    The FFTLog engine for the bin-averaged **double**-Bessel covariance
    integral: one transform per diagonal offset, with the Mellin
    coefficients summed before the inverse FFT.

NOTE: the two modules use **different** :math:`\Sigma_{\rm crit}`
conventions and the difference is exactly :math:`(1+z_l)^2`.
`sigma_critical` is physical, built from angular diameter distances;
`sigma_crit_comoving` is comoving, which is what `clenspy`'s comoving
:math:`\Delta\Sigma` needs for :math:`\gamma_t` to come out
dimensionless. Both are named for what they return.

"""

from .bessel import J2_SERIES_CUTOFF, j2_bin
from .lensing_kernel import LensingKernel, sigma_crit_comoving
from .limber import ARCMIN_TO_RAD, LimberProjector, limber
from .photoz import (
    Y3_Z_KERNEL_FILE,
    gaussian_cdf,
    photoz_counts,
    photoz_projection,
    y3_photoz_window,
)
from .sigma_crit import sigma_critical

__all__ = [
    "sigma_critical",
    "sigma_crit_comoving",
    "LensingKernel",
    "photoz_counts",
    "photoz_projection",
    "gaussian_cdf",
    "LimberProjector",
    "limber",
    "ARCMIN_TO_RAD",
    "j2_bin",
    "J2_SERIES_CUTOFF",
    "y3_photoz_window",
    "Y3_Z_KERNEL_FILE",
]
