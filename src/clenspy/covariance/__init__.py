r"""Covariance matrices: the `Estimator` layer.

Two blocks, and in both the physical components are stored **separately**
and summed at the end, with switches to isolate each one. The scientific
argument is almost always about which term dominates where, so a class that
returned only the total would make the paper unwritable.

`counts`
    :math:`{\rm Cov}[N_{ij}, N_{i'j'}]`: Poisson plus sample variance. The
    sample-variance term is rank one within each redshift slice, because
    every cluster in the slice sees the same window mode.
`deltasigma`
    :math:`{\rm Cov}[\Delta\Sigma(r_p), \Delta\Sigma(r_p')]`, the
    Gaussian-field expression of Wu et al. (2019), whose bracket expands
    into five terms.

NOTE: this is the top layer -- it imports from `clenspy.observables`,
`clenspy.kernels` and `clenspy.cosmology`, and nothing imports it.

NOTE: units are inherited. Counts covariance is dimensionless;
:math:`\Delta\Sigma` covariance is in
:math:`(M_\odot/{\rm Mpc}^2)^2`.

NOTE: **survey area appears twice, meaning two different things.**
:math:`\Omega(z)` normalises the counts (`clenspy.observables`); the sky
fraction :math:`f_{\rm sky}` sets the number of independent modes and
enters the :math:`\Delta\Sigma` covariance as
:math:`1/(4\pi f_{\rm sky})`. They are not interchangeable, and conflating
them is a factor of :math:`4\pi`.
"""

from . import counts, deltasigma
from .counts import CountsCovariance
from .deltasigma import ALL_TERMS, DeltaSigmaCovariance, j2_bin

__all__ = [
    "counts",
    "deltasigma",
    "CountsCovariance",
    "DeltaSigmaCovariance",
    "j2_bin",
    "ALL_TERMS",
]
