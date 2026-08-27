r"""Bessel and Hankel kernels for radial-bin projection.

The skill's layout puts Bessel/Hankel machinery in `kernels`, and that is
the right layer for a second reason: two consumers need
:math:`\hat J_2` and `kernels` is the lowest layer both may import --
`clenspy.covariance.deltasigma`, which contracts it directly, and
`clenspy.kernels.fftlog_cov`, which builds it into a Mellin kernel.

NOTE: one copy, deliberately. Two implementations of a kernel with a
delicate cancellation branch is how they drift apart, and they had:
a second version on another branch cut over to its series at
:math:`\ell\theta = 10^{-2}`, where the closed form still carries
:math:`\sim 3\times10^{-7}`. This one cuts over at 1 and is uniformly
accurate to :math:`8\times10^{-13}`.

NOTE: **units.** Dimensionless. ``ell`` is a multipole (or
:math:`k\chi_h`) and the annulus edges are radians -- or, equivalently,
:math:`k` in 1/Mpc against edges in Mpc, since only the product
:math:`\ell\theta` enters. That equivalence is what lets the
:math:`\Delta\Sigma` covariance work in :math:`(k, r_p)` while the formula
is written in :math:`(\ell, \theta)`.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0, j1

__all__ = ["J2_SERIES_CUTOFF", "J2_SERIES_TERMS", "j2_bin"]


#: Below :math:`\ell\theta_{\max}` of this, use the Taylor series rather
#: than the closed form. At 1.0 the two agree to 1e-13; the closed form
#: degrades below it and the series diverges above ~6.
J2_SERIES_CUTOFF = 1.0

#: Terms retained in the series branch. 14 is exact to fp64 at the cutoff.
J2_SERIES_TERMS = 14


def j2_bin(ell, theta_min, theta_max):
    r"""The radial-bin-averaged Bessel function :math:`\hat J_2`.

    .. math::
        \hat J_2 = \frac{2}{\ell^2(\theta_{\max}^2-\theta_{\min}^2)}
          \Big[2\big(J_0(\ell\theta_{\min}) - J_0(\ell\theta_{\max})\big)
          + \ell\big(\theta_{\min}J_1(\ell\theta_{\min})
                     - \theta_{\max}J_1(\ell\theta_{\max})\big)\Big]

    the average of :math:`J_2(\ell\theta)` over the annulus
    :math:`\theta_{\min} < \theta < \theta_{\max}` weighted by
    :math:`2\pi\theta\,d\theta`. Wu et al. (2019) ``eq:hJ2``.

    NOTE: **the closed form above is unusable for**
    :math:`\ell\theta_{\max} \lesssim 1`, and this function does not use it
    there. Its bracket is a near-total cancellation: both
    :math:`2(J_0 - J_0)` and :math:`\ell(\theta J_1 - \theta J_1)` are
    :math:`O(x^2)` with *opposite* signs and cancel to :math:`O(x^4)`, then
    get divided by :math:`\ell^2\Delta\theta^2`. At
    :math:`\ell\theta = 10^{-3}` the surviving value is nine orders below
    the terms that produced it, so fp64 returns roughly four correct
    digits -- measured at **4.8e-4** relative error against direct
    quadrature. Below `J2_SERIES_CUTOFF` the Taylor series is used instead,

    .. math::
        \hat J_2 = \sum_{m\ge0}\frac{(-1)^m\,\ell^{2m+2}
          \left(\theta_{\max}^{2m+4} - \theta_{\min}^{2m+4}\right)}
          {2^{2m+2}\,m!\,(m+2)!\,(m+2)
           \left(\theta_{\max}^2-\theta_{\min}^2\right)},

    obtained by averaging :math:`J_2`'s own series term by term. The two
    branches agree to 1e-13 at the cutoff, and the series diverges beyond
    :math:`x \sim 6` while the closed form is exact there -- so neither
    alone is sufficient.

    NOTE: this is an **average, not a sample**. :math:`J_2` peaks at
    :math:`\ell\theta = 2` and the first peak barely moves with bin width,
    but the decay does: a wider bin decays faster in :math:`\ell`. So
    replacing :math:`\hat J_2` by :math:`J_2` at the bin centre is
    accurate for the LSS terms (which fall steeply in :math:`\ell`, so only
    the first peak matters) and *wrong* for the shot- and shape-noise terms
    (which are :math:`\ell`-independent, so the whole tail contributes).
    That asymmetry is why the bin average cannot be skipped.

    NOTE: dimensionless. ``ell`` is dimensionless and the two angles are in
    radians; only their ratio to :math:`1/\ell` matters.

    Parameters
    ----------
    ell : float or array-like
        Multipole (or :math:`k\chi_h`). Must be positive.
    theta_min, theta_max : float or array-like
        Annulus edges in radians, ``theta_max > theta_min >= 0``.

    Returns
    -------
    np.ndarray
        :math:`\hat J_2`, broadcast over the inputs.
    """
    ell, theta_min, theta_max = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (ell, theta_min, theta_max))
    )
    if np.any(ell <= 0.0):
        raise ValueError("ell must be positive")
    if np.any(theta_max <= theta_min) or np.any(theta_min < 0.0):
        raise ValueError("require theta_max > theta_min >= 0")

    delta_sq = theta_max**2 - theta_min**2
    out = np.empty(ell.shape, dtype=float)

    # the closed form, wherever it is well conditioned
    big = ell * theta_max >= J2_SERIES_CUTOFF
    if np.any(big):
        x_lo, x_hi = ell[big] * theta_min[big], ell[big] * theta_max[big]
        out[big] = (2.0 / (ell[big] ** 2 * delta_sq[big])
                    * (2.0 * (j0(x_lo) - j0(x_hi))
                       + x_lo * j1(x_lo) - x_hi * j1(x_hi)))

    # ... and the term-by-term average of J_2's series, where it is not
    small = ~big
    if np.any(small):
        e, lo, hi = ell[small], theta_min[small], theta_max[small]
        total = np.zeros(e.shape, dtype=float)
        coefficient = 1.0
        for m in range(J2_SERIES_TERMS):
            if m:
                # (-1)^m / (2^(2m+2) m! (m+2)!) built by recurrence, so no
                # factorial overflows and no 2^28 literals
                coefficient *= -1.0 / (4.0 * m * (m + 2))
            else:
                coefficient = 1.0 / 8.0
            total += (coefficient / (m + 2)) * e ** (2 * m + 2) * (
                hi ** (2 * m + 4) - lo ** (2 * m + 4)
            ) / delta_sq[small]
        out[small] = total

    return out
