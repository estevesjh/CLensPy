r"""The linear growth factor :math:`D(z)`.

One quantity, one formula. The linear density contrast of a growing mode
evolves as :math:`\delta(a) \propto D^{+}(a)`, and for a
matter-plus-dark-energy background that has the closed-form quadrature

.. math::
    D^{+}(a) = \frac{5\Omega_m}{2}\,\frac{H(a)}{H_0}
               \int_0^{a} \frac{da'}{\left[a'H(a')/H_0\right]^{3}},
    \qquad
    D(a) = \frac{D^{+}(a)}{D^{+}(a=1)}

which is the form written down by Child et al. (2018) Sec. 4 -- the same
place their :math:`M_\star` and :math:`\sigma(R,z)` definitions come from,
so `clenspy.cosmology.concentration` and this module agree by construction.

`growth_factor` returns the **normalised** :math:`D`, with
:math:`D(z=0) = 1`, because that is the convention in which the linear
power spectrum scales as :math:`P_{\rm lin}(k,z) = D^2(z)P_{\rm lin}(k,0)`
and the convention of the frozen reference grid this is validated against.
`growth_unnormalised` exposes :math:`D^{+}` itself for the rare caller who
wants the growth *rate* normalisation.

NOTE: **units.** Everything here is dimensionless: :math:`D`, :math:`a`,
:math:`z`, and the integrand, which is written in units of :math:`H_0` so
that no length or time scale enters. The :math:`5\Omega_m/2` prefactor is
the standard normalisation making :math:`D^{+} \to a` in the
matter-dominated limit; it cancels in `growth_factor` and is kept only so
`growth_unnormalised` carries the usual convention.

NOTE: domain of validity. The quadrature is exact for any
:math:`H(a)` -- it assumes only that dark energy is smooth (does not
cluster) and that the growing mode is scale-independent. Both fail for
massive neutrinos and for modified gravity, where :math:`D` becomes
:math:`D(k,z)`; this module is then the wrong tool rather than an
approximation to it. It is also a *linear* growth factor, so it says
nothing about the nonlinear :math:`P(k)`.

NOTE: :math:`H(a)/H_0` is taken from the supplied `astropy` cosmology's
``efunc``, not re-derived, so whatever the cosmology includes -- radiation,
curvature, a :math:`w(z)` -- is automatically consistent between
:math:`D(z)` and every distance in `clenspy.cosmology.distances`.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from .fiducial import fiducial_cosmology

__all__ = ["growth_factor", "growth_unnormalised"]


def _growth_integral(a, efunc):
    r"""The :math:`\int_0^a da'/(a'E(a'))^3` piece, for one scalar ``a``.

    NOTE: the integrand is regular at the origin despite the
    :math:`a'^{-3}`: deep in matter domination :math:`E \simeq
    \sqrt{\Omega_m}\,a'^{-3/2}`, so :math:`(a'E)^{3} \propto a'^{-3/2}` and
    the integrand vanishes as :math:`a'^{3/2}`. No lower cutoff is needed
    and none is imposed -- one would be a silent approximation.
    """
    def integrand(ap):
        return 1.0 / (ap * efunc(1.0 / ap - 1.0)) ** 3

    value, _ = quad(integrand, 0.0, a, epsabs=0.0, epsrel=1e-10, limit=200)
    return value


def growth_unnormalised(z, cosmo=None):
    r"""The unnormalised growth factor :math:`D^{+}(a)`, dimensionless.

    .. math::
        D^{+}(a) = \frac{5\Omega_m}{2}\,E(a)
                   \int_0^{a}\frac{da'}{\left[a'E(a')\right]^{3}}

    with :math:`E = H/H_0` and :math:`a = 1/(1+z)`. Normalised so that
    :math:`D^{+} \to a` as :math:`a \to 0`; use `growth_factor` for the
    :math:`D(0) = 1` convention that :math:`P_{\rm lin}` scaling wants.

    Parameters
    ----------
    z : float or array-like
        Redshift, :math:`z > -1`.
    cosmo : astropy.cosmology.Cosmology, optional
        Defaults to `fiducial_cosmology()`.

    Returns
    -------
    np.ndarray
        :math:`D^{+}`, same shape as ``z``.
    """
    cosmo = fiducial_cosmology() if cosmo is None else cosmo
    z = np.asarray(z, dtype=float)
    if np.any(z <= -1.0):
        raise ValueError(f"z must exceed -1, got min {z.min()}")

    a = 1.0 / (1.0 + z)
    prefactor = 2.5 * cosmo.Om0
    integrals = np.array([_growth_integral(ai, cosmo.efunc)
                          for ai in np.atleast_1d(a).ravel()])
    out = prefactor * np.atleast_1d(cosmo.efunc(z)).ravel() * integrals
    return out.reshape(np.shape(a))


def growth_factor(z, cosmo=None):
    r"""The normalised linear growth factor, :math:`D(z)` with :math:`D(0)=1`.

    .. math::
        D(z) = \frac{D^{+}(a)}{D^{+}(1)},
        \qquad
        P_{\rm lin}(k, z) = D^{2}(z)\,P_{\rm lin}(k, 0)

    NOTE: this is the convention in which :math:`\sigma(R,z) =
    D(z)\,\sigma(R,0)`, which is how `clenspy.cosmology.mass_function`
    evolves its variance instead of recomputing a power spectrum per
    redshift. That factorisation is exactly the scale-independence
    assumption named in the module NOTE.

    Parameters
    ----------
    z : float or array-like
        Redshift.
    cosmo : astropy.cosmology.Cosmology, optional
        Defaults to `fiducial_cosmology()`.

    Returns
    -------
    np.ndarray
        :math:`D(z) \in (0, 1]` for :math:`z \ge 0`.
    """
    cosmo = fiducial_cosmology() if cosmo is None else cosmo
    # one visible division, and the 5*Om/2 prefactor cancels here
    return growth_unnormalised(z, cosmo) / growth_unnormalised(0.0, cosmo)


if __name__ == "__main__":
    cosmo = fiducial_cosmology()
    z = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 5.0])
    d = growth_factor(z, cosmo)

    print(f"Flat LambdaCDM, Om0 = {cosmo.Om0}, H0 = {cosmo.H0.value}\n")
    print(f"{'z':>6s}  {'D(z)':>10s}  {'D(z)(1+z)':>10s}  {'a':>7s}")
    for zi, di in zip(z, d):
        print(f"{zi:6.2f}  {di:10.6f}  {di * (1 + zi):10.6f}  "
              f"{1 / (1 + zi):7.4f}")
    # D(z)(1+z) = D+(a)/(a D+(1)) -> 1/D+(1) as a -> 0, NOT 1: the
    # normalisation D(0) = 1 is what puts the limit at 1/D+(1)
    suppression = 1.0 / growth_unnormalised(0.0, cosmo).item()
    print(f"  <- D(z)(1+z) rises to 1/D+(1) = {suppression:.6f}, not to 1.")
    print("     Growth tracks a in matter domination, so that limit IS the")
    print("     total suppression Lambda has caused since then: structure")
    print(f"     grew {suppression:.3f}x less than an EdS universe would give.")

    print(f"\nD(0) = {growth_factor(0.0, cosmo).item():.15f}  (exactly 1 by "
          "construction)")

    # the matter-dominated limit, as a number rather than an assertion
    z_hi = 200.0
    ratio = growth_unnormalised(z_hi, cosmo).item() * (1.0 + z_hi)
    print(f"D+(z={z_hi:.0f}) * (1+z) = {ratio:.6f}  <- the 5*Om/2 "
          "normalisation makes D+ -> a")

    # Lambda suppresses growth: compare against Einstein-de Sitter, where
    # D = a exactly
    eds = fiducial_cosmology(Om0=1.0)
    print("\nEinstein-de Sitter check (Om0 = 1), where D(z) = a exactly:")
    for zi in (0.5, 1.0, 3.0):
        got = growth_factor(zi, eds).item()
        print(f"  z = {zi:4.1f}:  D = {got:.10f}   a = {1 / (1 + zi):.10f}   "
              f"rel. err {abs(got * (1 + zi) - 1):.2e}")
