r"""Closed-form NFW: the analytic truth for the transform-chain bench.

The whole chain in closed form,

.. math::
    \rho(r) \;\xrightarrow{\rm 3D\,FT}\; \tilde\rho(k)
    \;\xrightarrow{\rm inverse}\; \rho(r),
    \qquad
    \rho(r) \;\xrightarrow{\rm Abel}\; \Sigma(R)
    \;\xrightarrow{\rm interior}\; \bar\Sigma(<R),
    \qquad
    \Delta\Sigma = \bar\Sigma - \Sigma,

so each numerical transform in ``validate_twohalo_chain.py`` is compared to
an exact answer and a failure localises to one stage.

NOTE: this is **deliberately a second copy** of formulae `clenspy.halo.nfw`
also carries. A validation reference that imports the code under test
validates nothing. These are transcribed from Wright & Brainerd (2000) and
checked against direct quadrature by `selfcheck` below, so the two
implementations are independent and their agreement is a result.

NOTE: **unit-agnostic.** Lengths in any unit L, ``rho_s`` in any U/L^3.
Then :math:`\tilde\rho` is in U and :math:`\Sigma` in :math:`{\rm U}/{\rm
L}^2 \times {\rm L}`. Callers own the bookkeeping -- and
``validate_twohalo_chain.py`` works in the h-ful convention of the
reference libraries, not `clenspy`'s h-free one.

The identity the bench exploits: :math:`\xi(r) = \int dk\,
\frac{k^2}{2\pi^2} P(k) j_0(kr)` is the **same integral** as the inverse
3-D Fourier transform of :math:`\tilde\rho`. So a code fed
:math:`P(k) \equiv \tilde\rho(k)` must return :math:`\xi(r) \equiv
\rho(r)`, and its :math:`\Sigma_{2h}` machinery must return the NFW
:math:`\Sigma`. Every stage acquires an exact reference for free.
"""

import numpy as np
from scipy.special import sici

__all__ = ["NfwAnalytic", "selfcheck"]


class NfwAnalytic:
    r""":math:`\rho(r) = \rho_s / [x(1+x)^2]`, :math:`x = r/r_s`.

    Parameters
    ----------
    rho_s : float
        Characteristic density :math:`\rho_s = \delta_c \rho_{\rm ref}`.
    r_s : float
        Scale radius, in whatever length unit the caller uses.
    c : float
        Concentration. Only used by the truncated Fourier transform.

    NOTE: :math:`\Sigma` and :math:`\Delta\Sigma` are the **untruncated**
    Wright & Brainerd forms. :math:`\tilde\rho` is offered both ways: the
    untruncated transform is log-divergent as :math:`k \to 0` (the total
    mass diverges), which is harmless for a Hankel quadrature evaluated at
    :math:`k > 0` but is what makes the FFTLog legs ripple at the 0.1%
    level.
    """

    def __init__(self, rho_s=1.0, r_s=1.0, c=5.0):
        self.rho_s, self.r_s, self.c = float(rho_s), float(r_s), float(c)

    @classmethod
    def from_m200m(cls, m200m, c=5.0, rho_ref=1.0):
        r"""The halo of mass ``m200m`` w.r.t. ``200 * rho_ref``.

        .. math::
            r_{200} = \left[\frac{3 M}{4\pi \cdot 200\,\rho_{\rm ref}}
                      \right]^{1/3},
            \qquad
            \delta_c = \frac{200}{3}\,
                       \frac{c^3}{\ln(1+c) - c/(1+c)}
        """
        r200 = (3.0 * m200m / (4.0 * np.pi * 200.0 * rho_ref)) ** (1.0 / 3.0)
        delta_c = (200.0 / 3.0) * c**3 / (np.log(1 + c) - c / (1 + c))
        return cls(rho_s=delta_c * rho_ref, r_s=r200 / c, c=c)

    @property
    def r200(self):
        return self.c * self.r_s

    # -- 3D ---------------------------------------------------------------

    def rho(self, r):
        x = np.asarray(r, dtype=float) / self.r_s
        return self.rho_s / (x * (1.0 + x) ** 2)

    def rho_tilde(self, k, truncated=False):
        r""":math:`\tilde\rho(k) = 4\pi\int r^2 \rho(r) j_0(kr)\,dr`.

        Untruncated, with :math:`\kappa = k r_s`:

        .. math::
            \tilde\rho = 4\pi\rho_s r_s^3
              \left[\sin\kappa\left(\tfrac{\pi}{2} - {\rm Si}\,\kappa\right)
                    - \cos\kappa \, {\rm Ci}\,\kappa\right]
        """
        kappa = np.asarray(k, dtype=float) * self.r_s
        amp = 4.0 * np.pi * self.rho_s * self.r_s**3
        si_k, ci_k = sici(kappa)
        if truncated:
            si_ck, ci_ck = sici((1.0 + self.c) * kappa)
            shape = (np.sin(kappa) * (si_ck - si_k)
                     + np.cos(kappa) * (ci_ck - ci_k)
                     - np.sin(self.c * kappa) / ((1.0 + self.c) * kappa))
        else:
            shape = (np.sin(kappa) * (np.pi / 2.0 - si_k)
                     - np.cos(kappa) * ci_k)
        return amp * shape

    def mass_3d(self, r):
        r""":math:`M(<r) = 4\pi\rho_s r_s^3[\ln(1+x) - x/(1+x)]`."""
        x = np.asarray(r, dtype=float) / self.r_s
        return (4.0 * np.pi * self.rho_s * self.r_s**3
                * (np.log(1.0 + x) - x / (1.0 + x)))

    # -- projected, Wright & Brainerd (2000) ------------------------------

    @staticmethod
    def _f(x):
        r""":math:`\Sigma = 2 r_s \rho_s f(x)`; W&B eq. 11."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        f = np.full_like(x, 1.0 / 3.0)  # the removable singularity at x = 1
        lo, hi = x < 1.0, x > 1.0
        xl, xh = x[lo], x[hi]
        f[lo] = (1.0 - 2.0 / np.sqrt(1.0 - xl**2)
                 * np.arctanh(np.sqrt((1.0 - xl) / (1.0 + xl)))) / (xl**2 - 1.0)
        f[hi] = (1.0 - 2.0 / np.sqrt(xh**2 - 1.0)
                 * np.arctan(np.sqrt((xh - 1.0) / (xh + 1.0)))) / (xh**2 - 1.0)
        return f

    @staticmethod
    def _g(x):
        r""":math:`\Delta\Sigma = r_s \rho_s g(x)`; W&B eq. 13-16."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        g = np.full_like(x, 10.0 / 3.0 + 4.0 * np.log(0.5))
        lo, hi = x < 1.0, x > 1.0
        xl, xh = x[lo], x[hi]
        al = np.arctanh(np.sqrt((1.0 - xl) / (1.0 + xl)))
        g[lo] = (8.0 * al / (xl**2 * np.sqrt(1.0 - xl**2))
                 + 4.0 / xl**2 * np.log(xl / 2.0)
                 - 2.0 / (xl**2 - 1.0)
                 + 4.0 * al / ((xl**2 - 1.0) * np.sqrt(1.0 - xl**2)))
        ah = np.arctan(np.sqrt((xh - 1.0) / (xh + 1.0)))
        g[hi] = (8.0 * ah / (xh**2 * np.sqrt(xh**2 - 1.0))
                 + 4.0 / xh**2 * np.log(xh / 2.0)
                 - 2.0 / (xh**2 - 1.0)
                 + 4.0 * ah / ((xh**2 - 1.0) ** 1.5))
        return g

    def sigma(self, R):
        return 2.0 * self.r_s * self.rho_s * self._f(
            np.asarray(R, dtype=float) / self.r_s)

    def delta_sigma(self, R):
        return self.r_s * self.rho_s * self._g(
            np.asarray(R, dtype=float) / self.r_s)

    def sigma_bar(self, R):
        return self.delta_sigma(R) + self.sigma(R)

    def __repr__(self):
        return (f"NfwAnalytic(rho_s={self.rho_s:.4e}, "
                f"r_s={self.r_s:.4f}, c={self.c})")


def selfcheck(tol=1e-5, verbose=True):
    """Every closed form above against direct quadrature.

    This is what makes the module usable as a reference: the formulae are
    transcribed by hand, so they are checked by an independent numerical
    route before anything is compared to them.

    Returns
    -------
    list of str
        The labels that failed; empty on success.
    """
    from scipy.integrate import quad

    p = NfwAnalytic(rho_s=2.0, r_s=0.5, c=5.0)
    failures = []

    def check(label, got, want, rtol=tol):
        rel = abs(got / want - 1.0)
        if rel >= rtol:
            failures.append(label)
        if verbose:
            print(f"  [{'ok  ' if rel < rtol else 'FAIL'}] {label:<26s} "
                  f"closed={want:.6e}  quad={got:.6e}  rel={rel:.2e}")

    def scalar(v):
        return float(np.asarray(v).reshape(-1)[0])

    # 1) rho_tilde -- only the TRUNCATED form is checkable by quadrature:
    #    the untruncated integral has a slowly decaying oscillatory tail.
    for k in (0.3, 2.0, 8.0):
        num, _ = quad(
            lambda r, k=k: (4 * np.pi * r**2 * scalar(p.rho(r))
                            * np.sinc(k * r / np.pi)),
            0, p.r200, limit=400,
        )
        check(f"rho_tilde_trunc(k={k})", num,
              scalar(p.rho_tilde(k, truncated=True)))
    # k -> 0 anchor: the truncated transform must equal the enclosed mass
    check("rho_tilde_trunc(k->0)", scalar(p.mass_3d(p.r200)),
          scalar(p.rho_tilde(1e-6, truncated=True)), 1e-4)

    # 2) Sigma against the Abel integral 2 int_0^inf rho(sqrt(R^2+u^2)) du
    for R in (0.2, 1.0, 3.0):
        num, _ = quad(lambda u, R=R: 2 * scalar(p.rho(np.hypot(R, u))),
                      0, np.inf, limit=400)
        check(f"sigma(R={R})", num, scalar(p.sigma(R)))

    # 3) sigma_bar against (2/R^2) int_0^R Sigma R' dR'
    for R in (0.5, 2.0):
        num, _ = quad(lambda t: scalar(p.sigma(t)) * t, 0, R, limit=400)
        check(f"sigma_bar(R={R})", 2 * num / R**2, scalar(p.sigma_bar(R)))

    # 4) the definition DeltaSigma = sigma_bar - sigma
    for R in (0.5, 2.0):
        check(f"delta_sigma(R={R})",
              scalar(p.sigma_bar(R)) - scalar(p.sigma(R)),
              scalar(p.delta_sigma(R)), 1e-10)

    return failures


if __name__ == "__main__":
    import sys

    print("closed-form NFW against direct quadrature:")
    bad = selfcheck()
    print("all self-checks passed" if not bad else f"FAILED: {bad}")
    sys.exit(0 if not bad else 1)
