r"""The Tinker et al. (2008) halo mass function.

.. math::
    f(\sigma) = A\left[\left(\frac{\sigma}{b}\right)^{-a} + 1\right]
                \exp\!\left(-\frac{c}{\sigma^{2}}\right)

.. math::
    \frac{dn}{d\ln R} = \frac{3}{4\pi}\,
        \frac{d\ln\nu}{d\ln R}\;\frac{f(\sigma)}{2}\;\frac{1}{R^{3}},
    \qquad
    \frac{dn}{d\ln M} = \frac13\,\frac{dn}{d\ln R},
    \qquad
    \nu = \frac{\delta_c^{2}}{\sigma^{2}}

Ported from ``y3_cluster_cpp`` branch ``docs/sphinx-site``,
``src/modules/mf_tinker_cpp/python/tinker_core.py`` (``tinker_mf`` and
``tinker_outputs``), itself a port of the Fortran ``mf_tinker.f90`` /
``compute_mf_tinker.f90``. The variance it consumes lives in
`clenspy.cosmology.sigma`; this module is only the multiplicity function
and the grid walk.

**Four constants here are transcriptions, and three of them are traps.**

- :math:`\delta_c = 1.6865`, **not** 1.686. `DELTA_C_TINKER` is the value
  ``compute_mf_tinker.f90`` uses, and it is *not* the 1.686 used by the
  Tinker (2010) bias in `clenspy.cosmology.BiasModel` or by :math:`M_\star` in
  `clenspy.cosmology.concentration`. The difference is 3e-4 relative in
  :math:`\delta_c`, which the :math:`\exp(-c/\sigma^2)` tail amplifies.
  Both values are kept, separately, because silently unifying them would
  change a calibrated production number.
- :math:`\pi \to` `PI_FORTRAN` = 3.1415926535, the truncated literal in
  ``compute_mf_tinker.f90``. Wrong in the 11th digit. Kept because it
  appears in both :math:`dn/d\ln R` and :math:`M(R)`, where it *partly*
  cancels -- so replacing it with ``np.pi`` shifts the mass axis rather
  than merely rounding it.
- :math:`\bar\rho = 2.775\times10^{11}`, and note what it multiplies:
  :math:`M_h = \frac{4\pi}{3}\,2.775\times10^{11}R^3` carries **no**
  :math:`\Omega_m`. The mass axis is therefore in units of
  :math:`\Omega_m\,h^{-1}M_\odot`, not :math:`h^{-1}M_\odot`. Downstream,
  ``HMF_t`` queries it at :math:`\ln[M(\Omega_m - \Omega_\nu)]`. This is
  the "HMF mass shift" convention; `consumed_mask` reproduces the query
  range.
- the factor :math:`\tfrac12` in front of :math:`f(\sigma)`.
  ``tinker_mf`` returns :math:`f(\sigma)/2`, so it is written where the
  reference writes it rather than folded into :math:`A`.

NOTE: **units** are h-scaled, inherited from
`clenspy.cosmology.sigma`: ``R`` in Mpc/h, mass in
:math:`\Omega_m h^{-1}M_\odot` (see above), :math:`dn/d\ln R` and
:math:`dn/d\ln M` in :math:`h^3\,\mathrm{Mpc}^{-3}`. Not the package's
h-free convention -- convert at the boundary, visibly.

NOTE: :math:`\Delta = 200` with respect to the **mean matter** density.
The Tinker (2008) calibration covers :math:`200 \le \Delta \le 3200` and
:math:`0 \le z \le 2.5`; the evolution exponents were fit only to
:math:`z \le 2`.

NOTE: :math:`\Delta` enters only through the coefficients, and only
:math:`A`, :math:`a` and :math:`b` evolve with redshift; :math:`c` does
not. At :math:`\Delta = 200` the :math:`b` evolution is nearly absent
(:math:`\alpha \simeq 0.0107`, so 1% out to :math:`z = 2.5`) -- kept
because dropping it changes a calibrated fit, not because it matters.
"""

from __future__ import annotations

import numpy as np

from ..utils.decorators import time_method
from .sigma import lnr_grid

__all__ = [
    "DELTA_C_TINKER",
    "PI_FORTRAN",
    "RHO_FACT",
    "TINKER08_TABLE2",
    "TinkerMassFunction",
    "consumed_mask",
]

#: :math:`\delta_c` as spelled in ``compute_mf_tinker.f90``. **Not** the
#: 1.686 of the Tinker (2010) bias -- see the module NOTE.
DELTA_C_TINKER = 1.6865

#: The truncated :math:`\pi` literal of ``compute_mf_tinker.f90``. Kept
#: deliberately; see the module NOTE.
PI_FORTRAN = 3.1415926535

#: :math:`\rho_m/(\Omega_m h^2)` in Msun/Mpc^3. Multiplied *without* an
#: :math:`\Omega_m`, which is what puts the mass axis in
#: :math:`\Omega_m h^{-1}M_\odot`.
RHO_FACT = 2.775e11

#: Tinker et al. (2008) Table 2 -- ``Delta -> (A0, a0, b0, c)``, with
#: :math:`\Delta` referred to the mean matter density. Interpolated
#: linearly in :math:`\log_{10}\Delta`, as the paper prescribes.
TINKER08_TABLE2 = {
    "delta": (200.0, 300.0, 400.0, 600.0, 800.0, 1200.0, 1600.0, 2400.0,
              3200.0),
    "A0": (0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260),
    "a0": (1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66),
    "b0": (2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41),
    "c": (1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44),
}

#: :math:`\log_{10}75`, the pivot of the :math:`b(z)` exponent (Tinker
#: 2008 Eq. 8). Written as the logarithm it is, rather than as the magic
#: constant 1.8750612633 that appears in other implementations.
_LOG10_75 = np.log10(75.0)


class TinkerMassFunction:
    r"""Tinker et al. (2008) :math:`f(\sigma)` and :math:`dn/d\ln M`.

    NOTE: units are h-scaled -- R in Mpc/h, mass in
    :math:`\Omega_m h^{-1}M_\odot`, :math:`dn/d\ln M` in
    :math:`h^3\mathrm{Mpc}^{-3}`. See the module NOTE.

    Parameters
    ----------
    sigma_grid : clenspy.cosmology.sigma.SigmaGrid
        Stored verbatim. Supplies :math:`\sigma^2(R)` and
        :math:`d\ln\sigma^2/d\ln R`.
    delta : float, optional
        Spherical overdensity w.r.t. mean matter (default: 200).
    """

    def __init__(self, sigma_grid, delta: float = 200.0):
        d = np.asarray(TINKER08_TABLE2["delta"], dtype=float)
        if not (d[0] <= delta <= d[-1]):
            raise ValueError(
                f"Tinker (2008) is calibrated for {d[0]:.0f} <= Delta <= "
                f"{d[-1]:.0f}, got {delta}"
            )
        self.sigma_grid = sigma_grid
        self.delta = float(delta)

        ld, target = np.log10(d), np.log10(self.delta)
        self.A0, self.a0, self.b0, self.c = (
            float(np.interp(target, ld,
                            np.asarray(TINKER08_TABLE2[key], dtype=float)))
            for key in ("A0", "a0", "b0", "c")
        )
        # eq. 8: the exponent of the b(z) evolution
        self.alpha = 10.0 ** (-((0.75 / (target - _LOG10_75)) ** 1.2))

    def coefficients(self, z):
        r"""``(A, a, b, c)`` at ``z`` -- Tinker (2008) Eqs. 5--8."""
        one_plus_z = 1.0 + np.asarray(z, dtype=float)
        return (
            self.A0 * one_plus_z**-0.14,        # eq. 5
            self.a0 * one_plus_z**-0.06,        # eq. 6
            self.b0 * one_plus_z**-self.alpha,  # eq. 7, exponent from eq. 8
            np.full_like(one_plus_z, self.c),   # c does not evolve
        )

    def multiplicity(self, ln_nu, z):
        r""":math:`\tfrac12 f(\sigma)` per :math:`\ln\nu`, dimensionless.

        Takes :math:`\ln\nu = 2\ln\delta_c - \ln\sigma^2` rather than
        :math:`\sigma`, because that is the variable the grid walk carries
        and converting twice is how a :math:`\delta_c` mismatch creeps in.

        NOTE: the :math:`\tfrac12` is the reference's, written here rather
        than folded into :math:`A`.
        """
        ln_nu = np.asarray(ln_nu, dtype=float)
        A, a, b, c = self.coefficients(z)
        sigma = DELTA_C_TINKER / np.sqrt(np.exp(ln_nu))
        return 0.5 * A * ((sigma / b) ** (-a) + 1.0) * np.exp(-c / sigma**2)

    def f_sigma(self, sigma, z=0.0):
        r"""The multiplicity function :math:`f(\sigma)` itself (no 1/2)."""
        sigma = np.asarray(sigma, dtype=float)
        if np.any(sigma <= 0.0):
            raise ValueError("sigma must be positive")
        ln_nu = 2.0 * np.log(DELTA_C_TINKER) - 2.0 * np.log(sigma)
        return 2.0 * self.multiplicity(ln_nu, z)

    def mass_of_radius(self, r_hinv):
        r""":math:`M(R) = \frac{4\pi}{3}\,2.775\times10^{11}R^3`.

        NOTE: in :math:`\Omega_m h^{-1}M_\odot` -- no :math:`\Omega_m`
        factor, by convention. See the module NOTE.
        """
        r_hinv = np.asarray(r_hinv, dtype=float)
        return (4.0 * PI_FORTRAN / 3.0) * RHO_FACT * r_hinv**3

    @time_method
    def walk(self, z, lnr=None, truncate: bool = True):
        r"""The grid walk: :math:`dn/d\ln R` and :math:`dn/d\ln M` at one z.

        Port of the ``compute_mf_tinker.f90`` body via
        ``tinker_core.tinker_outputs``.

        Parameters
        ----------
        z : float
            Redshift.
        lnr : array-like, optional
            :math:`\ln R` grid, R in Mpc/h. Defaults to the production
            969-point grid (`clenspy.cosmology.sigma.lnr_grid`).
        truncate : bool, optional
            Pass the :math:`k \le 20/R` truncation through to
            :math:`\sigma^2` (default: True, the production quantity).

        Returns
        -------
        dict
            ``r_h``, ``m_h``, ``dndlnrh``, ``dndlnmh``, ``lnsigma2``,
            ``dlnsigma2``.
        """
        lnr = lnr_grid() if lnr is None else np.asarray(lnr, dtype=float)
        r = np.exp(lnr)
        ln_sigma2 = np.array([
            np.log(self.sigma_grid.sigma2(ri, truncate=truncate)) for ri in r
        ])
        dln_sigma2 = np.array([
            self.sigma_grid.dlnsigma2_dlnr(ri, truncate=truncate) for ri in r
        ])
        return self.outputs(lnr, ln_sigma2, dln_sigma2, z)

    @time_method
    def outputs(self, lnr, ln_sigma2, dln_sigma2, z):
        r"""``(dn/dlnR, dn/dlnM)`` from :math:`\ln\sigma^2` and its slope.

        Split from `walk` so a caller with :math:`\sigma^2` from anywhere
        -- an FFTLog block, a cached grid -- reuses the same grid walk
        rather than a second copy of it.
        """
        lnr = np.asarray(lnr, dtype=float)
        r = np.exp(lnr)
        # nu = delta_c^2 / sigma^2, so ln nu = 2 ln delta_c - ln sigma^2
        ln_nu = 2.0 * np.log(DELTA_C_TINKER) - np.asarray(ln_sigma2, float)
        dln_nu = -np.asarray(dln_sigma2, dtype=float)
        mf = self.multiplicity(ln_nu, z)
        dndlnrh = (3.0 / (4.0 * PI_FORTRAN)) * dln_nu * mf / r**3
        return {
            "r_h": r,
            "m_h": self.mass_of_radius(r),
            "z": z,
            "dndlnrh": dndlnrh,
            "dndlnmh": dndlnrh / 3.0,
            "lnsigma2": np.asarray(ln_sigma2, dtype=float),
            "dlnsigma2": np.asarray(dln_sigma2, dtype=float),
        }

    def __repr__(self):
        return (f"TinkerMassFunction(Delta={self.delta:.0f}m, "
                f"A0={self.A0:.4f}, a0={self.a0:.3f}, b0={self.b0:.3f}, "
                f"c={self.c:.3f}, alpha={self.alpha:.5f})")


def consumed_mask(m_h, omega_m_minus_nu, lnm_low=29.9336, lnm_high=36.8414):
    r"""Which of the 969 grid points the downstream pipeline actually reads.

    ``HMF_t`` queries at :math:`\ln[M(\Omega_m - \Omega_\nu)]` over the
    pipeline's :math:`[\ln M_{\rm low}, \ln M_{\rm high}]`, and the
    ``m_h`` axis is already in :math:`\Omega_m h^{-1}M_\odot`.

    NOTE: this is the mask a validation comparison must apply. Errors on
    the full grid are dominated by the tails -- the grid runs to
    :math:`R = 0.0034` Mpc/h, far below any cluster -- so an unmasked
    residual is not the error the analysis sees.
    """
    lnm = np.log(np.asarray(m_h, dtype=float))
    shift = np.log(omega_m_minus_nu)
    return (lnm >= lnm_low + shift) & (lnm <= lnm_high + shift)


if __name__ == "__main__":
    from .sigma import KCUT_COEF, LinearPk, SigmaGrid

    # scale-free test spectrum, as in sigma.py's demo
    k = np.logspace(-5.0, 3.0, 600)
    pk = LinearPk(k, 2.0e4 * k**-1.5 * np.exp(-((k / 50.0) ** 2)))
    grid = SigmaGrid(pk)
    hmf = TinkerMassFunction(grid)
    print(hmf, "\n")

    print("the two delta_c in this package are different numbers:")
    print(f"  mass function (compute_mf_tinker.f90): {DELTA_C_TINKER}")
    print("  Tinker 2010 bias / M_star            : 1.686")
    print(f"  ratio - 1 = {DELTA_C_TINKER / 1.686 - 1:.3e}  <- small, but it")
    print("  enters exp(-c/sigma^2), which amplifies it in the tail.\n")

    lnr = np.log(np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0]))
    out = hmf.walk(0.0, lnr=lnr)
    print(f"{'R [Mpc/h]':>10s}  {'M_h':>11s}  {'sigma':>8s}  "
          f"{'0.5 f(sig)':>10s}  {'dn/dlnM':>11s}")
    for i, r in enumerate(out["r_h"]):
        sig = np.exp(0.5 * out["lnsigma2"][i])
        ln_nu = 2 * np.log(DELTA_C_TINKER) - out["lnsigma2"][i]
        print(f"{r:10.3f}  {out['m_h'][i]:11.4e}  {sig:8.4f}  "
              f"{hmf.multiplicity(ln_nu, 0.0):10.6f}  "
              f"{out['dndlnmh'][i]:11.4e}")

    shift = (PI_FORTRAN / np.pi) - 1.0
    print(f"\nthe Fortran pi literal: PI_FORTRAN/pi - 1 = {shift:.3e}.")
    print("  Numerically irrelevant -- 3e-11, far below the 1e-3 epsrel the")
    print("  production QUADPACK ran at. It is kept only so a bit-level")
    print("  comparison against the Fortran dump has nothing left to")
    print("  explain, not because it changes any answer.")

    print("\nmass axis units check, at R = 8 Mpc/h:")
    m8 = hmf.mass_of_radius(8.0).item()
    print(f"  M_h = {m8:.4e} Omega_m h^-1 Msun")
    print(f"  for Omega_m = 0.3 that is {m8 * 0.3:.4e} h^-1 Msun")
    print("  <- the Omega_m is the caller's to apply; see the module NOTE.")

    print("\nredshift evolution of the coefficients (Delta = 200):")
    print(f"{'z':>5s}  {'A':>8s}  {'a':>8s}  {'b':>8s}  {'c':>6s}")
    for z in (0.0, 0.5, 1.0, 2.0):
        A, a, b, c = (float(v) for v in hmf.coefficients(z))
        print(f"{z:5.2f}  {A:8.5f}  {a:8.5f}  {b:8.5f}  {c:6.3f}")
    print(f"  <- only A, a, b move; alpha = {hmf.alpha:.5f} makes b nearly")
    print("     static at Delta = 200.")

    # dn/dlnM must fall steeply with mass and with redshift
    print("\ndn/dlnM at R = 8 Mpc/h against z:")
    for z in (0.0, 0.5, 1.0):
        o = hmf.walk(z, lnr=np.log(np.array([8.0])))
        print(f"  z = {z:4.2f}:  dn/dlnM = {o['dndlnmh'][0]:.4e}")

    print(f"\nthe truncation is inherited from SigmaGrid (k <= "
          f"{KCUT_COEF:.0f}/R):")
    for tr in (True, False):
        o = hmf.walk(0.0, lnr=np.log(np.array([8.0])), truncate=tr)
        print(f"  truncate={str(tr):5s}:  dn/dlnM = {o['dndlnmh'][0]:.6e}")
    a = hmf.walk(0.0, lnr=np.log(np.array([8.0])), truncate=True)["dndlnmh"][0]
    b = hmf.walk(0.0, lnr=np.log(np.array([8.0])),
                 truncate=False)["dndlnmh"][0]
    print(f"  relative difference = {abs(a / b - 1):.2e}  <- this is what the")
    print("  FFTLog fast path cannot reproduce, by construction.")
