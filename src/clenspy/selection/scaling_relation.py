r"""The mass--observable relations, :math:`P(\lambda^{\rm tr}\mid M, z)`.

Two of them, because the two DES analyses this package reproduces use
different ones and the choice matters:

- `LogNormalMor` -- Costanzi et al. (2021), their Eq. 2. A log-normal in
  :math:`\lambda` with a Poisson-like variance floor. Used by DES+SPT.
- `HodMor` -- the halo occupation distribution of Costanzi et al. (2019b),
  mass-calibrated by McClintock et al. (2019), as used by DES Y1.
  :math:`\lambda^{\rm tr} = \lambda^{\rm cen} + \lambda^{\rm sat}` with a
  **continuous shifted-Poisson** law for the satellites.

Both expose the same three things, and that is the whole interface:
``pdf(lambda_true, ln_mass, z)``, ``mean(ln_mass, z)``,
``std(ln_mass, z)``. The last two exist so
`clenspy.selection.selection_function` can place its quadrature bracket
without knowing which relation it holds --
:math:`[\mu_{\rm eff} - L\sigma_{\rm eff},\;\mu_{\rm eff} +
L\sigma_{\rm eff}]`, clipped at zero.

**The HOD law.** The exact Poisson-plus-scatter distribution has no closed
form. Costanzi et al. (2019b) used a lookup-table skew-normal; this module
uses the continuous shifted-Poisson form (priv. comm. M. Costanzi) that
``y3_cluster_cpp/src/models/mor_hod_t.hh`` implements:

.. math::
    P(\lambda^{\rm tr}\mid M,z) = \exp\!\left[
        -\nu + (x - 1)\ln\nu - \ln\Gamma(x)\right],
    \qquad
    \begin{aligned}
    \delta &= (\sigma_{\rm intr}\langle\lambda^{\rm sat}\rangle)^2 \\
    \nu &= \langle\lambda^{\rm sat}\rangle + \delta \\
    x &= \lambda^{\rm tr} - \lambda^{\rm cen} + \delta
    \end{aligned}

It is closed-form, differentiable in :math:`(M,z)`, extends smoothly to the
non-integer richness the quadrature needs, and matches the exact law in the
low-:math:`\lambda` tail where the skew-normal breaks down.

NOTE: **units.** Richness is a dimensionless count. Mass enters as
:math:`\ln M` with :math:`M` in :math:`h^{-1}M_\odot`, because both
relations are calibrated with pivots in those units
(:math:`M_p = 3\times10^{14}h^{-1}M_\odot`;
:math:`M_{\min}, M_1` as :math:`\log_{10}` of :math:`h^{-1}M_\odot`).
The returned PDF is a density **per unit richness**, so
:math:`\int P\,d\lambda^{\rm tr} = 1` -- not a probability mass.

NOTE: :math:`\lambda^{\rm cen} = 1` above :math:`M_{\min}` and 0 below, so
`HodMor`'s support starts at :math:`\lambda^{\rm tr} = 1` for a halo massive
enough to host a central. A quadrature bracket reaching below that
integrates over exact zeros, which is wasteful but not wrong; reaching
*above* the support's lower edge silently discards probability.

NOTE: :math:`\ln\Gamma` is evaluated directly (`scipy.special.gammaln`),
never as :math:`\ln\Gamma(x)` from a computed :math:`\Gamma(x)`.
:math:`\Gamma` overflows for :math:`x \gtrsim 171` while
:math:`\ln\Gamma` is finite to :math:`10^{300}`, and cluster richnesses
reach 200.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

__all__ = ["HodMor", "LogNormalMor"]

#: Below this satellite mean the shifted-Poisson form degenerates
#: (:math:`\nu \to 0` makes :math:`\ln\nu` diverge), so the zero-satellite
#: limit is represented by a narrow Gaussian at
#: :math:`\lambda^{\rm cen}` instead. Both values from ``mor_hod_t.hh``.
POISSON_TOL = 1.0e-8
FALLBACK_SIGMA = 1.0e-3

#: Costanzi et al. (2021) Eq. 2 pivots.
M_PIVOT_HINV = 3.0e14
Z_PIVOT = 0.45


class LogNormalMor:
    r"""Costanzi et al. (2021) Eq. 2: a log-normal mass--richness relation.

    .. math::
        \langle\ln\lambda\rangle(M,z) = \ln A_\lambda
          + B_\lambda\ln\!\left(\frac{M}{M_p}\right)
          + C_\lambda\ln\!\left(\frac{1+z}{1+z_p}\right)

    .. math::
        \sigma^2_{\ln\lambda} = D_\lambda^2
          + \frac{\langle\lambda\rangle - 1}{\langle\lambda\rangle^2}

    so that :math:`\ln\lambda^{\rm tr} \sim
    \mathcal N(\langle\ln\lambda\rangle, \sigma^2_{\ln\lambda})`.

    NOTE: the second variance term is the Poisson floor from discrete
    galaxy counting, and it is **not** :math:`1/\langle\lambda\rangle`.
    :math:`(\langle\lambda\rangle-1)/\langle\lambda\rangle^2` subtracts the
    central galaxy, which contributes no shot noise. It also goes *negative*
    for :math:`\langle\lambda\rangle < 1`; the total variance stays positive
    only because :math:`D_\lambda^2` dominates there, and this class raises
    if it does not rather than returning a negative variance.

    NOTE: units -- ``ln_mass`` is :math:`\ln M` with M in
    :math:`h^{-1}M_\odot`, pivot :math:`3\times10^{14}h^{-1}M_\odot`.

    Parameters
    ----------
    A_lambda, B_lambda, C_lambda, D_lambda : float
        Amplitude, mass slope, redshift evolution, intrinsic scatter.
    m_pivot_hinv, z_pivot : float, optional
        The pivots. Defaults are the published ones.
    """

    def __init__(self, A_lambda=76.9, B_lambda=1.020, C_lambda=0.29,
                 D_lambda=0.23, m_pivot_hinv=M_PIVOT_HINV, z_pivot=Z_PIVOT):
        self.A_lambda = float(A_lambda)
        self.B_lambda = float(B_lambda)
        self.C_lambda = float(C_lambda)
        self.D_lambda = float(D_lambda)
        self.m_pivot_hinv = float(m_pivot_hinv)
        self.z_pivot = float(z_pivot)

    def mean_ln_lambda(self, ln_mass, z):
        r""":math:`\langle\ln\lambda\rangle`, dimensionless. Eq. 2."""
        ln_mass = np.asarray(ln_mass, dtype=float)
        z = np.asarray(z, dtype=float)
        return (np.log(self.A_lambda)
                + self.B_lambda * (ln_mass - np.log(self.m_pivot_hinv))
                + self.C_lambda * np.log((1.0 + z) / (1.0 + self.z_pivot)))

    def var_ln_lambda(self, ln_mass, z):
        r""":math:`\sigma^2_{\ln\lambda}`: intrinsic plus the Poisson floor."""
        mean_lambda = np.exp(self.mean_ln_lambda(ln_mass, z))
        # NOT 1/<lambda>: the central galaxy carries no shot noise
        poisson = (mean_lambda - 1.0) / mean_lambda**2
        total = self.D_lambda**2 + poisson
        if np.any(total <= 0.0):
            raise ValueError(
                "sigma^2_lnlambda is not positive: the Poisson term "
                "(<lambda>-1)/<lambda>^2 is negative for <lambda> < 1 and "
                f"D_lambda^2 = {self.D_lambda**2:g} does not cover it. The "
                "relation is being evaluated below its calibrated mass "
                "range."
            )
        return total

    def mean(self, ln_mass, z):
        r""":math:`\langle\lambda^{\rm tr}\rangle`, the log-normal mean.

        NOTE: :math:`e^{\mu + \sigma^2/2}`, not :math:`e^\mu`. The median
        is the smaller one, and using it as the quadrature centre biases
        the bracket low.
        """
        return np.exp(self.mean_ln_lambda(ln_mass, z)
                      + 0.5 * self.var_ln_lambda(ln_mass, z))

    def std(self, ln_mass, z):
        r"""Standard deviation of :math:`\lambda^{\rm tr}` (not of its log)."""
        var_ln = self.var_ln_lambda(ln_mass, z)
        return self.mean(ln_mass, z) * np.sqrt(np.expm1(var_ln))

    def pdf(self, lambda_true, ln_mass, z):
        r""":math:`P(\lambda^{\rm tr}\mid M,z)`, a density per richness."""
        lambda_true, ln_mass, z = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (lambda_true, ln_mass, z))
        )
        mu = self.mean_ln_lambda(ln_mass, z)
        var = self.var_ln_lambda(ln_mass, z)
        positive = lambda_true > 0.0
        safe = np.where(positive, lambda_true, 1.0)
        # log-normal density: the 1/lambda is the Jacobian d(ln l)/d l
        density = (np.exp(-0.5 * (np.log(safe) - mu) ** 2 / var)
                   / (safe * np.sqrt(2.0 * np.pi * var)))
        return np.where(positive, density, 0.0)

    def __repr__(self):
        return (f"LogNormalMor(A={self.A_lambda:g}, B={self.B_lambda:g}, "
                f"C={self.C_lambda:g}, D={self.D_lambda:g})")


class HodMor:
    r"""The DES Y1 HOD relation, with a continuous shifted-Poisson law.

    .. math::
        \langle\lambda^{\rm sat}\rangle(M,z) =
          \left(\frac{M - M_{\min}}{M_1 - M_{\min}}\right)^{\alpha}
          \left(\frac{1+z}{1+z_\star}\right)^{\epsilon},
        \qquad
        \lambda^{\rm tr} = \lambda^{\rm cen} + \lambda^{\rm sat}

    with :math:`\lambda^{\rm cen} = 1` for :math:`M \ge M_{\min}`, else 0,
    and the density given by the continuous shifted-Poisson form in the
    module docstring.

    NOTE: the variance is
    :math:`\sigma^2 \simeq \langle\lambda^{\rm sat}\rangle +
    (\sigma_{\rm intr}\langle\lambda^{\rm sat}\rangle)^2` -- Poissonian at
    low occupancy and super-Poissonian at high occupancy through the
    halo-to-halo term :math:`\sigma_{\rm intr}`.

    NOTE: units -- ``log10_Mmin`` and ``log10_M1`` are
    :math:`\log_{10}(M/h^{-1}M_\odot)`; ``ln_mass`` is :math:`\ln M` in the
    same units.

    NOTE: the satellite mean is exactly zero for :math:`M \le M_{\min}`,
    which makes :math:`\nu \to 0` and :math:`\ln\nu` diverge. Below
    `POISSON_TOL` the zero-satellite limit is represented by a narrow
    Gaussian of width `FALLBACK_SIGMA` at :math:`\lambda^{\rm cen}`,
    following ``mor_hod_t.hh``. It is a representation of a delta function,
    not a physical width: a bracket narrower than `FALLBACK_SIGMA` will
    miss its normalisation.

    NOTE: the source these constants come from stores them as
    :math:`10^{\log_{10}M}/h` in :math:`M_\odot`. This class keeps
    :math:`h^{-1}M_\odot` throughout, so the tabulated exponents are used
    **as they are written** and no :math:`h` appears -- which is one fewer
    place for it to be applied twice.

    Parameters
    ----------
    log10_Mmin, log10_M1 : float
        Minimum and satellite-normalisation masses,
        :math:`\log_{10}(M/h^{-1}M_\odot)`.
    alpha, epsilon : float
        Mass slope and redshift evolution.
    sigma_intr : float
        Halo-to-halo (super-Poisson) scatter, dimensionless.
    z_pivot : float, optional
        :math:`z_\star` (default 0.45).
    """

    def __init__(self, log10_Mmin=11.72, log10_M1=12.42, alpha=0.72,
                 epsilon=-0.30, sigma_intr=0.24, z_pivot=Z_PIVOT):
        self.log10_Mmin = float(log10_Mmin)
        self.log10_M1 = float(log10_M1)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.sigma_intr = float(sigma_intr)
        self.z_pivot = float(z_pivot)
        if self.log10_M1 <= self.log10_Mmin:
            raise ValueError(
                f"log10_M1 ({self.log10_M1}) must exceed log10_Mmin "
                f"({self.log10_Mmin}); M1 - Mmin is the normalisation"
            )

    @classmethod
    def des_y1(cls):
        r"""The DES Y1 NC+3x2pt best fit.

        NOTE: keeps :math:`\epsilon = 0`, the widePlanck convention. The
        Buzzard mocks were **not** generated with this set -- see
        `buzzard`. Comparing a Buzzard data vector against this relation
        mismatches the redshift evolution.
        """
        return cls(
            log10_Mmin=11.3852818,
            log10_M1=12.6964410,
            alpha=0.858693714,
            epsilon=0.0,
            sigma_intr=0.180949022,
        )

    @classmethod
    def buzzard(cls):
        r"""The exact constants the Buzzard mock data vectors were made with.

        From ``des-nersc-cluster-scripts/cosmosis-models/
        mock_mcmc_buzzard_values.ini`` (Tan Xing). Differs from `des_y1` in
        two places, both of which matter for a mock comparison:
        :math:`\epsilon = 0.283887020` rather than 0, and
        :math:`z_\star = 0.4544` rather than 0.45.

        NOTE: any Buzzard comparison must use **this** set. Measured, the
        :math:`\epsilon` difference tilts
        :math:`\langle\lambda^{\rm sat}\rangle` from **0.947x** at
        :math:`z = 0.2` to **1.036x** at :math:`z = 0.65` -- a 9% swing
        across the DES Y1 range, and a *tilt* rather than an offset, so it
        does not absorb into the amplitude. That is a redshift-dependent
        mass shift, not a rounding difference.
        """
        return cls(
            log10_Mmin=11.3852818,
            log10_M1=12.6964410,
            alpha=0.858693714,
            epsilon=0.283887020,
            sigma_intr=0.180949022,
            z_pivot=0.4544,
        )

    def mu_sat(self, ln_mass, z):
        r""":math:`\langle\lambda^{\rm sat}\rangle(M,z)`, zero below M_min."""
        mass = np.exp(np.asarray(ln_mass, dtype=float))
        z = np.asarray(z, dtype=float)
        m_min = 10.0 ** self.log10_Mmin
        interval = 10.0**self.log10_M1 - m_min
        above = np.maximum(mass - m_min, 0.0)
        fraction = np.where(above > 0.0, above / interval, 0.0)
        evolution = ((1.0 + z) / (1.0 + self.z_pivot)) ** self.epsilon
        return np.where(fraction > 0.0, fraction**self.alpha * evolution, 0.0)

    def lambda_central(self, ln_mass):
        r""":math:`\lambda^{\rm cen}`: 1 above :math:`M_{\min}`, else 0."""
        return (np.exp(np.asarray(ln_mass, dtype=float))
                >= 10.0**self.log10_Mmin).astype(float)

    def mean(self, ln_mass, z):
        r""":math:`\langle\lambda^{\rm tr}\rangle = \lambda^{\rm cen}
        + \langle\lambda^{\rm sat}\rangle`.

        NOTE: this is the **model's** mean occupation -- the calibrated
        quantity, central plus mean satellites. It is *not* the first
        moment of `pdf`, which comes out about **1 richness unit higher**:
        exactly :math:`+1.0000` when :math:`\sigma_{\rm intr} = 0`, and
        more as :math:`\sigma_{\rm intr}` grows (:math:`+3.5` at
        :math:`\sigma_{\rm intr} = 0.5`, :math:`M = 10^{15}`).

        That gap is an artifact of the continuous shifted-Poisson
        *interpolation* of a discrete law, not a bug in either: the density
        is built to match the discrete probabilities at integer richness and
        to extend smoothly between them, and its continuous first moment is
        not obliged to equal the discrete one.

        The consequence for callers: use this for **bracket placement**,
        where an offset of 1 is irrelevant beside :math:`L\sigma_{\rm eff}`
        with :math:`L = 8`, and do not use it as a prediction of the mean
        observed richness. `LogNormalMor.mean` has no such offset -- it
        reproduces its own density's first moment to 1e-14.
        """
        return self.lambda_central(ln_mass) + self.mu_sat(ln_mass, z)

    def std(self, ln_mass, z):
        r""":math:`\sqrt{\mu_{\rm sat} +
        (\sigma_{\rm intr}\mu_{\rm sat})^2}`, the HOD width."""
        mu = self.mu_sat(ln_mass, z)
        return np.sqrt(mu + (self.sigma_intr * mu) ** 2)

    def pdf(self, lambda_true, ln_mass, z):
        r"""The continuous shifted-Poisson density (module docstring)."""
        lambda_true, ln_mass, z = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (lambda_true, ln_mass, z))
        )
        central = self.lambda_central(ln_mass)
        mu_sat = self.mu_sat(ln_mass, z)

        # the intrinsic scatter shifts BOTH the Poisson mean and the argument
        delta = (self.sigma_intr * mu_sat) ** 2
        nu = mu_sat + delta
        x = lambda_true - central + delta

        support = (lambda_true >= central) & (x > 0.0)
        safe_x = np.where(support, x, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_pdf = (-nu + (safe_x - 1.0) * np.log(np.maximum(nu, 1e-300))
                       - gammaln(safe_x))
        density = np.where(support, np.exp(log_pdf), 0.0)

        # zero-satellite limit: a narrow Gaussian standing in for a delta
        degenerate = mu_sat <= POISSON_TOL
        fallback = (np.exp(-0.5 * ((lambda_true - central) / FALLBACK_SIGMA)**2)
                    / (np.sqrt(2.0 * np.pi) * FALLBACK_SIGMA))
        density = np.where(degenerate, fallback, density)
        return np.where(lambda_true >= 0.0, density, 0.0)

    def __repr__(self):
        return (f"HodMor(log10_Mmin={self.log10_Mmin:g}, "
                f"log10_M1={self.log10_M1:g}, alpha={self.alpha:g}, "
                f"epsilon={self.epsilon:g}, "
                f"sigma_intr={self.sigma_intr:g})")


if __name__ == "__main__":
    ln_m = np.log(np.array([1e13, 1e14, 3e14, 1e15]))

    for mor in (LogNormalMor(), HodMor()):
        print(mor)
        print(f"{'M [h^-1 Msun]':>14s}  {'<lambda>':>9s}  {'std':>9s}  "
              f"{'std/<l>':>8s}")
        for lm in ln_m:
            print(f"{np.exp(lm):14.2e}  {mor.mean(lm, 0.3).item():9.3f}  "
                  f"{mor.std(lm, 0.3).item():9.3f}  "
                  f"{(mor.std(lm, 0.3) / mor.mean(lm, 0.3)).item():8.4f}")

        # every pdf must integrate to 1 over its support -- the check that
        # catches a missing Jacobian or a wrong normalisation
        lam = np.linspace(1e-6, 600.0, 400001)
        for lm in ln_m[1:]:
            total = np.trapezoid(mor.pdf(lam, lm, 0.3), x=lam)
            print(f"  integral of P dlambda at M = {np.exp(lm):.1e}: "
                  f"{total:.6f}")
        print()

    print("the two relations disagree, and that is the point:")
    ln_mor, hod = LogNormalMor(), HodMor()
    print(f"{'M':>10s}  {'log-normal <l>':>15s}  {'HOD <l>':>10s}  "
          f"{'ratio':>7s}")
    for lm in ln_m:
        a = ln_mor.mean(lm, 0.3).item()
        b = hod.mean(lm, 0.3).item()
        print(f"{np.exp(lm):10.1e}  {a:15.3f}  {b:10.3f}  {a / b:7.3f}")
    print("  <- different calibrations (DES+SPT vs DES Y1), different")
    print("     parameters, and not interchangeable at fixed numbers.")

    print("\nthe log-normal Poisson floor is (<l>-1)/<l>^2, not 1/<l>:")
    for lam_mean in (0.5, 1.0, 2.0, 20.0, 100.0):
        naive = 1.0 / lam_mean
        correct = (lam_mean - 1.0) / lam_mean**2
        print(f"  <lambda> = {lam_mean:6.1f}:  1/<l> = {naive:8.5f}   "
              f"(<l>-1)/<l>^2 = {correct:8.5f}   "
              f"{'<- negative!' if correct < 0 else ''}")
    print("  the central galaxy carries no shot noise, hence the -1; below")
    print("  <lambda> = 1 the term is negative and D_lambda^2 must cover it.")

    print("\nHOD: the shifted-Poisson tail is super-Poisson at high "
          "occupancy:")
    for s in (0.0, 0.24, 0.5):
        h = HodMor(sigma_intr=s)
        lm = np.log(1e15)
        mu = h.mu_sat(lm, 0.3).item()
        print(f"  sigma_intr = {s:4.2f}:  mu_sat = {mu:8.3f}  "
              f"std = {h.std(lm, 0.3).item():8.3f}  "
              f"std/sqrt(mu) = {h.std(lm, 0.3).item() / np.sqrt(mu):6.3f}")
    print("  <- ratio 1 is Poisson; above 1 is the halo-to-halo term.")

    print("\ngammaln, not log(gamma): Gamma overflows past x ~ 171 and")
    print("cluster richness reaches 200.")
    from scipy.special import gamma as _gamma
    with np.errstate(over="ignore"):
        print(f"  Gamma(200) = {_gamma(200.0):.3e}   "
              f"gammaln(200) = {gammaln(200.0):.6f}")
