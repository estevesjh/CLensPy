"""Mass-observable relations P(lambda_tr | M, z) for optically selected clusters.

Physical units: masses in Msun (no little-h).  Published fit parameters are
usually quoted in Msun/h — use the h-converting classmethod constructors
(:meth:`HodParams.des_y1`, :meth:`LogNormalParams.costanzi21`) rather than
pasting raw h-unit numbers into the physical-unit fields.

Contents
--------
``MassObservableRelation``
    ABC defining the MOR protocol: ``pdf``, ``ltr_mean``, ``ltr_sigma``,
    plus the shared quadrature ``bracket``.
``HodMOR``
    Shifted-Poisson x lognormal-intrinsic HOD relation (Costanzi 2026 /
    DES Y1 NC+3x2pt form).
``LogNormalMOR``
    Lognormal relation with the Costanzi 2021 Poisson-augmented variance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

__all__ = [
    "MassObservableRelation",
    "HodParams",
    "HodMOR",
    "LogNormalParams",
    "LogNormalMOR",
]


class MassObservableRelation(ABC):
    r"""Protocol for :math:`P(\lambda^{\rm tr} \mid M, z)` relations.

    Masses are physical Msun.  All methods are broadcast-safe.
    """

    @abstractmethod
    def pdf(self, ltr, M, z):
        r""":math:`P(\lambda^{\rm tr} \mid M, z)`."""

    @abstractmethod
    def ltr_mean(self, M, z):
        r"""Effective mean of :math:`P(\lambda^{\rm tr} \mid M, z)`."""

    @abstractmethod
    def ltr_sigma(self, M, z):
        r"""Effective width of :math:`P(\lambda^{\rm tr} \mid M, z)`."""

    def bracket(self, M, z, L: float = 6.0):
        r"""Feature-placed quadrature bracket.

        .. math::

            [a, b] = [\max(0, \mu_{\rm eff} - L\sigma_{\rm eff}),\;
                      \mu_{\rm eff} + L\sigma_{\rm eff}]

        Returns ``(a, b)`` broadcast over ``(M, z)``.  This is the support
        on which fixed Gauss-Legendre nodes resolve the near-delta ridge of
        the MOR at low mass (y3_cluster_cpp convention, L ~ 6-8).
        """
        mu = self.ltr_mean(M, z)
        sig = self.ltr_sigma(M, z)
        a = np.maximum(0.0, mu - L * sig)
        b = mu + L * sig
        return a, b


@dataclass(frozen=True)
class HodParams:
    """HOD MOR parameters (physical Msun pivots).

    ``M_min`` and ``M1`` are physical masses; use :meth:`des_y1` to build
    from the published Msun/h best fit.
    """

    M_min: float
    M1: float
    alpha: float
    sigma_intr: float
    epsilon: float = 0.0
    z_pivot: float = 0.4544

    @classmethod
    def des_y1(cls, h: float) -> "HodParams":
        """DES Y1 NC+3x2pt best fit (Costanzi 2026), converted from Msun/h."""
        return cls(
            M_min=10.0**11.3852818 / h,
            M1=10.0**12.6964410 / h,
            alpha=0.858693714,
            sigma_intr=0.180949022,
        )


class HodMOR(MassObservableRelation):
    r"""Shifted-Poisson :math:`\otimes` lognormal-intrinsic HOD relation.

    .. math::

        \lambda_{\rm sat}(M, z) = \left(\frac{M - M_{\min}}{M_1 - M_{\min}}
            \right)^{\alpha} \left(\frac{1+z}{1+z_p}\right)^{\epsilon}

    with continuous Poisson-Gaussian convolved pdf

    .. math::

        P(\lambda^{\rm tr} \mid M, z) = \exp\!\left[-\nu
            + (x - 1)\ln\nu - \ln\Gamma(x)\right],
        \quad \nu = \lambda_{\rm sat} + (\sigma_{\rm intr}\lambda_{\rm sat})^2,
        \quad x = \lambda^{\rm tr} + (\sigma_{\rm intr}\lambda_{\rm sat})^2 .
    """

    def __init__(self, params: HodParams) -> None:
        self.params = params
        self.M_min = params.M_min
        self.M1 = params.M1
        self.M_pivot = params.M1 - params.M_min
        self.alpha = params.alpha
        self.sigma_intr = params.sigma_intr
        self.epsilon = params.epsilon
        self.z_pivot = params.z_pivot

    def l_sat(self, M, z):
        r"""Mean satellite count above the richness threshold."""
        M = np.asarray(M, dtype=float)
        z = np.asarray(z, dtype=float)
        frac = np.clip((M - self.M_min) / self.M_pivot, 1e-30, None)
        return frac**self.alpha * ((1.0 + z) / (1.0 + self.z_pivot)) ** self.epsilon

    def l_tr(self, M, z):
        return 1.0 + self.l_sat(M, z)

    def ltr_mean(self, M, z):
        return self.l_sat(M, z)

    def ltr_sigma(self, M, z):
        m = self.l_sat(M, z)
        return np.sqrt(m + (m * self.sigma_intr) ** 2)

    def pdf(self, ltr, M, z):
        ltr = np.asarray(ltr, dtype=float)
        m = self.l_sat(M, z)
        mi = (m * self.sigma_intr) ** 2
        lam = m + mi
        x = ltr + mi
        val = np.exp(
            -lam + (x - 1.0) * np.log(np.clip(lam, 1e-300, None)) - gammaln(x)
        )
        return np.where(ltr >= 0.0, val, 0.0)

    def lambda_mean_below(self, M, z, lob, ltr_n: int = 400):
        r""":math:`\langle\lambda^{\rm tr}\rangle_{<\lambda^{\rm ob}}(M, z)`.

        Trapezoidal :math:`\int_0^{\lambda^{\rm ob}} \lambda\,
        P(\lambda \mid M, z)\, d\lambda`, tracking the Poisson-Gaussian
        peak so the grid resolves it even when ``lob`` is far in the tail.
        Used by the selection-bias closure.
        """
        M_arr = np.atleast_1d(np.asarray(M, dtype=float))
        m = self.l_sat(M_arr, z)
        lam = m + (m * self.sigma_intr) ** 2
        upper = min(float(lob), float(np.max(lam + 15.0 * np.sqrt(lam + 1.0))) + 1.0)
        grid = np.linspace(0.0, upper, ltr_n)
        p = self.pdf(grid[:, None], M_arr[None, :], z)
        out = np.trapezoid(grid[:, None] * p, grid, axis=0)
        return float(out[0]) if np.ndim(M) == 0 else out


@dataclass(frozen=True)
class LogNormalParams:
    """Lognormal MOR parameters (physical Msun pivot).

    Use :meth:`costanzi21` to build from the published Msun/h pivot.
    """

    A_lambda: float = 76.9
    B_lambda: float = 1.020
    C_lambda: float = 0.29
    D_lambda: float = 0.23
    M_pivot: float = 3.0e14
    z_pivot: float = 0.45

    @classmethod
    def costanzi21(cls, h: float) -> "LogNormalParams":
        """Costanzi 2021 DES+SPT best fit; pivot 3e14 Msun/h -> physical."""
        return cls(M_pivot=3.0e14 / h)


class LogNormalMOR(MassObservableRelation):
    r"""Lognormal relation with Costanzi 2021 Poisson-augmented variance.

    .. math::

        \langle\ln\lambda\rangle = \ln A + B \ln\frac{M}{M_p}
            + C \ln\frac{1+z}{1+z_p},
        \qquad
        \sigma^2_{\ln\lambda} = D^2
            + \frac{\langle\lambda\rangle - 1}{\langle\lambda\rangle^2} .
    """

    def __init__(self, params: LogNormalParams = LogNormalParams()) -> None:
        self.params = params

    def ln_lambda_mean(self, M, z):
        p = self.params
        M = np.asarray(M, dtype=float)
        z = np.asarray(z, dtype=float)
        return (
            np.log(p.A_lambda)
            + p.B_lambda * np.log(M / p.M_pivot)
            + p.C_lambda * np.log((1.0 + z) / (1.0 + p.z_pivot))
        )

    def sigma_ln_lambda(self, M, z):
        lam_lin = np.exp(self.ln_lambda_mean(M, z))
        lam_lin = np.maximum(lam_lin, 1.0 + 1e-12)
        var = self.params.D_lambda**2 + (lam_lin - 1.0) / lam_lin**2
        return np.sqrt(var)

    def ltr_mean(self, M, z):
        mu = self.ln_lambda_mean(M, z)
        s2 = self.sigma_ln_lambda(M, z) ** 2
        return np.exp(mu + 0.5 * s2)

    def ltr_sigma(self, M, z):
        s2 = self.sigma_ln_lambda(M, z) ** 2
        return self.ltr_mean(M, z) * np.sqrt(np.expm1(s2))

    def pdf(self, ltr, M, z):
        ltr = np.asarray(ltr, dtype=float)
        mu_ln = self.ln_lambda_mean(M, z)
        sig_ln = self.sigma_ln_lambda(M, z)
        ltr_safe = np.clip(ltr, 1e-300, None)
        val = np.exp(-0.5 * ((np.log(ltr_safe) - mu_ln) / sig_ln) ** 2) / (
            ltr_safe * sig_ln * np.sqrt(2.0 * np.pi)
        )
        return np.where(ltr > 0.0, val, 0.0)
