"""
Einasto projected profiles, v3.

Implements docs/einasto_proj_density_v3.tex: Retana-Montenegro et al. (2012)
case-1 residue series with a single coefficient a_k and rational weights.

Master form for profiles (units mass/length^2):

    Q(x) = sqrt(pi) * rho_0 * h * [
        sum_{k=1}^K  w_k * a_k * x^{k/n + 1}
      + sum_{k=0}^J  wt_k * at_k * x^{2k}
    ]

where x = R/h, and:
    a_k  = Gamma(-3/2 - k/(2n)) / Gamma(-k/(2n)) * (-1)^k / k!
    at_k = Gamma(n - 2nk) / Gamma(1/2 - k) * (-1)^k / k!

Weight factors:
    Sigma:        w_k = -(3n+k)/(2n),     wt_k = 1
    Sigma_bar:    w_k = -1,               wt_k = 1/(k+1)
    DeltaSigma:   w_k = (n+k)/(2n),       wt_k = -k/(k+1)

M_2D = pi * h^2 * x^2 * Sigma_bar

First track (a_k): carries cusp/core shape via fractional power x^{k/n}.
Second track (at_k): carries large-scale geometry, converges super-factorially.

Valid for all non-integer n > 0. For integer n: falls back to native E_nu series.
"""

import numpy as np
from scipy.special import gamma as _gamma, rgamma as _rgamma, gammainc

from .einasto import EinastoProfile, expn_fast


class EinastoProfileV3(EinastoProfile):
    """
    Einasto profile using Retana-Montenegro (2012) case-1 closed-form series.

    Parameters
    ----------
    alpha : float
        Shape parameter; n = 1/alpha.
    rho_0 : float
        Central density.
    r_s : float
        Scale radius.
    K : int
        Terms in the first track (core-shape series). K=60 typical.
    J : int
        Terms in the second track (geometry series). J=5 typical.
    order : int
        Fallback order for integer-n native series.
    """

    def __init__(self, alpha, rho_0, r_s, K=60, J=5, order=200, tol=None):
        # Bypass parent n>1.5 check: RM case-1 works for any non-integer n>0
        self.alpha = alpha
        self.rho_0 = rho_0
        self.r_s = r_s
        self.n_index = 1.0 / alpha
        self.h = self.r_s / (2 * self.n_index) ** self.n_index

        self._integer_n = abs(self.n_index - round(self.n_index)) < 1e-9
        self.K = K
        self.J = J

        if self._integer_n:
            # Need parent _build for native series fallback
            self.order = order
            n = self.n_index
            k = np.arange(0, order + 1)
            self._k = k
            from .einasto import _catalan_over_4k
            self._ck = _catalan_over_4k(k)
            self._nu_k = 2 * k * n - n + 1
        else:
            self._build_coefficients()

    def _build_coefficients(self):
        """Precompute a_k and at_k (depend on n only)."""
        n = self.n_index
        K, J = self.K, self.J

        # --- First track: a_k, k=1..K ---
        k = np.arange(1, K + 1, dtype=float)
        self._k1 = k
        arg_num = -1.5 - k / (2.0 * n)     # Gamma(-3/2 - k/(2n))
        arg_den = -k / (2.0 * n)            # Gamma(-k/(2n))
        sign_k = np.where(k % 2 == 0, 1.0, -1.0)  # (-1)^k
        log_kfact = np.zeros(K)
        np.cumsum(np.log(np.arange(1, K + 1)), out=log_kfact)
        kfact = np.exp(log_kfact)

        self._ak = _gamma(arg_num) * _rgamma(arg_den) * sign_k / kfact

        # Weights for each quantity (first track)
        self._w_sigma = -(3.0 * n + k) / (2.0 * n)
        self._w_sigbar = np.full(K, -1.0)
        self._w_ds = (n + k) / (2.0 * n)

        # Exponent: k/n + 1
        self._exp1 = k / n + 1.0

        # --- Second track: at_k, k=0..J ---
        j = np.arange(J + 1, dtype=float)
        self._j = j
        arg1 = n - 2.0 * n * j        # Gamma(n - 2nk)
        arg2 = 0.5 - j                 # Gamma(1/2 - k)
        sign_j = np.where(j % 2 == 0, 1.0, -1.0)
        jfact = np.ones(J + 1)
        for i in range(1, J + 1):
            jfact[i] = jfact[i - 1] * i

        # Guard poles in Gamma(n-2nk): set to 0 if near non-positive integer
        near_pole = np.abs(arg1 - np.rint(arg1)) < 1e-9
        gamma1 = np.where(near_pole, 0.0,
                          _gamma(np.where(near_pole, 1.0, arg1)))
        gamma2 = _gamma(arg2)

        self._atk = gamma1 / gamma2 * sign_j / jfact

        # Weights for each quantity (second track)
        self._wt_sigma = np.ones(J + 1)
        self._wt_sigbar = 1.0 / (j + 1.0)
        self._wt_ds = -j / (j + 1.0)  # 0 at k=0 → stable

        # Exponent: 2k
        self._exp2 = 2.0 * j

        # Prefactor
        self._prefactor = np.sqrt(np.pi) * self.rho_0 * self.h

    # ------------------------------------------------------------------
    def _eval_profile(self, x, w1, wt):
        """
        Evaluate master profile formula for array of x = R/h.

        Q(x) = sqrt(pi)*rho0*h * [sum w_k*a_k*x^{k/n+1} + 2n*sum wt_k*at_k*x^{2k}]
        """
        # First track: (nR, K)
        s1 = np.sum(
            (w1 * self._ak)[None, :] * x[:, None] ** self._exp1[None, :],
            axis=1)
        # Second track: (nR, J+1) — carries extra factor 2n from RM prefactor
        s2 = np.sum(
            (wt * self._atk)[None, :] * x[:, None] ** self._exp2[None, :],
            axis=1)
        return self._prefactor * (s1 + 2.0 * self.n_index * s2)

    # ------------------------------------------------------------------
    def sigma(self, R):
        R = np.atleast_1d(np.asarray(R, float))
        if self._integer_n:
            return super().sigma(R)
        x = R / self.h
        out = self._eval_profile(x, self._w_sigma, self._wt_sigma)
        return out if out.size > 1 else out.item()

    def mean_sigma(self, R):
        """Mean surface density Sigma_bar = M_2D / (pi R^2)."""
        R = np.atleast_1d(np.asarray(R, float))
        if self._integer_n:
            return np.atleast_1d(self.enclosed_mass_2D(R)) / (np.pi * R ** 2)
        x = R / self.h
        out = self._eval_profile(x, self._w_sigbar, self._wt_sigbar)
        return out if out.size > 1 else out.item()

    def enclosed_mass_2D(self, R):
        R = np.atleast_1d(np.asarray(R, float))
        if self._integer_n:
            return super().enclosed_mass_2D(R)
        out = np.pi * R ** 2 * np.atleast_1d(self.mean_sigma(R))
        return out if out.size > 1 else out.item()

    def deltasigma(self, R):
        R = np.atleast_1d(np.asarray(R, float))
        if self._integer_n:
            return super().deltasigma(R)
        x = R / self.h
        out = self._eval_profile(x, self._w_ds, self._wt_ds)
        return out if out.size > 1 else out.item()

    # ------------------------------------------------------------------
    # 3D quantities (inherited, but override to remove n>1.5 restriction)
    # ------------------------------------------------------------------
    def density(self, r):
        x = np.asarray(r) / self.h
        return self.rho_0 * np.exp(-x ** (1.0 / self.n_index))

    def enclosed_mass(self, r):
        n, h = self.n_index, self.h
        z = (np.asarray(r) / h) ** (1.0 / n)
        from scipy.special import gamma as _gam
        gamma_lower = gammainc(3 * n, z) * _gam(3 * n)
        return 4 * np.pi * self.rho_0 * n * h ** 3 * gamma_lower

    @property
    def total_mass(self):
        n, h = self.n_index, self.h
        return 4 * np.pi * self.rho_0 * n * h ** 3 * _gamma(3 * n)
