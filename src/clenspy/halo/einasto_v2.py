"""
Einasto projected profiles, v2: inner-zone dual form.

This is the implementation companion to docs/einasto_proj_density_v2.tex.

For z = (R/h)^(1/n) < 1 it evaluates the master Catalan sum

    G_w(z) = sum_k w_k c_k E_{nu_k}(z),   w_k in {1, k+1, k},

via the "dual form" (Theorem of Sec. 2 of the v2 note):

    G_w(z) = Psi_w(z) + sum_{m>=0} (-z)^m / m! * Phi_w(n, m),

with
    Psi_w(z)    = sum_k w_k c_k Gamma(n(1-2k)) z^{n(2k-1)}   (reflection part),
    Phi_w(n,m)  = sum_k w_k c_k / (2 n k - n - m)            (regular part).

The Phi_w(n,m) and the k-arrays of Psi depend on the shape index n only and are
precomputed once per halo; each radius then costs a polynomial evaluation, with
NO per-point mpmath / special-function call. This only pays off for non-integer
n with many radii at z<1 (see docs). For z>=1, and for integer n, it defers to
the v1 native series, which is more robust there.

WARNING: this is a benchmark / research implementation. The native v1 path
(EinastoProfile) remains the production recommendation; see the v2 note's
conclusion.
"""

import numpy as np
from scipy.special import gamma as _gamma

from .einasto import EinastoProfile, _catalan_over_4k


class EinastoProfileV2(EinastoProfile):
    """
    Einasto profile with the inner-zone (z<1) dual-form projected quantities.

    Parameters mirror :class:`EinastoProfile`, plus ``m_order`` controlling the
    truncation of the regular power series in z.
    """

    def __init__(self, alpha, rho_0, r_s, order=100, tol=None, m_order=24):
        super().__init__(alpha, rho_0, r_s, order=order, tol=tol)
        self.m_order = m_order
        # Integer n: all nu_k integer, Gamma(n(1-2k)) is all poles and the
        # regular-series denominators 2nk-n-m hit zero densely. This is exactly
        # the case the v2 note says needs no reformulation -- defer to v1.
        self._integer_n = abs(self.n_index - round(self.n_index)) < 1e-9
        if not self._integer_n:
            self._build_v2()

    def _build_v2(self):
        """Precompute the n-only coefficient tables (independent of R)."""
        n = self.n_index
        k = self._k.astype(float)          # 0..order
        ck = self._ck                      # c_k

        self._w = {
            "m2d": np.ones_like(k),
            "sigma": k + 1.0,
            "deltasigma": k,
        }

        # Reflection prefactors Gamma(n(1-2k)); poles at integer arg are masked
        # (those k fall back to the native term, handled in _G_inner).
        arg = n * (1.0 - 2.0 * k)
        near_pole = np.abs(arg - np.rint(arg)) < 1e-9
        gamma_refl = np.empty_like(arg)
        gamma_refl[~near_pole] = _gamma(arg[~near_pole])
        gamma_refl[near_pole] = np.nan
        self._gamma_refl = gamma_refl
        self._refl_pole = near_pole
        self._refl_exp = n * (2.0 * k - 1.0)          # exponent of z

        # Phi_w(n,m) closed form (Retana-Montenegro+2012, case 1; v2 note Thm 1):
        #   Phi^(k+1)(m) =  sqrt(pi)/(2n) * Gamma(-1/2 - m/(2n)) / Gamma(-m/(2n))
        #   Phi^(1)  (m) = -sqrt(pi)/(2n) * Gamma(-3/2 - m/(2n)) / Gamma(-m/(2n))
        #                   - 2/(3n + m)
        #   Phi^(k)  (m) =  Phi^(k+1)(m) - Phi^(1)(m)
        # No k-sum, no truncation.
        m = np.arange(0, self.m_order + 1, dtype=float)
        a = -0.5 - m / (2.0 * n)
        b = -1.5 - m / (2.0 * n)
        c = -m / (2.0 * n)            # Gamma(c) -> pole at m=0 (c=0)
        # Gamma(c) is +/- inf at integer non-positive c; reciprocal gamma is 0 there.
        # Use 1/Gamma(c) (rgamma) to handle c<=0 integer cleanly.
        from scipy.special import rgamma as _rgamma
        sqrtpi_over_2n = np.sqrt(np.pi) / (2.0 * n)
        rgamma_c = _rgamma(c)         # 0 at c=0,-1,-2,... (when m=0 mod 2n)
        phi_kp1 = sqrtpi_over_2n * _gamma(a) * rgamma_c
        phi_1   = -sqrtpi_over_2n * _gamma(b) * rgamma_c - 2.0 / (3.0 * n + m)
        phi_k   = phi_kp1 - phi_1
        self._phi = {"sigma": phi_kp1, "m2d": phi_1, "deltasigma": phi_k}

        # m! and the alternating power factors are radius-independent in part.
        self._m = m
        self._logfact_m = np.cumsum(np.concatenate(([0.0], np.log(np.arange(1, self.m_order + 1)))))

    # ------------------------------------------------------------------
    def _G_inner(self, z, quantity):
        """
        Dual-form master sum G_w(z) for z < 1 (z is a 1-D array of scaled radii).

        Returns array shape (z.size,).
        """
        z = np.atleast_1d(np.asarray(z, float))
        w = self._w[quantity]
        ck = self._ck

        # --- regular part: sum_m (-z)^m / m! * Phi_w(m) ---
        m = self._m
        # (nz, M+1): (-z)^m / m!
        with np.errstate(over="ignore"):
            logterm = m[None, :] * np.log(z[:, None]) - self._logfact_m[None, :]
        sign = (-1.0) ** m
        reg = np.sum(sign[None, :] * np.exp(logterm) * self._phi[quantity][None, :], axis=1)

        # --- reflection part: sum_k w_k c_k Gamma(n(1-2k)) z^{n(2k-1)} ---
        good = ~self._refl_pole
        zexp = z[:, None] ** self._refl_exp[None, :]       # (nz, K+1)
        refl_terms = (w * ck * self._gamma_refl)[None, :] * zexp
        refl = np.sum(np.where(good[None, :], refl_terms, 0.0), axis=1)

        # --- pole-k fallback: add native w_k c_k E_{nu_k}(z) for masked k ---
        if self._refl_pole.any():
            from .einasto import expn_fast
            kp = np.where(self._refl_pole)[0]
            nu = self._nu_k[kp]
            E = expn_fast(nu[None, :], z[:, None])         # (nz, npole)
            refl = refl + np.sum((w[kp] * ck[kp])[None, :] * E, axis=1)

        return reg + refl

    # ------------------------------------------------------------------
    def sigma(self, R):
        if self._integer_n:
            return super().sigma(R)
        R = np.atleast_1d(np.asarray(R, float))
        z = (R / self.h) ** (1.0 / self.n_index)
        out = np.empty_like(R)
        inner = z < 1.0
        if inner.any():
            G = self._G_inner(z[inner], "sigma")
            out[inner] = 2 * self.rho_0 * self.n_index * R[inner] * G
        if (~inner).any():
            out[~inner] = np.atleast_1d(super().sigma(R[~inner]))
        return out if out.size > 1 else out.item()

    def enclosed_mass_2D(self, R):
        if self._integer_n:
            return super().enclosed_mass_2D(R)
        R = np.atleast_1d(np.asarray(R, float))
        z = (R / self.h) ** (1.0 / self.n_index)
        out = np.empty_like(R)
        inner = z < 1.0
        if inner.any():
            G = self._G_inner(z[inner], "m2d")
            out[inner] = self.enclosed_mass(R[inner]) \
                + 2 * np.pi * self.rho_0 * self.n_index * R[inner] ** 3 * G
        if (~inner).any():
            out[~inner] = np.atleast_1d(super().enclosed_mass_2D(R[~inner]))
        return out if out.size > 1 else out.item()

    def deltasigma(self, R):
        if self._integer_n:
            return super().deltasigma(R)
        R = np.atleast_1d(np.asarray(R, float))
        z = (R / self.h) ** (1.0 / self.n_index)
        out = np.empty_like(R)
        inner = z < 1.0
        if inner.any():
            G = self._G_inner(z[inner], "deltasigma")
            mean = self.enclosed_mass(R[inner]) / (np.pi * R[inner] ** 2)
            out[inner] = mean - 2 * self.rho_0 * self.n_index * R[inner] * G
        if (~inner).any():
            out[~inner] = np.atleast_1d(super().deltasigma(R[~inner]))
        return out if out.size > 1 else out.item()
