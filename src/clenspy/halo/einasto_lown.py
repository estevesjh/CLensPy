"""
Einasto projected lensing quantities, stable for any n > 0.

(Originally built for the n <= 3/2 regime of issue #3; later extended to
all n after the legacy Catalan evaluation of DeltaSigma was measured at
30-200% relative error for n = 3.3-10 at every radius -- its truncation
error is *absolute* O(K^{-1/2}) while DeltaSigma is small. Integer n is
handled through the eps -> 0 resonance-pairing limits, which reproduce the
case-2/3 logarithmic series.)

Implements the Retana-Montenegro et al. (2012) case-1 residue series in a
numerically stable fp64 form, with per-quantity coefficients read directly
off one Mellin-Barnes kernel per quantity (docs/einasto_proj_density_v3.tex;
no chained weight ladders):

    Sigma(R)      = sqrt(pi) rho_0 h [ sum_k A_k x^{k/n+1} + sum_j S_j x^{2j} ]
    DeltaSigma(R) = sqrt(pi) rho_0 h [ sum_k D_k x^{k/n+1} + sum_j T_j x^{2j} ]

    x = R/h,
    A_k = (-1)^k/k! Gamma(-1/2-k/(2n))/Gamma(-k/(2n))
    D_k = -(n+k)/(3n+k) A_k     [Gamma(z+1) = z Gamma(z), z = -1/2-k/(2n)]
    S_j = 2n (-1)^j/j! Gamma(n-2nj)/Gamma(1/2-j)
    T_j = -j/(j+1) S_j                                   (T_0 = 0)
    Sigma_bar = Sigma + DeltaSigma,  M_2D = pi R^2 Sigma_bar.

The series equals the exact profile for ALL x (the Mellin-Barnes contour
closes left for any n > 0; verified against Abel quadrature and a direct
contour integration to 1e-40).  In fp64 its use is limited only by
cancellation: the largest term grows like e^{z} (z = x^{1/n}) while
DeltaSigma stays algebraic, losing ~0.5 z digits (~0.87 z for Sigma).  The
switch point z_sw is therefore *measured at build time* (max-term/result
scan) and beyond it evaluation dispatches to the all-positive Catalan
E_nu representation (docs/einasto_proj_density.tex, Theorem 1):

    Sigma      = 2 rho_0 n R sum_{k>=0} (k+1) c_k E_{nu_k}(z)
    DeltaSigma = M_3D/(pi R^2) - 2 rho_0 n R sum_{k>=1} k c_k E_{nu_k}(z)

with E_nu by a Lentz continued fraction (stable for z >> nu, where the
upward recurrence in expn_fast explodes) below nu = 160 and the DLMF
8.20(ii) uniform expansion above, plus closed-form integral tail
corrections for the algebraic k^{-1/2}/(2nk+b) tail.

Resonances: for n = p/q with q odd, the two pole strings collide at
k = n(2j-1): both gamma coefficients diverge like 1/eps with
eps = k/n - (2j-1), and the pair sums to a finite log(x) term.  Pairs with
|eps| < EPS_PAIR are evaluated jointly:

    pair(x) = x^{2j} ( p ln(x) phi(eps ln x) + s ),  phi(y) = expm1(y)/y,

with p = c1*eps and s = c1+c2 precomputed once per n in mpmath (eps itself
must be formed in extended precision: in fp64 it retains only
16 - |log10 eps| digits).  The identity is exact for every eps, so pairing
is applied generously; it reproduces the integer-limit log series
continuously in n.

Validated against mpmath Abel/cap quadrature to <= 4e-9 relative error for
n in [0.35, 1.5] (including 6/5, 4/3, 7/5) and R/h in [0.01, 40], and to
<= 1e-13 for n in {2.5, 10/3, 5, 10} (integer-resonant cases) over
R/h in [0.01, 20].
"""

import numpy as np
from scipy.special import gamma as _gamma, gammainc, gammaln

SQPI = np.sqrt(np.pi)

#: pair first/second-track terms whenever |eps| is below this. The paired
#: form is exact for any eps; uniqueness of the (k -> j) map only needs
#: 1/n > 2*EPS_PAIR, satisfied for all n <= 2.
EPS_PAIR = 0.25

#: E_nu dispatch boundary: below, Lentz CF; above, DLMF 8.20(ii) uniform
#: asymptotic (5 terms, error ~ nu^-5 < 1e-11 uniformly in z/nu >= 0).
NU_ASYMP = 160.0


def _sinpi(y):
    """sin(pi*y) with argument reduction (accurate for large y)."""
    y = np.asarray(y, float)
    return np.sin(np.pi * (y - 2.0 * np.round(y / 2.0)))


def neg_gammaln(y):
    """(sign, log|Gamma(-y)|) for y > 0 via reflection:
    Gamma(-y) = pi / (sin(-pi y) Gamma(1+y)).
    At an exact pole (y integer): sign 0, logabs +inf.  Detected by
    y == rint(y), not by sin == 0: the reduced sin(pi*odd) is ~1e-16,
    not zero, and would silently turn an exact pole into a wrong tiny
    coefficient (matters for integer n, where half the arguments are
    exact integers)."""
    y = np.asarray(y, float)
    pole = y == np.rint(y)
    s = _sinpi(y)
    sign = np.where(pole, 0.0, -np.sign(s))
    with np.errstate(divide="ignore"):
        logabs = np.where(
            pole, np.inf,
            np.log(np.pi) - np.log(np.abs(np.where(pole, 1.0, s)))
            - gammaln(1.0 + y))
    return sign, logabs


def expn_cf(nu, z, iters=120):
    """E_nu(z) by the incomplete-gamma continued fraction (modified Lentz):

        E_nu(z) = e^{-z} / (z+nu - 1*nu/(z+nu+2 - 2(nu+1)/(z+nu+4 - ...)))

    Stable for z >~ 1.5 and any nu >= 1 -- in particular for z >> nu, where
    the upward recurrence used by expn_fast is exponentially unstable.
    Verified at machine precision for nu in [2, 4200] x z in [7, 60].
    """
    nu, z = np.broadcast_arrays(np.asarray(nu, float), np.asarray(z, float))
    b = z + nu
    c = np.full(b.shape, 1e300)
    d = 1.0 / b
    h = d.copy()
    for i in range(1, iters + 1):
        a = -i * (nu - 1.0 + i)
        b = b + 2.0
        d = 1.0 / (a * d + b)
        c = b + a / c
        h = h * (c * d)
    return np.exp(-z) * h


class EinastoLowN:
    """Series backend for any n > 0 (see module docstring).

    ``EinastoProfile`` handles the exact anchors n = 1/2 and n = 1 with
    closed forms and only constructs this class for other n.
    """

    Z_CAP = 30.0          # never use the power series beyond this z
    K_ENU = 6000          # E_nu branch truncation (tail-corrected)

    def __init__(self, n, rho_0, h, tol=1e-9):
        self.n = float(n)
        self.rho_0 = float(rho_0)
        self.h = float(h)
        self.tol = float(tol)
        n = self.n
        # z range the power series must cover: for n <= 3/2 the fp64 budget
        # caps it near 30 anyway; for larger n, z = x^{1/n} is compressed
        # (x = 40 h -> z = 40^{1/n}), so a much smaller cap suffices and
        # keeps the coefficient tables small.
        zmax = self.Z_CAP if n <= 1.5 else max(6.0, 1.2 * 40.0 ** (1.0 / n))
        self.z_cap_n = zmax
        K_conv = int(zmax + 12 * np.sqrt(zmax) + 25)
        # J must cover (i) the second track's own convergence and (ii) the
        # resonant partner j = (k/n+1)/2 of every retained k (bites n < 1/2)
        self.J = int(max(zmax / (2 * n) + 12 * np.sqrt(zmax / (2 * n)) + 15,
                         (K_conv / n + 1) / 2 + 2))
        # ...and K must cover the partner k = n(2j-1) of every retained j:
        # for integer n every j >= 1 is an exact pole and *must* be paired
        self.K = max(K_conv, int(n * (2 * self.J - 1)) + 3)
        self._build()
        # self-calibrated dispatch: measure the digits actually lost to
        # cancellation (max term / result) on a z grid and switch to the
        # E_nu branch where the loss exceeds the fp64 budget for tol.
        budget = 15.9 + np.log10(self.tol) - 1.2
        self.zsw_sig = self._zsw("sigma", budget)
        self.zsw_ds = self._zsw("ds", budget)

    # ------------------------------------------------------------------
    # build: coefficient tables (per n), resonant pairs
    # ------------------------------------------------------------------
    def _build(self):
        n, K, J = self.n, self.K, self.J
        k = np.arange(1, K + 1, dtype=float)
        kap = k / (2 * n)
        sgn_k = np.where(k.astype(int) % 2 == 0, 1.0, -1.0)      # (-1)^k
        # Gamma(1/2 - kappa): positive argument for kappa < 1/2, else
        # reflected negative-argument form.
        arg = kap - 0.5
        pos = arg <= 0
        sg_half = np.empty(K)
        lg_half = np.empty(K)
        if pos.any():
            sg_half[pos] = 1.0
            lg_half[pos] = gammaln(0.5 - kap[pos])
        if (~pos).any():
            s_, l_ = neg_gammaln(arg[~pos])
            sg_half[~pos], lg_half[~pos] = s_, l_
        sg_den, lg_den = neg_gammaln(kap)                        # Gamma(-kap)
        lg_fact = gammaln(k + 1.0)

        # D_k; an exact pole in the denominator gamma -> coefficient 0
        pole_den = np.isinf(lg_den)
        self._sD = np.where(pole_den, 0.0, sgn_k * sg_half * sg_den)
        with np.errstate(invalid="ignore"):
            self._lD = np.where(
                pole_den, -np.inf,
                np.log(2 * n / (3 * n + k)) + lg_half - lg_den - lg_fact)
        # single gamma-ratio family: D_k = -(n+k)/(3n+k) A_k, i.e.
        # A_k = -(3n+k)/(n+k) D_k  (Gamma(z+1) = z Gamma(z) at z=-1/2-k/(2n))
        fac = (3 * n + k) / (n + k)
        self._sA = -self._sD
        self._lA = self._lD + np.log(fac)
        self._exp1 = k / n + 1.0

        j = np.arange(0, J + 1, dtype=float)
        y2 = 2 * n * j - n                       # Gamma(n-2nj) = Gamma(-y2)
        sg2 = np.empty(J + 1)
        lg2 = np.empty(J + 1)
        sg2[0], lg2[0] = 1.0, gammaln(n)
        if J >= 1:
            s_, l_ = neg_gammaln(y2[1:])
            sg2[1:], lg2[1:] = s_, l_
        # S_j = 2n/j! Gamma(n-2nj) Gamma(j+1/2)/pi  (reflection of 1/G(1/2-j))
        self._sS = sg2
        self._lS = np.log(2 * n / np.pi) + lg2 + gammaln(j + 0.5) \
            - gammaln(j + 1.0)
        self._sT = -self._sS
        with np.errstate(divide="ignore"):
            self._lT = self._lS + np.log(j / (j + 1.0))   # T_0 = 0
        self._exp2 = 2.0 * j

        self._build_pairs()
        # every remaining +inf log-coefficient (numerator gamma at an exact
        # pole) must be negligible at all z <= z_cap_n (far past the term
        # peak, its pair partner may fall outside the truncation): zero it.
        zmax = self.z_cap_n
        k_neg = min(zmax + 8 * np.sqrt(zmax) + 10, n * (2 * self.J - 1))
        j_neg = zmax / (2 * n) + 8 * np.sqrt(zmax / (2 * n)) + 5
        for s_, l_, m_, idx_neg in (
                (self._sA, self._lA, self._mask1, k_neg - 1),
                (self._sD, self._lD, self._mask1, k_neg - 1),
                (self._sS, self._lS, self._mask2, j_neg),
                (self._sT, self._lT, self._mask2, j_neg)):
            bad = m_ & (np.isposinf(l_) | np.isnan(l_))
            if bad.any():
                if np.where(bad)[0].min() < idx_neg:
                    raise RuntimeError(
                        f"unpaired singular series coefficient at n={n}: "
                        f"indices {np.where(bad)[0][:5]}")
                s_[bad], l_[bad] = 0.0, -np.inf

    def _build_pairs(self):
        """Detect resonances k/n ~ 2j-1; precompute (j, eps, p, s) per
        quantity in mpmath (eps needs extended precision)."""
        n, K, J = self.n, self.K, self.J
        # exact identity for any eps, so pair generously -- but the k <-> j
        # map must stay one-to-one: window half-width < half the exponent
        # spacing 1/n (relevant once n > 2)
        eps_pair = min(EPS_PAIR, 0.4 / n)
        cand = []
        taken = set()
        for k in range(1, K + 1):
            t = k / n + 1.0
            j = int(round(t / 2.0))
            if 1 <= j <= J and j not in taken and abs(t - 2 * j) < eps_pair:
                cand.append((k, j))
                taken.add(j)
        self._pairs = {"sigma": [], "ds": []}
        self._mask1 = np.ones(K, bool)
        self._mask2 = np.ones(J + 1, bool)
        if not cand:
            return
        import mpmath as mp
        for k, j in cand:
            self._mask1[k - 1] = False
            self._mask2[j] = False
            with mp.workdps(80):
                nn = mp.mpf(n)
                eps = k / nn + 1 - 2 * j
                if eps == 0:
                    # exact resonance (integer n, or n = p/q with q odd and
                    # exactly representable): take the eps -> 0 limit by
                    # nudging n at a precision where the nudge is resolved;
                    # p and s converge to their limits with error O(nudge).
                    extra = 300
                else:
                    extra = max(0, int(-mp.log10(abs(eps))) + 10)
            with mp.workdps(80 + extra):
                nn = mp.mpf(n)
                if eps == 0:
                    nn = nn * (1 + mp.mpf(10) ** (-150))
                eps_b = k / nn + 1 - 2 * j
                kapb = k / (2 * nn)
                # one gamma per track: A_k, then D_k = -(n+k)/(3n+k) A_k
                a_k = (-1) ** k / mp.factorial(k) * mp.rgamma(-kapb) \
                    * mp.gamma(-mp.mpf(0.5) - kapb)
                s_j = 2 * nn * (-1) ** j / mp.factorial(j) \
                    * mp.gamma(nn - 2 * nn * j) * mp.rgamma(mp.mpf(0.5) - j)
                for which in ("sigma", "ds"):
                    if which == "sigma":
                        c1, c2 = a_k, s_j
                    else:
                        c1 = -(nn + k) / (3 * nn + k) * a_k
                        c2 = s_j * (-mp.mpf(j) / (j + 1))
                    self._pairs[which].append(
                        (j, float(eps_b), float(c1 * eps_b), float(c1 + c2)))

    # ------------------------------------------------------------------
    # power series (z <= z_switch)
    # ------------------------------------------------------------------
    def _series(self, R, which, want_maxterm=False):
        """sqrt(pi) rho_0 h [track1 + track2 + pairs], log-space terms."""
        x = np.atleast_1d(np.asarray(R, float)) / self.h
        out = np.zeros_like(x)
        mt = np.zeros_like(x)
        nz = x > 0
        L = np.log(x[nz])
        if which == "sigma":
            s1, l1, s2, l2 = self._sA, self._lA, self._sS, self._lS
        else:
            s1, l1, s2, l2 = self._sD, self._lD, self._sT, self._lT
        m1, m2 = self._mask1, self._mask2
        e1 = np.exp(l1[m1][None, :] + self._exp1[m1][None, :] * L[:, None])
        e2 = np.exp(l2[m2][None, :] + self._exp2[m2][None, :] * L[:, None])
        tot = (s1[m1][None, :] * e1).sum(axis=1) \
            + (s2[m2][None, :] * e2).sum(axis=1)
        for j, eps, p, s in self._pairs[which]:
            y = eps * L
            phi = np.where(y != 0, np.expm1(y) / np.where(y == 0, 1, y), 1.0)
            # x^{2j} * (p L phi + s), composed in log space: x^{2j} alone
            # can overflow while the product is finite (superfactorially
            # small pair coefficient at large j)
            base = p * L * phi + s
            with np.errstate(divide="ignore", over="ignore"):
                mag = 2 * j * L + np.log(np.abs(np.where(base == 0, 1, base)))
                tot += np.where(base == 0, 0.0, np.sign(base) * np.exp(
                    np.where(base == 0, -np.inf, mag)))
        pref = SQPI * self.rho_0 * self.h
        out[nz] = pref * tot
        # x = 0 limits: Sigma(0) = 2 rho_0 h Gamma(n+1), DeltaSigma(0) = 0
        if (~nz).any() and which == "sigma":
            out[~nz] = 2.0 * self.rho_0 * self.h * _gamma(self.n + 1.0)
        if want_maxterm:
            mt[nz] = pref * np.maximum(e1.max(axis=1) if e1.size else 0.0,
                                       e2.max(axis=1) if e2.size else 0.0)
            return out, mt
        return out

    def _zsw(self, which, budget):
        zg = np.linspace(2.5, self.z_cap_n, 56)
        val, mt = self._series(self.h * zg ** self.n, which,
                               want_maxterm=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            lost = np.log10(mt / np.abs(val))
        bad = np.maximum.accumulate(lost) > budget
        return self.z_cap_n if not bad.any() \
            else zg[max(np.argmax(bad) - 1, 0)]

    # ------------------------------------------------------------------
    # E_nu branch (z > z_switch)
    # ------------------------------------------------------------------
    def _enu(self, R, which):
        """Catalan E_nu representation with closed-form tail corrections.

        Tail (k > K_ENU), with b = z+1-n, K' = K_ENU + 1/2,
        s = sqrt(2nK'/b), P = pi/2 - atan(s), Q = s/(1+s^2):

            I12 = 2P/sqrt(2nb)                     [k^{-1/2}/(2nk+b)]
            I32 = 2/(b sqrt(K')) - 2 sqrt(2n) P/b^1.5
            G2  = (P-Q)/(sqrt(2n) b^1.5)           [k^{-1/2}/(2nk+b)^2]
            G3  = [3P/2 - 2Q - s(1-s^2)/(2(1+s^2)^2)] / (2 sqrt(2n) b^2.5)
            tail = e^{-z}/sqrt(pi) (I12 + G2 - z G3 + a1 I32)

        (I12: leading E ~ e^{-z}/(z+nu); G2 - z G3: DLMF 8.20(ii) A_1 term
        nu/(z+nu)^3; a1: next order of the Catalan weights, -1/8 for Sigma,
        -9/8 for DeltaSigma.)
        """
        from .einasto import _catalan_over_4k, expint_asymptotic, expn_fast

        R = np.atleast_1d(np.asarray(R, float))
        n = self.n
        z = (R / self.h) ** (1.0 / n)
        K_e = self.K_ENU
        if which == "sigma":
            k = np.arange(0, K_e + 1, dtype=float)
            w = (k + 1) * _catalan_over_4k(k)
            a1 = -1.0 / 8.0
        else:
            k = np.arange(1, K_e + 1, dtype=float)
            w = k * _catalan_over_4k(k)
            a1 = -9.0 / 8.0
        nu = 2 * k * n - n + 1
        E = np.empty((z.size, k.size))
        lo = nu < 1.0
        mid = (~lo) & (nu < NU_ASYMP)
        hi = nu >= NU_ASYMP
        if lo.any():
            E[:, lo] = expn_fast(nu[None, lo], z[:, None])
        if mid.any():
            E[:, mid] = expn_cf(nu[None, mid], z[:, None])
        if hi.any():
            E[:, hi] = expint_asymptotic(nu[None, hi], z[:, None])
        ssum = (w[None, :] * E).sum(axis=1)

        b = z + 1.0 - n
        Kp = K_e + 0.5
        s = np.sqrt(2 * n * Kp / b)
        P = np.pi / 2 - np.arctan(s)
        Q = s / (1 + s * s)
        r2n = np.sqrt(2 * n)
        I12 = 2.0 * P / (r2n * np.sqrt(b))
        I32 = 2.0 / (b * np.sqrt(Kp)) - 2.0 * r2n * P / b ** 1.5
        G2 = (P - Q) / (r2n * b ** 1.5)
        G3 = (1.5 * P - 2 * Q - s * (1 - s * s) / (2 * (1 + s * s) ** 2)) \
            / (2 * r2n * b ** 2.5)
        tail = np.exp(-z) / SQPI * (I12 + G2 - z * G3 + a1 * I32)

        core = 2 * self.rho_0 * n * R * (ssum + tail)
        if which == "sigma":
            return core
        m3d_over_area = 4 * self.rho_0 * n * self.h \
            * gammainc(3 * n, z) * _gamma(3 * n) / (R / self.h) ** 2
        return m3d_over_area - core

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def sigma(self, R):
        R = np.atleast_1d(np.asarray(R, float))
        z = (R / self.h) ** (1.0 / self.n)
        out = np.empty_like(R)
        m = z <= self.zsw_sig
        if m.any():
            out[m] = self._series(R[m], "sigma")
        if (~m).any():
            out[~m] = self._enu(R[~m], "sigma")
        # Sigma ~ e^{-z}: below the fp64 normal range the value is
        # physically zero; flush to avoid subnormal noise.
        out[z > 690.0] = 0.0
        return out

    def deltasigma(self, R):
        R = np.atleast_1d(np.asarray(R, float))
        z = (R / self.h) ** (1.0 / self.n)
        out = np.empty_like(R)
        m = z <= self.zsw_ds
        if m.any():
            out[m] = self._series(R[m], "ds")
        if (~m).any():
            out[~m] = self._enu(R[~m], "ds")
        return out

    def mean_sigma(self, R):
        return self.sigma(R) + self.deltasigma(R)

    def enclosed_mass_2D(self, R):
        R = np.atleast_1d(np.asarray(R, float))
        return np.pi * R ** 2 * self.mean_sigma(R)
