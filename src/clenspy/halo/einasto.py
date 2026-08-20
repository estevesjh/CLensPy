"""
Einasto dark-matter halo profile.

Notation follows docs/einasto_proj_density.tex and
EinastoPertubationTheory/einasto_power_spectrum.tex:

    rho(r) = rho_0 exp[-(r/h)^(1/n)],   n > 0, h > 0,

with the (alpha, b, r_s) translation alpha = 1/n, b = 2n,
h = r_s / (2n)^n.

Projected quantities use the Catalan series (Theorem 1 of the projected
density note):

    c_k   = Cat_k / 4^k = C(2k,k) / [(k+1) 4^k],   c_0 = 1,
    nu_k  = 2 k n - n + 1,
    x     = (R/h)^(1/n),

    Sigma(R)      = 2 rho_0 n R    sum_{k>=0} (k+1) c_k E_{nu_k}(x),
    M_2D(R)       = M_3D(R) + 2 pi rho_0 n R^3 sum_{k>=0} c_k E_{nu_k}(x),
    DeltaSigma(R) = M_3D(R)/(pi R^2) - 2 rho_0 n R sum_{k>=1} k c_k E_{nu_k}(x).

The generalized exponential integral E_nu(x) is evaluated by dispatch:
integer nu>=1 via scipy.special.expn; large nu via the DLMF 8.20 uniform
asymptotic expansion; otherwise via mpmath.expint.
"""

import mcfit
import numpy as np
from scipy.special import (expn, gamma, gammainc, gammaincc, gammaln, kv,
                           loggamma, roots_genlaguerre)

from ..utils.decorators import scalar_array_output
from ..utils.integrate import compute_sigma_quadvec, sigma_to_deltasigma_cumtrapz
from ..utils.interpolate import make_log_interpolation

SQPI_ = np.sqrt(np.pi)

# Cache for generalised Gauss-Laguerre nodes/weights, keyed by (alpha_w, N).
_GL_CACHE: dict = {}


def _gauss_laguerre_nodes(alpha_w, N):
    """Nodes/weights for weight u^{alpha_w} e^{-u} on (0, inf), cached."""
    key = (float(alpha_w), int(N))
    if key not in _GL_CACHE:
        u, w = roots_genlaguerre(N, alpha_w)
        _GL_CACHE[key] = (u.astype(float), w.astype(float))
    return _GL_CACHE[key]


def _einasto_pk_GL(kt, n, h, rho_0, N=96):
    """Master-integral GL evaluator: P(k) for kt small (deep plateau).

    P(k) = (rho_0 n h^3 / 4 pi) int_0^inf u^{3n-1} e^{-u} sinc(kt u^n) du,
    sinc(y) = sin y / y, sinc(0) = 1.  Exact at kt=0 (sum -> Gamma(3n)).
    Valid only when kt u_N^n stays small enough that the phase varies
    slowly across the weight; for n=4.3, N=96 covers kt <~ 1e-5.
    """
    u, w = _gauss_laguerre_nodes(3.0 * n - 1.0, N)
    kt = np.atleast_1d(np.asarray(kt, float))
    phase = kt[:, None] * (u[None, :] ** n)
    sinc = np.sinc(phase / np.pi)
    I = (w[None, :] * sinc).sum(axis=1)
    return rho_0 * n * h ** 3 / (4.0 * np.pi) * I


def _einasto_pk_wright_real(kt, n, h, rho_0, M=80):
    """Real Wright series Eq.(largek), simple log-space exp() with sign.

        P(k) = (rho_0 h^3 / 4 pi kt^3) sum_{m=1}^M A_m^- kt^{-m/n},
        A_m^- = (-1)^{m+1} Gamma(2 + m/n) sin(pi m / 2n) / m!.

    Stable in fp64 for n>1 and all kt > ~1e-4 with M=80: m! beats
    Gamma(2+m/n) by m~30 even at xi=10, so terms decay before reaching
    fp64 dynamic range.  No log-rescale, no theta_pm split, no PCHIP.
    """
    kt = np.atleast_1d(np.asarray(kt, float))
    m = np.arange(1, M + 1, dtype=float)
    log_coef = gammaln(2.0 + m / n) - gammaln(m + 1.0)
    sign = (-1.0) ** (m + 1) * np.sin(np.pi * m / (2.0 * n))
    log_xi = -np.log(kt)[:, None] / n                # (nk, 1)
    log_terms = log_coef[None, :] + m[None, :] * log_xi
    terms = sign[None, :] * np.exp(log_terms)
    series = terms.sum(axis=1)
    return rho_0 * h ** 3 / (4.0 * np.pi * kt ** 3) * series

try:
    import mpmath as _mp
except ImportError:  # pragma: no cover
    _mp = None

# Number of terms retained in the DLMF 8.20 asymptotic expansion.
NTERMS_ASYMP = 5


def _nu_asymp_threshold(rtol, nterms=NTERMS_ASYMP):
    """Minimum nu for asymptotic: error ~ nu^{-nterms} < rtol."""
    return rtol ** (-1.0 / nterms)


_EULER_GAMMA = 0.5772156649015329

# --------------------------------------------------------------------------
# P(k) for 0 < n < 1: analytic evaluators with computable error estimates
# (docs/einasto_proj_density_v4.tex, "The power spectrum"). All in the
# P = rho_tilde/(4 pi)^2 convention with rho_0 = h = 1; caller rescales.
# --------------------------------------------------------------------------
_PK_TOL = 1e-9


def _pk_build_kummer(n, M=320, dps_cap=300):
    """Build the Kummer coefficients b_m defined by
    e^z sum_m A_m^+ z^m = sum_m b_m z^m  (z = (kt/2)^2), i.e.

        b_m = sum_{i<=m} A_i^+ / (m-i)!,
        A_i^+ = (-1)^i Gamma(3n+2ni) / (i! (3/2)_i).

    The alternating cancellation is absorbed HERE, once, in mpmath with
    self-consistently chosen precision; at runtime the series in b_m is
    (near-)cancellation-free for n <~ 0.93. Returns (sign, log|b_m|,
    usable): as n -> 1 the entire order 1/(2-2n) of f diverges and the
    build precision explodes -- usable=False then, and the dispatch falls
    back to the plain series + asymptotics (+ GL in a small window).
    """
    import mpmath as mp

    def _pass(dps_):
        with mp.workdps(dps_):
            nn = mp.mpf(n)
            a = [mp.gamma(3 * nn + 2 * nn * i)
                 / (mp.factorial(i) * mp.rf(mp.mpf(1.5), i))
                 for i in range(M + 1)]
            b, lost = [], 0.0
            for m_ in range(M + 1):
                s = mp.mpf(0)
                big = mp.mpf(0)
                for i in range(m_ + 1):
                    t = (-1) ** i * a[i] / mp.factorial(m_ - i)
                    s += t
                    big = max(big, abs(t))
                if s != 0 and big > 0:
                    lost = max(lost, float(mp.log10(big / abs(s))))
                b.append(s)
            return b, lost

    dps = 40
    for _ in range(3):
        b, lost = _pass(dps)
        if lost + 25 <= dps:
            sign = np.array([float(mp.sign(x)) for x in b])
            with np.errstate(divide="ignore"):
                logb = np.array([float(mp.log(abs(x))) if x != 0 else -np.inf
                                 for x in b])
            return sign, logb, True
        dps = int(lost) + 30
        if dps > dps_cap:
            break
    return None, None, False


def _pk_kummer_eval(n, kt, sign, logb):
    """P and error estimate from the Kummer form,
    P = (n/4pi) sum_m sgn(b_m) exp(log|b_m| + m ln(zeta) - zeta),
    zeta = kt^2/4. The e^{-zeta} factor is folded into each term
    (overflow-free); estimate = 10^{lost - 15.6}."""
    kt = np.atleast_1d(np.asarray(kt, float))
    m = np.arange(logb.size, dtype=float)
    z = kt * kt / 4.0
    with np.errstate(divide="ignore", invalid="ignore"):
        lt = logb[None, :] + m[None, :] * np.log(z)[:, None] - z[:, None]
    if (z == 0).any():
        lt[z == 0] = np.where(m == 0, logb[0], -np.inf)
    ltmax = lt.max(axis=1, keepdims=True)
    terms = sign[None, :] * np.exp(lt - ltmax)
    s_scaled = terms.sum(axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        val = n / (4 * np.pi) * s_scaled * np.exp(ltmax[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        est = 10.0 ** (-np.log10(np.abs(s_scaled)) - 15.6)
    # under-truncation is invisible to the cancellation metric (terms still
    # growing at the cutoff look like a clean single-signed sum): if the
    # term peak sits at the end of the table, the value is unusable.
    est[lt.argmax(axis=1) >= logb.size - 2] = np.inf
    est[~np.isfinite(val)] = np.inf
    return val, est


def _pk_conv_eval(n, kt, M=400):
    """Plain convergent small-kt series (log-space terms) + estimate."""
    kt = np.atleast_1d(np.asarray(kt, float))
    m = np.arange(0, M + 1, dtype=float)
    logc = gammaln(3 * n + 2 * n * m) - gammaln(m + 1) \
        - (gammaln(1.5 + m) - gammaln(1.5))
    z = kt * kt / 4.0
    with np.errstate(divide="ignore", invalid="ignore"):
        lt = logc[None, :] + m[None, :] * np.log(z)[:, None]
    if (z == 0).any():
        lt[z == 0] = np.where(m == 0, logc[0], -np.inf)
    ltmax = lt.max(axis=1, keepdims=True)
    terms = ((-1.0) ** m)[None, :] * np.exp(lt - ltmax)
    s_scaled = terms.sum(axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        val = n / (4 * np.pi) * s_scaled * np.exp(ltmax[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        est = 10.0 ** (-np.log10(np.abs(s_scaled)) - 15.6)
    est[lt.argmax(axis=1) >= M - 2] = np.inf   # under-truncated: unusable
    est[~np.isfinite(val)] = np.inf
    return val, est


def _pk_mb_contour(n, kt, c=-0.5, h=0.08):
    """P(kt) by trapezoidal quadrature of the Mellin-Barnes kernel along
    w = c + i tau (rho_0 = h = 1 units):

        P = (n/(4 pi kt)) (1/2 pi i) int Gamma(w) sin(pi w/2)
            Gamma(2n - nw) kt^{-w} dw .

    The integrand decays like e^{-(pi/2) n |tau|} (the sin GROWS like
    e^{+(pi/2)|tau|}) and is analytic in -1 < Re w < 2 (the w = 0 pole is
    killed by the sin zero), so the trapezoidal rule converges
    geometrically (Aceto & Durastante 2022 setting). c = -1/2 keeps the
    small-kt amplitude cancellation at ~kt^{-1/2}. Validated to <= 8e-12
    for n in [0.45, 2.5] over kt in [1e-8, 12]; the phase gradient
    ~ n ln n of Gamma(2n-nw) undersamples for n >~ 3 -- callers must not
    use it there.
    """
    kt = np.atleast_1d(np.asarray(kt, float))
    tau_max = max((40.0 + 2 * np.abs(np.log(kt)).max())
                  / ((np.pi / 2) * n), 8.0)
    tau = np.arange(h / 2, tau_max, h)
    w = c + 1j * tau
    logG = loggamma(w) + loggamma(2 * n - n * w) \
        + np.log(np.sin(np.pi * w / 2))
    vals = np.exp(logG[None, :] - w[None, :] * np.log(kt)[:, None])
    integral = 2.0 * (h / (2 * np.pi)) * vals.real.sum(axis=1)
    return n / (4 * np.pi * kt) * integral


def _pk_plateau_eval(n, kt, Mmax=400):
    """P0 + small-kt series with optimal truncation (asymptotic for n>1).
    Returns (val, est); est = 10 x smallest retained term / |sum|."""
    kt = np.atleast_1d(np.asarray(kt, float))
    m = np.arange(1, Mmax + 1, dtype=float)
    logc = gammaln(3 * n + 2 * n * m) - gammaln(m + 1) \
        - (gammaln(1.5 + m) - gammaln(1.5))
    logP0 = gammaln(3 * n)
    out = np.empty(kt.size)
    est = np.empty(kt.size)
    for i, k_ in enumerate(kt):
        if k_ <= 0:
            out[i], est[i] = n / (4 * np.pi) * np.exp(logP0), 0.0
            continue
        lt = logc + m * np.log(k_ * k_ / 4.0) - logP0     # terms / P0
        grow = np.diff(lt) > 0
        mo = int(np.argmax(grow)) + 1 if grow.any() else Mmax
        s_rel = 1.0 + (((-1.0) ** m[:mo]) * np.exp(lt[:mo])).sum()
        out[i] = n / (4 * np.pi) * np.exp(logP0) * s_rel
        est[i] = 10.0 * np.exp(lt[min(mo, Mmax - 1)]) \
            / max(abs(s_rel), 1e-300)
    return out, est


def _pk_direct_eval(n, kt, M=600):
    """Direct large-kt series (convergent for n>1), log-space terms.
    est covers BOTH fp64 cancellation and the unsummed tail (the
    cancellation metric alone cannot see under-truncation)."""
    kt = np.atleast_1d(np.asarray(kt, float))
    m = np.arange(1, M + 1, dtype=float)
    logc = gammaln(2 + m / n) - gammaln(m + 1)
    sgn = (-1.0) ** (m + 1) * np.sin(np.pi * m / (2 * n))
    out = np.empty(kt.size)
    est = np.empty(kt.size)
    for i, k_ in enumerate(kt):
        if k_ <= 0:
            out[i], est[i] = 0.0, np.inf
            continue
        lt = logc - (m / n) * np.log(k_)
        ltmax = lt.max()
        s_scaled = (sgn * np.exp(lt - ltmax)).sum()
        out[i] = s_scaled * np.exp(ltmax) / (4 * np.pi * k_ ** 3)
        s_abs = max(abs(s_scaled), 1e-300)
        cancel = 10.0 ** (-np.log10(s_abs) - 15.6)
        tail = 10.0 * np.exp(lt[-1] - ltmax) / s_abs
        est[i] = max(cancel, tail)
        if lt.argmax() >= M - 2:
            est[i] = np.inf
    return out, est


def _pk_filon(n, kt, npts=50000):
    """P(kt) by Filon quadrature of the t-space master integral
    (rho_0 = h = 1 units):

        P = (1/(4 pi kt)) int_0^{t_hi} g(t) sin(kt t) dt,
        g(t) = t e^{-t^{1/n}},  t_hi = (2n + 45)^n
        [u = t^{1/n} substitution: u^{3n-1} du = (1/n) t^2 dt cancels the
        master integral's overall factor n].

    Piecewise-LINEAR interpolation of the smooth envelope g with the
    oscillatory factor integrated EXACTLY per interval, so the node count
    follows the envelope (not the oscillation count) -- the standard cure
    for large-n turnover kt, where the integrand oscillates ~kt t_hi >> 1
    times and Gauss-Laguerre undersamples. Cost ~ npts flops per kt.
    """
    kt = np.atleast_1d(np.asarray(kt, float))
    # envelope grid: uniform in z = t^{1/n} resolves g everywhere
    z = np.linspace(0.0, 2 * n + 45.0, npts)
    t = z ** n
    g = t * np.exp(-z)
    out = np.empty(kt.size)
    for i, k_ in enumerate(kt):
        a = k_ * t[:-1]
        b = k_ * t[1:]
        dt_ = t[1:] - t[:-1]
        good = dt_ > 0
        # exact int_{ta}^{tb} (g_a + s (g_b - g_a)) sin(k t) dt with
        # s = (t - ta)/dt:  use antiderivatives of sin, t sin
        ca, cb = np.cos(a), np.cos(b)
        sa, sb = np.sin(a), np.sin(b)
        with np.errstate(divide="ignore", invalid="ignore"):
            I0 = (ca - cb) / k_                       # int sin
            I1 = (sb - sa) / k_ ** 2 - (t[1:] * cb - t[:-1] * ca) / k_
            g0, g1 = g[:-1], g[1:]
            slope = np.where(good, (g1 - g0) / np.where(good, dt_, 1), 0.0)
            seg = (g0 - slope * t[:-1]) * I0 + slope * I1
        out[i] = seg[good].sum() / (4 * np.pi * k_)
    return out


def _pk_asym_eval(n, kt, Mmax=2000):
    """Large-kt series with optimal truncation. For n < 1 it is a valid
    asymptotic expansion (Watson/Erdelyi); smallest-term truncation gives
    error ~ exp(-c kt^{1/(1-n)}). Estimate = 10 x smallest term / |sum|."""
    kt = np.atleast_1d(np.asarray(kt, float))
    m = np.arange(1, Mmax + 1, dtype=float)
    logc = gammaln(2 + m / n) - gammaln(m + 1)
    sgn = (-1.0) ** (m + 1) * np.sin(np.pi * m / (2 * n))
    out = np.empty(kt.size)
    est = np.empty(kt.size)
    for i, k_ in enumerate(kt):
        if k_ <= 0:
            out[i], est[i] = 0.0, np.inf
            continue
        lt = logc - (m / n) * np.log(k_)
        grow = np.diff(lt) > 0
        mo = int(np.argmax(grow)) + 1 if grow.any() else Mmax
        s = (sgn[:mo] * np.exp(lt[:mo])).sum()
        out[i] = s / (4 * np.pi * k_ ** 3)
        est[i] = 10.0 * np.exp(lt[min(mo, Mmax - 1)]) / abs(s) \
            if s != 0 else np.inf
    return out, est


def _expdisk_deltasigma_factor(x):
    """DeltaSigma/(rho_0 h) for the n = 1 (exponential) profile.

    Closed form 8/x^2 - 4 K_2(x) - 2 x K_1(x) self-cancels as x -> 0
    (both 8/x^2 and 4K_2 ~ 8/x^2 while the result is O(x^2 ln x)); below
    x = 0.1 use its verified small-x expansion (error <= 1e-10 there):

        -(x^2/2)(Lt - 1/4) - (x^4/12)(Lt - 7/6) - (x^6/256)(Lt - 13/8),
        Lt = ln(x/2) + euler_gamma.
    """
    x = np.asarray(x, float)
    out = np.empty_like(x)
    small = x < 0.1
    if small.any():
        xs = x[small]
        with np.errstate(divide="ignore"):
            Lt = np.log(xs / 2.0) + _EULER_GAMMA
        t = -(xs ** 2 / 2) * (Lt - 0.25) - (xs ** 4 / 12) * (Lt - 7.0 / 6.0) \
            - (xs ** 6 / 256) * (Lt - 13.0 / 8.0)
        out[small] = np.where(xs > 0, t, 0.0)
    if (~small).any():
        xl = x[~small]
        out[~small] = 8.0 / xl ** 2 - 4.0 * kv(2, xl) - 2.0 * xl * kv(1, xl)
    return out


def _expdisk_m2d_factor(x):
    """M_2D/(4 pi rho_0 h^3) for the n = 1 profile: 2 - x^2 K_2(x), with
    the small-x expansion below x = 0.1 (2 and x^2 K_2 ~ 2 cancel):

        x^2/2 + (x^4/8) Lb + (x^6/96)(Lb - 2/3) + (x^8/3072)(Lb - 25/24),
        Lb = ln(x/2) + euler_gamma - 3/4.
    """
    x = np.asarray(x, float)
    out = np.empty_like(x)
    small = x < 0.1
    if small.any():
        xs = x[small]
        with np.errstate(divide="ignore"):
            Lb = np.log(xs / 2.0) + _EULER_GAMMA - 0.75
        t = xs ** 2 / 2 + (xs ** 4 / 8) * Lb + (xs ** 6 / 96) * (Lb - 2.0 / 3.0) \
            + (xs ** 8 / 3072) * (Lb - 25.0 / 24.0)
        out[small] = np.where(xs > 0, t, 0.0)
    if (~small).any():
        xl = x[~small]
        out[~small] = 2.0 - xl ** 2 * kv(2, xl)
    return out


def _catalan_over_4k(k):
    """
    c_k = Cat_k / 4^k = C(2k,k) / [(k+1) 4^k], computed in log space to stay
    finite for large k (both C(2k,k) and 4^k overflow individually near k~500).
    """
    k = np.asarray(k, dtype=float)
    log_ck = gammaln(2 * k + 1) - 2 * gammaln(k + 1) - np.log(k + 1) - k * np.log(4.0)
    return np.exp(log_ck)


def _asymptotic_polys(nterms):
    """
    DLMF 8.20.4 recurrence for the polynomials A_k(lambda):

        A_0 = 1,
        A_{k+1}(l) = (1 - 2 k l) A_k(l) + l (l+1) A_k'(l).

    Returns a list of numpy.polynomial.Polynomial of length ``nterms``.
    """
    from numpy.polynomial import Polynomial as P

    polys = [P([1.0])]
    lam = P([0.0, 1.0])           # lambda
    lam1 = P([0.0, 1.0, 1.0])     # lambda (lambda + 1)
    for k in range(nterms - 1):
        Ak = polys[-1]
        Akp1 = (P([1.0]) - 2 * k * lam) * Ak + lam1 * Ak.deriv()
        polys.append(Akp1)
    return polys


_ASYMP_POLYS = _asymptotic_polys(NTERMS_ASYMP)


def expint_asymptotic(nu, x, nterms=NTERMS_ASYMP):
    """
    Uniform large-index asymptotic of E_nu(x) (DLMF 8.20.6) with x = lambda p,
    p = nu:

        E_p(lambda p) ~ e^{-x} / [(lambda+1) p]
                        sum_{k=0}^{nterms-1} A_k(lambda) / [(lambda+1)^{2k} p^k].

    Parameters
    ----------
    nu, x : array_like
        Broadcastable; nu is the index p, x the argument.

    Returns
    -------
    ndarray
        E_nu(x) to the retained order.
    """
    nu = np.asarray(nu, dtype=float)
    x = np.asarray(x, dtype=float)
    lam = x / nu
    lam1 = lam + 1.0
    series = np.zeros(np.broadcast(nu, x).shape)
    for k in range(nterms):
        Ak = _ASYMP_POLYS[k](lam)
        series += Ak / (lam1 ** (2 * k) * nu ** k)
    return np.exp(-x) / (lam1 * nu) * series


def _expint_mpmath(nu, x):
    """Elementwise E_nu(x) via mpmath.expint (handles any real nu)."""
    if _mp is None:  # pragma: no cover
        raise ImportError("mpmath required for non-integer/small-nu E_nu(x)")

    def _one(s, z):
        return float(_mp.re(_mp.expint(float(s), float(z))))

    f = np.frompyfunc(_one, 2, 1)
    return f(nu, x).astype(float)


def _expint_gamma(nu, x):
    """
    E_p(z) = z^{p-1} Gamma(1-p, z) via scipy upper incomplete gamma.

    Valid when a = 1 - p > 0, i.e. p < 1. Uses the regularized form:
    Gamma(a, z) = gammaincc(a, z) * Gamma(a).
    """
    a = 1.0 - nu
    return x ** (nu - 1.0) * gammaincc(a, x) * gamma(a)


def _expint_recurrence(nu, x):
    """
    E_nu(x) for nu > 1 by upward recurrence, no mpmath (pitfalls.md S2).

    Start from the base index s0 = nu - floor(nu) < 1 (the scipy gamma branch,
    :func:`_expint_gamma`) and apply DLMF 8.19.12,

        s E_{s+1}(x) = e^{-x} - x E_s(x)   =>   E_{s+1} = (e^{-x} - x E_s)/s,

    stepping by integers up to nu. Stable to ~1e-13 for the nu >~ x regime
    relevant here (nu_k grows with k); degrades gracefully (~1e-10) only when
    x >> nu. nu and x are broadcast together.
    """
    nu, x = np.broadcast_arrays(np.asarray(nu, float), np.asarray(x, float))
    s = nu - np.floor(nu)                 # base in [0,1)
    E = _expint_gamma(s, x)
    ex = np.exp(-x)
    for _ in range(int(np.floor(nu).max()) if nu.size else 0):
        active = s + 1.0 <= nu + 1e-9
        if not active.any():
            break
        s_safe = np.where(s == 0.0, 1.0, s)
        E = np.where(active, (ex - x * E) / s_safe, E)
        s = np.where(active, s + 1.0, s)
    return E


def expn_fast(nu, x, rtol=1e-9, nterms=NTERMS_ASYMP):
    """
    Generalized exponential integral E_nu(x) by branch dispatch.

    - integer nu >= 1            : scipy.special.expn (exact, vectorized)
    - nu >= threshold(rtol)      : DLMF 8.20 uniform asymptotic
    - nu < 1 (1-nu > 0)         : E_p(z) = z^{p-1} Gamma(1-p, z) via scipy
    - otherwise                  : mpmath.expint (fallback)

    The asymptotic threshold is set adaptively so that the expansion error
    ~nu^{-nterms} < rtol.  E.g. rtol=1e-6 → nu_min ≈ 10; rtol=1e-10 → ~100.

    Parameters
    ----------
    nu, x : array_like
        Broadcastable arrays.
    rtol : float, optional
        Target relative accuracy for the asymptotic branch (default 1e-6).
    nterms : int, optional
        Number of asymptotic terms (default 5).

    Returns
    -------
    ndarray
        E_nu(x), shape = broadcast(nu, x).
    """
    nu_asymp = _nu_asymp_threshold(rtol, nterms)

    nu_b, x_b = np.broadcast_arrays(np.asarray(nu, float), np.asarray(x, float))
    out = np.empty(nu_b.shape, dtype=float)

    nu_round = np.rint(nu_b)
    is_int_pos = np.isclose(nu_b, nu_round) & (nu_round >= 1)
    is_asymp = (~is_int_pos) & (nu_b >= nu_asymp)
    is_gamma = (~is_int_pos) & (~is_asymp) & (nu_b < 1.0)
    is_rest = ~(is_int_pos | is_asymp | is_gamma)

    if is_int_pos.any():
        out[is_int_pos] = expn(nu_round[is_int_pos].astype(int), x_b[is_int_pos])
    if is_asymp.any():
        out[is_asymp] = expint_asymptotic(nu_b[is_asymp], x_b[is_asymp], nterms)
    if is_gamma.any():
        out[is_gamma] = _expint_gamma(nu_b[is_gamma], x_b[is_gamma])
    if is_rest.any():
        # Non-integer 1 < nu < threshold: upward recurrence (no mpmath).
        out[is_rest] = _expint_recurrence(nu_b[is_rest], x_b[is_rest])

    return out


class EinastoProfile:
    """
    Einasto profile rho(r) = rho_0 exp[-(r/h)^(1/n)].

    Parameters
    ----------
    alpha : float
        Shape parameter; n = 1/alpha.
    rho_0 : float
        Central amplitude rho_0 of the profile (the prefactor in rho(r)).
    r_s : float
        Scale radius; h = r_s / (2n)^n.
    order : int, optional
        Number of terms (k = 0..order) in the projected series. Only used
        when n > 3/2 (see Notes).
    tol : float, optional
        If given, the series order is chosen automatically at construction
        via :meth:`order_for_tol` (``order`` is then used only as the search
        ceiling). The Catalan series converges algebraically (~K^{-1/2}), so
        the required order grows steeply as the shape index n falls below ~2.
        Only used when n > 3/2 (see Notes).

    Notes
    -----
    For n = 1/alpha > 3/2, :meth:`sigma`, :meth:`deltasigma`, and
    :meth:`enclosed_mass_2D` use the closed-form Catalan series
    (docs/einasto_proj_density.tex). For n <= 3/2 (alpha >= 2/3) they use
    the stable low-n backend (:class:`~clenspy.halo.einasto_lown.EinastoLowN`):
    the Retana-Montenegro et al. (2012) case-1 residue series with resonance
    pairing at small/moderate z = (R/h)^(1/n), switching to the all-positive
    Catalan E_nu representation beyond a per-n calibrated z. Validated to
    ~4e-9 relative accuracy against mpmath quadrature for n in [0.35, 1.5]
    and R/h in [0.01, 40]. The purely numerical Abel/cumtrapz fallbacks
    (:meth:`_sigma_numerical`, :meth:`_deltasigma_numerical`) are retained
    for cross-checks only.

    :meth:`power_spectrum`/:meth:`fourier` use their own, independent split
    (docs/einasto_power_spectrum.tex): an analytic series for n > 1
    (converges for all k), and a FFTLog transform (`mcfit.xi2P`) of
    :meth:`density` for 0 < n < 1 away from n = 1/2 (the small-k series is
    convergent there too, but its finite-precision partial sums are not
    usable - see :meth:`power_spectrum`).

    Both n = 1 (exponential, rho = rho_0 exp(-r/h)) and n = 1/2 (Gaussian,
    rho = rho_0 exp(-(r/h)^2)) have exact closed forms and bypass both the
    series and the numerical fallbacks: `sigma`/`deltasigma` at n = 1 use the
    modified Bessel functions K_1/K_2, and `power_spectrum` at n = 1 and
    n = 1/2 uses the closed forms from docs/einasto_power_spectrum.tex.

    :meth:`density`, :meth:`enclosed_mass`, and :attr:`total_mass` use the
    incomplete-gamma closed form for any n and are unaffected by any of this.
    """

    def __init__(self, alpha, rho_0, r_s, order=100, tol=None):
        self.alpha = alpha
        self.rho_0 = rho_0
        self.r_s = r_s

        self.n_index = 1.0 / alpha
        if self.n_index <= 0:
            raise ValueError(f"n = 1/alpha = {self.n_index:.3f} must be positive.")
        self.h = self.r_s / (2 * self.n_index) ** self.n_index

        # sigma/deltasigma/enclosed_mass_2D: exact closed forms at the
        # anchors n = 1/2 (Gaussian) and n = 1 (exponential); the stable
        # residue-series + E_nu hybrid (einasto_lown) for every other n.
        # The legacy Catalan machinery is still built for n > 3/2 because
        # power_spectrum and order_for_tol use it (self.order / _ck / _nu_k),
        # but sigma/deltasigma/enclosed_mass_2D no longer evaluate through
        # it (its DeltaSigma truncation error is O(K^{-1/2}) *absolute*,
        # i.e. 30-200% relative -- see docs/einasto_proj_density_v4.tex).
        self._series = self.n_index > 1.5
        self._lown = None
        self._pk_bm = None          # lazy Kummer P(k) build, n < 1 only
        if self._series:
            if tol is not None:
                self._build(order)                       # ceiling for the search
                order = self.order_for_tol(tol, max_order=order)
            self._build(order)
        else:
            self.order = None
        if not self._is_anchor():
            from .einasto_lown import EinastoLowN
            self._lown = EinastoLowN(
                self.n_index, self.rho_0, self.h,
                tol=tol if tol is not None else 1e-9)

    def _is_anchor(self):
        # tight tolerance on purpose: np.isclose's default rtol=1e-5 would
        # silently evaluate e.g. n = 1 + 1e-7 with the n = 1 closed form
        # (an O(1e-7) profile error); the backend handles near-integer n
        # exactly via resonance pairing, so only true anchors bypass it.
        return abs(self.n_index - 0.5) < 1e-12 or abs(self.n_index - 1.0) < 1e-12

    def _build(self, order):
        """Precompute the index-dependent series arrays for k = 0..order."""
        self.order = order
        n = self.n_index
        k = np.arange(0, order + 1)
        self._k = k
        self._ck = _catalan_over_4k(k)                       # Cat_k / 4^k
        self._nu_k = 2 * k * n - n + 1                        # nu_k

    # ------------------------------------------------------------------
    # Numerical fallback (n <= 3/2): no closed-form Catalan series exists,
    # so Sigma, DeltaSigma, and P(k) are computed directly from `density`
    # by Abel projection / FFTLog instead. See the class Notes.
    # ------------------------------------------------------------------
    def _numerical_r_grid(self, n_grid=400):
        """Log-spaced r grid spanning density() from ~1e-4 h out to where
        it has decayed to ~exp(-40) of rho_0 (double-precision noise floor)."""
        r_min = self.h * 1e-4
        r_max = self.h * 40.0 ** self.n_index
        return np.logspace(np.log10(r_min), np.log10(r_max), n_grid)

    def _sigma_numerical(self, R):
        """Sigma(R) via the Abel (line-of-sight) projection of density(r)."""
        R = np.atleast_1d(np.asarray(R, float))
        r_max = R.max() + self.h * 40.0 ** self.n_index

        def xi_func(r, z):
            return self.density(r)

        return compute_sigma_quadvec(xi_func, R, np.array([0.0]), r_max=r_max).ravel()

    def _deltasigma_numerical(self, R):
        """DeltaSigma(R) from a dense numerical Sigma(R) grid (cumtrapz).

        The cumulative-trapezoid enclosed mass needs a well-resolved grid
        near R=0 (see `sigma_to_deltasigma_cumtrapz`'s caveat); 1600 points
        keeps the innermost decade accurate to ~0.3% at negligible extra
        cost (a few ms; `density`/Abel evaluations are cheap closed forms).
        """
        R = np.atleast_1d(np.asarray(R, float))
        r_max = R.max() + self.h * 40.0 ** self.n_index
        Rgrid = np.logspace(np.log10(self.h * 1e-4), np.log10(r_max), 1600)
        sigma_grid = self._sigma_numerical(Rgrid)
        ds_grid = sigma_to_deltasigma_cumtrapz(Rgrid, sigma_grid)
        return make_log_interpolation(Rgrid, ds_grid)(R)

    def _power_spectrum_numerical(self, k):
        """P(k) = rho_tilde(k)/(4 pi)^2 via FFTLog (mcfit.xi2P) of density(r)."""
        k = np.atleast_1d(np.asarray(k, float))
        rgrid = self._numerical_r_grid(2048)
        kgrid, Fk = mcfit.xi2P(rgrid, lowring=True)(self.density(rgrid))
        return make_log_interpolation(kgrid, Fk / (4 * np.pi) ** 2)(k)

    # ------------------------------------------------------------------
    # 3D quantities
    # ------------------------------------------------------------------
    def density(self, r):
        r"""
        Density :math:`\rho(r)`.

        .. math::
            \rho(r) = \rho_0\, \exp\!\left[-(r/h)^{1/n}\right]
        """
        x = np.asarray(r) / self.h
        return self.rho_0 * np.exp(-x ** (1.0 / self.n_index))

    def enclosed_mass(self, r):
        r"""
        Spherical enclosed mass.

        .. math::
            M_{\rm 3D}(r) = 4\pi \rho_0\, n\, h^3\,
            \gamma\!\left(3n,\, (r/h)^{1/n}\right)

        where :math:`\gamma` is the lower incomplete gamma function.
        """
        n, h = self.n_index, self.h
        x = (np.asarray(r) / h) ** (1.0 / n)
        gamma_lower = gammainc(3 * n, x) * gamma(3 * n)   # unnormalized
        return 4 * np.pi * self.rho_0 * n * h ** 3 * gamma_lower

    @property
    def total_mass(self):
        r"""
        Total mass.

        .. math::
            M_{\rm tot} = 4\pi \rho_0\, n\, h^3\, \Gamma(3n)
        """
        n, h = self.n_index, self.h
        return 4 * np.pi * self.rho_0 * n * h ** 3 * gamma(3 * n)

    # ------------------------------------------------------------------
    # Projected series
    # ------------------------------------------------------------------
    def _E_nu(self, R):
        """
        Evaluate E_{nu_k}((R/h)^(1/n)) for all k = 0..order.

        Returns
        -------
        ndarray
            Shape (R.size, order+1).
        """
        x = (np.atleast_1d(np.asarray(R, float)) / self.h) ** (1.0 / self.n_index)
        x_col = x[:, None]                # (nR, 1)
        nu_row = self._nu_k[None, :]      # (1, order+1)
        return expn_fast(nu_row, x_col)   # (nR, order+1)

    @scalar_array_output
    def sigma(self, R):
        r"""
        Surface density :math:`\Sigma(R)`.

        For n > 3/2, the Catalan series:

        .. math::
            \Sigma(R) = 2 \rho_0\, n\, R \sum_{k \ge 0} (k+1)\, c_k\,
            E_{\nu_k}(x), \qquad x = (R/h)^{1/n}

        For n = 1 (exponential profile), the exact closed form

        .. math::
            \Sigma(R) = 2 \rho_0\, R\, K_1(R/h)

        is used instead, where :math:`K_1` is the modified Bessel function
        of the second kind (standard Abel projection of
        :math:`\rho(r) = \rho_0 e^{-r/h}`, via the integral representation
        :math:`K_1(x) = \int_0^\infty e^{-x\cosh t}\cosh(t)\, dt`).

        For other n <= 3/2 the series is not used either; Sigma is instead
        computed by direct Abel (line-of-sight) projection of `density`
        (see the class Notes and :meth:`_sigma_numerical`).
        """
        R = np.atleast_1d(np.asarray(R, float))
        if abs(self.n_index - 1.0) < 1e-12:
            return 2.0 * self.rho_0 * R * kv(1, R / self.h)
        if abs(self.n_index - 0.5) < 1e-12:
            return SQPI_ * self.rho_0 * self.h * np.exp(-((R / self.h) ** 2))
        return self._lown.sigma(R)

    # Below this scaled radius z = (R/h)^(1/n) the native DeltaSigma series
    # loses all precision to catastrophic cancellation (M_3D/piR^2 and the
    # k-sum both ~z^n and nearly cancel; the true result is ~z^{n+1}). The
    # small-z asymptotic is used instead.
    _DS_ASYMP_ZMAX = 0.15
    _DS_ASYMP_NTERMS = 4

    def _deltasigma_asymp(self, R):
        """
        Small-z asymptotic of DeltaSigma, z = (R/h)^(1/n) -> 0.

        DeltaSigma(R) = sum_{p>=1} C_p z^{n+p},
            C_p = -A_p (n+p)/(3n+p),
            A_p = 2 rho_0 n h (-1)^p / p! * Phi_Sigma(p),
            Phi_Sigma(p) = sum_k (k+1) c_k / (2 n k - n - p),

        i.e. the regular power-series coefficients of the v2 dual form. The
        leading z^n pieces of M_3D/piR^2 and the native k-sum cancel exactly;
        Phi_Sigma(p) carries the surviving z^{n+p} term. With _DS_ASYMP_NTERMS
        terms the relative error is ~1% out to z ~ _DS_ASYMP_ZMAX; the series is
        asymptotic, so adding terms beyond ~4 does not help. See
        docs/einasto_proj_density_v2.tex and einasto_pitfalls.md S6.
        """
        n, h = self.n_index, self.h
        z = (R / h) ** (1.0 / n)
        # Phi_Sigma(p) = sum_k (k+1) c_k / (2nk - n - p) converges only as
        # K^{-1/2}, so it needs many more terms than the profile `order`. Use a
        # dedicated high-K sum (independent of self.order); it is a cheap 1-D
        # sum evaluated once.
        K = 200000
        kk = np.arange(0, K + 1, dtype=float)
        ck = _catalan_over_4k(kk)
        out = np.zeros_like(R)
        fact = 1.0
        for p in range(1, self._DS_ASYMP_NTERMS + 1):
            fact *= p
            phi_p = np.sum((kk + 1.0) * ck / (2.0 * n * kk - n - p))
            A_p = 2.0 * self.rho_0 * n * h * (-1.0) ** p / fact * phi_p
            C_p = -A_p * (n + p) / (3.0 * n + p)
            out += C_p * z ** (n + p)
        return out

    @scalar_array_output
    def deltasigma(self, R):
        r"""
        Excess surface density :math:`\Delta\Sigma(R) \equiv \bar\Sigma(<R) -
        \Sigma(R)`.

        For n > 3/2, the Catalan series:

        .. math::
            \Delta\Sigma(R) = \frac{M_{\rm 3D}(R)}{\pi R^2}
            - 2\rho_0\, n\, R \sum_{k \ge 1} k\, c_k\, E_{\nu_k}(x),
            \qquad x = (R/h)^{1/n}

        For :math:`z = (R/h)^{1/n} <` :attr:`_DS_ASYMP_ZMAX` the native
        series suffers catastrophic cancellation and the small-z asymptotic
        (:meth:`_deltasigma_asymp`) is used instead.

        For n = 1 (exponential profile), the exact closed form

        .. math::
            \Delta\Sigma(R) = \rho_0 h \left[\frac{8}{x^2} - 4 K_2(x)
            - 2 x K_1(x)\right], \qquad x = R/h

        is used instead (from :math:`\Sigma(R) = 2\rho_0 R K_1(R/h)` and
        :math:`M_{\rm 2D}(R) = 4\pi\rho_0 h^3 [2 - x^2 K_2(x)]`, using
        :math:`d(x^2 K_2(x))/dx = -x^2 K_1(x)`).

        For other n <= 3/2, neither series applies; DeltaSigma is instead
        computed from a dense numerical `sigma` grid (see the class Notes
        and :meth:`_deltasigma_numerical`).
        """
        R = np.atleast_1d(np.asarray(R, float))
        if abs(self.n_index - 1.0) < 1e-12:
            return self.rho_0 * self.h * _expdisk_deltasigma_factor(R / self.h)
        if abs(self.n_index - 0.5) < 1e-12:
            x2 = (R / self.h) ** 2
            with np.errstate(divide="ignore", invalid="ignore"):
                out = SQPI_ * self.rho_0 * self.h * (
                    -np.expm1(-x2) / x2 - np.exp(-x2))
            return np.where(x2 > 0, out, 0.0)
        return self._lown.deltasigma(R)

    @scalar_array_output
    def enclosed_mass_2D(self, R):
        r"""
        Cylindrical (projected) enclosed mass.

        .. math::
            M_{\rm 2D}(R) = M_{\rm 3D}(R) + 2\pi \rho_0\, n\, R^3
            \sum_{k \ge 0} c_k\, E_{\nu_k}(x), \qquad x = (R/h)^{1/n}

        For n <= 3/2: exact closed forms at the anchors (n = 1/2 Gaussian,
        n = 1 exponential), otherwise ``pi R^2 (Sigma + DeltaSigma)`` from
        the stable low-n series backend.
        """
        R = np.atleast_1d(np.asarray(R, float))
        if abs(self.n_index - 1.0) < 1e-12:
            return 4.0 * np.pi * self.rho_0 * self.h ** 3 \
                * _expdisk_m2d_factor(R / self.h)
        if abs(self.n_index - 0.5) < 1e-12:
            x2 = (R / self.h) ** 2
            return np.pi * SQPI_ * self.rho_0 * self.h ** 3 * (-np.expm1(-x2))
        return self._lown.enclosed_mass_2D(R)

    def order_for_tol(self, tol, R=None, max_order=5000, quantity="sigma"):
        """
        Smallest series order K whose estimated relative truncation error < tol.

        The Catalan series converge algebraically (terms u_k ~ C k^{-p}), so
        the tail remainder is NOT the last term: a last-term criterion
        underestimates the true error by a factor ~K. The asymptotic decay
        exponent is known per quantity (Sigma and DeltaSigma: p = 3/2; M_2D:
        p = 5/2), so the tail is estimated by integrating that power law,

            R_K ~ sum_{k>K} u_k ~ u_K * K / (p - 1),

        and the relative error is ``R_K / |S_K|``. This tracks the true error
        (validated against the Abel-transform ground truth) rather than the
        optimistic step size; for Sigma it gives R_K ~ 2 K u_K. Because the
        Sigma weight (k+1) c_k bounds the M_2D (c_k) and DeltaSigma (k c_k)
        weights for k>=1, the Sigma order is a conservative choice for all
        three quantities.

        Parameters
        ----------
        tol : float
            Target relative truncation error.
        R : array_like, optional
            Probe radii. Defaults to r_s * [0.1, 0.3, 1, 3, 10], spanning the
            range where convergence is slowest.
        max_order : int, optional
            Search ceiling; returned if ``tol`` is not met.
        quantity : {"sigma", "m2d", "deltasigma"}, optional
            Series whose convergence is tested.

        Returns
        -------
        int
            Series order K (k = 0..K).
        """
        if not self._series:
            raise NotImplementedError(
                "order_for_tol tunes the Catalan series order, which is not "
                "used for n <= 3/2 (alpha >= 2/3); sigma/deltasigma/"
                "power_spectrum are computed numerically for this n instead."
            )
        if R is None:
            R = self.r_s * np.array([0.1, 0.3, 1.0, 3.0, 10.0])
        R = np.atleast_1d(np.asarray(R, float))

        n, h = self.n_index, self.h
        x = (R / h) ** (1.0 / n)
        x_col = x[:, None]

        k = np.arange(0, max_order + 1)
        ck = _catalan_over_4k(k)
        nu_k = 2 * k * n - n + 1
        if quantity == "sigma":
            weight, p_decay = (k + 1) * ck, 1.5
        elif quantity == "deltasigma":
            weight, p_decay = k * ck, 1.5
        elif quantity == "m2d":
            weight, p_decay = ck, 2.5
        else:
            raise ValueError(f"unknown quantity {quantity!r}")

        terms = weight[None, :] * expn_fast(nu_k[None, :], x_col)  # (nR, K+1)
        partial = np.cumsum(terms, axis=1)
        u = np.abs(terms)
        kk = k.astype(float)

        # Tail estimate from the known power-law decay u_k ~ C k^{-p_decay}:
        # R_K ~ u_K * K / (p_decay - 1). Require k >= kmin so the asymptotic
        # regime holds (the k=0->1 step is not power-law).
        kmin = 2
        with np.errstate(divide="ignore", invalid="ignore"):
            tail = u[:, kmin:] * kk[kmin:] / (p_decay - 1.0)
            rel_err = tail / np.abs(partial[:, kmin:])

        converged = np.all(rel_err < tol, axis=0)
        idx = np.argmax(converged)
        if converged[idx]:
            return int(idx + kmin)
        return int(max_order)

    # ------------------------------------------------------------------
    # Fourier-space form factor / power spectrum
    # ------------------------------------------------------------------
    def power_spectrum(self, k, branch="auto", order=None):
        r"""
        Rescaled Fourier transform of the profile (einasto_power_spectrum.tex):

        .. math::
            P(k) = \frac{\tilde\rho(k)}{(4\pi)^2}

        In "auto" mode (the default), the shape index :math:`n = 1/\alpha`
        selects one of four exact or convergent representations, with
        :math:`\tilde k \equiv k h`:

        **n > 1** (large-k / small-scale series), analytic, converges for
        all k:

        .. math::
            P(k) = \frac{\rho_0 h^3}{4\pi \tilde k^3}
            \sum_{m \ge 1} A_m^- \tilde k^{-m/n}, \qquad
            A_m^- = \frac{(-1)^{m+1}}{m!}\, \Gamma\!\left(2+\frac{m}{n}\right)
            \sin\!\left(\frac{\pi m}{2n}\right)

        **n = 1** (boundary), closed form:

        .. math::
            P(k) = \frac{\rho_0 h^3}{2\pi \left(1 + \tilde k^2\right)^2}

        **n = 1/2**, exact Gaussian closed form:

        .. math::
            P(k) = \frac{\rho_0 h^3}{16\sqrt{\pi}}\, e^{-\tilde k^2/4}

        **0 < n < 1, n != 1/2**: the small-k series

        .. math::
            P(k) = \frac{\rho_0\, n\, h^3}{4\pi}
            \sum_{m \ge 0} A_m^+ \left(\frac{\tilde k^2}{4}\right)^m, \qquad
            A_m^+ = \frac{(-1)^m\, \Gamma(3n+2nm)}{m!\, (3/2)_m}

        converges for all k but self-cancels in fp64 beyond a modest
        :math:`\tilde k`. Evaluation therefore dispatches per point among
        three analytic forms with computable error estimates: the Kummer
        (anti-cancellation) decomposition

        .. math::
            P = \frac{\rho_0 n h^3}{4\pi}\, e^{-\tilde k^2/4}
            \sum_{m \ge 0} b_m \left(\frac{\tilde k}{2}\right)^{2m},
            \qquad b_m = \sum_{i=0}^{m} \frac{A_i^+}{(m-i)!}

        with build-time :math:`b_m` (exactly :math:`b_m = \delta_{m0}` at
        n = 1/2); the plain series; and the optimally-truncated large-k
        series above, a valid asymptotic expansion for n < 1 with error
        :math:`\sim e^{-c\tilde k^{1/(1-n)}}`. Trapezoidal Mellin-Barnes
        contour quadrature covers the narrow window (only
        :math:`n \gtrsim 0.93`) where no estimate meets ``_PK_TOL``.
        Validated to <= 1e-11 against mpmath for n = 0.45-0.97 (see
        docs/einasto_proj_density_v4.tex).

        **n > 1** dispatches through a cost-ordered analytic cascade
        (plateau series, direct series, MB contour / Filon quadrature) --
        see the inline comments in the auto branch and
        docs/einasto_math.md. Validated to <= 3.3e-10 for n = 1.05-4 and
        <= 8.5e-8 for n = 10 over the physical k range.

        Parameters
        ----------
        k : array_like
            Wavenumber [1/length].
        branch : {"auto", "small_k", "large_k", "closed"}, optional
            Series branch; "auto" picks by n as described above. The named
            branches evaluate the corresponding series directly (useful for
            comparison/research) but are not used automatically for n<=1
            because of the cancellation issue above.
        order : int, optional
            Number of series terms (defaults to self.order).

        Returns
        -------
        ndarray
            P(k), same shape as k.
        """
        n, h, rho_0 = self.n_index, self.h, self.rho_0
        M = self.order if order is None else order
        kt = np.asarray(k, float) * h

        if branch == "auto":
            if np.isclose(n, 1.0):
                branch = "closed"
            elif np.isclose(n, 0.5):
                # Exact Gaussian transform (einasto_power_spectrum.tex, eq. 21).
                return rho_0 * h ** 3 / (16.0 * np.sqrt(np.pi)) * np.exp(-(kt ** 2) / 4.0)
            elif n < 1.0:
                # Analytic dispatch (docs/einasto_proj_density_v4.tex,
                # "The power spectrum"): per point, the best of
                #   (a) the Kummer form e^{-zeta} sum b_m zeta^m (the
                #       anti-cancellation decomposition of the convergent
                #       small-kt series; exact e^{-zeta} at n=1/2),
                #   (b) the plain convergent series (when the Kummer build
                #       is unusable, n >~ 0.93), and
                #   (c) the optimally-truncated large-kt asymptotic series
                #       (valid for n<1 with error ~ exp(-c kt^{1/(1-n)})),
                # each carrying a computable error estimate; Gauss-Laguerre
                # quadrature of the master integral only where no estimate
                # meets _PK_TOL (a narrow kt window for n >~ 0.93).
                kt_arr = np.atleast_1d(np.asarray(kt, float))
                if self._pk_bm is None:
                    self._pk_bm = _pk_build_kummer(n)
                sgn_b, logb, usable = self._pk_bm
                va, ea = _pk_asym_eval(n, kt_arr)
                if usable:
                    vb, eb = _pk_kummer_eval(n, kt_arr, sgn_b, logb)
                else:
                    vb, eb = _pk_conv_eval(n, kt_arr)
                use_a = ea < eb
                val = np.where(use_a, va, vb)
                err = np.minimum(ea, eb)
                # Mellin-Barnes contour quadrature where the series
                # estimates fail (a narrow kt window for n >~ 0.93);
                # machine-exact and the cheapest evaluator in that window.
                bad = err > 1e-8
                if bad.any():
                    val[bad] = _pk_mb_contour(n, kt_arr[bad])
                out = rho_0 * h ** 3 * val
                return out if np.ndim(kt) else out[0]
            else:
                # n > 1: cost-ordered analytic cascade (each branch carries
                # a computable error estimate; later, costlier branches
                # only touch the points earlier ones could not certify):
                #   1. plateau series, optimally truncated (asymptotic for
                #      n>1; superb at small kt),
                #   2. direct large-k series (convergent for n>1; superb
                #      at moderate/large kt; estimate covers cancellation
                #      AND the unsummed tail),
                #   3. crack filler: Mellin-Barnes contour quadrature for
                #      n <= 3 (machine-exact, cheapest); Gauss-Laguerre
                #      (N=300; exact at small kt) for larger n, where the
                #      MB phase gradient ~ n ln n is under-sampled.
                # The Wright rotation branch is no longer used here (it
                # was misrouted into the deep plateau for n >~ 3 and is
                # dominated by the direct series where it is valid).
                kt_arr = np.atleast_1d(np.asarray(kt, float))
                vp, ep = _pk_plateau_eval(n, kt_arr)
                val = vp
                need = ep > _PK_TOL
                if need.any():
                    vd, ed = _pk_direct_eval(n, kt_arr[need])
                    better = ed < ep[need]
                    val[need] = np.where(better, vd, vp[need])
                    still = np.where(need)[0][np.minimum(ed, ep[need])
                                              > 1e-8]
                    if still.size:
                        if n <= 3.0:
                            val[still] = _pk_mb_contour(n, kt_arr[still])
                        else:
                            # large-n turnover: the master integrand
                            # oscillates ~kt (2n)^n times against the
                            # weight; Filon (envelope-resolved, exact
                            # sine integrals) instead of GL, which
                            # undersamples there (errors up to ~4e-2)
                            val[still] = _pk_filon(n, kt_arr[still])
                out = rho_0 * h ** 3 * val
                return out if np.ndim(kt) else out[0]

        if branch == "closed":
            return rho_0 * h ** 3 / (2 * np.pi * (1 + kt ** 2) ** 2)

        if branch == "small_k":
            m = np.arange(0, M + 1)
            # A_m^+ = (-1)^m Gamma(3n+2nm) / [m! (3/2)_m]
            log_coef = gammaln(3 * n + 2 * n * m) - gammaln(m + 1) \
                - (gammaln(1.5 + m) - gammaln(1.5))
            # For n>1 this series diverges; use optimal truncation per kt.
            zeta = (np.atleast_1d(np.asarray(kt, float)) ** 2) / 4.0  # (nk,)
            prefactor = rho_0 * n * h ** 3 / (4 * np.pi)
            result = np.empty_like(zeta)
            for i, z in enumerate(zeta):
                if z == 0:
                    result[i] = prefactor * np.exp(log_coef[0])
                    continue
                log_term = log_coef + m * np.log(z)
                # Optimal truncation: stop where |term| starts growing
                diffs = np.diff(log_term)
                m_opt = np.argmax(diffs > 0)
                if m_opt == 0 and diffs[0] <= 0:
                    m_opt = M  # all decreasing, use full series
                terms = (-1.0) ** m[:m_opt+1] * np.exp(
                    log_term[:m_opt+1] - log_term[0])
                result[i] = prefactor * np.exp(log_term[0]) * np.sum(terms)
            return result

        if branch == "large_k":
            # Wright psi-function with adaptive order per kt.
            # F(z) = sum_{m>=1} Gamma(2+m/n)/m! z^m, entire for n>1.
            # series = Im[F(xi e^{i theta_-}) - F(xi e^{i theta_+})] / 2
            # with xi = kt^{-1/n}.  Accumulate in log-space; stop when
            # new terms contribute < rtol to |F|.
            theta_m = np.pi * (2 * n - 1) / (2 * n)
            theta_p = np.pi * (2 * n + 1) / (2 * n)
            exp_ithm = np.exp(1j * theta_m)
            exp_ithp = np.exp(1j * theta_p)

            kt_arr = np.atleast_1d(np.asarray(kt, float))
            result = np.empty_like(kt_arr)
            rtol = 1e-14
            max_terms = max(M, 5000)

            for i, kti in enumerate(kt_arr):
                if kti <= 0:
                    result[i] = rho_0 * n * h ** 3 / (4 * np.pi) * gamma(3 * n)
                    continue
                log_xi = -np.log(kti) / n
                # Accumulate F_m, F_p as running complex sums in scaled form
                # term_m = exp(log_coef_m + m*log_xi) * exp(i*m*theta)
                # Use incremental: log_coef_m = gammaln(2+m/n) - gammaln(m+1)
                Fm = 0.0 + 0j
                Fp = 0.0 + 0j
                log_scale = 0.0  # running scale factor
                prev_abs = 0.0
                for m in range(1, max_terms + 1):
                    lc = gammaln(2 + m / n) - gammaln(m + 1)
                    log_mag_m = lc + m * log_xi
                    # Rescale: keep log_scale as the reference
                    if m == 1:
                        log_scale = log_mag_m
                        sm = 1.0
                    else:
                        sm = np.exp(log_mag_m - log_scale)
                        if not np.isfinite(sm):
                            # Rescale upward
                            Fm *= np.exp(log_scale - log_mag_m)
                            Fp *= np.exp(log_scale - log_mag_m)
                            log_scale = log_mag_m
                            sm = 1.0
                    zm = sm * exp_ithm ** m
                    zp = sm * exp_ithp ** m
                    Fm += zm
                    Fp += zp
                    # Convergence: the peak term is near m ~ xi^n.
                    # Only check after passing it.
                    m_peak = int(np.exp(n * log_xi)) + 50
                    if m >= m_peak and m % 20 == 0:
                        cur = abs(np.imag(Fm - Fp))
                        if prev_abs > 0 and cur > 0:
                            rel_change = abs(cur - prev_abs) / cur
                            if rel_change < rtol:
                                break
                        prev_abs = cur
                imag_diff = np.imag(Fm - Fp) / 2.0
                if imag_diff == 0:
                    result[i] = 0.0
                else:
                    sign = np.sign(imag_diff)
                    log_ans = (log_scale + np.log(abs(imag_diff))
                               + np.log(rho_0 * h ** 3 / (4 * np.pi))
                               - 3 * np.log(kti))
                    result[i] = sign * np.exp(log_ans)
            return result

        raise ValueError(f"unknown branch {branch!r}")

    def fourier(self, k, **kwargs):
        """Isotropic form factor rho_tilde(k) = (4 pi)^2 P(k)."""
        return (4 * np.pi) ** 2 * self.power_spectrum(k, **kwargs)

    # ------------------------------------------------------------------
    # Lensing observables
    # ------------------------------------------------------------------
    def convergence(self, R, sigma_crit=1.0):
        """Convergence kappa(R) = Sigma(R) / Sigma_crit."""
        return self.sigma(R) / sigma_crit

    def shear(self, R, sigma_crit=1.0):
        """Tangential shear gamma(R) = DeltaSigma(R) / Sigma_crit."""
        return self.deltasigma(R) / sigma_crit
