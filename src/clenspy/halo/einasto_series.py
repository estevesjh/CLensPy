r"""Einasto power-spectrum branch evaluators.

Split out of `clenspy.halo.einasto`: these are the analytic
:math:`P(k)` evaluators that `EinastoProfile.power_spectrum` dispatches
between, each valid on its own slice of :math:`(n, kt)` and each carrying a
computable error estimate. The branch-selection logic stays with the class;
only the evaluators live here.

Derivations in ``docs/einasto_proj_density_v4.tex`` ("The power spectrum").

NOTE: every routine here works in the
:math:`P = \tilde\rho / (4\pi)^2` convention with
:math:`\rho_0 = h = 1`, and the caller rescales. ``kt`` is the
dimensionless :math:`k h`.

NOTE: ``h`` is the Einasto scale radius, not :math:`H_0/100` -- see
`clenspy.halo.einasto`.
"""

import numpy as np
from scipy.special import gammaln, loggamma, roots_genlaguerre

#: Nothing here is public. The evaluators are selected by
#: `EinastoProfile.power_spectrum`, never called directly.
__all__: list[str] = []



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
