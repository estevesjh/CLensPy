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
from scipy.special import expn, gamma, gammainc, gammaincc, gammaln, kv, roots_genlaguerre

from ..utils.decorators import scalar_array_output
from ..utils.integrate import compute_sigma_quadvec, sigma_to_deltasigma_cumtrapz
from ..utils.interpolate import make_log_interpolation

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
    (docs/einasto_proj_density.tex). For n <= 3/2 (alpha >= 2/3) that series
    converges too slowly to be usable, so :meth:`sigma` and :meth:`deltasigma`
    are instead computed numerically: :meth:`sigma` via direct Abel
    (line-of-sight) projection of :meth:`density`, :meth:`deltasigma` from a
    dense :meth:`sigma` grid via cumulative-trapezoid enclosed mass.
    :meth:`enclosed_mass_2D` has no fallback and raises for n <= 3/2.

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

        # n > 3/2: closed-form Catalan series. n <= 3/2: numerical fallback
        # (see class Notes) -- the series converges too slowly to use there.
        self._series = self.n_index > 1.5
        if self._series:
            if tol is not None:
                self._build(order)                       # ceiling for the search
                order = self.order_for_tol(tol, max_order=order)
            self._build(order)
        else:
            self.order = None

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
        if np.isclose(self.n_index, 1.0):
            return 2.0 * self.rho_0 * R * kv(1, R / self.h)
        if not self._series:
            return self._sigma_numerical(R)
        weight = (self._k + 1) * self._ck
        sumval = np.sum(weight[None, :] * self._E_nu(R), axis=-1)
        return 2 * self.rho_0 * self.n_index * R * sumval

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
        if np.isclose(self.n_index, 1.0):
            x = R / self.h
            return self.rho_0 * self.h * (
                8.0 / x ** 2 - 4.0 * kv(2, x) - 2.0 * x * kv(1, x)
            )
        if not self._series:
            return self._deltasigma_numerical(R)
        z = (R / self.h) ** (1.0 / self.n_index)
        small = z < self._DS_ASYMP_ZMAX

        out = np.empty_like(R)
        if small.any():
            out[small] = self._deltasigma_asymp(R[small])
        if (~small).any():
            Rb = R[~small]
            weight = self._k * self._ck         # k=0 term vanishes
            sumval = np.sum(weight[None, :] * self._E_nu(Rb), axis=-1)
            mean_term = self.enclosed_mass(Rb) / (np.pi * Rb ** 2)
            out[~small] = mean_term - 2 * self.rho_0 * self.n_index * Rb * sumval
        return out

    @scalar_array_output
    def enclosed_mass_2D(self, R):
        r"""
        Cylindrical (projected) enclosed mass.

        .. math::
            M_{\rm 2D}(R) = M_{\rm 3D}(R) + 2\pi \rho_0\, n\, R^3
            \sum_{k \ge 0} c_k\, E_{\nu_k}(x), \qquad x = (R/h)^{1/n}

        Only available for n > 3/2 (the Catalan series); there is no
        numerical fallback for n <= 3/2 (unlike `sigma`/`deltasigma`/
        `power_spectrum`).
        """
        if not self._series:
            raise NotImplementedError(
                "enclosed_mass_2D has no numerical fallback for n <= 3/2 "
                "(alpha >= 2/3); use enclosed_mass (3D) or sigma/deltasigma, "
                "which are supported numerically for this n."
            )
        R = np.atleast_1d(np.asarray(R, float))
        sumval = np.sum(self._ck[None, :] * self._E_nu(R), axis=-1)
        return self.enclosed_mass(R) + 2 * np.pi * self.rho_0 * self.n_index * R ** 3 * sumval

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

        is a convergent Cauchy series for all k, but its finite-precision
        partial sums suffer unresolved catastrophic cancellation (no
        anti-cancellation decomposition like the n>1 Wright form below has
        been derived for it). ``power_spectrum`` instead computes this
        case numerically, via a FFTLog transform (`mcfit.xi2P`) of
        `density`; see :meth:`_power_spectrum_numerical`.

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
                # The small-k series (eq. 12) is a convergent Cauchy series
                # for all k when n<1, but its finite-precision partial sums
                # suffer unresolved catastrophic cancellation well before
                # the asymptotic regime (no anti-cancellation trick like the
                # n>1 Wright decomposition below has been derived for this
                # branch yet). Use a direct FFTLog transform of density()
                # instead; validated against the exact n=1/2 case above to
                # ~1e-4 relative accuracy.
                return self._power_spectrum_numerical(k)
            else:
                # n>1: analytic large-k series (eq. 16), converges for all k.
                # Dispatch valid for both n<1 and n>1 -- per-kt threshold:
                #   sum_{m=1..m_star} |t_m| / P(0) < tol.
                # Plateau-residual is valid where this holds; the largest
                # such kt is the auto-found kt_max for the plateau branch.
                # Where it fails: for n>1 there's a large-k asymptotic
                # (Wright) series; for n<=1 that series isn't valid (see
                # _einasto_pk_wright_real), but GL quadrature alone already
                # covers the practically-relevant range (validated against
                # the exact Gaussian n=1/2 closed form to ~1e-14 out to
                # P(k)/P(0) ~ 1e-17), so large-k points just fall to GL.
                tol = 1e-2
                N_GL = min(400, max(96, int(30 * max(n, 1.0))))
                if n > 1.0:
                    xi_fail = 32.0 * (1.0 - np.exp(-(n - 1.0) / 2.5))
                    kt_w = xi_fail ** (-n)
                else:
                    kt_w = np.inf

                pref_pl = rho_0 * n * h ** 3 / (4 * np.pi)
                P0 = pref_pl * gamma(3 * n)

                kt_arr = np.atleast_1d(kt)
                result = np.empty_like(kt_arr)

                J_max = 80
                m_arr = np.arange(1, J_max + 1, dtype=float)
                log_z = 2 * np.log(kt_arr)[:, None] - np.log(4.0)
                log_t_abs = (
                    gammaln(3 * n + 2 * n * m_arr)[None, :]
                    - gammaln(m_arr + 1.0)[None, :]
                    - gammaln(1.5 + m_arr)[None, :]
                    + gammaln(1.5)
                    + m_arr[None, :] * log_z
                )                                              # log|t_m|
                log_P0 = gammaln(3 * n)
                log_t_norm = log_t_abs - log_P0                 # log|t_m/P0|
                # Decision metric: scan all truncations M=1..J_max and
                # pick the smallest M where |sum_{m=1..M} t_m| / P(0) < tol.
                # If no such M exists (sum never dips below tol), the
                # plateau-residual branch is invalid for this kt.
                sgn = (-1.0) ** m_arr
                t_norm = sgn[None, :] * np.exp(log_t_norm)      # (nk, J_max)
                cumS = np.cumsum(t_norm, axis=1)                # partial sums
                small = np.abs(cumS) < tol                      # (nk, J_max)
                first_M = np.argmax(small, axis=1)              # 0-indexed
                use_pl = np.any(small, axis=1)
                m_trunc = first_M + 1                           # smallest M
                use_w = (~use_pl) & (kt_arr >= kt_w)
                use_gl = ~(use_pl | use_w)

                if use_pl.any():
                    sgn = (-1.0) ** m_arr
                    idxs = np.where(use_pl)[0]
                    out_pl = np.empty(idxs.size)
                    for k, i in enumerate(idxs):
                        J = int(m_trunc[i])
                        terms = sgn[:J] * np.exp(log_t_abs[i, :J])
                        out_pl[k] = P0 + pref_pl * np.sum(terms)
                    result[use_pl] = out_pl
                if use_gl.any():
                    result[use_gl] = _einasto_pk_GL(
                        kt_arr[use_gl], n, h, rho_0, N=N_GL)
                if use_w.any():
                    result[use_w] = _einasto_pk_wright_real(
                        kt_arr[use_w], n, h, rho_0, M=80)
                return result

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
