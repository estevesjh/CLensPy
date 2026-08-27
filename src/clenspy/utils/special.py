r"""Special functions for halo-profile series -- generalised E_nu and friends.

Split out of `clenspy.halo.einasto`: none of this is Einasto-specific, and
other profiles that need a Mellin-Barnes or Catalan series will want the
same pieces.

The load-bearing routine is `expn_fast`, the generalised exponential
integral :math:`E_\nu(x)` for arbitrary real :math:`\nu > 0`, evaluated by
dispatch rather than a single formula:

- integer :math:`\nu \ge 1` -- ``scipy.special.expn``;
- large :math:`\nu` -- the DLMF 8.20(ii) uniform asymptotic expansion,
  truncated at `NTERMS_ASYMP` terms, whose :math:`\nu^{-\rm nterms}` error
  sets the switch point via `_nu_asymp_threshold`;
- :math:`\nu < 1` -- the incomplete-gamma closed form;
- otherwise -- upward recurrence in :math:`\nu`, falling back to
  ``mpmath.expint``.

NOTE: dimensionless throughout. These are pure special functions; no
lengths, masses or densities appear.
"""

import numpy as np
from scipy.special import expn, gamma, gammaincc, gammaln

try:
    import mpmath as _mp
except ImportError:  # pragma: no cover
    _mp = None

__all__ = [
    "EULER_GAMMA",
    "NTERMS_ASYMP",
    "catalan_over_4k",
    "expint_asymptotic",
    "expn_fast",
]



# Number of terms retained in the DLMF 8.20 asymptotic expansion.
NTERMS_ASYMP = 5


def _nu_asymp_threshold(rtol, nterms=NTERMS_ASYMP):
    """Minimum nu for asymptotic: error ~ nu^{-nterms} < rtol."""
    return rtol ** (-1.0 / nterms)


EULER_GAMMA = 0.5772156649015329


def catalan_over_4k(k):
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


if __name__ == "__main__":
    import numpy as np

    print(f"EULER_GAMMA   = {EULER_GAMMA:.15f}")
    print(f"NTERMS_ASYMP  = {NTERMS_ASYMP}")

    print("\nexpn_fast(nu, x) against scipy.special.expn for integer nu:")
    from scipy.special import expn

    for nu in (1, 2, 3):
        x = np.array([0.5, 1.0, 5.0, 20.0])
        mine, ref = expn_fast(nu, x), expn(nu, x)
        err = np.max(np.abs(mine / ref - 1))
        print(f"  nu={nu}: max rel err {err:.3e}")

    print("\nnon-integer nu, where scipy has no expn:")
    for nu in (0.3, 1.7, 2.5):
        print(f"  E_{nu}(1.0) = {float(np.ravel(expn_fast(nu, 1.0))[0]):.10e}")

    print("\nasymptotic branch, x >> 1 (DLMF 8.20):")
    for x in (30.0, 100.0):
        v = float(np.ravel(expint_asymptotic(1.5, x))[0])
        print(f"  expint_asymptotic(1.5, {x}) = {v:.6e}  "
              f"(e^-x/x = {np.exp(-x) / x:.6e})")
