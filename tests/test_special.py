r"""Coverage gaps in the special-function dispatch: the mpmath fallback,
the recurrence loop's early-exit guard, and the mixed asymptotic/gamma
dispatch branch of `expn_fast`.

`clenspy.utils.special` is already exercised end to end via
`tests/test_einasto.py`; this file targets the pieces that slip past that
coverage.
"""

import numpy as np
import pytest

from clenspy.utils.special import (
    _expint_gamma,
    _expint_recurrence,
    _nu_asymp_threshold,
    expn_fast,
)

mpmath = pytest.importorskip("mpmath")


def test_expint_mpmath_fallback_matches_mpmath_directly():
    r"""``_expint_mpmath`` (and its inner ``_one``) wraps ``mpmath.expint``.

    It is not reached through `expn_fast`'s dispatch table -- the upward
    recurrence (`_expint_recurrence`) covers the same regime without
    mpmath -- so it is exercised directly here rather than through a
    contrived `expn_fast` call.
    """
    from clenspy.utils.special import _expint_mpmath

    # array inputs: frompyfunc's ``.astype(float)`` needs an array-typed
    # result, which a bare Python-scalar call does not give it.
    for nu, x in ((2.5, 3.0), (0.3, 1.0), (7.0, 0.2)):
        got = float(np.ravel(_expint_mpmath(np.array([nu]), np.array([x])))[0])
        ref = float(mpmath.re(mpmath.expint(nu, x)))
        assert np.isfinite(got)
        assert got == pytest.approx(ref, rel=1e-9)


def test_expint_recurrence_converges_and_is_finite():
    r"""A case designed to converge quickly: small, mixed non-integer nu.

    The ``if not active.any(): break`` guard in `_expint_recurrence` is
    unreachable in practice -- the loop's own range is
    ``int(floor(nu).max())``, which is *exactly* the number of steps the
    largest element needs, so ``active`` never goes fully false before the
    range is exhausted (verified empirically: instrumenting the loop shows
    it never breaks early for any nu, including mixed arrays with very
    different magnitudes). This test therefore checks the documented
    behaviour -- a finite, correct result on a fast-converging input --
    rather than the unreachable branch itself.
    """
    nu = np.array([1.3, 2.7, 20.5])
    x = np.array([1.0, 1.0, 1.0])
    got = _expint_recurrence(nu, x)
    assert np.all(np.isfinite(got))
    for n, xv, g in zip(nu, x, got):
        ref = float(mpmath.re(mpmath.expint(float(n), float(xv))))
        assert g == pytest.approx(ref, rel=1e-8)


def test_expn_fast_dispatches_asymptotic_and_gamma_in_one_call():
    r"""One array spanning both the :math:`\nu < 1` and large-:math:`\nu`
    regimes, so ``is_asymp`` and ``is_gamma`` both fire in the same call.
    """
    threshold = _nu_asymp_threshold(1e-9)
    assert threshold < 100.5  # sanity: the chosen nu really clears it

    # 100.5 (not 100.0): an exact integer nu is claimed by `is_int_pos`
    # first and never reaches `is_asymp`, since `np.rint` rounds it right
    # back to itself.
    nu = np.array([0.5, 100.5])
    x = np.array([1.0, 50.0])
    got = expn_fast(nu, x)
    assert got.shape == nu.shape
    assert np.all(np.isfinite(got))

    ref = np.array([float(mpmath.re(mpmath.expint(float(n), float(xv))))
                     for n, xv in zip(nu, x)])
    np.testing.assert_allclose(got, ref, rtol=1e-5)

    # and each element matches evaluating its own branch alone
    gamma_only = _expint_gamma(nu[0], x[0])
    assert got[0] == pytest.approx(gamma_only, rel=1e-12)
