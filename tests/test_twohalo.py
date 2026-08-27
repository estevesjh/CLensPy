"""Does `TwoHaloTerm` run, and are its three outputs mutually consistent?

NOTE: the comparison against `cluster_toolkit` and CLMM that used to be the
only test in this file is now `validation/validate_twohalo_chain.py`, where
it checks each transform stage against a closed-form NFW instead of only the
end of the chain. See ``docs/validation.md``.

These are cheap invariants that need no external library: shapes, signs,
monotonicity, and the identity relating the three projections.
"""

import numpy as np
import pytest

from clenspy.halo.twohalo import TwoHaloTerm

#: A pure power law, so xi(r) is a pure power law and the projections are
#: strictly monotonic -- the invariants below are then exact statements
#: rather than statements about this particular input. A cored P(k) gives a
#: genuinely flat Sigma core and would make the monotonicity check false for
#: physical reasons, which is not what these tests are for.
K = np.logspace(-3, 1, 64)  # 1/Mpc
PK = 2e4 * K ** (-1.5)
Z = 0.2


@pytest.fixture
def twohalo():
    return TwoHaloTerm(K, PK, zvec=Z)


def test_xi_is_finite_and_decreasing(twohalo):
    r = np.logspace(-1, 1.5, 30)
    xi = np.ravel(twohalo.xi(r, Z))
    assert np.all(np.isfinite(xi))
    assert np.all(np.diff(xi) < 0)


def test_sigma_and_deltasigma_are_positive_and_decreasing(twohalo):
    R = np.logspace(-1, 1, 25)
    for name in ("sigma", "deltasigma"):
        v = np.ravel(getattr(twohalo, name)(R, Z))
        assert np.all(np.isfinite(v)), f"{name} must be finite"
        assert np.all(v > 0), f"{name} must be positive"
        assert np.all(np.diff(v) < 0), f"{name} must decrease outward"


def test_sigma_falls_off_faster_than_xi():
    """Projection is shallower than the 3D profile, never steeper."""
    th = TwoHaloTerm(K, PK, zvec=Z)
    r = np.array([1.0, 4.0])
    slope_xi = np.diff(np.log(np.ravel(th.xi(r, Z)))) / np.diff(np.log(r))
    slope_sig = np.diff(np.log(np.ravel(th.sigma(r, Z)))) / np.diff(np.log(r))
    assert slope_sig > slope_xi


def test_p_kz_reproduces_the_input_spectrum(twohalo):
    """The P(k, z) interpolator must pass its own input through."""
    k = K[5:-5]  # off the interpolation edges
    np.testing.assert_allclose(
        np.ravel(twohalo.p_kz(k, Z)), PK[5:-5], rtol=2e-2
    )


def test_scalar_and_array_radii_agree(twohalo):
    """A scalar R gives the same number as the length-1 array."""
    for name in ("xi", "sigma", "deltasigma"):
        method = getattr(twohalo, name)
        np.testing.assert_allclose(
            np.ravel(method(1.5, Z)), np.ravel(method(np.array([1.5]), Z))
        )


def test_unsorted_k_is_accepted():
    """A descending k grid must give the same answer as an ascending one."""
    order = np.argsort(-K)
    shuffled = TwoHaloTerm(K[order], PK[order], zvec=Z)
    R = np.logspace(-1, 0.5, 8)
    np.testing.assert_allclose(
        np.ravel(shuffled.sigma(R, Z)),
        np.ravel(TwoHaloTerm(K, PK, zvec=Z).sigma(R, Z)),
        rtol=1e-6,
    )


@pytest.mark.parametrize("method", ["trapz", "quad_vec"])
def test_quadrature_backends_agree(method):
    """The Abel backends must agree to better than their own accuracy."""
    R = np.logspace(-1, 0.5, 10)
    ref = np.ravel(TwoHaloTerm(K, PK, zvec=Z, method="quad_vec").sigma(R, Z))
    got = np.ravel(TwoHaloTerm(K, PK, zvec=Z, method=method).sigma(R, Z))
    np.testing.assert_allclose(got, ref, rtol=1e-3)
