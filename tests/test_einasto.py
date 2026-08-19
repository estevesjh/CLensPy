# tests/test_einasto.py

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import kv

from clenspy.halo.einasto import EinastoProfile, expint_asymptotic, expn_fast

mpmath = pytest.importorskip("mpmath")


def sigma_abel(n, h, rho_0, R):
    """Ground-truth Sigma(R) via r = R cosh(u) (removes the Abel singularity)."""
    def f(u):
        r = R * np.cosh(u)
        return rho_0 * np.exp(-(r / h) ** (1.0 / n)) * np.cosh(u)
    umax = np.arccosh(min(1e12, (200.0 ** n) * h / R))
    val, _ = quad(f, 0.0, umax, limit=400)
    return 2.0 * R * val


class TestExpInt:
    """Generalized exponential integral E_nu(x)."""

    @pytest.mark.parametrize("nu,x", [
        (1.0, 2.0), (3.0, 0.5), (2.5, 1.0), (0.5, 1.0),
        (12.7, 4.0), (40.0, 3.0), (-3.0, 2.0),
    ])
    def test_dispatch_matches_mpmath(self, nu, x):
        got = float(np.ravel(expn_fast(nu, x))[0])
        ref = float(mpmath.re(mpmath.expint(nu, x)))
        assert got == pytest.approx(ref, rel=1e-5)

    @pytest.mark.parametrize("nu,x", [(20.0, 5.0), (50.0, 25.0), (100.0, 10.0)])
    def test_dlmf_asymptotic(self, nu, x):
        got = float(expint_asymptotic(nu, x))
        ref = float(mpmath.re(mpmath.expint(nu, x)))
        assert got == pytest.approx(ref, rel=1e-6)


class TestSpecialCases:
    """Validation and boundary behaviour."""

    def test_rejects_small_n(self):
        # n <= 3/2 should raise
        with pytest.raises(ValueError, match="must be > 3/2"):
            EinastoProfile(alpha=2.0, rho_0=1.0, r_s=1.0)   # n = 0.5
        with pytest.raises(ValueError, match="must be > 3/2"):
            EinastoProfile(alpha=1.0, rho_0=1.0, r_s=1.0)   # n = 1.0


class TestSpiralHalo:
    """Physically relevant indices n = 4, 5."""

    @pytest.mark.parametrize("n", [4.0, 5.0])
    def test_sigma_vs_abel(self, n):
        rho_0 = 1.0
        e = EinastoProfile(alpha=1.0 / n, rho_0=rho_0, r_s=1.0, order=200)
        h = e.h
        # probe x = (R/h)^(1/n) in [0.5, 3] — avoids the slow-convergence x>3 tail
        R = h * np.array([0.5, 1.0, 2.0, 3.0]) ** n
        ref = np.array([sigma_abel(n, h, rho_0, r) for r in R])
        assert np.allclose(e.sigma(R), ref, rtol=1e-2)

    @pytest.mark.parametrize("n,tol", [(4.0, 1e-2), (5.0, 1e-2), (5.0, 1e-3)])
    def test_order_for_tol_meets_target(self, n, tol):
        rho_0 = 1.0
        e = EinastoProfile(alpha=1.0 / n, rho_0=rho_0, r_s=1.0, order=2)
        h = e.h
        R = h * np.array([0.3, 0.5, 1.0, 2.0, 3.0]) ** n
        K = e.order_for_tol(tol, R=R, max_order=20000)
        ek = EinastoProfile(alpha=1.0 / n, rho_0=rho_0, r_s=1.0, order=K)
        ref = np.array([sigma_abel(n, h, rho_0, r) for r in R])
        err = np.max(np.abs(ek.sigma(R) / ref - 1.0))
        # estimate is calibrated to the true error within a small factor
        assert err <= 1.5 * tol

    def test_power_spectrum_n4_table(self):
        # A_m^- coefficients, einasto_power_spectrum.tex Table I
        n = 4.0
        m = np.arange(1, 8)
        from scipy.special import gammaln
        A = (-1.0) ** (m + 1) * np.exp(gammaln(2 + m / n) - gammaln(m + 1)) \
            * np.sin(np.pi * m / (2 * n))
        ref = np.array([0.43358, -0.46999, 0.24766, -0.08333,
                        0.01963, -0.00326, 0.00034])
        assert np.allclose(A, ref, atol=5e-5)


class TestScalarArrayOutput:
    """Scalar in -> scalar out, like NFW."""

    def test_scalar(self):
        e = EinastoProfile(alpha=0.25, rho_0=1.0, r_s=1.0, order=20)
        assert np.isscalar(e.sigma(1.0))
        assert np.isscalar(e.deltasigma(1.0))
        assert np.isscalar(e.enclosed_mass_2D(1.0))

    def test_array(self):
        e = EinastoProfile(alpha=0.25, rho_0=1.0, r_s=1.0, order=20)
        out = e.sigma(np.array([0.5, 1.0, 2.0]))
        assert out.shape == (3,)
