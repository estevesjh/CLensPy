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

    def test_rejects_nonpositive_n(self):
        with pytest.raises(ValueError, match="must be positive"):
            EinastoProfile(alpha=-1.0, rho_0=1.0, r_s=1.0)

    def test_small_n_uses_numerical_fallback(self):
        # n <= 3/2 no longer raises; sigma/deltasigma/power_spectrum fall
        # back to the numerical (Abel/FFTLog) path instead of the Catalan
        # series. See TestNumericalFallback for accuracy checks.
        e = EinastoProfile(alpha=2.0, rho_0=1.0, r_s=1.0)   # n = 0.5
        assert not e._series
        assert np.isfinite(e.sigma(1.0))
        assert np.isfinite(e.deltasigma(1.0))

    def test_enclosed_mass_2d_has_no_numerical_fallback(self):
        e = EinastoProfile(alpha=2.0, rho_0=1.0, r_s=1.0)   # n = 0.5
        with pytest.raises(NotImplementedError):
            e.enclosed_mass_2D(1.0)


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


class TestNumericalFallback:
    """n <= 3/2: exact anchors (n=1/2 Gaussian, n=1 exponential) and the
    general numerical (Abel/FFTLog) path, validated end to end from
    power_spectrum (Fourier space) through to deltasigma (real space)."""

    # ------------------------------------------------------------------
    # n = 1/2: exact Gaussian, rho(r) = rho_0 exp(-(r/h)^2)
    # ------------------------------------------------------------------
    def test_gaussian_power_spectrum_matches_closed_form(self):
        e = EinastoProfile(alpha=2.0, rho_0=1.0, r_s=1.0)  # n = 0.5
        k = np.logspace(-2, 1.5, 30)
        pk = e.power_spectrum(k)
        pk_true = e.rho_0 * e.h ** 3 / (16.0 * np.sqrt(np.pi)) * np.exp(-((k * e.h) ** 2) / 4.0)
        assert np.allclose(pk, pk_true, rtol=1e-12)

    def test_gaussian_sigma_matches_closed_form(self):
        e = EinastoProfile(alpha=2.0, rho_0=1.0, r_s=1.0)  # n = 0.5
        R = np.array([0.1, 0.3, 0.6, 1.0, 1.5])
        sigma_true = e.rho_0 * e.h * np.sqrt(np.pi) * np.exp(-((R / e.h) ** 2))
        assert np.allclose(e.sigma(R), sigma_true, rtol=1e-10)

    def test_gaussian_deltasigma_matches_closed_form(self):
        e = EinastoProfile(alpha=2.0, rho_0=1.0, r_s=1.0)  # n = 0.5
        R = np.array([0.1, 0.3, 0.6, 1.0, 1.5])
        sigma_true = e.rho_0 * e.h * np.sqrt(np.pi) * np.exp(-((R / e.h) ** 2))
        sigmabar_true = (
            e.rho_0 * e.h ** 3 * np.sqrt(np.pi) / R ** 2
            * (1.0 - np.exp(-((R / e.h) ** 2)))
        )
        deltasigma_true = sigmabar_true - sigma_true
        assert np.allclose(e.deltasigma(R), deltasigma_true, rtol=1e-2)

    # ------------------------------------------------------------------
    # n = 1: exact exponential, rho(r) = rho_0 exp(-r/h)
    # ------------------------------------------------------------------
    def test_exponential_sigma_deltasigma_match_bessel_closed_form(self):
        e = EinastoProfile(alpha=1.0, rho_0=1.0, r_s=1.0)  # n = 1
        R = np.array([0.3, 0.7, 1.0, 2.0, 4.0])
        x = R / e.h
        sigma_true = 2.0 * e.rho_0 * R * kv(1, x)
        deltasigma_true = e.rho_0 * e.h * (8.0 / x ** 2 - 4.0 * kv(2, x) - 2.0 * x * kv(1, x))

        assert np.allclose(e.sigma(R), sigma_true, rtol=1e-13)
        assert np.allclose(e.deltasigma(R), deltasigma_true, rtol=1e-13)

    def test_exponential_power_spectrum_matches_closed_form(self):
        e = EinastoProfile(alpha=1.0, rho_0=1.0, r_s=1.0)  # n = 1
        k = np.logspace(-2, 1.5, 30)
        pk = e.power_spectrum(k)
        kt = k * e.h
        pk_true = e.rho_0 * e.h ** 3 / (2.0 * np.pi * (1.0 + kt ** 2) ** 2)
        assert np.allclose(pk, pk_true, rtol=1e-12)

    # ------------------------------------------------------------------
    # Generic 0 < n < 1 (no exact anchor): power_spectrum cross-checked
    # against an independent brute-force quadrature of the Hankel integral.
    # ------------------------------------------------------------------
    def test_power_spectrum_below_one_matches_brute_force_quadrature(self):
        e = EinastoProfile(alpha=1.0 / 0.7, rho_0=1.0, r_s=1.0)  # n = 0.7

        def brute_force_Fk(k):
            f = lambda r: r ** 2 * e.density(r) * np.sinc(k * r / np.pi)
            val, _ = quad(f, 0.0, e.h * 40.0 ** e.n_index, limit=400)
            return 4.0 * np.pi * val

        k = np.array([0.05, 0.3, 1.0, 3.0, 8.0])
        pk_code = e.power_spectrum(k)
        pk_brute = np.array([brute_force_Fk(kk) for kk in k]) / (4.0 * np.pi) ** 2
        assert np.allclose(pk_code, pk_brute, rtol=2e-3)

    # ------------------------------------------------------------------
    # Full pipeline: power_spectrum (Fourier) -> xi (FFTLog) -> Sigma (Abel
    # projection) -> DeltaSigma, cross-checked against the class's own
    # sigma()/deltasigma() and the exact Gaussian closed form throughout.
    # ------------------------------------------------------------------
    def test_fourier_to_deltasigma_pipeline_gaussian(self):
        import mcfit

        from clenspy.utils.integrate import (
            compute_sigma_grid,
            sigma_to_deltasigma_cumtrapz,
        )
        from clenspy.utils.interpolate import make_log_interpolation

        e = EinastoProfile(alpha=2.0, rho_0=1.0, r_s=1.0)  # n = 0.5, exact Gaussian
        norm = (4.0 * np.pi) ** 2

        # Fourier: P(k) from the class, in its own P = rho_tilde/(4pi)^2 convention.
        kvec = np.logspace(-5, 6, 4096)
        Pk = e.power_spectrum(kvec)

        # xi(r) via FFTLog (once) should recover density(r)/(4pi)^2 exactly
        # (P2xi is the inverse of the xi2P transform power_spectrum is built
        # from). Build the interpolator once and reuse it below, rather than
        # re-running the FFTLog transform on every Abel-integrand call.
        r_fftlog, xi_r_fftlog = mcfit.P2xi(kvec, lowring=True)(Pk)
        xi_interp = make_log_interpolation(r_fftlog, xi_r_fftlog)

        rvals = np.logspace(-2, np.log10(4.0), 40)
        xi_num = xi_interp(rvals)
        xi_true = e.density(rvals) / norm
        assert np.max(np.abs(xi_num - xi_true) / xi_true) < 1e-3

        # Sigma(R) via the same Abel projection TwoHaloTerm uses, applied to
        # the FFTLog-recovered xi(r).
        def xi_func(r, z):
            return xi_interp(r)

        Rvals = np.logspace(-2, np.log10(1.5), 40)
        sigma_num = compute_sigma_grid(
            xi_func, Rvals, np.array([0.0]), method="quad_vec", rmax_integral=4.0
        ).ravel() * norm
        sigma_true = e.sigma(Rvals)
        assert np.max(np.abs(sigma_num - sigma_true) / sigma_true) < 1e-3

        # DeltaSigma(R) via cumtrapz on that Sigma(R) grid; the innermost
        # points are unreliable for cumulative-trapz (see
        # sigma_to_deltasigma_cumtrapz's docstring), so only check R >~ h.
        deltasigma_num = sigma_to_deltasigma_cumtrapz(Rvals, sigma_num / norm) * norm
        deltasigma_true = e.deltasigma(Rvals)
        outer = Rvals > 1.0 * e.h
        rel = np.abs(deltasigma_num[outer] - deltasigma_true[outer]) / deltasigma_true[outer]
        assert rel.max() < 1e-2


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
