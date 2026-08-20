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

    def test_small_n_uses_lown_backend(self):
        # n <= 3/2: generic non-integer n uses the EinastoLowN series
        # backend; the exact anchors (n = 1/2, 1) use closed forms and
        # build no backend.
        e = EinastoProfile(alpha=1.0 / 0.7, rho_0=1.0, r_s=1.0)   # n = 0.7
        assert not e._series
        assert e._lown is not None
        assert np.isfinite(e.sigma(1.0))
        assert np.isfinite(e.deltasigma(1.0))
        g = EinastoProfile(alpha=2.0, rho_0=1.0, r_s=1.0)         # n = 0.5
        assert g._lown is None

    def test_enclosed_mass_2d_low_n(self):
        # previously raised NotImplementedError for n <= 3/2; now exact
        e = EinastoProfile(alpha=2.0, rho_0=1.0, r_s=1.0)   # n = 0.5
        x2 = (1.0 / e.h) ** 2
        expected = np.pi ** 1.5 * e.rho_0 * e.h ** 3 * (-np.expm1(-x2))
        assert e.enclosed_mass_2D(1.0) == pytest.approx(expected, rel=1e-13)


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
        assert np.allclose(e.sigma(R), sigma_true, rtol=1e-14)

    def test_gaussian_deltasigma_matches_closed_form(self):
        # n = 1/2 now dispatches to the exact Gaussian closed form
        # (expm1-safe), including deep in the core (issue #3).
        e = EinastoProfile(alpha=2.0, rho_0=1.0, r_s=1.0)  # n = 0.5
        R = np.array([0.001, 0.01, 0.1, 0.3, 0.6, 1.0, 1.5])
        sigma_true = e.rho_0 * e.h * np.sqrt(np.pi) * np.exp(-((R / e.h) ** 2))
        sigmabar_true = (
            e.rho_0 * e.h ** 3 * np.sqrt(np.pi) / R ** 2
            * -np.expm1(-((R / e.h) ** 2))
        )
        deltasigma_true = sigmabar_true - sigma_true
        assert np.allclose(e.deltasigma(R), deltasigma_true, rtol=1e-12)

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


class TestLowNSeries:
    """n <= 3/2 backend (einasto_lown): residue series + resonance pairing
    + E_nu hybrid, against mpmath Abel/cap quadrature references (dps=50).

    Reference values are Sigma/(rho_0 h) and DeltaSigma/(rho_0 h) at
    x = R/h; the grid spans both dispatch zones (series and E_nu) and
    includes the resonant indices n = 6/5 and 4/3 (pole collisions
    k/n = 2j-1) plus generic n = 0.7 and 1.45.
    """

    # (n, x) -> (Sigma/(rho0 h), DeltaSigma/(rho0 h)), mpmath dps=50
    REF = {
        (0.7, 0.01): (1.817003682349393, 0.00013566122299910375),
        (0.7, 0.1): (1.7942828313351356, 0.01115107349413715),
        (0.7, 0.5): (1.4368739878387724, 0.17049853897851772),
        (0.7, 1.0): (0.8524592227540959, 0.37954573350492177),
        (0.7, 2.0): (0.18133875749283962, 0.4392197551210647),
        (0.7, 5.0): (0.00015820557250039238, 0.11702472276497673),
        (0.7, 10.0): (9.097489411246984e-12, 0.02930160370231706),
        (0.7, 25.0): (3.8507309501061355e-43, 0.00468825659390395),
        (1.2, 0.01): (2.202879520940828, 0.00033448256935875837),
        (1.2, 0.1): (2.1719431718956708, 0.013965283133629375),
        (1.2, 0.5): (1.8859746423394688, 0.12734022153324065),
        (1.2, 1.0): (1.4918240524658115, 0.2595055631510721),
        (1.2, 2.0): (0.8851037758906449, 0.4018520870902365),
        (1.2, 5.0): (0.1738827516107872, 0.35833797619810565),
        (1.2, 10.0): (0.01244377998337069, 0.16007710773742176),
        (1.2, 25.0): (8.316375992276452e-06, 0.028536854353221047),
        (1.3333333333333333, 0.01): (2.3804161399219796, 0.0003896285914460175),
        (1.3333333333333333, 0.1): (2.3486490219513603, 0.0140801547622757),
        (1.3333333333333333, 0.5): (2.080008353407883, 0.11843351843645963),
        (1.3333333333333333, 1.0): (1.7169745751673855, 0.2394806372573428),
        (1.3333333333333333, 2.0): (1.1367335362818212, 0.3848172623773606),
        (1.3333333333333333, 5.0): (0.33165226589426905, 0.4112805518853792),
        (1.3333333333333333, 10.0): (0.04899861400775265, 0.23511437829439555),
        (1.3333333333333333, 25.0): (0.0003186951131333833, 0.05078638000932623),
        (1.45, 0.01): (2.5674397976720695, 0.0004344708746326117),
        (1.45, 0.1): (2.535283477544214, 0.014049459798396214),
        (1.45, 0.5): (2.280641056140776, 0.11146826930332475),
        (1.45, 1.0): (1.9410226688496548, 0.22433911537244164),
        (1.45, 2.0): (1.383611267580755, 0.3699940018753547),
        (1.45, 5.0): (0.5177301734465429, 0.4472382811776997),
        (1.45, 10.0): (0.11744123954601715, 0.30747388948159315),
        (1.45, 25.0): (0.0026922828410213093, 0.08404074961202149),
        # n > 3/2 (backend replaces the legacy Catalan path, whose
        # DeltaSigma truncation error was 30-200%); n = 5, 10 are exact
        # resonances at every j (integer n, eps -> 0 pairing limits)
        (2.5, 0.01): (6.645065925954821, 0.0006519480921973524),
        (2.5, 0.1): (6.615935299296373, 0.011752149899686294),
        (2.5, 0.5): (6.447367214343599, 0.07160252787073458),
        (2.5, 1.0): (6.232718303164335, 0.1425955901686035),
        (2.5, 2.0): (5.8371075907780705, 0.2633855954855229),
        (2.5, 5.0): (4.896035505122096, 0.5095919123401352),
        (2.5, 10.0): (3.8261867920870047, 0.723108030922803),
        (2.5, 20.0): (2.563780645773324, 0.8682322148338406),
        (5.0, 0.01): (239.9983841725841, 0.0005813146521127401),
        (5.0, 0.1): (239.97973686743342, 0.007112110294466198),
        (5.0, 0.5): (239.88966049900083, 0.037735207369833726),
        (5.0, 1.0): (239.7765565128186, 0.07533058890140515),
        (5.0, 2.0): (239.55534736715825, 0.14742609754638938),
        (5.0, 5.0): (238.92911054717098, 0.3457576465268581),
        (5.0, 10.0): (237.97245028336874, 0.6389627892832612),
        (5.0, 20.0): (236.26001246829702, 1.1453237756578445),
        (10.0, 0.01): (7257599.998961903, 0.00035509463438595243),
        (10.0, 0.1): (7257599.988871102, 0.0037696072161091374),
        (10.0, 0.5): (7257599.942953309, 0.019160545633287864),
        (10.0, 1.0): (7257599.885478457, 0.038307008427562725),
        (10.0, 2.0): (7257599.771157261, 0.07620820237340951),
        (10.0, 5.0): (7257599.432902688, 0.1876462663268842),
        (10.0, 10.0): (7257598.880313727, 0.36853483004427656),
        (10.0, 20.0): (7257597.801987579, 0.7193285595729665),
    }

    @pytest.mark.parametrize("n", [0.7, 1.2, 4.0 / 3.0, 1.45, 2.5, 5.0, 10.0])
    def test_sigma_deltasigma_vs_mpmath(self, n):
        e = EinastoProfile(alpha=1.0 / n, rho_0=1.0, r_s=1.0)
        xs = np.array([x for (nn, x) in self.REF if nn == n])
        R = xs * e.h
        sig_ref = np.array([self.REF[(n, x)][0] for x in xs]) * e.h
        ds_ref = np.array([self.REF[(n, x)][1] for x in xs]) * e.h
        assert np.allclose(e.sigma(R), sig_ref, rtol=1e-8)
        assert np.allclose(e.deltasigma(R), ds_ref, rtol=1e-8)

    def test_enclosed_mass_2d_consistency(self, ):
        # M2D = pi R^2 (Sigma + DeltaSigma) against the reference table
        n = 1.2
        e = EinastoProfile(alpha=1.0 / n, rho_0=1.0, r_s=1.0)
        for x in (0.1, 1.0, 10.0):
            R = x * e.h
            m2d_ref = np.pi * R ** 2 * sum(self.REF[(n, x)]) * e.h
            assert e.enclosed_mass_2D(R) == pytest.approx(m2d_ref, rel=1e-8)

    def test_near_integer_resonance_continuity(self):
        # n = 1 + 1e-7: every odd k is nearly resonant (eps ~ 1e-7); the
        # paired evaluation must stay continuous with the exact n = 1
        # Bessel closed form. Unpaired evaluation loses ~7 digits here.
        e = EinastoProfile(alpha=1.0 / (1.0 + 1e-7), rho_0=1.0, r_s=1.0)
        b = EinastoProfile(alpha=1.0, rho_0=1.0, r_s=1.0)
        R = np.array([0.05, 0.3, 1.0, 3.0]) * e.h
        assert np.allclose(e.deltasigma(R), b.deltasigma(R * b.h / e.h),
                           rtol=1e-5)

    def test_deltasigma_core_limit(self):
        # DeltaSigma -> 0 smoothly as R -> 0 (no cumtrapz breakdown;
        # issue #3's R < 0.05 h regime)
        e = EinastoProfile(alpha=1.0 / 0.7, rho_0=1.0, r_s=1.0)
        R = np.array([1e-4, 1e-3, 1e-2]) * e.h
        ds = e.deltasigma(R)
        assert np.all(ds > 0)
        assert np.all(np.diff(ds) > 0)
        assert ds[0] < 1e-5 * e.deltasigma(e.h)

    def test_exponential_deltasigma_small_x_taylor(self):
        # n = 1 closed form self-cancels at small x; Taylor branch below
        # x = 0.1. Reference: mpmath 8/x^2 - 4K2 - 2xK1 at dps 40.
        e = EinastoProfile(alpha=1.0, rho_0=1.0, r_s=1.0)
        refs = {
            0.05: 0.004204308322748063,    # mpmath dps=40
            0.02: 0.0008556601672464003,
        }
        for x, val in refs.items():
            assert e.deltasigma(x * e.h) == pytest.approx(val * e.h, rel=1e-9)


class TestPowerSpectrumLowN:
    """Analytic P(k) dispatch for 0 < n < 1 (Kummer / plain convergent /
    optimally-truncated asymptotic series, GL bridge) vs mpmath quadrature
    of the master integral (dps=40). Values are P(kt) with rho_0 = h = 1;
    n = 0.45 exercises the oscillating (non-positive-definite) regime,
    n = 0.97 the near-integer regime where the Kummer build is disabled
    and GL bridges a narrow kt window."""

    PKREF = {
        (0.45, 0.1): 0.031844457990516314,
        (0.45, 0.5): 0.03026380750897322,
        (0.45, 1.0): 0.025795030061562037,
        (0.45, 1.6): 0.018436694644710133,
        (0.45, 2.5): 0.00814310907999036,
        (0.45, 4.0): 0.0006954999534419217,
        (0.45, 7.0): -1.6863776606589706e-05,
        (0.45, 12.0): -5.74652803986715e-07,
        (0.45, 25.0): -1.1153977545821377e-08,
        (0.45, 60.0): -1.1292611036658844e-10,
        (0.7, 0.1): 0.05798610437009312,
        (0.7, 0.5): 0.05114651713723428,
        (0.7, 1.0): 0.035167280710398076,
        (0.7, 1.6): 0.017476045026332987,
        (0.7, 2.5): 0.004764436430386136,
        (0.7, 4.0): 0.0006266970530007417,
        (0.7, 7.0): 4.354291276310888e-05,
        (0.7, 12.0): 3.552655343274688e-06,
        (0.7, 25.0): 1.2819215075659588e-07,
        (0.7, 60.0): 2.581522985581913e-09,
        (0.9, 0.1): 0.10925349698751724,
        (0.9, 0.5): 0.08227120864479952,
        (0.9, 1.0): 0.039903132135854356,
        (0.9, 1.6): 0.014132728511247193,
        (0.9, 2.5): 0.003424343977041789,
        (0.9, 4.0): 0.0005801569798791829,
        (0.9, 7.0): 6.033344579095651e-05,
        (0.9, 12.0): 6.549044832987129e-06,
        (0.9, 25.0): 3.164194929676816e-07,
        (0.9, 60.0): 8.57621210602538e-09,
        (0.97, 0.1): 0.13986860234714463,
        (0.97, 0.5): 0.09585493730469372,
        (0.97, 1.0): 0.03998035546438639,
        (0.97, 1.6): 0.013003784057819713,
        (0.97, 2.5): 0.003135275635216576,
        (0.97, 4.0): 0.0005596122357434227,
        (0.97, 7.0): 6.289802420805039e-05,
        (0.97, 12.0): 7.2949925642853385e-06,
        (0.97, 25.0): 3.8012238537370375e-07,
        (0.97, 60.0): 1.1141707418134902e-08,
    }

    @pytest.mark.parametrize("n", [0.45, 0.7, 0.9, 0.97])
    def test_pk_vs_mpmath(self, n):
        e = EinastoProfile(alpha=1.0 / n, rho_0=1.0, r_s=1.0)
        kts = np.array([kt for (nn, kt) in self.PKREF if nn == n])
        ref = np.array([self.PKREF[(n, kt)] for kt in kts]) \
            * e.rho_0 * e.h ** 3
        pk = e.power_spectrum(kts / e.h)
        assert np.allclose(pk, ref, rtol=1e-8)


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
