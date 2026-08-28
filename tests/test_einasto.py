# tests/test_einasto.py

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import kv

from clenspy.halo.einasto import EinastoProfile, _expdisk_m2d_factor
from clenspy.halo.einasto_series import (
    _PK_TOL,
    _pk_direct_eval,
    _pk_filon,
    _pk_mb_contour,
    _pk_plateau_eval,
)
from clenspy.utils.special import expint_asymptotic, expn_fast

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
        # every non-anchor n (both above and below 3/2) uses the EinastoLowN
        # series backend; the exact anchors (n = 1/2, 1) use closed forms
        # and build no backend.
        e = EinastoProfile(alpha=1.0 / 0.7, rho_0=1.0, r_s=1.0)   # n = 0.7
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


class TestExplicitPowerSpectrumBranches:
    """The named `branch=` overrides of `power_spectrum` (not used by
    "auto" but exposed for comparison/research) -- see the class
    docstring's "closed"/"small_k"/"large_k" descriptions."""

    def test_closed_branch_is_n_independent(self):
        # the "closed" formula doesn't reference n at all -- any profile
        # must reproduce it exactly.
        e = EinastoProfile(alpha=1.0 / 2.5, rho_0=3.0, r_s=0.7)  # n = 2.5
        k = np.logspace(-2, 1.5, 20)
        kt = k * e.h
        pk_true = e.rho_0 * e.h ** 3 / (2.0 * np.pi * (1.0 + kt ** 2) ** 2)
        assert np.allclose(e.power_spectrum(k, branch="closed"), pk_true,
                            rtol=1e-14)

    def test_small_k_branch_matches_the_m0_closed_form_at_kt_zero(self):
        # NOTE: self.order is only built for n > 1.5 (see EinastoProfile
        # .__init__); the "small_k" branch needs an explicit `order=` for
        # n <= 1.5, since `self.order` is None there and `M = self.order
        # if order is None else order` then crashes with a bare TypeError
        # (`np.arange(0, None + 1)`) rather than a clear error -- a real
        # latent bug in the explicit-branch API, flagged separately.
        from scipy.special import gamma
        n = 0.7
        e = EinastoProfile(alpha=1.0 / n, rho_0=1.0, r_s=1.0)
        pk0 = e.power_spectrum(np.array([0.0]), branch="small_k", order=40)
        expected = e.rho_0 * n * e.h ** 3 / (4.0 * np.pi) * gamma(3 * n)
        assert np.allclose(pk0, expected, rtol=1e-12)

    def test_small_k_branch_agrees_with_auto_for_n_below_1(self):
        # for n < 1 "auto" dispatches through the same convergent series
        # (via a numerically-stabilised route); at small kt both must
        # agree closely.
        n = 0.7
        e = EinastoProfile(alpha=1.0 / n, rho_0=1.0, r_s=1.0)
        kt = np.array([1e-4, 1e-3, 1e-2])
        k = kt / e.h
        small_k = e.power_spectrum(k, branch="small_k", order=40)
        auto = e.power_spectrum(k)
        assert np.allclose(small_k, auto, rtol=1e-6)

    def test_small_k_branch_without_explicit_order_works_on_low_n(self):
        # self.order now always defaults to the constructor's own default
        # (previously None for n <= 1.5, which raised a raw TypeError here
        # -- see git history), so the small_k branch works without an
        # explicit order= and agrees with "auto".
        n = 0.7
        e = EinastoProfile(alpha=1.0 / n, rho_0=1.0, r_s=1.0)
        assert e.order == 100
        k = np.array([0.5, 1.0])
        assert np.allclose(
            e.power_spectrum(k, branch="small_k"), e.power_spectrum(k, branch="auto")
        )

    def test_large_k_branch_agrees_with_auto_direct_series_for_n_above_1(self):
        # "large_k" is a legacy Wright psi-function evaluator, independent
        # of the _pk_direct_eval cascade "auto" uses -- both are valid
        # large-kt expansions for n > 1 and must agree.
        n = 2.0
        e = EinastoProfile(alpha=1.0 / n, rho_0=1.0, r_s=1.0)
        kt = np.array([5.0, 20.0, 50.0])
        k = kt / e.h
        large_k = e.power_spectrum(k, branch="large_k")
        auto = e.power_spectrum(k)
        assert np.allclose(large_k, auto, rtol=1e-6)

    def test_large_k_branch_at_kt_zero_matches_gamma_3n(self):
        from scipy.special import gamma
        n = 2.0
        e = EinastoProfile(alpha=1.0 / n, rho_0=1.0, r_s=1.0)
        pk0 = e.power_spectrum(np.array([0.0]), branch="large_k")
        expected = e.rho_0 * n * e.h ** 3 / (4.0 * np.pi) * gamma(3 * n)
        assert np.allclose(pk0, expected, rtol=1e-10)

    def test_unknown_branch_raises(self):
        e = EinastoProfile(alpha=1.0, rho_0=1.0, r_s=1.0)
        with pytest.raises(ValueError, match="unknown branch"):
            e.power_spectrum(1.0, branch="bogus")


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

    def test_sigma_exact_limit_at_r_zero(self):
        # Sigma(0) = 2 rho_0 h Gamma(n+1) exactly (the x=0 special case in
        # einasto_lown.py, distinct from the x>0 series/E_nu evaluation);
        # DeltaSigma(0) = 0. Mixed array so both the x=0 and x>0 branches
        # run in the same call.
        from scipy.special import gamma as _gamma
        n = 0.7
        e = EinastoProfile(alpha=1.0 / n, rho_0=1.0, r_s=1.0)
        R = np.array([0.0, 1.0]) * e.h
        sigma = e.sigma(R)
        expected0 = 2.0 * e.rho_0 * e.h * _gamma(n + 1.0)
        assert sigma[0] == pytest.approx(expected0, rel=1e-12)
        assert sigma[1] == pytest.approx(self.REF[(n, 1.0)][0] * e.rho_0 * e.h,
                                          rel=1e-8)
        assert e.deltasigma(np.array([0.0])) == pytest.approx(0.0, abs=1e-300)


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


class TestPowerSpectrumHighN:
    """n > 1 cascade: plateau series (small kt) -> direct series
    (moderate/large kt), each carrying its own computable error estimate
    (einasto_series._pk_plateau_eval / _pk_direct_eval). No mpmath ground
    truth is used here: the (n, kt) points below were pinned down by
    inspecting each evaluator's OWN returned error estimate directly, and
    the check is that power_spectrum's auto dispatch reproduces exactly
    the evaluator with the smaller one -- "the whole reason the selection
    logic in einasto.py exists" (see the __main__ block at the bottom of
    einasto_series.py).
    """

    @pytest.mark.parametrize("n,kt", [
        (1.5, 1e-3), (1.5, 1e-2),
        (3.0, 1e-6), (3.0, 1e-5),
    ])
    def test_plateau_authoritative_at_small_kt(self, n, kt):
        vp, ep = _pk_plateau_eval(n, np.array([kt]))
        assert ep[0] < _PK_TOL  # confirms plateau is certified here
        e = EinastoProfile(alpha=1.0 / n, rho_0=1.0, r_s=1.0)
        pk = e.power_spectrum(kt / e.h)
        assert pk == pytest.approx(float(vp[0]) * e.rho_0 * e.h ** 3, rel=1e-10)

    @pytest.mark.parametrize("n,kt", [
        (1.5, 0.5), (1.5, 1.0), (1.5, 3.0), (1.5, 10.0), (1.5, 50.0),
        (3.0, 3e-3), (3.0, 1e-2), (3.0, 0.1), (3.0, 1.0), (3.0, 10.0),
        (3.0, 50.0),
    ])
    def test_direct_authoritative_at_moderate_large_kt(self, n, kt):
        vp, ep = _pk_plateau_eval(n, np.array([kt]))
        vd, ed = _pk_direct_eval(n, np.array([kt]))
        # confirms direct is certified (est <= 1e-8) and beats plateau here
        assert ed[0] <= 1e-8
        assert ed[0] <= ep[0]
        e = EinastoProfile(alpha=1.0 / n, rho_0=1.0, r_s=1.0)
        pk = e.power_spectrum(kt / e.h)
        assert pk == pytest.approx(float(vd[0]) * e.rho_0 * e.h ** 3, rel=1e-10)


class TestPowerSpectrumCrackFiller:
    """The narrow "crack" between the plateau and direct series' validity
    windows for n > 1 (neither evaluator's own error estimate meets
    _PK_TOL/1e-8 there), patched by Mellin-Barnes contour quadrature for
    n <= 3 and Filon quadrature for n > 3 (see the n > 1 branch of
    power_spectrum). Each crack filler is checked against whichever
    neighbouring evaluator is authoritative at the SAME kt -- the
    small-kt edge of the crack (plateau) or the large-kt edge (direct) --
    with generous tolerances set from that neighbour's own estimate.
    """

    def test_mb_contour_bridges_the_n3_gap(self):
        n = 3.0
        # kt = 1e-4: inside the plateau/direct gap for n = 3 (both
        # evaluators' own error estimates exceed 1e-8).
        kt_gap = np.array([1e-4])
        vp, ep = _pk_plateau_eval(n, kt_gap)
        vd, ed = _pk_direct_eval(n, kt_gap)
        assert min(ep[0], ed[0]) > 1e-8
        vmb = _pk_mb_contour(n, kt_gap)
        e = EinastoProfile(alpha=1.0 / n, rho_0=1.0, r_s=1.0)
        pk = e.power_spectrum(kt_gap[0] / e.h)
        assert pk == pytest.approx(float(vmb[0]) * e.rho_0 * e.h ** 3, rel=1e-10)
        # plateau's own estimate (~6e-8) is still fairly tight this close
        # to the gap's small-kt edge; MB must track it.
        assert vmb[0] == pytest.approx(vp[0], rel=1e-5)

        # kt = 3e-3: just outside the gap, direct is authoritative
        # (est ~ 5.5e-10); MB is not dispatched there but is still valid
        # per its own docstring (n up to 2.5-3) and must agree with direct.
        kt_out = np.array([3e-3])
        vd2, ed2 = _pk_direct_eval(n, kt_out)
        assert ed2[0] < 1e-8
        vmb2 = _pk_mb_contour(n, kt_out)
        assert vmb2[0] == pytest.approx(vd2[0], rel=1e-6)

    def test_filon_bridges_the_n4_gap(self):
        n = 4.0
        # kt = 1e-4: inside the plateau/direct gap for n = 4 (n > 3, so
        # Filon replaces the MB contour as the crack filler).
        kt_gap = np.array([1e-4])
        vp, ep = _pk_plateau_eval(n, kt_gap)
        vd, ed = _pk_direct_eval(n, kt_gap)
        assert min(ep[0], ed[0]) > 1e-8
        vf = _pk_filon(n, kt_gap)
        e = EinastoProfile(alpha=1.0 / n, rho_0=1.0, r_s=1.0)
        pk = e.power_spectrum(kt_gap[0] / e.h)
        assert pk == pytest.approx(float(vf[0]) * e.rho_0 * e.h ** 3, rel=1e-10)
        # direct's own estimate is still moderately good here (~2e-7);
        # Filon must track it.
        assert vf[0] == pytest.approx(vd[0], rel=5e-5)

        # Deep in the plateau's validity window (est ~ 3e-29), Filon is
        # not dispatched, but the standalone quadrature must still
        # reproduce the plateau value there.
        kt_in = np.array([1e-7])
        vp2, ep2 = _pk_plateau_eval(n, kt_in)
        assert ep2[0] < 1e-20
        vf2 = _pk_filon(n, kt_in)
        assert vf2[0] == pytest.approx(vp2[0], rel=1e-6)

    def test_n10_power_spectrum_finite_positive_and_monotonic(self):
        # Large-n turnover regime the module docstring calls out
        # ("<= 8.5e-8 for n = 10"). Direct alone certifies the whole
        # physical range at this n (its own estimate stays ~1e-14 from
        # kt ~ 1e-15 to kt ~ 1e6), so there is no second analytic series
        # to cross-check against here; instead assert the full
        # auto-dispatch cascade produces a sane P(k) -- finite, positive,
        # and monotonically non-increasing, with no sign flips or wild
        # jumps -- both over an ordinary k grid and over kt in [20, 100].
        e = EinastoProfile(alpha=0.1, rho_0=1.0, r_s=1.0)  # n = 10

        k = np.logspace(-3, 3, 200)
        pk = e.power_spectrum(k)
        assert np.all(np.isfinite(pk))
        assert np.all(pk > 0)
        assert np.all(np.diff(pk) <= 0)

        kt = np.linspace(20.0, 100.0, 20)
        pk_kt = e.power_spectrum(kt / e.h)
        assert np.all(np.isfinite(pk_kt))
        assert np.all(pk_kt > 0)
        assert np.all(np.diff(pk_kt) <= 0)


class TestExpDiskM2DFactor:
    """M_2D/(4 pi rho_0 h^3) for the exact n = 1 profile (used inside
    EinastoProfile.enclosed_mass_2D): 2 - x^2 K_2(x), with a small-x
    Taylor branch below x = 0.1 that avoids the 2-vs-x^2K_2(x)
    cancellation there."""

    @pytest.mark.parametrize("x,ref", [
        # mpmath dps=50 evaluation of 2 - x^2 K_2(x) directly, independent
        # of the module's own scipy.special.kv call and Taylor coefficients.
        (0.01, 4.9993161058937644776878351447052552095396434759002e-05),
        (0.15, 0.011074743385586977724239725125765712321572292304681),
    ])
    def test_matches_mpmath(self, x, ref):
        got = _expdisk_m2d_factor(np.array([x]))[0]
        assert got == pytest.approx(ref, rel=1e-9)

    def test_taylor_branch_matches_naive_closed_form_near_boundary(self):
        # At x = 0.099 (just inside the Taylor branch, x < 0.1) the fp64
        # cancellation in 2 - x^2 K_2(x) has not yet become severe, so a
        # direct scipy.special.kv evaluation is still trustworthy there,
        # giving an independent check that the Taylor branch continuously
        # extends the direct formula (no jump at the x = 0.1 switch).
        x = 0.099
        taylor = _expdisk_m2d_factor(np.array([x]))[0]
        naive = 2.0 - x ** 2 * kv(2, x)
        assert taylor == pytest.approx(naive, rel=1e-8)

    def test_continuous_across_taylor_branch_boundary(self):
        # Straddle x = 0.1 tightly enough that the function's own smooth
        # variation is negligible, so any branch-switch discontinuity
        # would show up cleanly.
        lo = _expdisk_m2d_factor(np.array([0.1 - 1e-6]))[0]
        hi = _expdisk_m2d_factor(np.array([0.1 + 1e-6]))[0]
        assert lo == pytest.approx(hi, rel=1e-4)


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


class TestLensingObservablePassthroughs:
    """`fourier`, `convergence`, `shear` -- thin wrappers around
    `power_spectrum`/`sigma`/`deltasigma`."""

    def test_fourier_is_4pi_squared_times_power_spectrum(self):
        e = EinastoProfile(alpha=2.0, rho_0=1.0, r_s=1.0)  # n = 0.5
        k = np.array([0.1, 1.0, 5.0])
        assert np.allclose(e.fourier(k), (4 * np.pi) ** 2 * e.power_spectrum(k))

    def test_convergence_is_sigma_over_sigma_crit(self):
        e = EinastoProfile(alpha=1.0, rho_0=1.0, r_s=1.0)  # n = 1
        R = np.array([0.3, 1.0, 2.0])
        sigma_crit = 3.5
        assert np.allclose(e.convergence(R, sigma_crit), e.sigma(R) / sigma_crit)

    def test_shear_is_deltasigma_over_sigma_crit(self):
        e = EinastoProfile(alpha=1.0, rho_0=1.0, r_s=1.0)  # n = 1
        R = np.array([0.3, 1.0, 2.0])
        sigma_crit = 3.5
        assert np.allclose(e.shear(R, sigma_crit), e.deltasigma(R) / sigma_crit)
