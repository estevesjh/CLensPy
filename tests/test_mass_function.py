r"""The variance, the growth factor, and the Tinker (2008) mass function.

Most of these are identities rather than tolerances, because a power-law
:math:`P(k) \propto k^{n}` has an exactly known variance slope,
:math:`d\ln\sigma^2/d\ln R = -(3+n)`, and Einstein--de Sitter has
:math:`D(z) = a` exactly. Where a tolerance is unavoidable the reference is
an independent implementation (`pyccl`, `cluster_toolkit`) or a finite
difference of the very quantity being differentiated.

The remaining tests guard the transcription traps this port carries: the
two different :math:`\delta_c`, the :math:`\Omega_m` in the mass axis, the
:math:`k \le 20/R` truncation, and the Leibniz boundary term.
"""

import numpy as np
import pytest

from clenspy.cosmology.fiducial import fiducial_cosmology
from clenspy.cosmology.growth import growth_factor, growth_unnormalised
from clenspy.cosmology.halo_mass_function import (
    DELTA_C_TINKER,
    PI_FORTRAN,
    RHO_FACT,
    TINKER08_TABLE2,
    TinkerMassFunction,
    consumed_mask,
)
from clenspy.cosmology.sigma import (
    KCUT_COEF,
    LNR1,
    LNR2,
    NR,
    STEP,
    LinearPk,
    SigmaGrid,
    lnr_grid,
)
from clenspy.utils.special import TOPHAT_SERIES_CUTOFF, tophat_dw, tophat_w

#: A pure power law, so sigma^2 has an exact analytic slope. The k range is
#: wide enough that the R values tested sit far from both ends.
K = np.logspace(-6.0, 4.0, 900)
N_SPEC = -1.5


def power_law_grid(n=N_SPEC):
    return SigmaGrid(LinearPk(K, 2.0e4 * K**n))


# -- the top-hat window -----------------------------------------------------


def test_tophat_w_is_one_at_the_origin():
    assert tophat_w(0.0).item() == pytest.approx(1.0)
    assert tophat_dw(0.0).item() == pytest.approx(0.0)


def test_tophat_series_and_closed_form_agree_at_the_cutoff():
    """The branch seam must be invisible; this is what fixes the cutoff."""
    x = TOPHAT_SERIES_CUTOFF
    closed_w = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    closed_dw = (3.0 * (x * x * np.sin(x) - 3 * np.sin(x) + 3 * x * np.cos(x))
                 / x**4)
    assert tophat_w(x).item() == pytest.approx(closed_w, rel=1e-12)
    assert tophat_dw(x).item() == pytest.approx(closed_dw, rel=1e-8)


def test_tophat_dw_is_the_derivative_of_tophat_w():
    for x0 in (0.05, 0.5, 2.0, 7.0, 30.0):
        fd = (tophat_w(x0 + 1e-6) - tophat_w(x0 - 1e-6)).item() / 2e-6
        assert tophat_dw(x0).item() == pytest.approx(fd, rel=1e-6, abs=1e-12)


def test_tophat_w_oscillates_and_decays():
    x = np.linspace(0.1, 60.0, 2000)
    w = tophat_w(x)
    assert np.any(w < 0.0)                       # it does go negative
    assert abs(w[-1]) < 0.01                     # and decays as 1/x^2


# -- the growth factor ------------------------------------------------------


def test_growth_is_exactly_the_scale_factor_in_einstein_de_sitter():
    r"""EdS has :math:`D = a` in closed form -- the quadrature's own test."""
    eds = fiducial_cosmology(Om0=1.0)
    for z in (0.1, 0.5, 1.0, 3.0, 10.0):
        assert growth_factor(z, eds).item() == pytest.approx(
            1.0 / (1.0 + z), rel=1e-10
        )


def test_growth_is_one_at_z_zero():
    assert growth_factor(0.0).item() == 1.0


def test_growth_falls_monotonically_with_redshift():
    z = np.linspace(0.0, 5.0, 30)
    d = growth_factor(z)
    assert np.all(np.diff(d) < 0.0)
    assert np.all(d > 0.0) and np.all(d <= 1.0)


def test_unnormalised_growth_tends_to_a_at_high_redshift():
    """The 5*Om/2 prefactor is what makes D+ -> a in matter domination."""
    z = 300.0
    assert growth_unnormalised(z).item() * (1.0 + z) == pytest.approx(
        1.0, rel=1e-3
    )


def test_lambda_suppresses_growth_relative_to_eds():
    lcdm, eds = fiducial_cosmology(Om0=0.3), fiducial_cosmology(Om0=1.0)
    # normalised to 1 today, LCDM must have grown *less* since a given z,
    # i.e. its D at that z is larger relative to a
    for z in (0.5, 1.0, 2.0):
        assert growth_factor(z, lcdm).item() > growth_factor(z, eds).item()


def test_growth_matches_the_frozen_reference_value():
    """D(z=2) = 0.421446 for Om0 = 0.3 (cluster-lensing-cov frozen grid)."""
    assert growth_factor(2.0, fiducial_cosmology(Om0=0.3)).item() == (
        pytest.approx(0.421446, abs=1e-6)
    )


def test_growth_rejects_z_at_or_below_minus_one():
    with pytest.raises(ValueError, match="exceed -1"):
        growth_factor(-1.0)


# -- sigma^2: the power-law identity ---------------------------------------


def test_sigma2_slope_is_minus_three_plus_n_for_a_power_law():
    r""":math:`\sigma^2 \propto R^{-(3+n)}` exactly, for :math:`P \propto k^n`.

    The single strongest check on the quadrature, the window, *and* the
    derivative at once: all three must be right for the slope to come out
    at the analytic value.
    """
    for n in (-1.0, -1.5, -2.0):
        grid = power_law_grid(n)
        for r in (0.5, 2.0, 8.0):
            got = grid.dlnsigma2_dlnr(r, truncate=False)
            assert got == pytest.approx(-(3.0 + n), rel=2e-3), (n, r)


def test_sigma2_amplitude_scales_linearly_with_pk():
    grid_a = SigmaGrid(LinearPk(K, 1.0e4 * K**N_SPEC))
    grid_b = SigmaGrid(LinearPk(K, 3.0e4 * K**N_SPEC))
    assert grid_b.sigma2(8.0) / grid_a.sigma2(8.0) == pytest.approx(3.0)


def test_sigma_falls_monotonically_with_radius():
    grid = power_law_grid()
    r = np.logspace(-1.0, 1.5, 25)
    s = np.array([grid.sigma(ri) for ri in r])
    assert np.all(np.diff(s) < 0.0)


def test_derivative_under_the_integral_matches_finite_differences():
    """Untruncated, where no boundary term exists to confuse the check."""
    grid = power_law_grid()
    for r in (0.5, 2.0, 8.0):
        h = 1e-5
        fd = (grid.sigma2(r * np.exp(h), truncate=False)
              - grid.sigma2(r * np.exp(-h), truncate=False)) / (2 * h)
        assert grid.dsigma2_dlnr(r, truncate=False) == pytest.approx(
            fd, rel=1e-7
        )


def test_the_quadrature_is_converged_in_panel_order():
    """24 vs 48 points per panel must not move the answer."""
    pk = LinearPk(K, 2.0e4 * K**N_SPEC)
    a, b = SigmaGrid(pk, nquad=24), SigmaGrid(pk, nquad=48)
    for r in (0.1, 1.0, 8.0):
        assert a.sigma2(r) == pytest.approx(b.sigma2(r), rel=1e-12)


def test_sigma2_agrees_with_cluster_toolkit():
    """An independent implementation of the same untruncated integral."""
    ct = pytest.importorskip("cluster_toolkit")
    pk_vals = 2.0e4 * K**N_SPEC
    grid = SigmaGrid(LinearPk(K, pk_vals))
    for r in (0.5, 2.0, 8.0):
        theirs = ct.peak_height.sigma2_at_R(r, K, pk_vals)
        mine = grid.sigma2(r, truncate=False)
        assert mine == pytest.approx(theirs, rel=1e-4), r


# -- the truncation and its boundary term ----------------------------------


def test_truncation_is_inactive_when_twenty_over_r_exceeds_k_max():
    """It can only cut what the table contains."""
    grid = power_law_grid()
    r_small = KCUT_COEF / K[-1] * 0.5      # 20/R > k_max
    assert grid.sigma2(r_small, truncate=True) == (
        pytest.approx(grid.sigma2(r_small, truncate=False), rel=1e-14)
    )


def test_truncation_lowers_sigma2_when_active():
    """Removing power can only reduce the variance."""
    grid = power_law_grid()
    for r in (1.0, 8.0, 30.0):
        assert grid.sigma2(r, truncate=True) < grid.sigma2(r,
                                                           truncate=False)


def test_the_leibniz_boundary_term_is_present_and_signed():
    r"""Without it, :math:`d\sigma^2/d\ln R` is wrong when truncating.

    Check it the only way that cannot be circular: finite-difference the
    *truncated* sigma^2, which moves its own upper limit with R, and
    compare against the analytic derivative including the boundary term.
    """
    grid = power_law_grid()
    for r in (1.0, 8.0):
        h = 1e-6
        fd = (grid.sigma2(r * np.exp(h), truncate=True)
              - grid.sigma2(r * np.exp(-h), truncate=True)) / (2 * h)
        with_term = grid.dsigma2_dlnr(r, truncate=True)
        # the finite difference of the truncated sigma^2 moves its own upper
        # limit with R, so matching it is only possible *with* the boundary
        # term. This is the whole test.
        assert with_term == pytest.approx(fd, rel=1e-5), r

        # and omitting the term must break that agreement: reconstruct the
        # no-boundary derivative and show it does not match the same fd.
        # (Its sign relative to `with_term` is not fixed -- the derivative
        # integrand 2 W W' x oscillates -- so no sign is asserted here.)
        edges = grid._edges(r, True)
        pts, wts = grid._panel_points(edges)
        without_term = float(np.dot(wts, grid._d_integrand(pts, r)))
        assert without_term != pytest.approx(fd, rel=1e-5), r
        assert abs(without_term - with_term) > 0.0


# -- the FFTLog fast path --------------------------------------------------


def test_fftlog_matches_the_untruncated_reference():
    """It is the truncate=False quantity, and only that."""
    grid = power_law_grid()
    lnr = np.log(np.array([0.5, 2.0, 8.0]))
    ln_s2, dln_s2 = grid.sigma2_fftlog(lnr)
    for lr, ls, dls in zip(lnr, ln_s2, dln_s2):
        r = np.exp(lr)
        assert ls == pytest.approx(
            np.log(grid.sigma2(r, truncate=False)), rel=1e-5
        )
        assert dls == pytest.approx(
            grid.dlnsigma2_dlnr(r, truncate=False), rel=1e-3
        )


def test_fftlog_refuses_rather_than_extrapolating():
    """A silent extrapolation here would look like a converged answer."""
    grid = power_law_grid()
    with pytest.raises(RuntimeError, match="does not cover"):
        grid.sigma2_fftlog(np.log(np.array([1e12])), pad_decades=0.0)


# -- the production grid ---------------------------------------------------


def test_the_lnr_grid_is_the_production_grid():
    g = lnr_grid()
    assert g.size == NR == 969
    assert g[0] == pytest.approx(LNR1)
    np.testing.assert_allclose(np.diff(g), STEP)
    # NOTE: the last grid point is -5.684 + 0.01*968 = 3.996, which is NOT
    # LNR2 = 4.0. LNR2 is the upper edge of the Chebyshev fitting interval;
    # the grid stops one step short of it. Conflating the two puts the last
    # point outside the fit.
    assert g[-1] == pytest.approx(LNR1 + STEP * (NR - 1))
    assert g[-1] == pytest.approx(3.996)
    assert g[-1] < LNR2


# -- Tinker (2008) --------------------------------------------------------


def test_tinker_table_is_the_published_table():
    assert TINKER08_TABLE2["delta"][0] == 200.0
    assert TINKER08_TABLE2["A0"][0] == 0.186
    assert TINKER08_TABLE2["a0"][0] == 1.47
    assert TINKER08_TABLE2["b0"][0] == 2.57
    assert TINKER08_TABLE2["c"][0] == 1.19


def test_the_b_evolution_exponent_matches_eq_8_at_delta_200():
    hmf = TinkerMassFunction(power_law_grid())
    expected = 10.0 ** (-((0.75 / np.log10(200.0 / 75.0)) ** 1.2))
    assert hmf.alpha == pytest.approx(expected, rel=1e-14)
    assert hmf.alpha == pytest.approx(0.0107, abs=1e-4)


def test_only_A_a_b_evolve_with_redshift():
    hmf = TinkerMassFunction(power_law_grid())
    A0, a0, b0, c0 = (float(v) for v in hmf.coefficients(0.0))
    A2, a2, b2, c2 = (float(v) for v in hmf.coefficients(2.0))
    assert A2 < A0 and a2 < a0 and b2 < b0
    assert c2 == c0 == hmf.c


def test_f_sigma_matches_pyccl():
    """Same formula, independent implementation, at matched Delta and z."""
    ccl = pytest.importorskip("pyccl")
    hmf = TinkerMassFunction(power_law_grid())
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8,
                          n_s=0.96)
    ccl_mf = ccl.halos.MassFuncTinker08(mass_def="200m")
    for z in (0.0, 0.5, 1.0):
        a = 1.0 / (1.0 + z)
        for sigma in (0.5, 1.0, 2.0):
            theirs = ccl_mf._get_fsigma(cosmo, np.array([sigma]), a, None)
            mine = hmf.f_sigma(sigma, z=z)
            # 1e-10, not machine epsilon: pyccl writes the evolution as
            # a**0.14 where this module writes (1+z)**-0.14. Identical
            # algebra, different rounding, agreeing to ~6e-12.
            assert float(np.ravel(mine)[0]) == pytest.approx(
                float(np.ravel(theirs)[0]), rel=1e-10
            ), (z, sigma)


def test_multiplicity_is_half_of_f_sigma():
    """The reference's 1/2, kept explicit rather than folded into A."""
    hmf = TinkerMassFunction(power_law_grid())
    sigma = 1.3
    ln_nu = 2.0 * np.log(DELTA_C_TINKER) - 2.0 * np.log(sigma)
    assert hmf.multiplicity(ln_nu, 0.0).item() == pytest.approx(
        0.5 * hmf.f_sigma(sigma, 0.0).item()
    )


def test_dndlnm_is_a_third_of_dndlnr():
    hmf = TinkerMassFunction(power_law_grid())
    out = hmf.walk(0.0, lnr=np.log(np.array([1.0, 8.0])))
    np.testing.assert_allclose(out["dndlnmh"], out["dndlnrh"] / 3.0,
                               rtol=1e-14)


def test_dndlnm_falls_steeply_with_mass():
    hmf = TinkerMassFunction(power_law_grid())
    out = hmf.walk(0.0, lnr=np.log(np.logspace(-0.5, 1.3, 15)))
    assert np.all(np.diff(out["dndlnmh"]) < 0.0)
    # and it is steep: several decades across the range
    assert out["dndlnmh"][0] / out["dndlnmh"][-1] > 1e3


def test_mass_of_radius_carries_no_omega_m():
    r"""The mass axis is in :math:`\Omega_m h^{-1}M_\odot`, by convention."""
    hmf = TinkerMassFunction(power_law_grid())
    r = 8.0
    expected = (4.0 * PI_FORTRAN / 3.0) * RHO_FACT * r**3
    assert hmf.mass_of_radius(r).item() == pytest.approx(expected, rel=1e-14)
    # explicitly: it is NOT the h-free Omega_m-weighted mass
    assert hmf.mass_of_radius(r).item() != pytest.approx(
        expected * 0.3, rel=1e-3
    )


def test_mass_scales_as_r_cubed():
    hmf = TinkerMassFunction(power_law_grid())
    assert (hmf.mass_of_radius(2.0) / hmf.mass_of_radius(1.0)).item() == (
        pytest.approx(8.0)
    )


def test_the_two_delta_c_are_different_and_both_kept():
    """1.6865 in the mass function, 1.686 in the bias. Not unified."""
    from clenspy.cosmology.concentration import DELTA_COLLAPSE

    assert DELTA_C_TINKER == 1.6865
    assert DELTA_COLLAPSE == 1.686
    assert DELTA_C_TINKER != DELTA_COLLAPSE


def test_the_fortran_pi_is_the_truncated_literal():
    assert PI_FORTRAN == 3.1415926535
    assert PI_FORTRAN != np.pi
    assert abs(PI_FORTRAN / np.pi - 1.0) < 1e-10


def test_outputs_and_walk_agree():
    """`outputs` is the reusable grid walk; `walk` must just feed it."""
    hmf = TinkerMassFunction(power_law_grid())
    lnr = np.log(np.array([1.0, 4.0, 10.0]))
    out = hmf.walk(0.5, lnr=lnr)
    again = hmf.outputs(lnr, out["lnsigma2"], out["dlnsigma2"], 0.5)
    np.testing.assert_allclose(again["dndlnmh"], out["dndlnmh"], rtol=1e-14)


def test_rejects_delta_outside_the_calibration():
    grid = power_law_grid()
    with pytest.raises(ValueError, match="calibrated"):
        TinkerMassFunction(grid, delta=100.0)
    with pytest.raises(ValueError, match="calibrated"):
        TinkerMassFunction(grid, delta=5000.0)


def test_f_sigma_rejects_non_positive_sigma():
    hmf = TinkerMassFunction(power_law_grid())
    with pytest.raises(ValueError, match="sigma must be positive"):
        hmf.f_sigma(np.array([1.0, -0.5, 2.0]))
    with pytest.raises(ValueError, match="sigma must be positive"):
        hmf.f_sigma(0.0)


def test_tinker_mass_function_repr_contains_the_class_name():
    hmf = TinkerMassFunction(power_law_grid())
    assert "TinkerMassFunction" in repr(hmf)


def test_sigma_grid_refuses_a_bare_array():
    """The LinearPk input policy is part of sigma^2's definition."""
    with pytest.raises(TypeError, match="LinearPk"):
        SigmaGrid(K)


def test_sigma2_rejects_non_positive_r():
    grid = power_law_grid()
    with pytest.raises(ValueError, match="R must be positive"):
        grid.sigma2(-1.0)
    with pytest.raises(ValueError, match="R must be positive"):
        grid.sigma2(0.0)


def test_edges_are_none_when_the_window_is_degenerate():
    r"""``20/R`` below the table's lower edge collapses the panel range."""
    grid = power_law_grid()
    r_huge = 2.5e5  # 20/R << 1e-4 h/Mpc, below the table's own lower limit
    assert grid._edges(r_huge, truncate=True) is None


def test_sigma2_and_its_derivative_vanish_when_the_window_is_degenerate():
    grid = power_law_grid()
    r_huge = 2.5e5
    assert grid.sigma2(r_huge, truncate=True) == 0.0
    assert grid.dsigma2_dlnr(r_huge, truncate=True) == 0.0


def test_dlnsigma2_dlnr_rejects_a_non_positive_sigma2():
    """Reached through the same degenerate window as the tests above."""
    grid = power_law_grid()
    r_huge = 2.5e5
    with pytest.raises(ValueError, match="not positive"):
        grid.dlnsigma2_dlnr(r_huge, truncate=True)


def test_fftlog_refuses_a_non_positive_variance():
    """Ringing from a spiky P(k), caught rather than silently splined."""
    k = np.logspace(-3, 3, 40)
    pk_vals = np.full_like(k, 1e-12)
    pk_vals[20] = 1e12  # a narrow spike, to force FFTLog ringing
    grid = SigmaGrid(LinearPk(k, pk_vals))
    lnr = np.log(np.logspace(-3, 3, 20))
    with pytest.raises(RuntimeError, match="non-positive"):
        grid.sigma2_fftlog(lnr, n_fine=128, pad_decades=0.05)


def test_linear_pk_repr_contains_the_class_name():
    pk = LinearPk(K, 2.0e4 * K**N_SPEC)
    assert "LinearPk" in repr(pk)


def test_sigma_grid_repr_contains_the_class_name():
    grid = power_law_grid()
    assert "SigmaGrid" in repr(grid)


def test_linear_pk_is_zero_outside_the_table():
    pk = LinearPk(K, 2.0e4 * K**N_SPEC)
    assert pk(np.log(K[0] * 0.5)).item() == 0.0
    assert pk(np.log(K[-1] * 2.0)).item() == 0.0
    assert pk(np.log(K[len(K) // 2])).item() > 0.0


def test_linear_pk_validates_its_input():
    with pytest.raises(ValueError, match="ascending"):
        LinearPk(K[::-1], 2.0e4 * K**N_SPEC)
    with pytest.raises(ValueError, match="positive"):
        LinearPk(K, np.zeros_like(K))
    with pytest.raises(ValueError, match="same shape"):
        LinearPk(K, K[:-1])


# -- the consumed-mass mask -----------------------------------------------


def test_consumed_mask_selects_an_interior_band():
    hmf = TinkerMassFunction(power_law_grid())
    m_h = hmf.mass_of_radius(np.exp(lnr_grid()))
    mask = consumed_mask(m_h, 0.3)
    assert mask.any() and not mask.all()
    # it must be a contiguous band, not scattered points
    idx = np.flatnonzero(mask)
    assert np.all(np.diff(idx) == 1)


def test_consumed_mask_shifts_with_omega_m():
    hmf = TinkerMassFunction(power_law_grid())
    m_h = hmf.mass_of_radius(np.exp(lnr_grid()))
    lo = np.flatnonzero(consumed_mask(m_h, 0.2))[0]
    hi = np.flatnonzero(consumed_mask(m_h, 0.4))[0]
    assert hi > lo


# -- the shared sigma grid (step 13b) --------------------------------------


def test_bias_and_mass_function_read_the_same_sigma():
    r"""The dedup: two fits to one peak height must see one :math:`\sigma`.

    `BiasModel` used to compute its own sigma(M) by a second FFTLog. It now
    delegates to a `SigmaGrid`, so the two layers cannot drift.
    """
    from clenspy.cosmology.bias import BiasModel

    pk_vals = 2.0e4 * K**N_SPEC
    model = BiasModel(K, pk_vals)
    assert isinstance(model.sigma_grid, SigmaGrid)

    # sigma at a mass, both ways round: through the bias and through the
    # grid directly, at the same Lagrangian radius
    m = 1.0e14
    r = (3 * m / (4 * np.pi * model.rhom)) ** (1 / 3)
    direct = np.sqrt(model.sigma_grid.sigma2(r, truncate=False))
    assert model.sigma_tophat(m).item() == pytest.approx(direct, rel=2e-5)


def test_bias_no_longer_caches_nu_across_different_masses():
    """The old `bias` returned the *first* mass's bias for every later one."""
    from clenspy.cosmology.bias import BiasModel

    model = BiasModel(K, 2.0e4 * K**N_SPEC)
    b_small = model.bias(1.0e13)
    b_large = model.bias(1.0e15)
    # and now the other order, on a fresh object, to show order-independence
    model2 = BiasModel(K, 2.0e4 * K**N_SPEC)
    assert model2.bias(1.0e15) == pytest.approx(b_large)
    assert model2.bias(1.0e13) == pytest.approx(b_small)
    assert b_large > b_small


def test_the_bias_grid_is_built_once_and_reused():
    """The FFTLog was previously rebuilt on every sigma_tophat call."""
    from clenspy.cosmology.bias import BiasModel

    model = BiasModel(K, 2.0e4 * K**N_SPEC)
    first = model.sigma_grid
    model.sigma_tophat(1e14)
    model.sigma_tophat(1e15)
    assert model.sigma_grid is first


def test_sigma_grid_is_unit_agnostic_only_above_the_fixed_lower_limit():
    r"""Why an h-free class may share an h-scaled module's evaluator.

    :math:`\sigma^2` depends only on :math:`kR` being dimensionless and
    :math:`P` scaling as :math:`k^{-3}`, so rescaling k and asking at the
    rescaled R should give the same number.

    But **both** quadrature limits are dimensionful, not just
    :math:`20/R`: ``LNK_LO`` is :math:`10^{-4}` **h/Mpc**. Invariance
    therefore holds only when the tabulated k range lies entirely above
    it, so that the lower limit is the table's own edge and no absolute
    scale enters. This test pins both halves of that statement.
    """
    h = 0.7
    # entirely above 1e-4 in both scalings, so LNK_LO never binds
    k_hi = np.logspace(-3.0, 4.0, 900)
    pk_vals = 2.0e4 * k_hi**N_SPEC
    a = SigmaGrid(LinearPk(k_hi, pk_vals))
    # k -> k/h  (h/Mpc -> 1/Mpc),  P -> P*h^3,  R -> R*h
    b = SigmaGrid(LinearPk(k_hi / h, pk_vals * h**3))
    for r in (0.5, 2.0, 8.0):
        assert b.sigma2(r * h, truncate=False) == pytest.approx(
            a.sigma2(r, truncate=False), rel=1e-10
        )


def test_the_fixed_lower_limit_breaks_unit_invariance_when_it_binds():
    """The other half: a k range straddling 1e-4 is not rescalable."""
    h = 0.7
    k_lo = np.logspace(-6.0, 4.0, 900)      # extends below LNK_LO = 1e-4
    pk_vals = 2.0e4 * k_lo**N_SPEC
    a = SigmaGrid(LinearPk(k_lo, pk_vals))
    b = SigmaGrid(LinearPk(k_lo / h, pk_vals * h**3))
    # the same 1e-4 cut now removes a different physical range in each
    assert b.sigma2(8.0 * h, truncate=False) != pytest.approx(
        a.sigma2(8.0, truncate=False), rel=1e-9
    )


def test_the_truncation_breaks_that_unit_invariance():
    """Which is exactly why the bias must not use truncate=True."""
    h = 0.7
    pk_vals = 2.0e4 * K**N_SPEC
    a = SigmaGrid(LinearPk(K, pk_vals))
    b = SigmaGrid(LinearPk(K / h, pk_vals * h**3))
    # 20/R is a dimensionful cut, so rescaling changes what it removes
    assert b.sigma2(8.0 * h, truncate=True) != pytest.approx(
        a.sigma2(8.0, truncate=True), rel=1e-6
    )
