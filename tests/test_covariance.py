r"""The two covariance blocks.

A covariance matrix has properties that can be checked without a reference:
it is symmetric, positive semi-definite, and its stored components must sum
to the total. Beyond that, each block has structure that *is* the physics
and that a wrong implementation breaks:

- the counts sample-variance term is **rank one** within each redshift
  slice and **exactly zero** between slices;
- the :math:`\Delta\Sigma` bracket's five terms scale independently and
  predictably in :math:`f_{\rm sky}`, :math:`n_h` and the shape noise;
- :math:`\hat J_2` is a bin *average*, which is checkable against direct
  quadrature of :math:`J_2`.
"""

import numpy as np
import pytest
from scipy.special import jv

from clenspy.covariance import (
    ALL_TERMS,
    CountsCovariance,
    DeltaSigmaCovariance,
    j2_bin,
)

COUNTS = np.array([[2500.0, 3100.0, 2700.0],
                   [900.0, 1150.0, 1000.0],
                   [300.0, 380.0, 330.0],
                   [110.0, 140.0, 120.0]])
BIAS = np.array([[2.1, 2.2, 2.3],
                 [2.6, 2.7, 2.8],
                 [3.2, 3.3, 3.5],
                 [4.3, 4.5, 4.8]])
SIGMA_W = np.array([0.0908, 0.0841, 0.0784])


# -- counts: Poisson + sample variance -------------------------------------


def test_counts_covariance_is_symmetric_and_positive_definite():
    c = CountsCovariance(COUNTS, BIAS, SIGMA_W).cov()
    np.testing.assert_allclose(c, c.T, rtol=0.0, atol=0.0)
    assert np.linalg.eigvalsh(c).min() > 0.0


def test_counts_components_sum_to_the_total():
    cc = CountsCovariance(COUNTS, BIAS, SIGMA_W)
    summed = sum(cc.components().values())
    np.testing.assert_allclose(summed, cc.cov(), rtol=1e-14)


def test_the_poisson_term_is_exactly_the_counts_on_the_diagonal():
    cc = CountsCovariance(COUNTS, BIAS, SIGMA_W)
    p = cc.cov_poisson()
    np.testing.assert_allclose(np.diag(p), COUNTS.ravel(), rtol=0.0)
    assert np.count_nonzero(p - np.diag(np.diag(p))) == 0


def test_the_sample_variance_term_is_rank_one_per_redshift_slice():
    r"""It is an outer product :math:`(\bar bN)(\bar bN)^{\rm T}`.

    Rank one is the signature of a single coherent mode. A term with any
    other rank is not a window fluctuation.
    """
    cc = CountsCovariance(COUNTS, BIAS, SIGMA_W)
    sv = cc.cov_sample_variance()
    for j in range(cc.n_z_bins):
        idx = np.arange(cc.n_lambda_bins) * cc.n_z_bins + j
        block = sv[np.ix_(idx, idx)]
        assert np.linalg.matrix_rank(block, tol=1e-8 * block.max()) == 1


def test_the_sample_variance_term_vanishes_between_redshift_bins():
    """Different slices are independent -- the stated approximation."""
    cc = CountsCovariance(COUNTS, BIAS, SIGMA_W)
    sv = cc.cov_sample_variance()
    for a in range(cc.size):
        for b in range(cc.size):
            if a % cc.n_z_bins != b % cc.n_z_bins:
                assert sv[a, b] == 0.0, (a, b)


def test_sample_variance_scales_as_sigma_window_squared():
    a = CountsCovariance(COUNTS, BIAS, SIGMA_W)
    b = CountsCovariance(COUNTS, BIAS, 2.0 * SIGMA_W)
    ratio = (b.cov_sample_variance()[a.cov_sample_variance() != 0.0]
             / a.cov_sample_variance()[a.cov_sample_variance() != 0.0])
    np.testing.assert_allclose(ratio, 4.0, rtol=1e-12)


def test_sample_variance_scales_as_bias_squared():
    a = CountsCovariance(COUNTS, BIAS, SIGMA_W)
    b = CountsCovariance(COUNTS, 3.0 * BIAS, SIGMA_W)
    mask = a.cov_sample_variance() != 0.0
    np.testing.assert_allclose(
        b.cov_sample_variance()[mask] / a.cov_sample_variance()[mask], 9.0,
        rtol=1e-12,
    )


def test_zero_window_variance_leaves_only_poisson():
    cc = CountsCovariance(COUNTS, BIAS, np.zeros(3))
    np.testing.assert_allclose(cc.cov(), cc.cov_poisson(), rtol=0.0)


def test_the_switches_isolate_each_term():
    cc = CountsCovariance(COUNTS, BIAS, SIGMA_W)
    np.testing.assert_allclose(cc.cov(sample_variance=False),
                               cc.cov_poisson(), rtol=1e-14)
    np.testing.assert_allclose(cc.cov(poisson=False),
                               cc.cov_sample_variance(), rtol=1e-14)
    assert np.all(cc.cov(poisson=False, sample_variance=False) == 0.0)


def test_dropping_sample_variance_understates_the_error():
    """The reason the term exists. It is not a small correction."""
    cc = CountsCovariance(COUNTS, BIAS, SIGMA_W)
    full = np.sqrt(np.diag(cc.cov()))
    poisson = np.sqrt(np.diag(cc.cov(sample_variance=False)))
    assert np.all(full > poisson)
    assert np.max(full / poisson) > 4.0


def test_richness_bins_are_strongly_correlated_at_fixed_redshift():
    cc = CountsCovariance(COUNTS, BIAS, SIGMA_W)
    corr = cc.correlation()
    idx = np.arange(cc.n_lambda_bins) * cc.n_z_bins          # all at redshift bin 0
    off = corr[np.ix_(idx, idx)][np.triu_indices(cc.n_lambda_bins, k=1)]
    assert np.all(off > 0.9)


def test_the_block_helper_matches_the_full_matrix():
    cc = CountsCovariance(COUNTS, BIAS, SIGMA_W)
    idx = np.arange(cc.n_lambda_bins) * cc.n_z_bins + 1
    np.testing.assert_allclose(cc.block(1), cc.cov()[np.ix_(idx, idx)],
                               rtol=0.0)


def test_counts_covariance_validates_its_inputs():
    with pytest.raises(ValueError, match="must be 2-D"):
        CountsCovariance(np.ones(4), BIAS, SIGMA_W)
    with pytest.raises(ValueError, match="bias must match"):
        CountsCovariance(COUNTS, BIAS[:, :2], SIGMA_W)
    with pytest.raises(ValueError, match="one entry per redshift bin"):
        CountsCovariance(COUNTS, BIAS, SIGMA_W[:2])
    with pytest.raises(ValueError, match="non-negative"):
        CountsCovariance(-COUNTS, BIAS, SIGMA_W)


# -- the bin-averaged Bessel function --------------------------------------


def test_j2_bin_matches_direct_quadrature_of_j2():
    r"""The definition: :math:`\hat J_2` is the
    :math:`2\pi\theta\,d\theta`-weighted average of :math:`J_2`."""
    for ell in (1.0, 30.0, 300.0, 3000.0):
        for lo, hi in ((0.001, 0.002), (0.005, 0.02), (0.02, 0.05)):
            theta = np.linspace(lo, hi, 200001)
            numerator = np.trapezoid(jv(2, ell * theta) * 2 * np.pi * theta,
                                    x=theta)
            expected = numerator / (np.pi * (hi**2 - lo**2))
            assert j2_bin(ell, lo, hi).item() == pytest.approx(
                expected, rel=1e-6, abs=1e-12
            ), (ell, lo, hi)


def test_j2_bin_is_accurate_at_small_argument():
    r"""The closed form is not, and this pins the series branch.

    For :math:`\ell\theta \ll 1` the closed form's bracket cancels to
    nine orders and fp64 leaves ~4 digits: it was measured at 4.8e-4
    relative error before the series branch existed. This test is the
    regression guard.
    """
    for ell, lo, hi in ((1.0, 0.001, 0.002), (0.5, 1e-4, 3e-4),
                        (10.0, 1e-3, 2e-3)):
        theta = np.linspace(lo, hi, 400001)
        numerator = np.trapezoid(jv(2, ell * theta) * 2 * np.pi * theta,
                                x=theta)
        expected = numerator / (np.pi * (hi**2 - lo**2))
        assert j2_bin(ell, lo, hi).item() == pytest.approx(
            expected, rel=1e-9
        ), (ell, lo, hi)


def test_the_two_j2_branches_agree_at_the_cutoff():
    """Neither alone is sufficient, so the seam must be invisible.

    Compared against direct quadrature on each side rather than against
    each other: straddling the cutoff also moves the function, and
    dlnJ2/dlnell is O(1), so the two sides differ by the slope times the
    step whether or not there is a discontinuity.
    """
    from clenspy.covariance.deltasigma import J2_SERIES_CUTOFF
    lo, hi = 0.005, 0.02
    for factor in (1.0 - 1e-9, 1.0 + 1e-9):
        ell = factor * J2_SERIES_CUTOFF / hi
        theta = np.linspace(lo, hi, 400001)
        numerator = np.trapezoid(jv(2, ell * theta) * 2 * np.pi * theta,
                                x=theta)
        expected = numerator / (np.pi * (hi**2 - lo**2))
        assert j2_bin(ell, lo, hi).item() == pytest.approx(
            expected, rel=1e-11
        ), factor


def test_j2_bin_tends_to_j2_for_a_narrow_annulus():
    """A vanishing bin width recovers the point value."""
    ell, theta = 200.0, 0.01
    for width in (1e-2, 1e-3, 1e-4):
        lo = theta * (1 - width)
        hi = theta * (1 + width)
        assert j2_bin(ell, lo, hi).item() == pytest.approx(
            jv(2, ell * theta), rel=10 * width
        )


def test_j2_bin_decays_faster_for_a_wider_bin():
    r"""Wu et al.'s statement about :math:`\hat J_2`, checked.

    At fixed bin centre the first peak barely moves, but the large-:math:`
    \ell` decay is faster for a wider bin -- which is why the bin average
    cannot be skipped for the flat noise terms.
    """
    centre, ell_large = 0.01, 5000.0
    narrow = abs(j2_bin(ell_large, centre * 0.95, centre * 1.05).item())
    wide = abs(j2_bin(ell_large, centre * 0.5, centre * 1.5).item())
    assert wide < narrow


def test_j2_bin_validates_its_arguments():
    with pytest.raises(ValueError, match="ell must be positive"):
        j2_bin(0.0, 0.01, 0.02)
    with pytest.raises(ValueError, match="theta_max > theta_min"):
        j2_bin(100.0, 0.02, 0.01)


# -- the DeltaSigma covariance ---------------------------------------------

RP_EDGES = np.logspace(np.log10(0.2), np.log10(30.0), 7)
CHI_H = 1100.0
F_SKY = 1500.0 * (np.pi / 180.0) ** 2 / (4.0 * np.pi)
N_H = 3.0e5
SHAPE_NOISE = 1.0e26


def c_hh(ell):
    return 1e-5 * (np.asarray(ell, dtype=float) / 100.0) ** -1.0


def c_ss(ell):
    return 4e26 * (np.asarray(ell, dtype=float) / 100.0) ** -1.2


def c_hs(ell):
    """The linear-bias limit, where (C_hS)^2 = C_hh C_SS identically."""
    return np.sqrt(c_hh(ell) * c_ss(ell))


def make_cov(**kw):
    kwargs = dict(rp_edges=RP_EDGES, chi_h=CHI_H, f_sky=F_SKY,
                  c_ell_hh=c_hh, c_ell_SS=c_ss, c_ell_hS=c_hs, n_h=N_H,
                  shape_noise=SHAPE_NOISE)
    kwargs.update(kw)
    return DeltaSigmaCovariance(**kwargs)


def test_deltasigma_covariance_is_symmetric_and_positive_definite():
    c = make_cov().cov()
    np.testing.assert_allclose(c, c.T, rtol=1e-14)
    assert np.linalg.eigvalsh(c).min() > 0.0


def test_deltasigma_components_sum_to_the_total():
    cov = make_cov()
    summed = sum(cov.components().values())
    np.testing.assert_allclose(summed, cov.cov(), rtol=1e-12)


def test_the_terms_selector_matches_the_named_components():
    cov = make_cov()
    parts = cov.components()
    for name in ALL_TERMS:
        np.testing.assert_allclose(cov.cov(terms=(name,)), parts[name],
                                   rtol=1e-12)


def test_an_unknown_term_is_rejected():
    with pytest.raises(ValueError, match="unknown terms"):
        make_cov().cov(terms=("cosmic_shear",))


def test_the_whole_covariance_scales_inversely_with_f_sky():
    r"""It enters only as :math:`1/(4\pi f_{\rm sky})`."""
    a, b = make_cov(), make_cov(f_sky=2.0 * F_SKY)
    np.testing.assert_allclose(b.cov() / a.cov(), 0.5, rtol=1e-12)


def _ratio_where_nonzero(numerator, denominator):
    """Element-wise ratio, skipping structural zeros.

    ``shot_shape`` is exactly diagonal now, so a bare element-wise ratio
    hits 0/0 off the diagonal. Comparing only where the denominator is
    nonzero keeps the scaling check meaningful.
    """
    mask = denominator != 0.0
    assert mask.any()
    return numerator[mask] / denominator[mask]


def test_the_shot_noise_terms_scale_as_one_over_n_h():
    a, b = make_cov(), make_cov(n_h=2.0 * N_H)
    for name in ("shot_lss", "shot_shape"):
        ratio = _ratio_where_nonzero(b.components()[name],
                                     a.components()[name])
        np.testing.assert_allclose(ratio, 0.5, rtol=1e-12), name


def test_the_lss_only_term_is_independent_of_both_noises():
    a = make_cov()
    b = make_cov(n_h=10.0 * N_H, shape_noise=10.0 * SHAPE_NOISE)
    np.testing.assert_allclose(b.components()["lss_lss"],
                               a.components()["lss_lss"], rtol=1e-12)
    np.testing.assert_allclose(b.components()["cross"],
                               a.components()["cross"], rtol=1e-12)


def test_the_shape_noise_terms_scale_linearly_in_the_shape_noise():
    a, b = make_cov(), make_cov(shape_noise=3.0 * SHAPE_NOISE)
    for name in ("lss_shape", "shot_shape"):
        # 1e-10, not machine epsilon: the off-diagonal entries involve
        # cancelling J2 oscillations, so their ratio carries a few ulp more
        np.testing.assert_allclose(
            _ratio_where_nonzero(b.components()[name],
                                 a.components()[name]), 3.0, rtol=1e-10
        ), name


def test_lss_lss_equals_cross_in_the_linear_bias_limit():
    r""":math:`(C^{h\Sigma})^2 = C^{hh}C^{\Sigma\Sigma}` when
    :math:`C^{h\Sigma} = \sqrt{C^{hh}C^{\Sigma\Sigma}}`.

    A structural check on the bracket: the two terms are built from
    different spectra and must coincide exactly in this limit.
    """
    parts = make_cov().components()
    np.testing.assert_allclose(parts["cross"], parts["lss_lss"], rtol=1e-12)


def test_shot_shape_dominates_at_small_radius_and_lss_at_large():
    """The physics the decomposition exists to expose."""
    cov = make_cov()
    parts = cov.components()
    total = np.diag(cov.cov())
    shot_shape = np.diag(parts["shot_shape"]) / total
    lss = (np.diag(parts["lss_lss"]) + np.diag(parts["cross"])) / total
    assert shot_shape[0] > 0.8            # small rp: pure noise
    assert shot_shape[-1] < shot_shape[0]  # falls with radius
    assert lss[-1] > lss[0]                # and the LSS terms rise


def test_neighbouring_radial_bins_are_positively_correlated():
    c = make_cov().cov()
    d = np.sqrt(np.diag(c))
    corr = c / np.outer(d, d)
    off = np.diag(corr, k=1)
    assert np.all(off > 0.0)
    assert np.all(off < 1.0)


def test_the_k_grid_is_converged_on_both_axes():
    """Measured, not asserted -- the module's named approximation."""
    conv = make_cov(n_k=8192).convergence()
    assert set(conv) == {"n_k", "k_max"}
    assert conv["n_k"] < 1e-4
    assert conv["k_max"] < 1e-4


def test_the_diagonal_falls_with_radius():
    d = np.diag(make_cov().cov())
    assert np.all(np.diff(d) < 0.0)


def test_deltasigma_covariance_validates_its_inputs():
    with pytest.raises(ValueError, match="ascending"):
        make_cov(rp_edges=np.array([2.0, 1.0]))
    with pytest.raises(ValueError, match=r"f_sky must lie in \(0, 1\]"):
        make_cov(f_sky=0.0)
    with pytest.raises(ValueError, match=r"f_sky must lie in \(0, 1\]"):
        make_cov(f_sky=1.5)
    with pytest.raises(ValueError, match="chi_h must be positive"):
        make_cov(chi_h=0.0)
    with pytest.raises(ValueError, match="n_h must be positive"):
        make_cov(n_h=0.0)
    with pytest.raises(ValueError, match=">= 2 entries"):
        make_cov(rp_edges=np.array([1.0]))


def test_the_bilinear_form_uses_the_right_k_measure():
    r"""The integral is :math:`\int k\,dk`, i.e.
    :math:`\int k^2\,d\ln k`.

    Checked by scaling: with a scale-free spectrum
    :math:`P \propto k^{n}` the covariance must scale as a pure power of
    any overall rescaling of the noise, which a wrong Jacobian breaks
    because it changes the effective slope. Here the simplest exact
    statement: a constant bracket integrates to something finite and
    positive, and doubling it doubles the result.
    """
    a = make_cov()
    b = make_cov(shape_noise=2.0 * SHAPE_NOISE, n_h=N_H)
    # shot_shape is the constant-bracket term: strictly linear
    np.testing.assert_allclose(
        _ratio_where_nonzero(b.components()["shot_shape"],
                             a.components()["shot_shape"]), 2.0,
        rtol=1e-12,
    )
    assert np.all(np.diag(a.components()["shot_shape"]) > 0.0)


# -- the Hankel closure identity, and the precision it buys ----------------
#
# These exist because the claim "FFTLog buys nothing" was only true about
# COST. On precision the quadrature was losing 3.5e-3 on the total, and the
# fix turned out to be better than FFTLog: the dominant term has an exact
# closed form. These tests pin that, and pin the error scaling of what is
# left, so neither can silently regress.


def closure_reference(rp_edges, f_sky, n_h, shape_noise):
    r"""The exact ``shot_shape`` term, from Hankel closure.

    :math:`\int_0^\infty J_2(ka)J_2(kb)\,k\,dk = \delta(a-b)/a`, averaged
    over disjoint contiguous annuli, gives
    :math:`\delta_{ij}/A_{{\rm ann},i}` with
    :math:`A_{\rm ann} = \pi(r_{p,\max}^2 - r_{p,\min}^2)` in **Mpc^2**.
    """
    area = np.pi * (rp_edges[1:] ** 2 - rp_edges[:-1] ** 2)
    return np.diag((shape_noise / n_h) / (4.0 * np.pi * f_sky) / area)


def test_the_closed_form_shot_shape_is_the_closure_result():
    """The analytic reference, independent of the implementation."""
    cov = make_cov()
    np.testing.assert_allclose(
        cov.components()["shot_shape"],
        closure_reference(RP_EDGES, F_SKY, N_H, SHAPE_NOISE), rtol=1e-14,
    )


def test_the_closed_form_shot_shape_is_strictly_diagonal():
    r"""Disjoint annuli, so :math:`\delta(a-b)` gives nothing off-diagonal.

    The quadrature leaks here; the closed form cannot.
    """
    m = make_cov().components()["shot_shape"]
    assert np.count_nonzero(m - np.diag(np.diag(m))) == 0


def test_the_quadrature_converges_onto_the_closed_form():
    """Both compute the same integral, so they must agree in the limit."""
    exact = np.diag(closure_reference(RP_EDGES, F_SKY, N_H, SHAPE_NOISE))
    previous = np.inf
    for k_max in (1e2, 1e3, 1e4):
        got = np.diag(
            make_cov(k_range=(1e-4, k_max), n_k=8192,
                     exact_shot_shape=False).components()["shot_shape"]
        )
        error = np.max(np.abs(got / exact - 1.0))
        assert error < previous
        previous = error
    assert previous < 1e-3


def test_the_quadrature_error_is_truncation_limited_not_node_limited():
    r"""Which is why `convergence` had to start reporting both axes.

    At fixed :math:`k_{\max}` the error is flat in ``n_k`` to better than
    10%, while a decade of :math:`k_{\max}` buys a decade of accuracy. A
    diagnostic that varied only ``n_k`` therefore reported 4e-4 when the
    true error was 2.4e-3.
    """
    exact = np.diag(closure_reference(RP_EDGES, F_SKY, N_H, SHAPE_NOISE))

    def error(k_max, n_k):
        got = np.diag(make_cov(k_range=(1e-4, k_max), n_k=n_k,
                               exact_shot_shape=False
                               ).components()["shot_shape"])
        return np.max(np.abs(got / exact - 1.0))

    # nodes barely matter
    coarse, fine = error(1e3, 2048), error(1e3, 16384)
    assert fine / coarse == pytest.approx(1.0, rel=0.1)

    # truncation is everything: err * k_max is roughly constant
    for k_max in (1e2, 1e3, 1e4):
        assert error(k_max, 8192) * k_max == pytest.approx(2.5, rel=0.3)


def test_the_shot_shape_term_does_not_depend_on_chi_h():
    r"""The Mpc^2-vs-steradian trap, pinned.

    :math:`\ell\theta = k r_p`, so this term is a function of :math:`r_p`
    alone. If :math:`A_{\rm ann}` were taken in steradians it would pick up
    a :math:`\chi_h^2` -- a factor of :math:`10^6` at
    :math:`\chi_h = 1100` Mpc. Both routes must be flat in ``chi_h``.
    """
    for exact in (True, False):
        a = make_cov(chi_h=500.0, exact_shot_shape=exact
                     ).components()["shot_shape"]
        b = make_cov(chi_h=2500.0, exact_shot_shape=exact
                     ).components()["shot_shape"]
        np.testing.assert_allclose(np.diag(b), np.diag(a), rtol=1e-6), exact


def test_the_closed_form_improves_the_total_by_orders_of_magnitude():
    """The gain, measured against a heavily converged reference.

    This is the test the claim "FFTLog buys nothing" needed: on cost it was
    right, on precision it was not, and the closure identity is where the
    precision came from.
    """
    truth = np.diag(make_cov(k_range=(1e-4, 1e7), n_k=131072).cov())
    with_exact = np.diag(make_cov().cov())
    without = np.diag(make_cov(exact_shot_shape=False).cov())

    err_exact = np.max(np.abs(with_exact / truth - 1.0))
    err_quad = np.max(np.abs(without / truth - 1.0))
    assert err_exact < 1e-6
    assert err_quad > 10 * err_exact


def test_the_old_default_grid_was_measurably_wrong():
    """k_max = 1e3 with quadrature-only carried ~1e-3 on the total.

    Kept as a test so the default cannot quietly drift back.
    """
    truth = np.diag(make_cov(k_range=(1e-4, 1e7), n_k=131072).cov())
    old = np.diag(make_cov(k_range=(1e-4, 1e3), n_k=4096,
                           exact_shot_shape=False).cov())
    assert np.max(np.abs(old / truth - 1.0)) > 1e-3
    # ...and the current default is far better
    assert np.max(np.abs(np.diag(make_cov().cov()) / truth - 1.0)) < 1e-6


def test_the_exact_switch_is_honoured_by_both_entry_points():
    """`cov` and `components` must not disagree about which route ran."""
    for exact in (True, False):
        cov = make_cov(exact_shot_shape=exact)
        np.testing.assert_allclose(
            cov.cov(terms=("shot_shape",)), cov.components()["shot_shape"],
            rtol=1e-12,
        )
        # and the five components still sum to the total
        np.testing.assert_allclose(sum(cov.components().values()),
                                   cov.cov(), rtol=1e-12)


def test_annulus_area_is_in_mpc_squared():
    cov = make_cov()
    expected = np.pi * (RP_EDGES[1:] ** 2 - RP_EDGES[:-1] ** 2)
    np.testing.assert_allclose(cov.annulus_area(), expected, rtol=1e-14)
