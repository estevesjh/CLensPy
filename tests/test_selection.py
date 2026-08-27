r"""The selection function and its two analytic pieces.

The EMG CDF is checked three ways, because it is the one place where the
*mathematically* correct expression and the *numerically* usable one differ:
against the textbook form where that form is safe, against
`scipy.stats.exponnorm` (an independent implementation), and against the
places where the textbook form returns ``nan`` and this one must not.

The rest are conservation statements. Probabilities integrate to one,
contiguous bins tile without gap or overlap, and
:math:`\mathcal S_{ij}` factorises exactly.
"""

import numpy as np
import pytest

from clenspy.kernels.photoz import gaussian_cdf
from clenspy.selection import (
    EmgParams,
    HodMor,
    LogNormalMor,
    SelectionFunction,
    emg_cdf,
    richness_bin_probability,
)

LAM_EDGES = np.array([20.0, 30.0, 45.0, 60.0, 200.0])   # DES Y1
Z_EDGES = np.array([0.20, 0.35, 0.50, 0.65])
PARAMS = EmgParams(delta_mu=-1.5, sigma=3.0, f_prj=0.3, tau=0.12)


# -- the EMG CDF ------------------------------------------------------------


def test_emg_cdf_matches_the_textbook_form_where_it_is_safe():
    r"""Small :math:`\tau\sigma`, where :math:`e^{A}` does not overflow."""
    mu, sigma, tau = 35.0, 3.0, 0.1        # tau*sigma = 0.3
    for x in (25.0, 33.0, 35.0, 40.0, 60.0):
        z = (x - mu) / sigma
        A = -tau * (x - mu) + 0.5 * (tau * sigma) ** 2
        textbook = (gaussian_cdf(z).item()
                    - np.exp(A) * gaussian_cdf(z - tau * sigma).item())
        assert emg_cdf(x, mu, sigma, tau).item() == pytest.approx(
            textbook, rel=1e-12
        ), x


def test_emg_cdf_matches_scipy_exponnorm():
    """An independent implementation of the same distribution.

    ``scipy.stats.exponnorm`` parametrises by K = 1/(tau*sigma) with
    loc = mu and scale = sigma.
    """
    from scipy.stats import exponnorm

    for sigma, tau in ((3.0, 0.1), (3.0, 0.5), (1.0, 2.0), (5.0, 0.05)):
        mu = 35.0
        K = 1.0 / (tau * sigma)
        x = np.linspace(mu - 4 * sigma, mu + 20 / tau, 40)
        theirs = exponnorm.cdf(x, K, loc=mu, scale=sigma)
        mine = emg_cdf(x, mu, sigma, tau)
        np.testing.assert_allclose(mine, theirs, rtol=1e-9, atol=1e-12)


def test_emg_cdf_survives_where_the_textbook_form_is_nan():
    r"""The reason this module exists.

    For :math:`\tau\sigma \gtrsim 40`, :math:`e^{A}` overflows to ``inf``
    while :math:`\Phi(z-\tau\sigma)` underflows to 0, so the textbook
    product is ``inf * 0 = nan``. The true value is an ordinary number.
    """
    mu, sigma = 35.0, 3.0
    for tau_sigma in (40.0, 60.0, 100.0):
        tau = tau_sigma / sigma
        x = 40.0
        z = (x - mu) / sigma
        A = -tau * (x - mu) + 0.5 * tau_sigma**2
        with np.errstate(over="ignore", invalid="ignore"):
            naive = np.exp(A) * gaussian_cdf(z - tau_sigma).item()
        assert not np.isfinite(naive)              # the form we do not use
        got = emg_cdf(x, mu, sigma, tau).item()    # the form we do
        assert np.isfinite(got) and 0.0 <= got <= 1.0


def test_emg_cdf_is_monotone_and_bounded():
    x = np.linspace(0.0, 400.0, 2000)
    f = emg_cdf(x, 35.0, 3.0, 0.12)
    assert np.all(np.diff(f) >= -1e-15)
    assert f[0] == pytest.approx(0.0, abs=1e-12)
    assert f[-1] == pytest.approx(1.0, abs=1e-9)


def test_emg_cdf_branches_agree_at_the_seam():
    r"""The two branches meet at :math:`u = 0`, i.e. :math:`z = \tau\sigma`.

    Straddling the seam in ``x`` also moves the function itself, so the
    honest check is that the straddling slope stays between the two
    one-sided slopes -- a branch discontinuity would throw it outside.
    """
    mu, sigma, tau = 35.0, 3.0, 0.5
    x_seam = mu + (tau * sigma) * sigma
    d = 1e-4
    xs = x_seam + d * np.array([-2.0, -1.0, 1.0, 2.0])
    f = np.array([emg_cdf(x, mu, sigma, tau).item() for x in xs])
    below = (f[1] - f[0]) / d
    above = (f[3] - f[2]) / d
    across = (f[2] - f[1]) / (2 * d)
    assert min(below, above) <= across <= max(below, above)
    assert across == pytest.approx(0.5 * (below + above), rel=1e-6)


def test_emg_tends_to_the_gaussian_as_tau_grows():
    r"""A large rate means a vanishing tail, and it converges as
    :math:`O(1/\tau)`.

    The exponential's mean is :math:`1/\tau`, so the EMG is the Gaussian
    shifted by that much: the difference falls like :math:`1/\tau`, not
    faster. Asserting a fixed tight tolerance at one :math:`\tau` would
    hide the rate; this asserts the rate itself.
    """
    mu, sigma = 35.0, 3.0
    x = np.linspace(20.0, 60.0, 50)
    gauss = gaussian_cdf((x - mu) / sigma)
    errors = [np.max(np.abs(emg_cdf(x, mu, sigma, tau) - gauss))
              for tau in (1e2, 1e3, 1e4, 1e5)]
    # each decade in tau buys exactly a decade in accuracy
    for lo, hi in zip(errors[:-1], errors[1:]):
        assert lo / hi == pytest.approx(10.0, rel=0.01)
    assert errors[-1] < 2e-6


def test_emg_cdf_rejects_non_positive_parameters():
    with pytest.raises(ValueError, match="sigma"):
        emg_cdf(30.0, 35.0, 0.0, 0.1)
    with pytest.raises(ValueError, match="tau"):
        emg_cdf(30.0, 35.0, 3.0, 0.0)


# -- the bin probabilities --------------------------------------------------


def test_bin_probabilities_sum_to_the_total_cdf_difference():
    """Contiguous bins tile: differencing shared edges guarantees it."""
    for lam in (18.0, 25.0, 40.0, 90.0):
        s = richness_bin_probability(LAM_EDGES, lam, 0.3, PARAMS)
        mu, sigma, tau, f = PARAMS.at(lam, 0.3)
        total = ((1 - f) * (gaussian_cdf((LAM_EDGES[-1] - mu) / sigma)
                            - gaussian_cdf((LAM_EDGES[0] - mu) / sigma))
                 + f * (emg_cdf(LAM_EDGES[-1], mu, sigma, tau)
                        - emg_cdf(LAM_EDGES[0], mu, sigma, tau)))
        assert s.sum() == pytest.approx(float(np.ravel(total)[0]), rel=1e-14)


def test_bin_probabilities_are_probabilities():
    lam = np.linspace(1.0, 300.0, 200)
    s = richness_bin_probability(LAM_EDGES, lam, 0.3, PARAMS)
    assert np.all(s >= 0.0) and np.all(s <= 1.0)
    assert np.all(s.sum(axis=-1) <= 1.0 + 1e-12)


def test_f_prj_zero_recovers_the_pure_gaussian_model():
    """Costanzi et al. (2021)'s BKG limit, exactly."""
    gauss_only = EmgParams(delta_mu=-1.5, sigma=3.0, f_prj=0.0, tau=0.12)
    lam = 35.0
    s = richness_bin_probability(LAM_EDGES, lam, 0.3, gauss_only)
    mu = lam - 1.5
    expected = np.diff([gaussian_cdf((e - mu) / 3.0).item()
                        for e in LAM_EDGES])
    np.testing.assert_allclose(s, expected, rtol=1e-14)


def test_the_projection_boost_only_moves_richness_upward():
    r"""One-sided by construction: :math:`\Delta^{\rm prj} \ge 0`.

    Turning on :math:`f^{\rm prj}` must take probability out of the bin
    containing :math:`\lambda^{\rm tr}` and put it in *higher* bins, never
    lower.
    """
    lam = 25.0                                   # sits in bin 0, [20, 30)
    no_prj = EmgParams(delta_mu=-1.5, sigma=3.0, f_prj=0.0, tau=0.12)
    change = (richness_bin_probability(LAM_EDGES, lam, 0.3, PARAMS)
              - richness_bin_probability(LAM_EDGES, lam, 0.3, no_prj))
    assert change[0] < 0.0                       # loses from its own bin
    assert np.all(change[1:] > 0.0)              # gains in every higher one


def test_bin_probability_rejects_bad_edges():
    with pytest.raises(ValueError, match="ascending"):
        richness_bin_probability([30.0, 20.0], 25.0, 0.3, PARAMS)
    with pytest.raises(ValueError, match="at least two"):
        richness_bin_probability([20.0], 25.0, 0.3, PARAMS)


def test_emg_params_accept_callables():
    r"""Production supplies splines in :math:`(\lambda^{\rm tr}, z)`."""
    params = EmgParams(
        delta_mu=lambda lam, z: -0.05 * np.asarray(lam),
        sigma=lambda lam, z: 2.0 + 0.02 * np.asarray(lam),
        f_prj=lambda lam, z: 0.2 + 0.1 * np.asarray(z),
        tau=0.12,
    )
    mu, sigma, tau, f = params.at(30.0, 0.4)
    assert float(np.ravel(mu)[0]) == pytest.approx(30.0 - 1.5)
    assert float(np.ravel(sigma)[0]) == pytest.approx(2.6)
    assert float(np.ravel(f)[0]) == pytest.approx(0.24)


def test_emg_params_reject_an_out_of_range_f_prj():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        EmgParams(-1.5, 3.0, 1.4, 0.12).at(30.0, 0.3)


# -- the mass-observable relations ------------------------------------------


@pytest.mark.parametrize("mor", [LogNormalMor(), HodMor()],
                         ids=["lognormal", "hod"])
def test_mor_pdf_normalises_to_one(mor):
    """A density, not a probability mass. Catches a missing Jacobian."""
    lam = np.linspace(1e-8, 900.0, 600001)
    for m in (1e14, 3e14):
        total = np.trapezoid(mor.pdf(lam, np.log(m), 0.3), x=lam)
        assert total == pytest.approx(1.0, abs=2e-3), m


@pytest.mark.parametrize("mor", [LogNormalMor(), HodMor()],
                         ids=["lognormal", "hod"])
def test_mor_pdf_is_non_negative_and_vanishes_below_zero(mor):
    lam = np.linspace(-5.0, 300.0, 500)
    p = mor.pdf(lam, np.log(1e14), 0.3)
    assert np.all(p >= 0.0)
    assert np.all(p[lam < 0.0] == 0.0)


def test_lognormal_mean_is_its_own_densitys_first_moment():
    r"""``mean`` must be the mean of ``pdf``, not its median or its mode."""
    mor = LogNormalMor()
    lam = np.linspace(1e-8, 900.0, 600001)
    for m in (1e14, 3e14):
        p = mor.pdf(lam, np.log(m), 0.3)
        moment = np.trapezoid(lam * p, x=lam) / np.trapezoid(p, x=lam)
        assert moment == pytest.approx(mor.mean(np.log(m), 0.3).item(),
                                       rel=1e-10), m


def test_the_hod_density_first_moment_sits_exactly_one_above_mean():
    r"""A pinned artifact of the continuous interpolation, not a bug.

    ``HodMor.mean`` returns the *model's* occupation,
    :math:`\lambda^{\rm cen} + \langle\lambda^{\rm sat}\rangle`. The
    continuous shifted-Poisson density's own first moment is exactly
    **1.0 higher** at :math:`\sigma_{\rm intr} = 0` -- the density
    interpolates a discrete law and its continuous moment is not obliged to
    match the discrete one.

    Pinned here so the offset cannot drift silently, and so nobody
    "corrects" ``mean`` into disagreeing with the calibrated relation.
    """
    mor = HodMor(sigma_intr=0.0)
    lam = np.linspace(1e-8, 900.0, 600001)
    # the offset tightens onto 1 as the occupancy grows: at mu_sat ~ 3 it
    # is 0.995, by mu_sat ~ 37 it is 1.000. Low occupancy is where a
    # continuous interpolation of a discrete law is least faithful.
    for m, tol in ((1e13, 1e-2), (1e14, 2e-3), (3e14, 1e-3)):
        p = mor.pdf(lam, np.log(m), 0.3)
        moment = np.trapezoid(lam * p, x=lam) / np.trapezoid(p, x=lam)
        offset = moment - mor.mean(np.log(m), 0.3).item()
        assert offset == pytest.approx(1.0, abs=tol), m


def test_the_hod_offset_grows_with_intrinsic_scatter():
    """And it is not confined to 1 once sigma_intr is on."""
    lam = np.linspace(1e-8, 1500.0, 900001)
    lm = np.log(1e15)
    offsets = []
    for s in (0.0, 0.24, 0.5):
        mor = HodMor(sigma_intr=s)
        p = mor.pdf(lam, lm, 0.3)
        moment = np.trapezoid(lam * p, x=lam) / np.trapezoid(p, x=lam)
        offsets.append(moment - mor.mean(lm, 0.3).item())
    assert offsets[0] == pytest.approx(1.0, abs=1e-2)
    assert offsets[1] > offsets[0]
    assert offsets[2] > offsets[1]


def test_the_hod_offset_is_negligible_for_bracket_placement():
    r"""Which is the only thing ``mean`` is used for.

    An offset of ~1 against a half-width of :math:`L\sigma_{\rm eff}` with
    :math:`L = 8` shifts the bracket by well under a percent of its width.
    """
    mor = HodMor()
    for m in (1e14, 1e15):
        lm = np.log(m)
        half_width = 8.0 * mor.std(lm, 0.3).item()
        assert 1.0 / half_width < 0.03, m


@pytest.mark.parametrize("mor", [LogNormalMor(), HodMor()],
                         ids=["lognormal", "hod"])
def test_mor_richness_rises_with_mass(mor):
    m = np.logspace(13.0, 15.0, 20)
    means = np.array([mor.mean(np.log(mi), 0.3).item() for mi in m])
    assert np.all(np.diff(means) > 0.0)


def test_lognormal_mean_is_not_the_median():
    r""":math:`e^{\mu+\sigma^2/2}`, not :math:`e^\mu`."""
    mor = LogNormalMor()
    lm = np.log(1e14)
    median = np.exp(mor.mean_ln_lambda(lm, 0.3))
    assert mor.mean(lm, 0.3).item() > median.item()


def test_lognormal_poisson_floor_subtracts_the_central_galaxy():
    r""":math:`(\langle\lambda\rangle-1)/\langle\lambda\rangle^2`,
    not :math:`1/\langle\lambda\rangle`."""
    mor = LogNormalMor(D_lambda=0.23)
    lm = np.log(1e14)
    mean_lambda = np.exp(mor.mean_ln_lambda(lm, 0.3)).item()
    expected = 0.23**2 + (mean_lambda - 1.0) / mean_lambda**2
    assert mor.var_ln_lambda(lm, 0.3).item() == pytest.approx(expected)
    # and the two candidate floors are genuinely different
    assert (mean_lambda - 1.0) / mean_lambda**2 != pytest.approx(
        1.0 / mean_lambda, rel=1e-4
    )


def test_lognormal_refuses_a_negative_variance():
    r"""The floor goes negative for :math:`\langle\lambda\rangle < 1`."""
    tiny = LogNormalMor(A_lambda=1e-3, D_lambda=1e-4)
    with pytest.raises(ValueError, match="not positive"):
        tiny.var_ln_lambda(np.log(1e10), 0.3)


def test_hod_central_is_a_step_at_Mmin():
    mor = HodMor(log10_Mmin=11.72)
    m_min = 10.0**11.72
    assert mor.lambda_central(np.log(m_min * 1.01)).item() == 1.0
    assert mor.lambda_central(np.log(m_min * 0.99)).item() == 0.0


def test_hod_satellite_mean_vanishes_below_Mmin():
    mor = HodMor(log10_Mmin=11.72)
    assert mor.mu_sat(np.log(10.0**11.7), 0.3).item() == 0.0
    assert mor.mu_sat(np.log(1e14), 0.3).item() > 0.0


def test_hod_scatter_is_poisson_at_zero_intrinsic_scatter():
    r""":math:`\sigma = \sqrt{\mu_{\rm sat}}` exactly when
    :math:`\sigma_{\rm intr} = 0`."""
    mor = HodMor(sigma_intr=0.0)
    lm = np.log(1e15)
    mu = mor.mu_sat(lm, 0.3).item()
    assert mor.std(lm, 0.3).item() == pytest.approx(np.sqrt(mu))


def test_hod_scatter_is_super_poisson_when_sigma_intr_is_on():
    lm = np.log(1e15)
    poisson = HodMor(sigma_intr=0.0).std(lm, 0.3).item()
    assert HodMor(sigma_intr=0.24).std(lm, 0.3).item() > poisson


def test_hod_rejects_M1_below_Mmin():
    with pytest.raises(ValueError, match="must exceed"):
        HodMor(log10_Mmin=13.0, log10_M1=12.0)


def test_hod_uses_gammaln_not_gamma():
    r"""Gamma overflows past :math:`x \sim 171`; richness reaches 200."""
    mor = HodMor()
    # a mass whose mu_sat puts the shifted argument well past 171
    lm = np.log(1e16)
    assert mor.mu_sat(lm, 0.3).item() > 171.0
    p = mor.pdf(250.0, lm, 0.3)
    assert np.isfinite(p).all() and p.item() > 0.0


# -- the selection function ------------------------------------------------


def _selection(**kw):
    return SelectionFunction(LAM_EDGES, Z_EDGES, LogNormalMor(), PARAMS,
                             sigma_z=0.01, **kw)


def test_S_ij_factorises_exactly():
    r""":math:`\mathcal S_{ij} = S_i\,\mathcal S_j` -- not approximately."""
    sel = _selection()
    lm, z = np.log(3e14), 0.3
    np.testing.assert_allclose(
        sel.S_ij(lm, z), np.outer(sel.S_i(lm, z), sel.S_j(z)), rtol=0.0,
        atol=0.0,
    )


def test_S_ij_has_the_documented_shape():
    sel = _selection()
    assert sel.S_ij(np.log(1e14), 0.3).shape == (4, 3)
    assert sel.S_i(np.log(1e14), 0.3).shape == (4,)
    assert sel.S_j(0.3).shape == (3,)


def test_S_i_is_a_probability():
    sel = _selection()
    for m in (1e13, 1e14, 1e15):
        s = sel.S_i(np.log(m), 0.3)
        assert np.all(s >= 0.0)
        assert s.sum() <= 1.0 + 1e-9


def test_S_i_is_not_monotonic_in_mass():
    r"""It peaks and falls: a 1e15 halo has
    :math:`\langle\lambda\rangle > 200` and leaves the binning.

    Asserting monotonicity here would be wrong, and is the kind of
    plausible-looking test that would have to be deleted later.
    """
    sel = _selection()
    sums = [sel.S_i(np.log(m), 0.3).sum() for m in (1e13, 1e14, 3e14, 1e15)]
    assert sums[0] < sums[1] < sums[2]
    assert sums[3] < sums[2]


def test_S_j_is_a_half_at_a_shared_edge_and_the_pair_sums_to_one():
    """No cluster is lost between contiguous redshift bins."""
    sel = _selection()
    s = sel.S_j(Z_EDGES[1])
    assert s[0] == pytest.approx(0.5, abs=1e-12)
    assert s[1] == pytest.approx(0.5, abs=1e-12)
    assert s.sum() == pytest.approx(1.0, abs=1e-12)


def test_the_bracket_is_clipped_at_zero_not_at_the_lowest_bin_edge():
    r"""Upscatter is the physical effect; clipping at
    :math:`\lambda_i^{\min}` would delete it."""
    sel = _selection()
    a, b = sel.bracket(np.log(1e13), 0.3)
    assert a == pytest.approx(0.0)                 # not 20.0
    assert b > 0.0
    # and the integrand is genuinely non-zero below the lowest edge
    assert sel.mor.pdf(5.0, np.log(1e13), 0.3).item() > 0.0


def test_a_wider_bracket_captures_more_probability():
    """And the residual reports exactly how much was missing."""
    lm = np.log(1e14)
    narrow, wide = _selection(bracket_width=2.0), _selection(
        bracket_width=12.0)
    assert narrow.residual(lm, 0.3) > wide.residual(lm, 0.3)
    assert narrow.S_i(lm, 0.3).sum() < wide.S_i(lm, 0.3).sum()
    # at L = 12 the bracket is essentially complete
    assert wide.residual(lm, 0.3) < 1e-6


def test_the_residual_predicts_the_shortfall():
    r"""The named approximation, made measurable.

    The probability the bracket misses must account for the gap between a
    narrow bracket's :math:`S_i` sum and a converged one's -- it cannot
    exceed it, since only part of the missed mass would have landed inside
    the richness bins at all.
    """
    lm = np.log(1e14)
    narrow, converged = _selection(bracket_width=3.0), _selection(
        bracket_width=14.0)
    shortfall = (converged.S_i(lm, 0.3).sum() - narrow.S_i(lm, 0.3).sum())
    missed = narrow.residual(lm, 0.3)
    assert 0.0 < shortfall <= missed + 1e-12


def test_the_quadrature_is_converged_at_the_default_order():
    lm = np.log(1e14)
    a, b = _selection(n_quad=64), _selection(n_quad=192)
    np.testing.assert_allclose(a.S_i(lm, 0.3), b.S_i(lm, 0.3), rtol=1e-6)


def test_selection_function_is_vectorised_over_mass():
    sel = _selection()
    lm = np.log(np.array([1e13, 1e14, 1e15]))
    got = sel.S_i(lm, 0.3)
    assert got.shape == (3, 4)
    for i, single in enumerate(lm):
        np.testing.assert_allclose(got[i], sel.S_i(single, 0.3), rtol=1e-12)


def test_the_mor_is_swappable():
    """The point of the interface: three methods, two implementations."""
    log_normal = SelectionFunction(LAM_EDGES, Z_EDGES, LogNormalMor(),
                                   PARAMS, sigma_z=0.01)
    hod = SelectionFunction(LAM_EDGES, Z_EDGES, HodMor(), PARAMS,
                            sigma_z=0.01)
    lm = np.log(1e14)
    assert log_normal.S_i(lm, 0.3).sum() != pytest.approx(
        hod.S_i(lm, 0.3).sum(), rel=1e-3
    )


def test_sigma_z_may_be_per_bin():
    sel = SelectionFunction(LAM_EDGES, Z_EDGES, LogNormalMor(), PARAMS,
                            sigma_z=[0.01, 0.015, 0.02])
    np.testing.assert_allclose(sel.sigma_z, [0.01, 0.015, 0.02])
    with pytest.raises(ValueError, match="one value per redshift bin"):
        SelectionFunction(LAM_EDGES, Z_EDGES, LogNormalMor(), PARAMS,
                          sigma_z=[0.01, 0.02])


def test_selection_function_rejects_bad_edges():
    with pytest.raises(ValueError, match="ascending"):
        SelectionFunction([30.0, 20.0], Z_EDGES, LogNormalMor(), PARAMS,
                          sigma_z=0.01)
    with pytest.raises(ValueError, match="positive"):
        SelectionFunction(LAM_EDGES, Z_EDGES, LogNormalMor(), PARAMS,
                          sigma_z=0.0)
