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
    richness_edge_steps,
)
from clenspy.selection.geometry import area_overlap
from clenspy.selection.selection_function import EDGE_WIDTH, N_QUAD

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


def test_emg_pdf_rejects_non_positive_parameters():
    from clenspy.selection.richness_kernel import emg_pdf

    with pytest.raises(ValueError, match="sigma"):
        emg_pdf(30.0, 35.0, 0.0, 0.1)
    with pytest.raises(ValueError, match="tau"):
        emg_pdf(30.0, 35.0, 3.0, 0.0)


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


def test_emg_params_repr_shows_scalars_formatted():
    r = repr(EmgParams(delta_mu=-1.5, sigma=3.0, f_prj=0.3, tau=0.12))
    assert r.startswith("EmgParams(")
    assert "delta_mu=-1.5" in r
    assert "sigma=3" in r
    assert "f_prj=0.3" in r
    assert "tau=0.12" in r
    assert "mu_slope=1" in r
    assert "callable" not in r


def test_emg_params_repr_shows_callable_fields():
    r = repr(EmgParams(delta_mu=lambda lam, z: -1.5, sigma=3.0, f_prj=0.3,
                        tau=0.12))
    assert "delta_mu=callable" in r
    assert "sigma=3" in r  # the scalar branch still fires for the others


def test_richness_bin_first_moment_rejects_bad_edges():
    from clenspy.selection.richness_kernel import richness_bin_first_moment

    with pytest.raises(ValueError, match="ascending"):
        richness_bin_first_moment([30.0, 20.0], 25.0, 0.3, PARAMS)
    with pytest.raises(ValueError, match="at least two"):
        richness_bin_first_moment([20.0], 25.0, 0.3, PARAMS)


def test_emg_params_from_y3_table_rejects_wrong_row_count():
    import tempfile

    from clenspy.selection.richness_kernel import Y3_PRJ_COEFFICIENTS

    ncol = len(Y3_PRJ_COEFFICIENTS)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        row = " ".join(["1.0"] * ncol) + "\n"
        fh.write(row * 3)          # only 3 rows, but 15 z-nodes expected
        bad = fh.name
    with pytest.raises(ValueError, match="redshift rows"):
        EmgParams.from_y3_table(bad)


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


def test_the_hod_density_first_moment_matches_mean():
    r"""``HodMor.mean`` (:math:`\lambda^{\rm cen} +
    \langle\lambda^{\rm sat}\rangle`) must equal `pdf`'s own first moment,
    the same contract `LogNormalMor` satisfies.

    Regression for issue #5: an earlier version of `pdf` used
    ``x = lambda_true - lambda_central + delta``, double-counting the
    central galaxy against the shifted-Poisson family's own built-in +1
    (module docstring) and inflating this moment by exactly 1.0 richness
    unit at every mass -- silently biasing every lambda-weighted integral
    that consumes `pdf` directly (`SelBiasEngine.operators`'s P1/I1/I2).
    See ``validation/validate_mor_notebook.py``.
    """
    mor = HodMor(sigma_intr=0.0)
    lam = np.linspace(1e-8, 900.0, 600001)
    # exact at moderate-to-high occupancy; the residual continuous-vs-
    # discrete gap only reopens at mu_sat ~ 3 (M=1e13), and even there it
    # is 1.4% of the mean, not the old bug's flat +1 absolute unit.
    for m, tol in ((1e13, 6e-2), (1e14, 1e-3), (3e14, 1e-3)):
        p = mor.pdf(lam, np.log(m), 0.3)
        moment = np.trapezoid(lam * p, x=lam) / np.trapezoid(p, x=lam)
        offset = moment - mor.mean(np.log(m), 0.3).item()
        assert offset == pytest.approx(0.0, abs=tol), m


def test_the_hod_moment_match_holds_with_intrinsic_scatter():
    """Offset stays small and grows gently with sigma_intr, unlike the old
    bug's flat +1 unit regardless of scatter."""
    lam = np.linspace(1e-8, 1500.0, 900001)
    lm = np.log(1e15)
    offsets = []
    for s in (0.0, 0.24, 0.5):
        mor = HodMor(sigma_intr=s)
        p = mor.pdf(lam, lm, 0.3)
        moment = np.trapezoid(lam * p, x=lam) / np.trapezoid(p, x=lam)
        mean = mor.mean(lm, 0.3).item()
        offsets.append((moment - mean) / mean)
    assert offsets[0] == pytest.approx(0.0, abs=1e-4)
    assert 0.0 <= offsets[1] < offsets[2] < 0.05


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


def test_lognormal_repr_contains_class_name():
    assert "LogNormalMor" in repr(LogNormalMor())


def test_hod_repr_contains_class_name():
    assert "HodMor" in repr(HodMor())


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
    r"""Default panels against three times the nodes, over the masses where
    one 64-node rule across the whole bracket rippled by 1% of the
    :math:`[60, 200)` peak."""
    lm = np.log(np.geomspace(1e14, 10**15.5, 41))
    a = _selection().S_i(lm, 0.4)
    b = _selection(n_quad=3 * N_QUAD).S_i(lm, 0.4)
    assert np.max(np.abs(a - b) / b.max(axis=0)) < 1e-8


@pytest.mark.parametrize("mor, params", [
    (LogNormalMor(), PARAMS),
    (HodMor.buzzard(), PARAMS),
    (HodMor.buzzard(), EmgParams.from_y3_table()),
])
def test_S_i_converges_exponentially_in_the_panel_order(mor, params):
    r"""Every four nodes per panel buy two orders of magnitude -- the
    signature of an integrand that is smooth on every panel.

    A rule straddling a step resolves it only once the nodes are dense on
    the step's own scale, and one straddling a jump (the HOD density at
    :math:`\lambda^{\rm cen}`) gains a factor of two per doubling at best.
    """
    sel = SelectionFunction(LAM_EDGES, Z_EDGES, mor, params, sigma_z=0.01,
                            n_quad=96)
    lm = np.log(np.geomspace(1e13, 10**15.5, 61))
    ref = sel.S_i(lm, 0.4)
    err = []
    for n in (4, 8, 12):
        s = SelectionFunction(LAM_EDGES, Z_EDGES, mor, params, sigma_z=0.01,
                              n_quad=n).S_i(lm, 0.4)
        err.append(np.max(np.abs(s - ref) / ref.max(axis=0)))
    assert err[0] / err[1] > 30.0
    assert err[1] / err[2] > 30.0


def test_S_i_matches_adaptive_quadrature_where_one_rule_rippled():
    r"""Independent of the panel layout: `scipy.integrate.quad_vec`, told
    only where the steps are, at the masses where one 64-node rule missed
    :math:`S_i` of the :math:`[60, 200)` bin by up to 1.3% of its peak."""
    from scipy.integrate import quad_vec

    sel, z = _selection(), 0.4
    for m in (8e14, 1.14e15, 1.5e15):
        lm = np.log(m)
        a, b = (float(np.ravel(v)[0]) for v in sel.bracket(lm, z))
        # this kernel steps at lambda_tr = edge - delta_mu
        points = [e + 1.5 for e in LAM_EDGES if a < e + 1.5 < b]
        reference, _ = quad_vec(
            lambda lam: sel.mor.pdf(lam, lm, z) * richness_bin_probability(
                LAM_EDGES, lam, z, PARAMS),
            a, b, points=points, epsabs=1e-14, epsrel=1e-12, limit=2000,
        )
        np.testing.assert_allclose(sel.S_i(lm, z), np.ravel(reference),
                                   rtol=0.0, atol=1e-9)


def test_the_production_setup_matches_adaptive_quadrature():
    r"""`HodMor.buzzard` with the Y3 kernel, the mock-validation setup."""
    from scipy.integrate import quad_vec

    params, z = EmgParams.from_y3_table(), 0.4
    sel = SelectionFunction(LAM_EDGES, Z_EDGES, HodMor.buzzard(), params,
                            sigma_z=0.01)
    centre, _ = richness_edge_steps(LAM_EDGES, z, params)
    for m in (3e13, 3e14, 1e15):
        lm = np.log(m)
        a, b = (float(np.ravel(v)[0]) for v in sel.bracket(lm, z))
        points = sorted(x for x in [*centre, 1.0] if a < x < b)
        reference, _ = quad_vec(
            lambda lam: sel.mor.pdf(lam, lm, z) * richness_bin_probability(
                LAM_EDGES, lam, z, params),
            a, b, points=points, epsabs=1e-14, epsrel=1e-12, limit=2000,
        )
        np.testing.assert_allclose(sel.S_i(lm, z), np.ravel(reference),
                                   rtol=0.0, atol=1e-12)


def test_first_moment_i_converges_at_the_default_order():
    r"""The same rule, contracted against :math:`M_i` -- with the Y3 kernel,
    whose own :math:`\lambda^{\rm ob}` rule is converged in these bins."""
    params = EmgParams.from_y3_table()
    lm = np.log(np.geomspace(1e13, 10**15.5, 41))
    a = SelectionFunction(LAM_EDGES, Z_EDGES, HodMor.buzzard(), params,
                          sigma_z=0.01).first_moment_i(lm, 0.4)
    b = SelectionFunction(LAM_EDGES, Z_EDGES, HodMor.buzzard(), params,
                          sigma_z=0.01, n_quad=3 * N_QUAD).first_moment_i(
                              lm, 0.4)
    assert np.max(np.abs(a - b) / b.max(axis=0)) < 1e-10


def test_the_panels_tile_the_bracket():
    """Ascending from a to b, with every break inside the bracket a panel
    boundary: both ends of each edge's step panel, and the relation's
    mean."""
    sel, z = _selection(), 0.4
    lm = np.log(np.geomspace(1e13, 10**15.5, 7))
    edges = sel.panel_edges(lm, z)
    a, b = sel.bracket(lm, z)
    np.testing.assert_array_equal(edges[:, 0], a)
    np.testing.assert_array_equal(edges[:, -1], b)
    assert np.all(np.diff(edges, axis=-1) >= 0.0)
    centre, width = richness_edge_steps(LAM_EDGES, z, PARAMS)
    breaks = np.concatenate([centre - EDGE_WIDTH * width,
                             centre + EDGE_WIDTH * width])
    for row, lo, hi, mu in zip(edges, a, b, sel.mor.mean(lm, z)):
        for x in (*breaks, mu):
            if lo < x < hi:
                assert np.min(np.abs(row - x)) < 1e-12


def test_the_hod_jump_is_a_panel_boundary():
    r""":math:`\lambda^{\rm cen} = 1` is a panel edge whenever the bracket
    reaches below it -- the HOD density jumps there."""
    sel = SelectionFunction(LAM_EDGES, Z_EDGES, HodMor.buzzard(), PARAMS,
                            sigma_z=0.01)
    lm = np.log(3e13)
    assert sel.bracket(lm, 0.4)[0] < 1.0
    assert 1.0 in sel.panel_edges(lm, 0.4)


def test_hod_support_min_is_where_the_density_jumps():
    mor = HodMor.buzzard()
    m_min = 10.0**mor.log10_Mmin
    np.testing.assert_array_equal(
        mor.support_min(np.log([0.5 * m_min, 2.0 * m_min, 1e14]), 0.4),
        [0.0, 1.0, 1.0],
    )
    # <lambda_sat> = 1.8 at 1e13: a jump of 0.16 per unit richness
    lm = np.log(1e13)
    assert mor.pdf(1.0 - 1e-9, lm, 0.4).item() == 0.0
    assert mor.pdf(1.0, lm, 0.4).item() > 0.1


def test_richness_edge_steps_sit_where_the_kernel_centre_crosses_the_edge():
    r""":math:`\mu(\lambda^{\rm tr}_e) = \lambda_e`, and the width is
    :math:`\sigma/b_\mu` -- with the production fit, whose :math:`b_\mu`
    is not 1."""
    params = EmgParams.from_y3_table()
    for z in (0.2, 0.5, 0.8):
        centre, width = richness_edge_steps(LAM_EDGES, z, params)
        mu, sigma, _, _ = params.at(centre, z)
        np.testing.assert_allclose(mu, LAM_EDGES, rtol=1e-12)
        np.testing.assert_allclose(
            width, sigma / params.mu_slope(centre, z), rtol=1e-12
        )
    centre, width = richness_edge_steps(LAM_EDGES, np.array([0.2, 0.5]),
                                        params)
    assert centre.shape == width.shape == (2, LAM_EDGES.size)


def test_without_projection_an_edge_step_is_half_way_at_its_centre():
    """The Gaussian kernel puts exactly half its probability above the edge
    at the step's centre."""
    gauss = EmgParams(delta_mu=-1.5, sigma=3.0, f_prj=0.0, tau=0.12)
    centre, width = richness_edge_steps(LAM_EDGES, 0.3, gauss)
    np.testing.assert_allclose(centre, LAM_EDGES + 1.5)
    np.testing.assert_allclose(width, 3.0)
    for edge, c in zip(LAM_EDGES, centre):
        above = richness_bin_probability([edge, 1e6], c, 0.3, gauss)
        assert above.item() == pytest.approx(0.5, abs=1e-12)


def test_a_degenerate_bracket_integrates_to_zero():
    r"""`HodMor` below :math:`M_{\min}`: mean and width both zero, so every
    panel is empty and the kernel is never evaluated -- at
    :math:`\lambda^{\rm tr} = 0` the Y3 :math:`\sigma` vanishes and
    `emg_cdf` would refuse it."""
    sel = SelectionFunction(LAM_EDGES, Z_EDGES, HodMor.buzzard(),
                            EmgParams.from_y3_table(), sigma_z=0.01)
    lm = np.log(1e11)
    np.testing.assert_array_equal(sel.S_i(lm, 0.4), 0.0)
    np.testing.assert_array_equal(sel.first_moment_i(lm, 0.4), 0.0)
    assert sel.residual(lm, 0.4) == 1.0
    # and alongside a live cell, without disturbing it
    both = sel.S_i(np.log([1e11, 1e14]), 0.4)
    np.testing.assert_array_equal(both[0], 0.0)
    np.testing.assert_allclose(both[1], sel.S_i(np.log(1e14), 0.4),
                               rtol=1e-14)


def test_n_quad_must_be_a_positive_integer():
    for bad in (0, -3, 2.5):
        with pytest.raises(ValueError, match="n_quad"):
            _selection(n_quad=bad)


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


def test_selection_function_rejects_too_short_edges():
    with pytest.raises(ValueError, match="lambda_edges.*>= 2 entries"):
        SelectionFunction([20.0], Z_EDGES, LogNormalMor(), PARAMS,
                          sigma_z=0.01)
    with pytest.raises(ValueError, match="z_edges.*>= 2 entries"):
        SelectionFunction(LAM_EDGES, [0.2], LogNormalMor(), PARAMS,
                          sigma_z=0.01)


def test_n_lambda_bins_property():
    sel = _selection()
    assert sel.n_lambda_bins == LAM_EDGES.size - 1 == 4


def test_selection_function_repr_contains_class_name():
    assert "SelectionFunction" in repr(_selection())


# -- the calibrated parameter sets ------------------------------------------


def test_the_two_hod_parameter_sets_differ_only_in_the_evolution():
    """des_y1 keeps epsilon = 0; buzzard carries the epsilon fix."""
    a, b = HodMor.des_y1(), HodMor.buzzard()
    assert a.log10_Mmin == b.log10_Mmin
    assert a.log10_M1 == b.log10_M1
    assert a.alpha == b.alpha
    assert a.sigma_intr == b.sigma_intr
    assert a.epsilon == 0.0
    assert b.epsilon == pytest.approx(0.283887020)
    assert b.z_pivot == pytest.approx(0.4544)


def test_the_buzzard_set_tilts_the_evolution_not_the_amplitude():
    r"""A 9% swing across the DES Y1 range, and a tilt, not an offset.

    An offset would absorb into the amplitude and be harmless for a mass
    calibration. A tilt does not, which is why Buzzard comparisons must use
    the Buzzard set.
    """
    a, b = HodMor.des_y1(), HodMor.buzzard()
    lm = np.log(3e14)
    ratios = np.array([b.mu_sat(lm, z).item() / a.mu_sat(lm, z).item()
                       for z in (0.20, 0.35, 0.50, 0.65)])
    assert np.all(np.diff(ratios) > 0.0)              # monotonic tilt
    assert ratios[0] == pytest.approx(0.947, abs=2e-3)
    assert ratios[-1] == pytest.approx(1.036, abs=2e-3)
    assert ratios.max() / ratios.min() == pytest.approx(1.094, abs=5e-3)


def test_the_parameter_sets_stay_in_hinv_msun():
    r"""The exponents are used as written -- no h anywhere.

    The source stores :math:`10^{\log_{10}M}/h` in Msun; this class keeps
    :math:`h^{-1}M_\odot`, so the tabulated exponent is the value. One
    fewer place for an h to be applied twice.
    """
    assert HodMor.buzzard().log10_Mmin == pytest.approx(11.3852818)
    assert HodMor.buzzard().log10_M1 == pytest.approx(12.6964410)


# -- the EMG density and the production coefficient table ------------------


def test_emg_pdf_is_the_derivative_of_emg_cdf():
    """The only check that pins both at once."""
    from clenspy.selection.richness_kernel import emg_pdf

    mu, sigma, tau = 35.0, 3.0, 0.12
    x = np.linspace(20.0, 120.0, 25)
    h = 1e-5
    fd = (emg_cdf(x + h, mu, sigma, tau)
          - emg_cdf(x - h, mu, sigma, tau)) / (2 * h)
    np.testing.assert_allclose(emg_pdf(x, mu, sigma, tau), fd, rtol=1e-5)


def test_emg_pdf_normalises_to_one():
    from clenspy.selection.richness_kernel import emg_pdf

    x = np.linspace(0.0, 800.0, 800001)
    for sigma, tau in ((3.0, 0.12), (5.0, 0.5), (1.0, 2.0)):
        total = np.trapezoid(emg_pdf(x, 35.0, sigma, tau), x=x)
        assert total == pytest.approx(1.0, abs=1e-6), (sigma, tau)


def test_emg_pdf_survives_where_the_naive_product_would_not():
    """Same erfcx cure as the CDF, and the same failure without it."""
    from clenspy.selection.richness_kernel import emg_pdf

    for tau_sigma in (40.0, 80.0):
        got = emg_pdf(40.0, 35.0, 3.0, tau_sigma / 3.0).item()
        assert np.isfinite(got) and got > 0.0


def test_richness_pdf_is_the_mixture_and_normalises():
    from clenspy.selection.richness_kernel import emg_pdf, richness_pdf

    params = EmgParams(-1.5, 3.0, 0.3, 0.12)
    x = np.linspace(0.0, 800.0, 800001)
    assert np.trapezoid(richness_pdf(x, 35.0, 0.3, params), x=x) == (
        pytest.approx(1.0, abs=1e-6)
    )
    # and it really is (1-f) N + f EMG
    mu, sigma, tau, f = params.at(35.0, 0.3)
    gauss = np.exp(-0.5 * ((40.0 - mu) / sigma) ** 2) / (
        sigma * np.sqrt(2 * np.pi))
    expected = (1 - f) * gauss + f * emg_pdf(40.0, mu, sigma, tau)
    assert float(np.ravel(richness_pdf(40.0, 35.0, 0.3, params))[0]) == (
        pytest.approx(float(np.ravel(expected)[0]))
    )


def test_the_mu_slope_is_not_one_in_the_production_fit():
    r"""So hardcoding :math:`\mu = \lambda^{\rm tr} + \Delta\mu` is wrong.

    :math:`b_\mu` runs 0.984 to 1.172 across the table -- up to 17% -- which
    is why `EmgParams` carries ``mu_slope`` instead of assuming it.
    """
    params = EmgParams.from_y3_table()
    lam = 100.0
    for z, expected_min in ((0.2, 0.98), (0.8, 1.10)):
        mu = float(np.ravel(params.at(lam, z)[0])[0])
        slope = float(np.ravel(params.mu_slope(lam, z))[0])
        assert slope > expected_min
        # mu is slope*lam + a_mu, not lam + a_mu
        assert mu != pytest.approx(lam + float(
            np.ravel(params.delta_mu(lam, z))[0]), rel=1e-6)


def test_the_production_table_gives_a_normalised_kernel():
    from clenspy.selection.richness_kernel import richness_pdf

    params = EmgParams.from_y3_table()
    x = np.linspace(0.0, 1500.0, 1500001)
    for lam, z in ((20.0, 0.2), (40.0, 0.5), (100.0, 0.8)):
        total = np.trapezoid(richness_pdf(x, lam, z, params), x=x)
        assert total == pytest.approx(1.0, abs=1e-5), (lam, z)


def test_the_production_bin_probabilities_are_probabilities():
    params = EmgParams.from_y3_table()
    for lam in (20.0, 40.0, 100.0):
        s = richness_bin_probability(LAM_EDGES, lam, 0.5, params)
        assert np.all(s >= 0.0)
        assert s.sum() <= 1.0 + 1e-9


def test_f_prj_is_clipped_at_one_by_the_fit():
    """b_f reaches 0.998 and the sigmoid can push the raw value past 1."""
    params = EmgParams.from_y3_table()
    for z in np.linspace(0.1, 0.8, 15):
        for lam in (20.0, 60.0, 200.0):
            f = float(np.ravel(params.at(lam, z)[3])[0])
            assert 0.0 <= f <= 1.0, (lam, z)


def test_the_masking_coefficients_are_read_and_discarded():
    """Explicitly, so their absence is visible rather than a silent slice."""
    params = EmgParams.from_y3_table()
    assert params.unused_coefficients == ("afmsk", "bfmsk")


def test_the_production_table_shape_is_validated():
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("1.0 2.0 3.0\n4.0 5.0 6.0\n")
        bad = fh.name
    with pytest.raises(ValueError, match="coefficient columns"):
        EmgParams.from_y3_table(bad)


# -- the aperture geometry: area_overlap's elementwise branch ---------------


def test_area_overlap_pairs_elementwise_when_theta_matches_ltrs_shape():
    r"""``theta.shape[-1] == theta_ltr.shape[-1]`` -- the pairing branch.

    When ``theta`` is a 1-D array whose length equals ``theta_ltr``'s, the
    function pairs them index-by-index rather than broadcasting into the
    full :math:`(N_\theta, N_{\lambda^{\rm tr}})` grid the general branch
    would produce. The proof is the shape itself: the general broadcast of
    two length-4 arrays would be ``(4, 4)``, not ``(4,)``. Cross-checking
    against scalar-by-scalar calls confirms it is the *same* pairing, not
    some other reduction.
    """
    theta_lob = 1e-3
    theta_ltr = np.array([0.5e-3, 1.0e-3, 1.5e-3, 2.0e-3])
    theta = np.array([0.4e-3, 0.9e-3, 1.6e-3, 2.5e-3])

    out = area_overlap(theta, theta_lob, theta_ltr)
    assert out.shape == (4,)          # not (4, 4): the pairing branch fired

    expected = np.array([
        area_overlap(theta[i], theta_lob, theta_ltr[i]).item()
        for i in range(theta.size)
    ])
    np.testing.assert_allclose(out, expected, rtol=1e-14)


def test_area_overlap_pairing_branch_agrees_with_a_scalar_theta_lob_sweep():
    """Same pairing branch, checked against the geometry it encodes.

    A target strictly inside a much larger projector must be fully
    contained (area = (theta_lob/theta_ltr)^2), and a target far outside a
    much smaller, distant projector must have zero overlap -- both read
    off the paired output, not a broadcast grid.
    """
    theta_lob = 2.0e-3
    theta_ltr = np.array([10.0e-3, 1.0e-6])
    theta = np.array([0.0, 1.0])          # separations [rad]

    out = area_overlap(theta, theta_lob, theta_ltr)
    assert out.shape == (2,)
    assert out[0] == pytest.approx((theta_lob / theta_ltr[0]) ** 2)
    assert out[1] == pytest.approx(0.0)
