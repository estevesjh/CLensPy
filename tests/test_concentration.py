r"""The concentration--mass relations, and the ways they get misused.

Three kinds of check here:

- **transcription** -- the coefficient tables against the published values,
  and each formula against a hand-written second copy;
- **limits** -- the plateau :math:`c \to c_0` above :math:`M_T`, the power
  law below it, the sign of every derivative;
- **the traps** -- the mass definition, the :math:`M_\star` normalisation,
  and the two unrelated quantities both called :math:`\delta_c`. These are
  the ones that produce a plausible wrong number.
"""

import numpy as np
import pytest

from clenspy.cosmology.concentration import (
    CHILD18_TABLE1,
    CHILD18_TABLE2,
    DELTA_COLLAPSE,
    DUFFY08_PIVOT_HINV,
    DUFFY08_TABLE1,
    Y3_FIXED_CONCENTRATION,
    child18,
    child18_powerlaw,
    delta_c,
    duffy08,
    m_star_from_sigma,
    m_star_hinv,
    scatter,
)

M14 = 1.0e14  # h^-1 Msun, cluster scale


# -- transcription ----------------------------------------------------------


def test_child18_table1_is_the_published_table():
    """Child et al. (2018) Table 1, (m, A, M_T/M*, c_0), typed from the paper."""
    assert CHILD18_TABLE1 == {
        "individual_all": (-0.10, 3.44, 430.49, 3.19),
        "individual_relaxed": (-0.09, 2.88, 1644.53, 3.54),
        "stacked_nfw": (-0.07, 4.61, 638.65, 3.59),
        "stacked_einasto": (-0.01, 63.2, 431.48, 3.36),
    }


def test_child18_table2_is_the_published_table():
    """Child et al. (2018) Table 2, (A, d, m) for Eq. 19."""
    assert CHILD18_TABLE2 == {
        "individual_all": (75.4, -0.422, -0.089),
        "individual_relaxed": (68.4, -0.347, -0.083),
        "stacked_nfw": (57.6, -0.376, -0.078),
        "stacked_einasto": (122.0, -0.446, -0.101),
    }


def test_duffy08_table1_is_the_published_table():
    """Duffy et al. (2008) Table 1, full sample, keyed by mass definition."""
    assert DUFFY08_TABLE1 == {
        "vir": (7.85, -0.081, -0.71),
        "200m": (10.14, -0.081, -1.01),
        "200c": (5.71, -0.084, -0.47),
    }
    assert DUFFY08_PIVOT_HINV == 2.0e12


def test_child18_matches_eq18_written_out():
    """A second, independent copy of the algebra."""
    m, A, b, c0 = CHILD18_TABLE1["individual_all"]
    ms = m_star_hinv(0.3)
    x = (M14 / ms) / b
    expected = A * (x**m * (1.0 + x) ** (-m) - 1.0) + c0
    assert child18(M14, 0.3, ms).item() == pytest.approx(expected, rel=1e-14)


def test_child18_powerlaw_matches_eq19_written_out():
    A, d, m = CHILD18_TABLE2["individual_all"]
    expected = A * 1.3**d * M14**m
    assert child18_powerlaw(M14, 0.3).item() == pytest.approx(
        expected, rel=1e-14
    )


def test_duffy08_matches_the_written_out_power_law():
    A, B, C = DUFFY08_TABLE1["200m"]
    expected = A * (M14 / 2.0e12) ** B * 1.3**C
    assert duffy08(M14, 0.3).item() == pytest.approx(expected, rel=1e-14)


def test_duffy08_agrees_with_pyccl():
    """The same relation, from an independent implementation."""
    ccl = pytest.importorskip("pyccl")
    h = 0.7
    cosmo = ccl.Cosmology(Omega_c=0.24, Omega_b=0.046, h=h, sigma8=0.82,
                          n_s=0.96)
    for md in ("vir", "200m", "200c"):
        ccl_c = ccl.halos.ConcentrationDuffy08(mass_def=md)
        for z in (0.0, 0.3, 1.0):
            # pyccl takes h-free mass, this module takes h^-1 Msun
            mine = duffy08(M14, z, mass_def=md).item()
            theirs = ccl_c(cosmo, M14 / h, 1.0 / (1.0 + z))
            assert mine == pytest.approx(theirs, rel=1e-12)


# -- the M_star anchor line -------------------------------------------------


def test_m_star_reproduces_child18_quoted_anchors():
    r"""log10(M*/h^-1 Msun) = 12.5, 11, 9.5, 8 at z = 0, 1, 2, 3."""
    z = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.log10(m_star_hinv(z)),
                               [12.5, 11.0, 9.5, 8.0], atol=1e-12)


def test_m_star_falls_steeply_and_monotonically():
    z = np.linspace(0.0, 3.0, 40)
    ms = m_star_hinv(z)
    assert np.all(np.diff(ms) < 0.0)
    # 1.5 dex per unit redshift
    assert np.log10(ms[0] / m_star_hinv(1.0)) == pytest.approx(1.5)


def test_the_y3_commented_out_m_star_approximation_is_bad_at_low_z():
    """Why it stays commented out. Recorded so nobody revives it.

    The y3 source carries ``Mstarc = 10**(14.76 * .808**z)``. It converges
    onto the anchors only for z >~ 2, and at z = 0 it is 180x too high --
    which is precisely the cluster regime.
    """
    def y3_approx(z):
        return 10.0 ** (14.76 * 0.808**z)

    assert y3_approx(0.0) / m_star_hinv(0.0) > 100.0
    # by z = 2 it has come good
    assert y3_approx(2.0) / m_star_hinv(2.0) == pytest.approx(1.0, rel=0.5)


# -- limits and shape -------------------------------------------------------


def test_child18_asymptotes_to_the_plateau_above_the_threshold():
    r""":math:`c \to c_0` for :math:`M \gg M_T = b M_\star`."""
    for fit, (_, _, b, c0) in CHILD18_TABLE1.items():
        ms = m_star_hinv(0.0)
        c = child18(1e8 * b * ms, 0.0, ms, fit=fit).item()
        assert c == pytest.approx(c0, rel=2e-2), fit


def test_child18_plateau_is_between_three_and_four():
    """The paper's claim about c_0, checked against the table."""
    for _, _, _, c0 in CHILD18_TABLE1.values():
        assert 3.0 < c0 < 4.0


def test_child18_falls_with_mass_and_with_redshift():
    ms = m_star_hinv(0.3)
    m = np.logspace(13.0, 15.5, 30)
    assert np.all(np.diff(child18(m, 0.3, ms)) < 0.0)
    # and with z, at fixed mass, purely because M_star drops
    z = np.linspace(0.05, 1.0, 20)
    assert np.all(np.diff(child18(M14, z, m_star_hinv(z))) < 0.0)


def test_child18_is_redshift_independent_at_fixed_m_over_mstar():
    """The whole point of the M/M* scaling: one relation for all z."""
    ratio = 100.0
    cs = [child18(ratio * m_star_hinv(z), z, m_star_hinv(z)).item()
          for z in (0.0, 0.5, 1.0, 2.0, 3.0)]
    np.testing.assert_allclose(cs, cs[0], rtol=1e-12)


def test_the_two_child18_fits_agree_where_both_are_valid():
    r"""Eq. 18 and Eq. 19 are independent fits to the same simulations.

    They are calibrated on different redshift ranges and different
    functional forms, so agreement at the few-percent level is a real
    cross-check on both transcriptions -- a typo in either table would
    show up here.
    """
    for z in (0.0, 0.3, 0.6, 1.0):
        c18 = child18(M14, z, m_star_hinv(z)).item()
        c19 = child18_powerlaw(M14, z).item()
        assert c18 == pytest.approx(c19, rel=0.05), z


def test_child18_sits_above_duffy08_at_cluster_scales():
    """Child et al. Fig. 12: Duffy et al. (WMAP-5, low sigma_8) falls below."""
    for z in (0.0, 0.3, 0.6):
        assert (child18(M14, z, m_star_hinv(z)).item()
                > duffy08(M14, z, mass_def="200c").item())


def test_duffy08_falls_with_mass_and_redshift_in_every_row():
    for md in DUFFY08_TABLE1:
        m = np.logspace(12.0, 15.5, 30)
        assert np.all(np.diff(duffy08(m, 0.3, mass_def=md)) < 0.0)
        z = np.linspace(0.0, 2.0, 20)
        assert np.all(np.diff(duffy08(M14, z, mass_def=md)) < 0.0)


def test_duffy08_is_A_at_the_pivot_and_z_zero():
    for md, (A, _, _) in DUFFY08_TABLE1.items():
        got = duffy08(DUFFY08_PIVOT_HINV, 0.0, mass_def=md).item()
        assert got == pytest.approx(A)


def test_concentrations_are_physical_over_the_cluster_range():
    """No relation may return an unphysical c anywhere we will call it."""
    m = np.logspace(13.5, 15.5, 25)
    for z in (0.05, 0.35, 0.65):
        for c in (child18(m, z, m_star_hinv(z)),
                  child18_powerlaw(m, min(z, 1.0)),
                  duffy08(m, z)):
            assert np.all(c > 1.0) and np.all(c < 12.0)


# -- the traps --------------------------------------------------------------


def test_the_mass_definition_changes_duffy08_by_tens_of_percent():
    """Why `mass_def` is required in the signature and not a global."""
    c_200m = duffy08(M14, 0.3, mass_def="200m").item()
    c_200c = duffy08(M14, 0.3, mass_def="200c").item()
    assert c_200m / c_200c > 1.5


def test_clenspy_default_mass_def_is_200m_not_pyccls_200c():
    """A silent 55% error if a caller assumes pyccl's default."""
    assert duffy08(M14, 0.3).item() == pytest.approx(
        duffy08(M14, 0.3, mass_def="200m").item()
    )


def test_m_star_error_biases_concentration_the_stated_way():
    r"""The comoving/physical trap in `m_star_from_sigma`, as a number.

    Using :math:`\rho_m(z)` where the comoving :math:`\Omega_{m,0}
    \rho_{c,0}` belongs inflates :math:`M_\star` by :math:`(1+z)^3`. On
    Eq. 18's sloped branch a larger :math:`M_\star` means a smaller
    :math:`M/M_\star`, hence a *larger* c -- so the error biases
    concentration high, and the one-halo profile too concentrated.
    """
    z = 0.5
    right = m_star_hinv(z)
    wrong = right * (1.0 + z) ** 3
    assert child18(M14, z, wrong).item() > child18(M14, z, right).item()
    # and it is not a small effect
    assert child18(M14, z, wrong).item() / child18(M14, z, right).item() > 1.05


def test_m_star_from_sigma_finds_the_root_of_sigma_minus_delta_c():
    """The definition, on a power law whose root is known analytically."""
    #   sigma(R) = 0.82 (R/8)^-0.6 = 1.686  ->  R = 8 (0.82/1.686)^(1/0.6)
    r_expected = 8.0 * (0.82 / DELTA_COLLAPSE) ** (1.0 / 0.6)
    rho = 0.286 * 2.775e11
    m_star, r_star = m_star_from_sigma(lambda r: 0.82 * (r / 8.0) ** -0.6, rho)
    assert r_star == pytest.approx(r_expected, rel=1e-10)
    assert m_star == pytest.approx(4.0 / 3.0 * np.pi * rho * r_star**3)


def test_m_star_from_sigma_says_so_when_sigma_never_reaches_delta_c():
    """At high z it may not -- fail loudly rather than return a bracket end.

    A power law always crosses eventually, so the case to guard is a
    sigma(R) that stays below delta_c across the whole bracket -- what a
    heavily suppressed, high-redshift grid looks like.
    """
    with pytest.raises(ValueError, match="does not change sign"):
        m_star_from_sigma(lambda r: 0.5 * np.ones_like(r), 1.0)


def test_the_two_delta_c_are_unrelated_quantities():
    """`delta_c` (NFW, ~1e4) against DELTA_COLLAPSE (1.686). Same name."""
    assert DELTA_COLLAPSE == 1.686
    assert delta_c(4.0).item() > 1000.0
    assert delta_c(4.0).item() / DELTA_COLLAPSE > 1000.0


def test_delta_c_matches_the_y3_expression_at_the_y3_concentration():
    """Transcribed from nfw_dsigma_mis.hh; second copy here."""
    c = Y3_FIXED_CONCENTRATION
    expected = (200.0 * c**3 / 3.0) / (np.log(1.0 + c) - c / (1.0 + c))
    assert delta_c(c).item() == pytest.approx(expected, rel=1e-14)
    assert Y3_FIXED_CONCENTRATION == 4.0


def test_delta_c_rises_steeply_with_concentration():
    """Roughly c^3 at large c -- why a 10% c error is a 30% rho_s error."""
    c = np.array([3.0, 4.0, 5.0, 6.0])
    d = delta_c(c)
    assert np.all(np.diff(d) > 0.0)
    assert d[-1] / d[0] > 4.0


def test_scatter_is_a_third_of_the_concentration():
    """Child et al.'s note under both tables: sigma_c = c/3."""
    assert scatter(4.5).item() == pytest.approx(1.5)
    np.testing.assert_allclose(scatter(np.array([3.0, 6.0])), [1.0, 2.0])


# -- domains are enforced, not just documented ------------------------------


def test_child18_rejects_redshifts_outside_its_calibration():
    ms = m_star_hinv(0.0)
    with pytest.raises(ValueError, match="0 <= z <= 4"):
        child18(M14, 4.5, ms)
    with pytest.raises(ValueError, match="0 <= z <= 4"):
        child18(M14, -0.1, ms)


def test_child18_powerlaw_rejects_z_above_one():
    """Its stated domain is half that of Eq. 18, and it must say so."""
    child18_powerlaw(M14, 1.0)  # the boundary is allowed
    with pytest.raises(ValueError, match="0 <= z <= 1"):
        child18_powerlaw(M14, 1.5)


def test_every_relation_rejects_a_non_positive_mass():
    for call in (lambda: child18(0.0, 0.3, m_star_hinv(0.3)),
                 lambda: child18_powerlaw(-1.0, 0.3),
                 lambda: duffy08(0.0, 0.3)):
        with pytest.raises(ValueError, match="positive"):
            call()


def test_unknown_fit_or_mass_def_names_are_rejected():
    with pytest.raises(ValueError, match="fit must be one of"):
        child18(M14, 0.3, m_star_hinv(0.3), fit="stacked")
    with pytest.raises(ValueError, match="fit must be one of"):
        child18_powerlaw(M14, 0.3, fit="individual")
    with pytest.raises(ValueError, match="mass_def must be one of"):
        duffy08(M14, 0.3, mass_def="500c")


def test_delta_c_rejects_a_non_positive_concentration():
    with pytest.raises(ValueError, match="positive"):
        delta_c(0.0)


@pytest.mark.parametrize("fit", sorted(CHILD18_TABLE1))
def test_all_child18_rows_are_vectorised_and_scalar_safe(fit):
    m = np.array([1e13, 1e14, 1e15])
    ms = m_star_hinv(0.3)
    assert np.ravel(child18(m, 0.3, ms, fit=fit)).shape == m.shape
    assert np.size(child18(1e14, 0.3, ms, fit=fit)) == 1
