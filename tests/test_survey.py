r"""The survey layer: :math:`\Omega(z)`, the source population, the bins.

The important tests here are the **transcription** ones: the
:math:`\Omega(z)` fits are copied coefficient-by-coefficient out of
``y3_cluster_cpp``, so what matters is that they reproduce that C++ to
machine precision, and that the pathologies of the DES Y1 fit -- two
discontinuities and a zero crossing -- stay where they were measured. A
polynomial silently re-typed one digit wrong is a normalisation error that
nothing else would catch.
"""

import numpy as np
import pytest

from clenspy.protocols import Survey as SurveyProtocol
from clenspy.survey import (
    Survey,
    available_configs,
    deg2,
    load_config,
    omega_des_y1,
    omega_des_y3,
    omega_sdss,
    omega_y3xspt,
    survey_area,
    survey_bins,
)
from clenspy.survey.survey import DES_Y1_Z_RANGE

OMEGAS = (omega_des_y1, omega_des_y3, omega_sdss, omega_y3xspt)
_OMEGA_BY_NAME = {"des_y1": omega_des_y1, "des_y3": omega_des_y3,
                  "sdss": omega_sdss, "y3xspt": omega_y3xspt}


# -- Omega(z): transcription ------------------------------------------------

#: Reference values in deg^2, evaluated from the C++ coefficient arrays.
#: These are the numbers the fits must reproduce; the footprints they imply
#: (1494 for Y1 at z=0.2, 10263 for SDSS at its z=0.2 peak) are what
#: identifies which survey each fit belongs to.
DES_Y1_REFERENCE_DEG2 = {
    0.10: 1487.9,
    0.20: 1494.0,
    0.35: 1502.9,
    0.50: 1511.3,
    0.65: 649.1,
}
SDSS_REFERENCE_DEG2 = {
    0.10: 10232.2,
    0.20: 10263.0,
    0.30: 10247.9,
    0.33: 10233.0,
}


@pytest.mark.parametrize("z,expected", sorted(DES_Y1_REFERENCE_DEG2.items()))
def test_omega_des_y1_matches_the_cpp(z, expected):
    """The three-piece DES Y1 fit, against values from the C++ coefficients."""
    assert deg2(omega_des_y1(z)).item() == pytest.approx(expected, abs=0.1)


@pytest.mark.parametrize("z,expected", sorted(SDSS_REFERENCE_DEG2.items()))
def test_omega_sdss_matches_the_cpp(z, expected):
    """The degree-11 SDSS fit. A retyped digit shows up here."""
    assert deg2(omega_sdss(z)).item() == pytest.approx(expected, abs=0.1)


def test_the_fits_identify_their_surveys_by_footprint():
    """Each fit's peak area is the footprint that names it.

    This is what settles that ``OMEGA_Z_DES`` is DES **Y1**: it gives
    ~1494 deg^2, close to the published Y1 area of 1437, and nothing like
    Y3's 4143.
    """
    assert 1400 < deg2(omega_des_y1(0.2)).item() < 1600      # DES Y1, not Y3
    assert deg2(omega_des_y3(0.2)).item() == pytest.approx(4143.0)  # Y3 gold
    assert 10000 < deg2(omega_sdss(0.2)).item() < 10500      # SDSS
    assert deg2(omega_y3xspt(0.2)).item() == pytest.approx(2500.0, abs=1e-6)


def test_des_y3_is_the_gold_footprint_not_the_forecast():
    """4143 deg^2, the data. Not the 5000 of the downstream forecast config."""
    assert deg2(omega_des_y3(0.4)).item() == pytest.approx(4143.0)
    assert deg2(omega_des_y3(0.4)).item() != pytest.approx(5000.0)


# -- Omega(z): the pathologies, pinned --------------------------------------


def test_des_y1_seam_discontinuities_are_where_they_were_measured():
    """Both breaks jump. Documented, from the C++, and must not drift.

    If a future edit smooths these, the fit no longer matches the C++ and
    the counts change; this test is here so that would be a decision rather
    than an accident.
    """
    for z_break, expected_jump in ((0.504, -0.0037), (0.700, -0.3063)):
        below = omega_des_y1(z_break - 1e-9).item()
        above = omega_des_y1(z_break).item()
        assert above / below - 1 == pytest.approx(expected_jump, abs=5e-4)


def test_omega_never_returns_negative_area():
    """The DES Y1 fit crosses zero at z = 0.9378; SDSS diverges. Both clamp."""
    z = np.linspace(0.0, 3.0, 200)
    for omega in OMEGAS:
        assert np.all(omega(z) >= 0.0), omega.__name__


def test_des_y1_is_clamped_above_its_zero_crossing():
    assert omega_des_y1(0.93).item() > 0.0
    assert omega_des_y1(0.94).item() == 0.0
    assert omega_des_y1(2.0).item() == 0.0


def test_omega_is_positive_across_each_analysis_range():
    """Inside its stated domain of validity, each fit is usable."""
    z = np.linspace(*DES_Y1_Z_RANGE, 50)
    assert np.all(omega_des_y1(z) > 0)
    assert np.all(omega_sdss(np.linspace(0.10, 0.33, 50)) > 0)


# -- Omega(z): plumbing -----------------------------------------------------


@pytest.mark.parametrize("omega", OMEGAS, ids=lambda f: f.__name__)
def test_omega_is_vectorised_and_scalar_safe(omega):
    z = np.array([0.15, 0.30, 0.45])
    assert np.ravel(omega(z)).shape == z.shape
    assert np.size(omega(0.3)) == 1
    np.testing.assert_allclose(np.ravel(omega(z))[1], np.ravel(omega(0.30)))


def test_survey_area_registry_returns_the_same_functions():
    assert survey_area("des_y1") is omega_des_y1
    assert survey_area("DES_Y3") is omega_des_y3  # case-insensitive
    assert survey_area("sdss") is omega_sdss


def test_unknown_survey_names_the_known_ones():
    with pytest.raises(KeyError, match="des_y1"):
        survey_area("des_y2")


def test_deg2_round_trips():
    rad2 = omega_sdss(0.2)
    np.testing.assert_allclose(deg2(rad2) * (np.pi / 180) ** 2, rad2)


# -- the source population --------------------------------------------------


def populations():
    return [
        Survey.from_config("des_y1"),
        Survey.from_config("des_y3"),
        Survey.top_hat(zs_min=0.8, zs_max=1.2),
        Survey.tabulated(
            z=np.linspace(0.1, 2.0, 40),
            dndz=np.exp(-((np.linspace(0.1, 2.0, 40) - 0.8) ** 2) / 0.08),
        ),
    ]


@pytest.mark.parametrize("pop", populations(), ids=lambda p: p.name)
def test_surveys_conform_to_the_protocol(pop):
    """The concrete class and the protocol share a name, so alias one.

    NOTE: importing both as ``Survey`` makes this check compare the class
    against itself and pass vacuously -- which is what it did for one
    commit.
    """
    assert isinstance(pop, SurveyProtocol)


@pytest.mark.parametrize("pop", populations(), ids=lambda p: p.name)
def test_pz_src_is_normalised(pop):
    """Every shape integrates to one over its own support."""
    z = np.linspace(*pop.zs_range(), 4001)
    assert np.trapezoid(pop.pz_src(z), x=z) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("pop", populations(), ids=lambda p: p.name)
def test_pz_src_vanishes_outside_the_support(pop):
    """So a caller may integrate wider without picking up phantom weight."""
    outside = np.array([pop.zs_min - 0.1, pop.zs_max + 0.1])
    assert np.all(pop.pz_src(outside) == 0.0)


@pytest.mark.parametrize("pop", populations(), ids=lambda p: p.name)
def test_pz_src_is_non_negative_and_finite(pop):
    z = np.linspace(0.0, 4.0, 200)
    p = pop.pz_src(z)
    assert np.all(np.isfinite(p)) and np.all(p >= 0.0)


def test_des_y1_carries_the_config_numbers():
    """Transcribed from cluster-lensing-cov/configs/des_y1.json."""
    pop = Survey.from_config("des_y1")
    assert pop.sigma_gamma == 0.3
    assert pop.n_src_arcmin == 6.28


def test_des_y3_has_y3_noise_but_the_y1_pz_shape():
    """The config says so itself; the test records that it is a placeholder."""
    y1, y3 = Survey.from_config("des_y1"), Survey.from_config("des_y3")
    assert y3.sigma_gamma == 0.261 and y3.n_src_arcmin == 5.59
    z = np.linspace(0.1, 2.5, 50)
    np.testing.assert_allclose(y3.pz_src(z), y1.pz_src(z))


def test_a_narrow_top_hat_approaches_a_single_source_plane():
    """The analytic limit: p(z) -> delta(z - z_s) as the width shrinks."""
    for width in (0.2, 0.02, 0.002):
        pop = Survey.top_hat(zs_min=1.0 - width / 2, zs_max=1.0 + width / 2)
        z = np.linspace(0.9, 1.1, 20001)
        mean = np.trapezoid(z * pop.pz_src(z), x=z)
        assert mean == pytest.approx(1.0, abs=1e-3)


def test_tabulated_does_not_extrapolate():
    z = np.linspace(0.5, 1.5, 21)
    pop = Survey.tabulated(z=z, dndz=np.ones_like(z))
    assert pop.zs_range() == (0.5, 1.5)
    assert pop.pz_src(0.4).item() == 0.0
    assert pop.pz_src(1.6).item() == 0.0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(sigma_gamma=-0.1), "sigma_gamma"),
        (dict(n_src_arcmin=0.0), "n_src_arcmin"),
        (dict(zs_min=2.0, zs_max=1.0), "must increase"),
    ],
)
def test_source_population_validates_eagerly(kwargs, match):
    base = dict(pz_shape=lambda z: np.ones_like(z), sigma_gamma=0.3,
                n_src_arcmin=6.0)
    with pytest.raises(ValueError, match=match):
        Survey(**{**base, **kwargs})


def test_a_pz_that_integrates_to_zero_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        Survey(pz_shape=lambda z: np.zeros_like(z),
                         sigma_gamma=0.3, n_src_arcmin=6.0)


def test_tabulated_rejects_bad_tables():
    with pytest.raises(ValueError, match="increasing"):
        Survey.tabulated(z=[1.0, 0.5, 2.0], dndz=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        Survey.tabulated(z=[0.5, 1.0], dndz=[1.0, -1.0])
    with pytest.raises(ValueError, match="shape"):
        Survey.tabulated(z=[0.5, 1.0], dndz=[1.0, 1.0, 1.0])


# -- the bins ---------------------------------------------------------------


def test_des_y1_bins_are_the_production_grid():
    bins = survey_bins("des_y1")
    assert len(bins) == 12
    assert (bins.n_lam, bins.n_z) == (4, 3)
    assert bins.at(0, 0).lam_edges == (20.0, 30.0)
    assert bins.at(3, 0).lam_edges == (60.0, 200.0)  # 200, not 1000
    assert bins.at(0, 2).z_edges == (0.50, 0.65)
    # sigma_z is the scatter, 0.01. The 0.03 that appears as SIGMA_Z in the
    # y3 production config is the 3-sigma window, not the scatter.
    assert all(b.sigma_z == 0.01 for b in bins)


def test_des_y3_bins_match_y1():
    """No distinct Y3 binning is recorded; the config exists to be edited."""
    y1, y3 = survey_bins("des_y1"), survey_bins("des_y3")
    assert [b.index for b in y1] == [b.index for b in y3]
    assert [b.lam_edges for b in y1] == [b.lam_edges for b in y3]


def test_bin_redshift_range_matches_the_omega_validity_range():
    """The bins must lie inside the fit's stated domain of validity."""
    bins = survey_bins("des_y1")
    z_lo = min(b.z_min for b in bins)
    z_hi = max(b.z_max for b in bins)
    assert (z_lo, z_hi) == DES_Y1_Z_RANGE
    assert np.all(omega_des_y1(np.linspace(z_lo, z_hi, 30)) > 0)


def test_every_config_names_an_omega_fit_that_exists():
    """A config's ``omega_z`` must resolve, or its counts are unnormalised."""
    for name in available_configs():
        cfg = load_config(name)
        assert survey_area(cfg["omega_z"]) is not None


def test_config_and_omega_fit_are_addressed_consistently():
    for name in ("des_y1", "des_y3"):
        cfg = load_config(name)
        assert len(survey_bins(cfg)) == 12
        assert survey_area(cfg["omega_z"]) is _OMEGA_BY_NAME[name]


def test_sdss_has_an_omega_fit_but_no_config():
    """The asymmetry is the point: the fit is transcribed, the choices are not."""
    assert omega_sdss(0.2).item() > 0.0
    with pytest.raises(FileNotFoundError, match="transcribes"):
        load_config("sdss")
    with pytest.raises(FileNotFoundError):
        Survey.from_config("sdss")
    with pytest.raises(FileNotFoundError):
        survey_bins("sdss")


def test_a_missing_config_lists_what_exists():
    with pytest.raises(FileNotFoundError, match="des_y1"):
        load_config("des_y2")


def test_configs_carry_provenance_on_every_group():
    """A number with no provenance is a number nobody can check."""
    for name in available_configs():
        cfg = load_config(name)
        for section in ("bins", "sources"):
            keys = cfg[section].keys()
            assert any(k.startswith("_provenance") for k in keys), (
                f"{name}.{section} has no _provenance"
            )


def _minimal_config():
    return {
        "name": "test-cfg",
        "omega_z": "des_y1",
        "bins": {
            "lam_edges": [20.0, 30.0, 45.0],
            "z_edges": [0.2, 0.35, 0.5],
            "sigma_z": [0.01, 0.01],
        },
        "sources": {
            "pz_model": "smail",
            "sigma_gamma": 0.3,
            "n_src_arcmin": 6.28,
        },
    }


def test_survey_bins_requires_a_bins_section():
    cfg = _minimal_config()
    del cfg["bins"]
    with pytest.raises(KeyError, match="bins"):
        survey_bins(cfg)


def test_from_config_requires_a_sources_section():
    cfg = _minimal_config()
    del cfg["sources"]
    with pytest.raises(KeyError, match="sources"):
        Survey.from_config(cfg)


def test_from_config_rejects_an_unknown_pz_model():
    cfg = _minimal_config()
    cfg["sources"]["pz_model"] = "bogus_model"
    with pytest.raises(KeyError, match="bogus_model"):
        Survey.from_config(cfg)


def test_tabulated_rejects_scalar_or_too_short_z():
    with pytest.raises(ValueError, match="at least two nodes"):
        Survey.tabulated(z=0.5, dndz=1.0)
    with pytest.raises(ValueError, match="at least two nodes"):
        Survey.tabulated(z=[0.5], dndz=[1.0])


def test_survey_repr_contains_class_name_and_sigma_gamma():
    pop = Survey.top_hat(zs_min=0.8, zs_max=1.2, sigma_gamma=0.27)
    r = repr(pop)
    assert "Survey" in r
    assert "sigma_gamma" in r
    assert "0.27" in r


def test_sigma_z_is_the_scatter_not_the_window():
    """0.01 is sigma_z; 0.03 is 3 sigma. Confusing them widens every bin 3x."""
    for name in available_configs():
        b = load_config(name)["bins"]
        assert all(s == 0.01 for s in b["sigma_z"])
        assert b["sigma_z_n_sigma_window"] == 3.0
