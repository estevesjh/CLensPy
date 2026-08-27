r"""The Limber projection, against the identities Wu et al. (2019) implies.

The strongest check needs no reference data. With **linear bias**,
:math:`P_{\rm hm}^2 = P_{\rm hh} P_{\rm mm}` identically, so over the same
:math:`\chi` range

.. math::
    \left(C_\ell^{\rm h\Sigma}\right)^2
      = C_\ell^{\rm hh}\, C_\ell^{\Sigma\Sigma}

**exactly**, for every :math:`\ell`. The two windows cancel between the
numerator and the denominator, so this holds whatever :math:`F_{\rm h}` and
:math:`F_\Sigma` happen to be -- which makes it a check that the three
spectra are wired to the *right* windows rather than a check on the windows
themselves. It fails immediately if any of the three carries the wrong
power of :math:`\bar\rho`, :math:`V` or :math:`\chi`.

The dimensional checks are the other half: the paper states that
:math:`C_\ell^{\Sigma\Sigma}` has the dimensions of :math:`\Sigma^2` and
:math:`C_\ell^{\rm h\Sigma}` those of :math:`\Sigma`, with
:math:`C_\ell^{\rm hh}` dimensionless. Scaling :math:`\bar\rho` and
checking the response pins all three.
"""

import numpy as np
import pytest

from clenspy.cosmology import fiducial_cosmology, mean_matter_density
from clenspy.kernels import ARCMIN_TO_RAD, LensingKernel, LimberProjector
from clenspy.survey import Survey

COSMO = fiducial_cosmology()
Z_MIN, Z_MAX, Z_HALO, BIAS = 0.35, 0.50, 0.425, 3.5


def pk_lin(k, z):
    """A smooth power-law stand-in, so no Boltzmann solver is needed."""
    k = np.asarray(k, dtype=float)
    return 2e4 * k**-1.5 / (1.0 + (k / 0.2) ** 2) / (1.0 + z) ** 2


def build(n_ell=200, **kwargs):
    survey = Survey.from_config("des_y1")
    lk = LensingKernel(survey, COSMO)
    opts = dict(
        chi=lambda z: COSMO.comoving_distance(z).value,
        pk_lin=pk_lin,
        rho_mean0=mean_matter_density(COSMO),
        q_sigma=lk.q_sigma,
        mean_sigma_crit=lk.mean_sigma_crit,
        f_src_behind=lk.f_src_behind,
        sigma_gamma=survey.sigma_gamma,
        n_src_arcmin2=survey.n_src_arcmin,
        n_ell=n_ell,
    )
    opts.update(kwargs)
    return LimberProjector(**opts)


@pytest.fixture
def proj():
    return build()


# -- the exact identity -----------------------------------------------------


def test_linear_bias_saturates_the_cross_correlation(proj):
    r""":math:`(C_\ell^{\rm h\Sigma})^2 = C_\ell^{\rm hh} C_\ell^{\Sigma\Sigma}`
    exactly, on a common range, for linear bias.

    The single most informative test in this file: it pins that all three
    spectra use the windows the paper assigns them, to machine precision,
    with no reference data.
    """
    c_hh = proj.C_ell_hh(Z_MIN, Z_MAX, BIAS)
    c_SS = proj.C_ell_SS(Z_MIN, Z_MAX, Z_HALO)
    c_hS = proj.C_ell_hS(Z_MIN, Z_MAX, BIAS, Z_HALO)
    np.testing.assert_allclose(c_hS**2, c_hh * c_SS, rtol=1e-12)


def test_the_identity_breaks_when_a_window_is_wrong():
    r"""The identity has teeth: a stochastic :math:`P_{\rm hm}` breaks it.

    Decorrelating the halo and matter fields must push
    :math:`(C^{\rm h\Sigma})^2` *below* :math:`C^{\rm hh}C^{\Sigma\Sigma}`,
    which is the Cauchy-Schwarz direction.
    """
    proj = build()
    c_hh = proj.C_ell_hh(Z_MIN, Z_MAX, BIAS)
    c_SS = proj.C_ell_SS(Z_MIN, Z_MAX, Z_HALO)
    # 70% correlated: P_hm -> 0.7 b P_lin
    c_hS = proj.C_ell_hS(Z_MIN, Z_MAX, BIAS, Z_HALO,
                         pk_hm=lambda k, z: 0.7 * BIAS * pk_lin(k, z))
    ratio = c_hS**2 / (c_hh * c_SS)
    np.testing.assert_allclose(ratio, 0.49 * np.ones_like(ratio), rtol=1e-10)


def test_widening_the_lss_range_lowers_the_correlation(proj):
    """Extra LSS adds variance that does not correlate with the haloes."""
    c_hh = proj.C_ell_hh(Z_MIN, Z_MAX, BIAS)
    c_hS = proj.C_ell_hS(Z_MIN, Z_MAX, BIAS, Z_HALO)
    narrow = proj.C_ell_SS(Z_MIN, Z_MAX, Z_HALO)
    wide = proj.C_ell_SS(0.1, 2.0, Z_HALO)
    assert np.all(wide > narrow)
    assert np.all(c_hS**2 / (c_hh * wide) < 1.0)


# -- dimensions, as the paper states them -----------------------------------


def test_c_ell_hh_is_dimensionless_in_rho():
    r""":math:`C_\ell^{\rm hh}` carries no :math:`\bar\rho`."""
    a = build().C_ell_hh(Z_MIN, Z_MAX, BIAS)
    b = build(rho_mean0=3.0 * mean_matter_density(COSMO)).C_ell_hh(
        Z_MIN, Z_MAX, BIAS)
    np.testing.assert_allclose(a, b, rtol=1e-14)


def test_c_ell_SS_scales_as_rho_squared():
    r""":math:`C_\ell^{\Sigma\Sigma} \propto \bar\rho^2` -- dimensions of
    :math:`\Sigma^2`, per the paper."""
    a = build().C_ell_SS(Z_MIN, Z_MAX, Z_HALO)
    b = build(rho_mean0=3.0 * mean_matter_density(COSMO)).C_ell_SS(
        Z_MIN, Z_MAX, Z_HALO)
    np.testing.assert_allclose(b / a, 9.0 * np.ones_like(a), rtol=1e-12)


def test_c_ell_hS_scales_as_rho():
    r""":math:`C_\ell^{\rm h\Sigma} \propto \bar\rho` -- dimensions of
    :math:`\Sigma`."""
    a = build().C_ell_hS(Z_MIN, Z_MAX, BIAS, Z_HALO)
    b = build(rho_mean0=3.0 * mean_matter_density(COSMO)).C_ell_hS(
        Z_MIN, Z_MAX, BIAS, Z_HALO)
    np.testing.assert_allclose(b / a, 3.0 * np.ones_like(a), rtol=1e-12)


def test_bias_scaling_follows_the_linear_bias_forms(proj):
    r""":math:`C^{\rm hh} \propto b^2`, :math:`C^{\rm h\Sigma} \propto b`."""
    hh1 = proj.C_ell_hh(Z_MIN, Z_MAX, 1.0)
    hh2 = proj.C_ell_hh(Z_MIN, Z_MAX, 2.0)
    np.testing.assert_allclose(hh2 / hh1, 4.0 * np.ones_like(hh1), rtol=1e-12)
    hS1 = proj.C_ell_hS(Z_MIN, Z_MAX, 1.0, Z_HALO)
    hS2 = proj.C_ell_hS(Z_MIN, Z_MAX, 2.0, Z_HALO)
    np.testing.assert_allclose(hS2 / hS1, 2.0 * np.ones_like(hS1), rtol=1e-12)


# -- the windows themselves -------------------------------------------------


def test_F_h_is_normalised_on_its_own_slabs(proj):
    r""":math:`\sum \Delta\chi\, F_{\rm h}(\chi) = 1` discretely.

    Eq. ``F_h`` divides by :math:`V = \int \chi^2 d\chi`, so the window
    integrates to one -- and it must do so on the *same* slabs the spectra
    are summed on, not merely in the continuum.
    """
    _, chi_mid, dchi, volume = proj._slabs(Z_MIN, Z_MAX)
    F_h = proj.F_h(chi_mid, volume)
    assert float(np.sum(dchi * F_h)) == pytest.approx(1.0, rel=1e-12)


def test_F_Sigma_is_rho_times_q_sigma(proj):
    r"""The identification of eq. ``F_Sigma`` with :math:`q_\Sigma`, exactly.

    This is the paper's own definition, and getting it wrong by a factor of
    :math:`\bar\rho` is invisible in a ratio test.
    """
    lk = LensingKernel(Survey.from_config("des_y1"), COSMO)
    z_mid, _, _, _ = proj._slabs(Z_MIN, Z_MAX)
    np.testing.assert_allclose(
        proj.F_Sigma(z_mid, Z_HALO),
        mean_matter_density(COSMO) * np.ravel(lk.q_sigma(z_mid, Z_HALO)),
        rtol=1e-12,
    )


def test_spectra_fall_with_ell(proj):
    """A smooth P(k) gives monotonically falling spectra."""
    for c in (proj.C_ell_hh(Z_MIN, Z_MAX, BIAS),
              proj.C_ell_SS(0.1, 2.0, Z_HALO),
              proj.C_ell_hS(Z_MIN, Z_MAX, BIAS, Z_HALO)):
        assert np.all(np.isfinite(c))
        assert np.all(np.diff(c) < 0)


# -- the noise terms --------------------------------------------------------


def test_shot_noise_is_the_inverse_surface_density(proj):
    r""":math:`1/n_{\rm h}^{(2D)} = ` area / counts."""
    assert proj.shot_noise_h(counts=1500.0, area_sr=0.455) == pytest.approx(
        0.455 / 1500.0
    )


def test_shape_noise_scales_as_sigma_gamma_squared_over_n_src():
    n_a = build(sigma_gamma=0.3, n_src_arcmin2=6.28).shape_noise_Sigma(Z_HALO)
    n_b = build(sigma_gamma=0.6, n_src_arcmin2=6.28).shape_noise_Sigma(Z_HALO)
    n_c = build(sigma_gamma=0.3, n_src_arcmin2=12.56).shape_noise_Sigma(Z_HALO)
    assert n_b / n_a == pytest.approx(4.0, rel=1e-12)
    assert n_c / n_a == pytest.approx(0.5, rel=1e-12)


def test_shape_noise_carries_the_f_src_deviation_from_the_paper():
    r"""The paper has :math:`\sigma_\gamma^2/n_s`; this divides by
    :math:`n_s f_{\rm src}`.

    Passing ``f_src_behind = 1`` must recover the paper's form exactly, and
    the ratio is :math:`1/f_{\rm src}`.
    """
    lk = LensingKernel(Survey.from_config("des_y1"), COSMO)
    f_src = float(np.ravel(lk.f_src_behind(Z_HALO))[0])
    assert 0.0 < f_src < 1.0

    ours = build().shape_noise_Sigma(Z_HALO)
    paper = build(f_src_behind=lambda z: 1.0).shape_noise_Sigma(Z_HALO)
    assert ours / paper == pytest.approx(1.0 / f_src, rel=1e-12)
    assert ours > paper  # counting fewer sources raises the noise


def test_shape_noise_is_infinite_with_no_sources_behind_the_lens():
    """Rather than dividing by zero."""
    assert np.isinf(build(f_src_behind=lambda z: 0.0).shape_noise_Sigma(Z_HALO))


def test_arcmin_conversion_is_the_only_unit_crossing():
    """n_src enters per steradian; the factor is 1/ARCMIN_TO_RAD^2."""
    lk = LensingKernel(Survey.from_config("des_y1"), COSMO)
    f_src = float(np.ravel(lk.f_src_behind(Z_HALO))[0])
    sc = float(np.ravel(lk.mean_sigma_crit(Z_HALO))[0])
    expected = 0.3**2 / (6.28 * f_src / ARCMIN_TO_RAD**2) * sc**2
    got = build(sigma_gamma=0.3, n_src_arcmin2=6.28).shape_noise_Sigma(Z_HALO)
    assert got == pytest.approx(expected, rel=1e-12)


# -- the grid and the API ---------------------------------------------------


def test_the_ell_grid_is_log_spaced_over_the_stated_range():
    p = build(n_ell=1000, ell_range=(1e-1, 2e7))
    assert p.ell.size == 1000
    assert p.ell[0] == pytest.approx(1e-1)
    assert p.ell[-1] == pytest.approx(2e7)
    dlog = np.diff(np.log(p.ell))
    np.testing.assert_allclose(dlog, dlog[0] * np.ones_like(dlog), rtol=1e-12)


def test_the_constructor_stores_and_computes_nothing_but_the_grid(proj):
    """No spectrum is evaluated until asked for."""
    assert not any(k.startswith("C_ell") for k in proj.__dict__)


def test_deprecated_aliases_return_the_new_methods(proj):
    """The downstream calls the old names; they must agree exactly."""
    np.testing.assert_allclose(proj.c_ell_sigma(0.1, 2.0, Z_HALO),
                               proj.C_ell_SS(0.1, 2.0, Z_HALO), rtol=0)
    c, shot = proj.c_ell_h(Z_MIN, Z_MAX, BIAS, 1500.0, 0.455)
    np.testing.assert_allclose(c, proj.C_ell_hh(Z_MIN, Z_MAX, BIAS), rtol=0)
    assert shot == proj.shot_noise_h(1500.0, 0.455)
    np.testing.assert_allclose(
        proj.c_ell_h_sigma(Z_MIN, Z_MAX, BIAS, Z_HALO),
        proj.C_ell_hS(Z_MIN, Z_MAX, BIAS, Z_HALO), rtol=0)
    assert proj.shape_noise_sigma(Z_HALO) == proj.shape_noise_Sigma(Z_HALO)


def test_the_old_import_path_still_works_with_a_warning():
    """cluster-lensing-cov imports clenspy.lensing.limber."""
    import importlib
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod = importlib.reload(
            importlib.import_module("clenspy.lensing.limber"))
    assert mod.LimberProjector is LimberProjector
    assert any(w.category is DeprecationWarning for w in caught)
