r"""The lensing kernel, against its analytic limits.

Every quantity here is an integral, so the tests that matter are the ones
with a closed-form answer:

- :math:`\Sigma_{\rm crit}^{\rm com} \times (1+z_l)^2` must equal the
  physical :math:`\Sigma_{\rm crit}` **exactly** -- that is the convention
  boundary, and it is the easiest thing in this module to get wrong.
- :math:`q_\Sigma(z_l; z_h)` at :math:`z_l = z_h` has integrand 1, so it
  must reduce to :math:`f_{\rm src}(z_h)`.
- a narrow top-hat source distribution must reproduce the single-source-
  plane :math:`1/\Sigma_{\rm crit}`.

Plus the four traps of errata E.1, each asserted directly.
"""

import numpy as np
import pytest

from clenspy.cosmology import fiducial_cosmology
from clenspy.kernels import LensingKernel, sigma_crit_comoving, sigma_critical
from clenspy.kernels.lensing_kernel import (
    _SIGMA_CRIT_AMPLITUDE,
    MIN_LENS_SOURCE_SEPARATION,
)
from clenspy.survey import Survey

COSMO = fiducial_cosmology()
Z_LENS = np.array([0.2, 0.35, 0.5, 0.65])


@pytest.fixture
def kernel():
    return LensingKernel(survey=Survey.from_config("des_y1"), cosmology=COSMO)


# -- the convention boundary ------------------------------------------------


@pytest.mark.parametrize("z_lens", [0.1, 0.35, 0.8])
@pytest.mark.parametrize("z_source", [1.0, 1.5, 2.5])
def test_comoving_and_physical_differ_by_exactly_one_plus_z_squared(
    z_lens, z_source
):
    """The whole convention question, in one assertion.

    ``sigma_critical`` is physical (angular diameter distances);
    ``sigma_crit_comoving`` is comoving. Anything other than
    :math:`(1+z_l)^2` between them means one of the two is wrong.
    """
    comoving = sigma_crit_comoving(z_lens, z_source, COSMO).item()
    physical = sigma_critical(z_lens, z_source, COSMO)
    assert physical / comoving == pytest.approx((1.0 + z_lens) ** 2, rel=1e-10)


def test_comoving_sigma_crit_matches_its_closed_form():
    """Transcription check against the formula in the docstring."""
    z_l, z_s = 0.35, 1.2
    chi_l = COSMO.comoving_distance(z_l).value
    chi_s = COSMO.comoving_distance(z_s).value
    expected = (_SIGMA_CRIT_AMPLITUDE * chi_s
                / (chi_l * (chi_s - chi_l) * (1.0 + z_l)))
    assert sigma_crit_comoving(z_l, z_s, COSMO).item() == pytest.approx(
        expected, rel=1e-12
    )


def test_a_source_in_front_of_the_lens_gives_infinite_sigma_crit():
    """So its inverse is zero and it drops out of an average by itself."""
    assert np.isinf(sigma_crit_comoving(1.0, 0.5, COSMO).item())
    assert np.isinf(sigma_crit_comoving(1.0, 1.0, COSMO).item())  # z_s == z_l
    assert np.isfinite(sigma_crit_comoving(1.0, 1.5, COSMO).item())


def test_sigma_crit_falls_with_source_redshift_and_diverges_at_the_lens():
    z_s = np.array([0.36, 0.5, 1.0, 2.0, 3.0])
    sc = sigma_crit_comoving(0.35, z_s, COSMO)
    assert np.all(np.diff(sc) < 0)
    assert sc[0] > 10 * sc[-1]  # the 1/(chi_s - chi_l) divergence


# -- the analytic limits ----------------------------------------------------


@pytest.mark.parametrize("z_halo", [0.2, 0.35, 0.5])
def test_q_sigma_reduces_to_f_src_behind_when_the_redshifts_coincide(
    kernel, z_halo
):
    r"""At :math:`z_l = z_h` the ratio in the integrand is exactly 1.

    So :math:`q_\Sigma` collapses to :math:`\int p(z_s)\,dz_s` over the
    same range, which is :math:`f_{\rm src}(z_h)`. Nothing about the
    geometry survives, which makes this a clean check on the bookkeeping --
    and it holds only because both integrals key their range on the same
    redshift.
    """
    assert kernel.q_sigma(z_halo, z_halo).item() == pytest.approx(
        kernel.f_src_behind(z_halo).item(), rel=1e-12
    )


def test_q_sigma_is_signed_and_is_not_clamped(kernel):
    r"""The frozen definition keeps the sign; clamping changes the covariance.

    Its source range is keyed on :math:`z_l`, so for :math:`z_l < z_h` the
    integral includes sources in front of the halo where
    :math:`\Sigma_{\rm crit}(z_h, z_s) < 0`. The frozen reference runs from
    -2.29 to +3.91, so a q_sigma that is everywhere positive is wrong.
    """
    q = kernel.q_sigma(np.linspace(0.1, 0.6, 40), 0.9)
    assert np.any(q < 0.0), "q_sigma must be able to go negative"


def test_q_sigma_keys_its_range_on_the_lens_not_the_halo(kernel):
    r"""Which is what makes the negative lobe reachable at all.

    Keyed on :math:`\max(z_l, z_h)` the range would start behind the halo
    and the sign could never flip; this pins the choice.
    """
    z_l, z_h = 0.2, 0.9
    nodes = kernel._zs_nodes(z_l)
    assert nodes[0] == pytest.approx(z_l + MIN_LENS_SOURCE_SEPARATION)
    assert nodes[0] < z_h  # sources in front of the halo are included


def test_a_narrow_top_hat_recovers_the_single_source_plane():
    r"""As :math:`p(z_s) \to \delta(z_s - z_0)`,
    :math:`\langle\Sigma_{\rm crit}^{-1}\rangle \to 1/\Sigma_{\rm crit}`."""
    z_0, z_l = 1.0, 0.35
    exact = 1.0 / sigma_crit_comoving(z_l, z_0, COSMO).item()
    errors = []
    for width in (0.2, 0.05, 0.01):
        su = Survey.top_hat(zs_min=z_0 - width / 2, zs_max=z_0 + width / 2)
        got = LensingKernel(su, COSMO).mean_inverse_sigma_crit(z_l).item()
        errors.append(abs(got / exact - 1))
    # converges, and the narrowest is already tight
    assert errors[-1] < 2e-4, errors
    assert errors[0] > errors[-1], errors


def test_mean_sigma_crit_recovers_the_single_source_plane_too():
    z_0, z_h, width = 1.0, 0.35, 0.01
    su = Survey.top_hat(zs_min=z_0 - width / 2, zs_max=z_0 + width / 2)
    got = LensingKernel(su, COSMO).mean_sigma_crit(z_h).item()
    exact = sigma_crit_comoving(z_h, z_0, COSMO).item()
    assert got == pytest.approx(exact, rel=2e-4)


# -- the four traps of errata E.1 -------------------------------------------


def test_averaging_the_inverse_is_not_inverting_the_average(kernel):
    """E.1 item 1. The difference *is* the source weighting.

    If these ever agreed, one of the two integrals would be wrong.
    """
    inv = kernel.mean_inverse_sigma_crit(Z_LENS)
    mean = kernel.mean_sigma_crit(Z_LENS)
    product = inv * mean

    # NOTE: no Cauchy-Schwarz bound applies here, tempting as it is. The two
    # averages are not taken against the same measure: p(z_s) is not
    # renormalised on the truncated range, and mean_sigma_crit is
    # cutoff-defined while its inverse is not. So the product is simply not
    # 1, in either direction -- which is the whole claim.
    # measured: 1.089, 1.102, 0.893, 0.549 across Z_LENS -- it crosses 1,
    # so not even the direction of the inequality is fixed
    assert np.all(np.abs(product - 1.0) > 0.05), product
    assert product.max() > 1.0 and product.min() < 1.0, product


def test_the_integrand_is_clamped_so_foreground_sources_never_subtract(kernel):
    """E.1 item 2. Every output is non-negative for any lens redshift."""
    z = np.linspace(0.0, 3.5, 60)
    assert np.all(kernel.mean_inverse_sigma_crit(z) >= 0.0)
    assert np.all(kernel.mean_sigma_crit(z) >= 0.0)
    assert np.all(kernel.f_src_behind(z) >= 0.0)
    # NOTE: q_sigma is deliberately NOT clamped -- see
    # test_q_sigma_is_signed_and_is_not_clamped.


def test_the_flat_subtraction_form_is_used_not_the_naive_difference():
    """E.1 item 3, checked against astropy's own z1z2 distance.

    Working in comoving distance makes ``chi_s - chi_l`` the right answer
    automatically; this pins that it agrees with the angular-diameter
    subtraction form, and that the naive difference does not.
    """
    z_l, z_s = 0.35, 1.0
    d_ls_correct = COSMO.angular_diameter_distance_z1z2(z_l, z_s).value
    d_l = COSMO.angular_diameter_distance(z_l).value
    d_s = COSMO.angular_diameter_distance(z_s).value

    flat = d_s - (1.0 + z_l) / (1.0 + z_s) * d_l
    assert flat == pytest.approx(d_ls_correct, rel=1e-12)

    naive = d_s - d_l
    assert abs(naive / d_ls_correct - 1) > 0.3  # wrong by ~34%

    # and the comoving form agrees with the physical one built from d_ls
    comoving = sigma_crit_comoving(z_l, z_s, COSMO).item()
    physical = _SIGMA_CRIT_AMPLITUDE * d_s / (d_l * d_ls_correct)
    assert physical / comoving == pytest.approx((1 + z_l) ** 2, rel=1e-10)


def test_photoz_bias_is_an_argument_and_moves_the_answer(kernel):
    """E.1 item 4. delta_z shifts p(z_s), so it must change the result.

    Shifting the source distribution to higher z (delta_z < 0 samples
    p at lower argument, i.e. moves sources back) raises the lensing
    efficiency, so the kernel is monotonic in delta_z.
    """
    values = [kernel.mean_inverse_sigma_crit(0.35, delta_z=dz).item()
              for dz in (-0.04, -0.02, 0.0, 0.02, 0.04)]
    assert np.all(np.diff(values) < 0), values
    assert values[0] / values[-1] > 1.02  # a 4% shift matters at the % level


# -- f_src_behind -----------------------------------------------------------


def test_f_src_behind_is_one_below_the_distribution_and_zero_above(kernel):
    """p(z_s) is normalised, so the fraction behind z=0 is the whole of it."""
    assert kernel.f_src_behind(0.0).item() == pytest.approx(1.0, abs=1e-4)
    assert kernel.f_src_behind(3.0).item() == 0.0
    assert kernel.f_src_behind(5.0).item() == 0.0


def test_f_src_behind_decreases_monotonically(kernel):
    z = np.linspace(0.0, 2.9, 40)
    f = kernel.f_src_behind(z)
    assert np.all(np.diff(f) <= 1e-12)


def test_f_src_behind_of_a_top_hat_is_the_linear_ramp():
    r"""A closed form: for a uniform p(z), the fraction behind is linear.

    NOTE: the ramp is offset by `MIN_LENS_SOURCE_SEPARATION`, because the
    integral starts at :math:`z_h + 0.01` rather than :math:`z_h`. That is
    the definition, so the closed form has to carry it -- writing the naive
    ramp and loosening the tolerance would hide a 1% bias.
    """
    zs_min, zs_max = 0.5, 1.5
    su = Survey.top_hat(zs_min=zs_min, zs_max=zs_max)
    lk = LensingKernel(su, COSMO)
    for z_h in (0.5, 1.0, 1.25, 1.5):
        lo = max(z_h + MIN_LENS_SOURCE_SEPARATION, zs_min)
        expected = max(0.0, (zs_max - lo) / (zs_max - zs_min))
        assert lk.f_src_behind(z_h).item() == pytest.approx(expected, abs=1e-6)


# -- the unity seam ---------------------------------------------------------


def test_unity_makes_every_consumer_emit_deltasigma():
    """The clean protocol seam of errata E.1, as the y3 pipeline uses it."""
    lk = LensingKernel(Survey.from_config("des_y1"), COSMO, unity=True)
    np.testing.assert_array_equal(lk.mean_inverse_sigma_crit(Z_LENS),
                                  np.ones(Z_LENS.shape))
    # it affects only that one quantity -- the others are still physical
    assert lk.f_src_behind(0.35).item() < 1.0
    assert lk.mean_sigma_crit(0.35).item() > 1e14


# -- numerics and plumbing --------------------------------------------------


def test_the_convergent_quantities_converge(kernel, monkeypatch):
    """<Sigma_crit^-1> and f_src are honest integrals: refining is a no-op."""
    import clenspy.kernels.lensing_kernel as lk_mod

    coarse_inv = kernel.mean_inverse_sigma_crit(Z_LENS)
    coarse_f = kernel.f_src_behind(Z_LENS)
    monkeypatch.setattr(lk_mod, "N_ZS_NODES", 8 * lk_mod.N_ZS_NODES)
    fine = LensingKernel(Survey.from_config("des_y1"), COSMO)
    np.testing.assert_allclose(coarse_inv,
                               fine.mean_inverse_sigma_crit(Z_LENS), rtol=1e-3)
    np.testing.assert_allclose(coarse_f, fine.f_src_behind(Z_LENS), rtol=1e-3)


def test_mean_sigma_crit_does_not_converge_and_that_is_the_point(kernel):
    """It is logarithmically divergent: refining lowers it, not settles it.

    Recorded as a test because it is the reason <Sigma_crit^-1> is the
    quantity to prefer, and because someone will otherwise "fix" the node
    count thinking it is a tolerance.
    """
    z_h = 0.35
    values = [kernel.mean_sigma_crit(z_h, n_nodes=n).item()
              for n in (100, 200, 400, 800)]
    assert np.all(np.diff(values) < 0), values
    # 100 -> 800 nodes moves it ~5%: not a tolerance, a definition
    assert values[0] / values[-1] > 1.04, values


def test_going_below_the_separation_floor_is_refused(kernel):
    """0.01 is a definition of a lens-source pair, not a tolerance."""
    with pytest.raises(ValueError, match="below"):
        kernel.mean_sigma_crit(0.35, min_separation=1e-4)
    with pytest.raises(ValueError, match="definition"):
        kernel.q_sigma(0.2, 0.35, min_separation=0.0)
    # and the floor itself is accepted
    assert np.isfinite(
        kernel.mean_sigma_crit(0.35, min_separation=0.01).item())


def test_kernel_z_interpolant_matches_the_direct_integral(kernel):
    """Same quantity, two routes."""
    z = np.array([0.25, 0.4, 0.7, 1.1])
    np.testing.assert_allclose(kernel.kernel_z(z),
                               kernel.mean_inverse_sigma_crit(z), rtol=1e-2)


def test_kernel_z_does_not_extrapolate(kernel):
    """A lens behind every source is unlensed, so the interpolant is zero."""
    assert kernel.kernel_z(4.0).item() == 0.0


def test_kernel_z_cache_is_keyed_on_delta_z(kernel):
    """Sweeping the photo-z bias must not reuse the previous grid."""
    a = kernel.kernel_z(0.35, delta_z=0.0).item()
    b = kernel.kernel_z(0.35, delta_z=0.05).item()
    c = kernel.kernel_z(0.35, delta_z=0.0).item()
    assert a != b
    assert a == c


@pytest.mark.parametrize(
    "method", ["mean_inverse_sigma_crit", "mean_sigma_crit", "f_src_behind"]
)
def test_methods_are_vectorised_and_scalar_safe(kernel, method):
    f = getattr(kernel, method)
    assert np.ravel(f(Z_LENS)).shape == Z_LENS.shape
    assert np.size(f(0.35)) == 1
    np.testing.assert_allclose(np.ravel(f(Z_LENS))[1], np.ravel(f(0.35)))


def test_q_sigma_is_vectorised_in_the_lens_redshift(kernel):
    assert np.ravel(kernel.q_sigma(Z_LENS, 0.35)).shape == Z_LENS.shape
    np.testing.assert_allclose(np.ravel(kernel.q_sigma(Z_LENS, 0.35))[1],
                               np.ravel(kernel.q_sigma(0.35, 0.35)))


def test_q_sigma_falls_once_the_structure_is_behind_the_halo(kernel):
    """Monotonic only for z_l > z_h -- it is not monotonic overall.

    Below the halo the integrand carries the signed lobe and the
    :math:`z_s = z_h` pole sits inside the range, so q_sigma is
    non-monotonic there. That was an assumption worth testing rather than
    asserting: it is false.
    """
    z_h = 0.35
    q = kernel.q_sigma(np.linspace(z_h + 0.05, 1.5, 20), z_h)
    assert np.all(np.diff(q) < 0)


def test_the_constructor_computes_nothing(kernel):
    """It stores collaborators; the z_l interpolant is built on first use."""
    lk = LensingKernel(Survey.from_config("des_y1"), COSMO)
    assert lk._kernel_z is None
    lk.kernel_z(0.35)
    assert lk._kernel_z is not None
