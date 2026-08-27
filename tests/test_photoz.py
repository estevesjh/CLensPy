r"""The two photo-z kernels, and the ways they must not be confused.

Each has a closed-form property that pins it:

- the counts kernel integrates over :math:`z^{\rm tr}` to the **bin width**
  (it is a probability of landing in the bin, and every true redshift lands
  somewhere);
- the projection weight integrates to :math:`\frac43 n_\sigma\sigma_z` and
  is **exactly zero** outside :math:`\pm n_\sigma\sigma_z`, with
  :math:`n_\sigma = 3` -- which is where the 0.03 in the y3 configs comes
  from, and the single most confusable number in this module.

So the tests are mostly not tolerance checks -- they are identities. The
remaining ones assert that the two are distinguishable, because using one
where the other belongs is the failure this module exists to prevent.
"""

import numpy as np
import pytest

from clenspy.kernels import gaussian_cdf, photoz_counts, photoz_projection

Z_MIN, Z_MAX = 0.35, 0.50
Z_OB = 0.5 * (Z_MIN + Z_MAX)
SIGMA_Z = 0.01
N_SIGMA = 3.0
#: The projection window's half-width -- the 0.03.
WINDOW = N_SIGMA * SIGMA_Z

#: Dense enough that the trapezoid identities below hold to 1e-6.
FINE = np.linspace(0.0, 1.5, 300001)


# -- the standard normal CDF ------------------------------------------------


def test_gaussian_cdf_matches_scipy():
    """Written out by hand for speed, so check it against the reference."""
    from scipy.stats import norm

    x = np.linspace(-8.0, 8.0, 401)
    np.testing.assert_allclose(gaussian_cdf(x), norm.cdf(x), atol=1e-14)


def test_gaussian_cdf_endpoints_and_symmetry():
    assert gaussian_cdf(0.0).item() == pytest.approx(0.5)
    assert gaussian_cdf(-40.0).item() == 0.0
    assert gaussian_cdf(40.0).item() == 1.0
    x = np.linspace(0.1, 5.0, 20)
    np.testing.assert_allclose(gaussian_cdf(x) + gaussian_cdf(-x),
                               np.ones_like(x), atol=1e-15)


# -- the counts kernel: a probability ---------------------------------------


def test_counts_kernel_integrates_to_the_bin_width():
    r"""The identity: :math:`\int \mathcal{S}_j(z^{\rm tr})\,dz^{\rm tr}
    = z_j^{\max} - z_j^{\min}`.

    Every true redshift is observed somewhere, so the kernel redistributes
    the bin's width without creating or destroying any of it.
    """
    integral = np.trapezoid(photoz_counts(FINE, Z_MIN, Z_MAX, SIGMA_Z), x=FINE)
    assert integral == pytest.approx(Z_MAX - Z_MIN, rel=1e-9)


def test_counts_kernel_is_a_probability():
    s = photoz_counts(FINE, Z_MIN, Z_MAX, SIGMA_Z)
    assert np.all(s >= 0.0) and np.all(s <= 1.0)


def test_counts_kernel_is_a_half_at_each_edge():
    """At a bin edge exactly half the scatter falls inside."""
    for edge in (Z_MIN, Z_MAX):
        assert photoz_counts(edge, Z_MIN, Z_MAX, SIGMA_Z).item() == (
            pytest.approx(0.5, abs=1e-12)
        )


def test_counts_kernel_is_one_deep_inside_the_bin():
    z_mid = 0.5 * (Z_MIN + Z_MAX)
    assert photoz_counts(z_mid, Z_MIN, Z_MAX, SIGMA_Z).item() == (
        pytest.approx(1.0, abs=1e-12)
    )


def test_counts_kernel_has_support_everywhere():
    """Unlike the projection weight. Small, but never exactly zero.

    This is the property that makes substituting it into a line-of-sight
    integral wrong: it puts weight at every redshift.
    """
    tail = photoz_counts(Z_MIN - 5 * SIGMA_Z, Z_MIN, Z_MAX, SIGMA_Z).item()
    assert 0.0 < tail < 1e-6


def test_counts_kernel_six_sigma_tail_justifies_the_L_z_envelope():
    """The y3 grid hard-zeroes at L_z = 6; this is what that discards."""
    tail = photoz_counts(Z_MIN - 6 * SIGMA_Z, Z_MIN, Z_MAX, SIGMA_Z).item()
    assert tail < 1e-8


def test_counts_kernel_edges_are_not_interchangeable():
    """Swapping z_min and z_max flips the sign -- the order is load-bearing."""
    a = photoz_counts(0.40, Z_MIN, Z_MAX, SIGMA_Z).item()
    b = photoz_counts(0.40, Z_MAX, Z_MIN, SIGMA_Z).item()
    assert a == pytest.approx(-b)
    assert a > 0.0


def test_counts_kernel_narrows_with_sigma_z():
    """A tighter photo-z makes the bin sharper, not just smaller."""
    z = Z_MIN - 0.02
    wide = photoz_counts(z, Z_MIN, Z_MAX, 0.03).item()
    tight = photoz_counts(z, Z_MIN, Z_MAX, 0.01).item()
    assert wide > tight
    # and the 3x width claim: 0.03 is the 3-sigma window of sigma_z = 0.01
    assert photoz_counts(Z_MIN - 3 * 0.01, Z_MIN, Z_MAX, 0.01).item() < 2e-3


def test_counts_kernel_rejects_a_non_positive_sigma():
    with pytest.raises(ValueError, match="sigma_z"):
        photoz_counts(0.4, Z_MIN, Z_MAX, 0.0)


# -- the projection weight: a compact window --------------------------------


def test_projection_weight_integrates_to_four_thirds_of_the_window():
    r""":math:`\int (1 - u^2)\,dz = \frac43 n_\sigma\sigma_z`. It is
    **not** a probability, and this is the number that says so."""
    integral = np.trapezoid(photoz_projection(FINE, Z_OB, SIGMA_Z), x=FINE)
    assert integral == pytest.approx(4.0 / 3.0 * WINDOW, rel=1e-6)


def test_projection_window_is_three_sigma_not_one():
    """The 0.03, pinned. This is the number the two kernels disagree on.

    The parabola's half-width is 3*sigma_z = 0.03 for sigma_z = 0.01, which
    is why 0.03 appears in the y3 configs. It is the window, never the
    scatter.
    """
    w = photoz_projection(FINE, Z_OB, SIGMA_Z)
    support = FINE[w > 0]
    half_width = 0.5 * (support[-1] - support[0])
    assert half_width == pytest.approx(WINDOW, rel=1e-3)
    assert half_width == pytest.approx(0.03, rel=1e-3)
    # and the kernel is comfortably nonzero one sigma out, where a
    # sigma-wide window would already have vanished
    assert photoz_projection(Z_OB + SIGMA_Z, Z_OB, SIGMA_Z).item() > 0.8


def test_projection_weight_vanishes_outside_the_window():
    """Exactly zero, not small -- the support is compact."""
    w = photoz_projection(FINE, Z_OB, SIGMA_Z)
    outside = np.abs(FINE - Z_OB) >= WINDOW
    assert np.all(w[outside] == 0.0)
    assert np.all(w[~outside] > 0.0)


def test_n_sigma_scales_the_window_linearly():
    """So a caller reproducing a different convention can say so."""
    for n in (1.0, 2.0, 3.0, 5.0):
        w = photoz_projection(FINE, Z_OB, SIGMA_Z, n_sigma=n)
        integral = np.trapezoid(w, x=FINE)
        assert integral == pytest.approx(4.0 / 3.0 * n * SIGMA_Z, rel=1e-5)


def test_passing_the_window_as_the_scatter_is_three_times_too_wide():
    """The mistake this signature exists to make visible."""
    right = photoz_projection(FINE, Z_OB, SIGMA_Z)              # 0.01 scatter
    wrong = photoz_projection(FINE, Z_OB, 3 * SIGMA_Z)          # 0.03 as scatter
    assert (np.trapezoid(wrong, x=FINE) / np.trapezoid(right, x=FINE)
            == pytest.approx(3.0, rel=1e-4))


def test_projection_weight_peaks_at_one_at_the_centre():
    assert photoz_projection(Z_OB, Z_OB, SIGMA_Z).item() == pytest.approx(1.0)


def test_projection_weight_is_the_parabola_it_claims_to_be():
    """Transcription check against 1 - u^2 written out."""
    z = np.linspace(Z_OB - 0.9 * WINDOW, Z_OB + 0.9 * WINDOW, 25)
    u = (z - Z_OB) / WINDOW
    np.testing.assert_allclose(photoz_projection(z, Z_OB, SIGMA_Z),
                               1.0 - u**2, rtol=1e-14)


def test_projection_weight_is_symmetric():
    for d in (0.2, 0.5, 0.9):
        lo = photoz_projection(Z_OB - d * WINDOW, Z_OB, SIGMA_Z).item()
        hi = photoz_projection(Z_OB + d * WINDOW, Z_OB, SIGMA_Z).item()
        assert lo == pytest.approx(hi)


def test_projection_weight_accepts_a_callable_sigma_z():
    r""":math:`\sigma_z(z)` is a table in production, so a callable must work.

    NOTE: it is evaluated at :math:`z`, not at :math:`z^{\rm ob}` -- which
    is what the exact C++ core does, and it makes the window asymmetric
    when the width varies.
    """
    def width(z):
        return 0.01 * (1.0 + np.asarray(z, dtype=float))

    z = np.array([Z_OB - 0.02, Z_OB, Z_OB + 0.02])
    got = photoz_projection(z, Z_OB, width)
    u = (z - Z_OB) / (N_SIGMA * width(z))
    np.testing.assert_allclose(got, np.maximum(0.0, 1.0 - u**2), rtol=1e-14)
    # a varying width breaks the symmetry a constant one has
    assert got[0] != pytest.approx(got[2])


def test_projection_weight_rejects_a_non_positive_sigma():
    with pytest.raises(ValueError, match="positive"):
        photoz_projection(0.4, Z_OB, 0.0)
    with pytest.raises(ValueError, match="positive"):
        photoz_projection(np.array([0.4]), Z_OB, lambda z: np.zeros_like(z))
    with pytest.raises(ValueError, match="n_sigma"):
        photoz_projection(0.4, Z_OB, SIGMA_Z, n_sigma=0.0)


# -- the two are not interchangeable ----------------------------------------


def test_the_two_kernels_have_different_support():
    """The single most important difference: one is compact, one is not."""
    w = photoz_projection(FINE, Z_OB, SIGMA_Z)
    s = photoz_counts(FINE, Z_MIN, Z_MAX, SIGMA_Z)
    assert np.count_nonzero(w) < 0.10 * w.size    # a narrow window
    assert np.count_nonzero(s) > 0.10 * s.size    # weight far from the bin


def test_the_two_kernels_have_different_normalisation():
    """4/3 sigma_z against the bin width -- an order of magnitude apart."""
    proj = np.trapezoid(photoz_projection(FINE, Z_OB, SIGMA_Z), x=FINE)
    counts = np.trapezoid(photoz_counts(FINE, Z_MIN, Z_MAX, SIGMA_Z), x=FINE)
    assert counts / proj == pytest.approx(
        (Z_MAX - Z_MIN) / (4.0 / 3.0 * WINDOW), rel=1e-5
    )
    assert counts / proj > 3.0


@pytest.mark.parametrize(
    "kernel", [photoz_counts, photoz_projection],
    ids=["counts", "projection"],
)
def test_both_are_vectorised_and_scalar_safe(kernel):
    z = np.array([0.36, 0.42, 0.48])
    args = ((Z_MIN, Z_MAX, SIGMA_Z) if kernel is photoz_counts
            else (Z_OB, SIGMA_Z))
    assert np.ravel(kernel(z, *args)).shape == z.shape
    assert np.size(kernel(0.42, *args)) == 1
    np.testing.assert_allclose(np.ravel(kernel(z, *args))[1],
                               np.ravel(kernel(0.42, *args)))
