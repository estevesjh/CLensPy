r"""The binned observables: the weight, and its two contractions.

Almost every test here is an identity, because the design makes them
available. :math:`\langle N_{ij}\rangle` and
:math:`\Delta\Sigma_{ij}` are the *same* weight contracted against
different things, so:

- contracting against 1 must give exactly 1;
- contracting against :math:`M` must give exactly
  :math:`\langle M\rangle_{ij}`;
- :math:`\Omega(z)` must scale the counts linearly and cancel identically
  in any average;
- the centring mixture is linear, so it must commute with the stack.

An implementation with two separate weights cannot pass these. That is
what they are for.
"""

import numpy as np
import pytest

from clenspy.cosmology.fiducial import fiducial_cosmology
from clenspy.observables import ClusterAbundance, StackedDeltaSigma
from clenspy.observables.deltasigma import F_MIS_Y3, TAU_MIS_Y3
from clenspy.selection import EmgParams, LogNormalMor, SelectionFunction

COSMO = fiducial_cosmology()
LAM_EDGES = np.array([20.0, 30.0, 45.0, 60.0, 200.0])
Z_EDGES = np.array([0.20, 0.35, 0.50, 0.65])


def toy_mass_function(ln_mass, z):
    r"""A smooth analytic ``dn/dlnM``: no CAMB, no sigma grid.

    Exponentially cut off in mass and falling with redshift, which is
    enough to exercise every contraction. The absolute normalisation is
    arbitrary and no test depends on it.
    """
    lnm, zz = np.broadcast_arrays(np.asarray(ln_mass, dtype=float),
                                 np.asarray(z, dtype=float))
    m = np.exp(lnm)
    return 1e-5 * (m / 1e14) ** -1.0 * np.exp(-m / 5e14) / (1.0 + zz)


def flat_omega(z):
    """A constant 1500 deg^2 footprint, in steradians."""
    return np.full_like(np.asarray(z, dtype=float),
                        1500.0 * (np.pi / 180.0) ** 2)


def make_selection():
    return SelectionFunction(LAM_EDGES, Z_EDGES, LogNormalMor(),
                             EmgParams(-1.5, 3.0, 0.3, 0.12), sigma_z=0.01)


def make_abundance(n_m=16, n_z=20, omega=flat_omega):
    return ClusterAbundance(
        np.log(np.logspace(13.5, 15.3, n_m)),
        np.linspace(0.16, 0.70, n_z),
        toy_mass_function, make_selection(), COSMO, omega,
    )


# -- the weight and its shape ----------------------------------------------


def test_weight_has_the_documented_shape():
    ab = make_abundance()
    assert ab.weight().shape == (16, 20, 4, 3)


def test_counts_have_the_documented_shape_and_are_positive():
    ab = make_abundance()
    n = ab.counts()
    assert n.shape == (4, 3)
    assert np.all(n > 0.0)


def test_counts_are_dimensionless_and_finite():
    n = make_abundance().counts()
    assert np.all(np.isfinite(n))


# -- the contraction identities --------------------------------------------


def test_averaging_unity_returns_exactly_one():
    r"""The identity that proves `average` normalises by its own weight."""
    ab = make_abundance()
    ones = np.ones((ab.ln_mass.size, ab.z.size))
    np.testing.assert_allclose(ab.average(ones), 1.0, rtol=1e-14)


def test_averaging_mass_reproduces_mean_mass():
    ab = make_abundance()
    mass = np.broadcast_to(np.exp(ab.ln_mass)[:, None],
                           (ab.ln_mass.size, ab.z.size))
    np.testing.assert_allclose(ab.average(mass), ab.mean_mass(), rtol=1e-14)


def test_averaging_redshift_reproduces_mean_redshift():
    ab = make_abundance()
    z = np.broadcast_to(ab.z[None, :], (ab.ln_mass.size, ab.z.size))
    np.testing.assert_allclose(ab.average(z), ab.mean_redshift(), rtol=1e-14)


def test_average_is_linear():
    r""":math:`\langle aX + bY\rangle = a\langle X\rangle
    + b\langle Y\rangle`."""
    ab = make_abundance()
    shape = (ab.ln_mass.size, ab.z.size)
    x = np.exp(ab.ln_mass)[:, None] * np.ones(shape)
    y = ab.z[None, :] * np.ones(shape)
    combined = ab.average(3.0 * x - 2.0 * y)
    separate = 3.0 * ab.average(x) - 2.0 * ab.average(y)
    np.testing.assert_allclose(combined, separate, rtol=1e-12)


def test_average_rejects_a_mismatched_grid():
    ab = make_abundance()
    with pytest.raises(ValueError, match="must start with shape"):
        ab.average(np.ones((3, 4)))


# -- Omega belongs to the counts, and cancels in the average ---------------


def test_omega_scales_the_counts_linearly():
    single = make_abundance()
    doubled = make_abundance(omega=lambda z: 2.0 * flat_omega(z))
    np.testing.assert_allclose(doubled.counts() / single.counts(), 2.0,
                               rtol=1e-12)


def test_omega_cancels_identically_in_every_average():
    r"""Which is why a footprint must never be applied to a profile too."""
    single = make_abundance()
    doubled = make_abundance(omega=lambda z: 2.0 * flat_omega(z))
    np.testing.assert_allclose(doubled.mean_mass(), single.mean_mass(),
                               rtol=1e-14)
    np.testing.assert_allclose(doubled.mean_redshift(),
                               single.mean_redshift(), rtol=1e-14)


def test_a_z_dependent_omega_does_not_cancel_in_the_counts():
    """It reweights redshift, so it must change the counts' z-distribution."""
    flat = make_abundance()
    tilted = make_abundance(omega=lambda z: flat_omega(z) * (1.0 + z) ** 3)
    ratio = tilted.counts() / flat.counts()
    # the highest-z bin must be boosted relative to the lowest
    assert ratio[0, -1] > ratio[0, 0]


# -- physical sanity -------------------------------------------------------


def test_mean_mass_rises_with_richness_bin():
    """The single most diagnostic property of a correct weight."""
    m = make_abundance().mean_mass()
    for j in range(m.shape[1]):
        assert np.all(np.diff(m[:, j]) > 0.0), j


def test_mean_redshift_lies_inside_its_bin():
    z = make_abundance().mean_redshift()
    for j in range(z.shape[1]):
        assert np.all(z[:, j] > Z_EDGES[j])
        assert np.all(z[:, j] < Z_EDGES[j + 1])


def test_mean_mass_is_within_the_grid_range():
    ab = make_abundance()
    m = ab.mean_mass()
    assert np.all(m > np.exp(ab.ln_mass[0]))
    assert np.all(m < np.exp(ab.ln_mass[-1]))


def test_the_grid_is_converged_at_the_default_resolution():
    """Measured, not asserted -- the module's named approximation."""
    assert make_abundance(n_m=32, n_z=40).convergence() < 1e-2


def test_convergence_improves_with_resolution():
    coarse = make_abundance(n_m=12, n_z=14).convergence()
    fine = make_abundance(n_m=48, n_z=56).convergence()
    assert fine < coarse


def test_abundance_rejects_bad_grids():
    sel = make_selection()
    with pytest.raises(ValueError, match="ascending"):
        ClusterAbundance(np.log([1e15, 1e14]), np.linspace(0.2, 0.6, 5),
                         toy_mass_function, sel, COSMO, flat_omega)
    with pytest.raises(ValueError, match=">= 2 points"):
        ClusterAbundance(np.log([1e14]), np.linspace(0.2, 0.6, 5),
                         toy_mass_function, sel, COSMO, flat_omega)
    with pytest.raises(ValueError, match="z must be positive"):
        ClusterAbundance(np.log(np.logspace(14, 15, 4)),
                         np.array([0.0, 0.3, 0.6]),
                         toy_mass_function, sel, COSMO, flat_omega)


# -- the stacked profile ---------------------------------------------------


def toy_profile(r, mass, z):
    r""":math:`\Delta\Sigma \propto M/R`, so the stack has a known limit."""
    return mass / np.asarray(r, dtype=float)


#: A mutable default would be shared across tests; module-level instead.
RADII = np.array([0.2, 1.0, 5.0])


def make_stack(radii=None):
    radii = RADII if radii is None else radii
    ab = make_abundance()
    return StackedDeltaSigma.from_profile(ab, toy_profile, radii), ab


def test_stacked_profile_has_the_documented_shape():
    stack, _ = make_stack()
    assert stack.profile().shape == (4, 3, 3)


def test_a_profile_linear_in_mass_stacks_to_the_mean_mass():
    r"""With :math:`\Delta\Sigma = M/R` the stack is
    :math:`\langle M\rangle_{ij}/R` exactly.

    The strongest possible check that the stack uses the counts' own
    weight: any second, inconsistent weight breaks it.
    """
    stack, ab = make_stack(RADII)
    expected = ab.mean_mass()[..., None] / (RADII * ab.h)
    np.testing.assert_allclose(stack.profile(), expected, rtol=1e-12)


def test_from_profile_applies_the_h_conversion_once():
    r""":math:`M[M_\odot] = M[h^{-1}M_\odot]/h`, and only once."""
    ab = make_abundance()
    radii = np.array([1.0])
    stack = StackedDeltaSigma.from_profile(ab, toy_profile, radii)
    # toy_profile returns mass/r, so the grid must be exp(lnM)/h
    expected = np.exp(ab.ln_mass) / ab.h
    np.testing.assert_allclose(stack.profile_grid[:, 0, 0], expected,
                               rtol=1e-14)


def test_from_profile_honours_an_explicit_h():
    ab = make_abundance()
    a = StackedDeltaSigma.from_profile(ab, toy_profile, np.array([1.0]))
    b = StackedDeltaSigma.from_profile(ab, toy_profile, np.array([1.0]),
                                       h=1.0)
    np.testing.assert_allclose(b.profile_grid / a.profile_grid, ab.h,
                               rtol=1e-14)


def test_stack_rejects_a_mismatched_profile_grid():
    ab = make_abundance()
    with pytest.raises(ValueError, match=r"\(n_m, n_z, n_r\)"):
        StackedDeltaSigma(ab, np.ones((3, 4, 5)), np.array([1.0, 2.0]))


def test_the_profile_falls_with_radius_and_rises_with_richness():
    stack, _ = make_stack()
    ds = stack.profile()
    assert np.all(np.diff(ds, axis=-1) < 0.0)          # falls with R
    assert np.all(np.diff(ds, axis=0) > 0.0)           # rises with richness


# -- the centring mixture -------------------------------------------------


def test_the_mixture_commutes_with_the_stack():
    r"""It is linear, so mixing then stacking equals stacking then mixing.

    This is why `mixture` acts on results rather than being buried in the
    weight: the caller may do whichever is cheaper.
    """
    stack, ab = make_stack()
    mis_grid = 0.4 * stack.profile_grid
    mixed_first = ab.average((1 - F_MIS_Y3) * stack.profile_grid
                             + F_MIS_Y3 * mis_grid)
    stacked_first = stack.mixture(stack.profile(), ab.average(mis_grid))
    np.testing.assert_allclose(mixed_first, stacked_first, rtol=1e-12)


def test_the_mixture_interpolates_between_its_two_endpoints():
    stack, _ = make_stack()
    cen, mis = stack.profile(), 0.4 * stack.profile()
    np.testing.assert_allclose(stack.mixture(cen, mis, f_mis=0.0), cen,
                               rtol=1e-14)
    np.testing.assert_allclose(stack.mixture(cen, mis, f_mis=1.0), mis,
                               rtol=1e-14)
    half = stack.mixture(cen, mis, f_mis=0.5)
    np.testing.assert_allclose(half, 0.5 * (cen + mis), rtol=1e-14)


def test_the_mixture_rejects_an_unphysical_fraction():
    stack, _ = make_stack()
    cen = stack.profile()
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            stack.mixture(cen, cen, f_mis=bad)


def test_the_mixture_rejects_mismatched_shapes():
    stack, _ = make_stack()
    cen = stack.profile()
    with pytest.raises(ValueError, match="same shape"):
        stack.mixture(cen, cen[..., :1])


def test_the_des_y3_miscentring_constants():
    """Kelly et al. (2024), the values the paper adopts."""
    assert F_MIS_Y3 == 0.25
    assert TAU_MIS_Y3 == 0.17
