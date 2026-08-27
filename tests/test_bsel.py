r"""The selection-affected halo bias, :math:`b_{\rm sel}(\theta)`.

The paper's Section 4.1. Most of these are structural rather than
numerical, because the physical amplitude needs a self-consistent halo
model and the point of the tests is the *machinery*: that the three
operators are positive and ordered, that the closure inverts, that the
:math:`\lambda^{\rm tr}` marginalisation commutes with the sigmoid (which
is what makes a two-scalar table exact), and that the sigmoid has the
shape the paper specifies.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from clenspy.cosmology.fiducial import fiducial_cosmology
from clenspy.selection import (
    PhysicalMassMor,
    SelBiasEngine,
    SelectionBiasTable,
    SigmoidBias,
)
from clenspy.selection.geometry import sigmoid_theta
from clenspy.selection.scaling_relation import HodMor

COSMO = fiducial_cosmology()
H = COSMO.h
LOB, ZOB = 40.0, 0.4


def _hmf(mass, z):
    m, zz = np.broadcast_arrays(np.asarray(mass, float),
                                np.asarray(z, float))
    return 1e-19 * (m / 1e14) ** -2.0 * np.exp(-m / 5e14) / (1.0 + zz)


def _bias(mass, z):
    m, zz = np.broadcast_arrays(np.asarray(mass, float),
                                np.asarray(z, float))
    return 1.0 + 0.9 * (m / 3e14) ** 0.3 * (1.0 + zz) ** 0.5


def _xi(r, zob):
    r = np.asarray(r, dtype=float)
    return np.maximum((np.maximum(r, 1e-3) / 5.0) ** -1.8, 0.0)


def _engine(**kw):
    kwargs = dict(cosmology=COSMO, xi_nl=_xi, hmf=_hmf, bias=_bias,
                  mor=PhysicalMassMor(HodMor.des_y1(), H),
                  n_z=24, n_M=12, n_theta=6, n_ltr=30, ltr_grid_size=8)
    kwargs.update(kw)
    return SelBiasEngine(**kwargs)


# -- the three operators ---------------------------------------------------


def test_the_operators_are_positive_and_ordered():
    r""":math:`P_1, I_1, I_2 > 0` and :math:`I_1 < I_2`.

    :math:`I_1` carries an extra :math:`\sigma(\theta) \in (0,1)` factor
    relative to :math:`I_2`, so it must be the smaller of the two. If it
    is not, the closure's denominator flips sign and :math:`b_{\rm small}`
    is meaningless.
    """
    p1, i1, i2 = _engine().operators(LOB, ZOB)
    assert p1 > 0.0 and i1 > 0.0 and i2 > 0.0
    assert i1 < i2


def test_the_operators_are_cached():
    engine = _engine()
    first = engine.operators(LOB, ZOB)
    assert engine.operators(LOB, ZOB) is first


def test_a_vanishing_correlation_kills_I1_and_I2_but_not_P1():
    r""":math:`P_1 = \mathcal P[1]` has no :math:`\xi_{\rm NL}` in it."""
    engine = _engine(xi_nl=lambda r, zob: np.zeros_like(
        np.asarray(r, dtype=float)))
    p1, i1, i2 = engine.operators(LOB, ZOB)
    assert p1 > 0.0
    assert i1 == 0.0 and i2 == 0.0


def test_b_eff_is_a_bias_weighted_average_within_range():
    engine = _engine()
    b_eff = engine.b_eff(LOB, ZOB)
    masses = np.array([engine.min_mass, 10.0**engine.log10_M_max])
    assert _bias(masses, ZOB).min() < b_eff < _bias(masses, ZOB).max()


# -- the closure -----------------------------------------------------------


def test_the_closure_reproduces_its_own_algebra():
    engine = _engine()
    p1, i1, i2 = engine.operators(LOB, ZOB)
    b_eff = engine.b_eff(LOB, ZOB)
    ltr = np.array([30.0, 35.0])
    delta, b_small, b_large = engine._closure(LOB, p1, i1, i2, b_eff, ltr)

    d_rnd = p1 + b_eff * i2
    np.testing.assert_allclose(delta, (LOB - ltr) / d_rnd - 1.0, rtol=1e-14)
    np.testing.assert_allclose(
        b_large, b_eff * (1.0 + engine.boost_slope * delta), rtol=1e-14)
    np.testing.assert_allclose(
        b_small, ((LOB - ltr) - p1 - b_large * i1) / (i2 - i1), rtol=1e-12)


def test_the_boost_slope_is_the_buzzard_calibrated_number():
    """0.13, the one non-closed-form number in the model."""
    assert SelBiasEngine.boost_slope == 0.13


def test_a_degenerate_denominator_falls_back_rather_than_diverging():
    r"""When :math:`I_2 \to I_1`, :math:`b_{\rm small}` is not defined.

    A named degradation: it returns :math:`b_{\rm large}` instead of an
    arbitrarily large number.
    """
    engine = _engine()
    ltr = np.array([30.0, 35.0])
    _, b_small, b_large = engine._closure(LOB, 1.0, 0.5, 0.5, 2.0, ltr)
    np.testing.assert_allclose(b_small, b_large, rtol=0.0)


# -- the marginalisation, and why the table is two columns wide ------------


def test_the_marginalisation_commutes_with_the_sigmoid():
    r"""The identity that makes a two-scalar table **exact**.

    :math:`\sigma(\theta)` carries no :math:`\lambda^{\rm tr}`, so
    averaging the plateaus and then building the sigmoid equals building a
    sigmoid per :math:`\lambda^{\rm tr}` and averaging. Without this the
    table would have to store a whole :math:`\theta` grid per bin.
    """
    engine = _engine()
    p1, i1, i2 = engine.operators(LOB, ZOB)
    b_eff = engine.b_eff(LOB, ZOB)
    ltr, weights = engine._ltr_weights(LOB, ZOB)
    _, b_small_vec, b_large_vec = engine._closure(
        LOB, p1, i1, i2, b_eff, ltr)
    profile = engine.marginalised_bias(LOB, ZOB)
    theta_lam = profile.theta_lambda

    for fraction in (0.1, 0.5, 1.0, 3.0):
        theta = fraction * theta_lam
        sigma = sigmoid_theta(theta, theta_lam)
        built_then_averaged = float(np.sum(
            weights * (b_small_vec + (b_large_vec - b_small_vec) * sigma)))
        assert profile(theta) == pytest.approx(built_then_averaged,
                                              rel=1e-12), fraction


def test_the_ltr_weights_are_normalised():
    _, weights = _engine()._ltr_weights(LOB, ZOB)
    assert np.sum(weights) == pytest.approx(1.0)
    assert np.all(weights >= 0.0)


# -- the sigmoid profile ---------------------------------------------------


def test_the_sigmoid_is_half_way_at_half_the_aperture():
    """The paper's theta_0 = theta_lambda/2, checked through the profile."""
    profile = SigmoidBias(lob=40.0, zob=0.4, theta_lambda=1e-3,
                          b_small=2.0, b_large=3.0)
    assert profile(0.5e-3) == pytest.approx(2.5)


def test_the_sigmoid_tends_to_its_two_plateaus():
    profile = SigmoidBias(lob=40.0, zob=0.4, theta_lambda=1e-3,
                          b_small=2.0, b_large=3.0)
    assert profile(0.0) == pytest.approx(2.0, abs=0.25)   # k=2.5 is gentle
    assert profile(1e-2) == pytest.approx(3.0, abs=1e-6)


def test_the_sigmoid_is_monotone_between_the_plateaus():
    profile = SigmoidBias(lob=40.0, zob=0.4, theta_lambda=1e-3,
                          b_small=2.0, b_large=3.0)
    theta = np.linspace(0.0, 5e-3, 200)
    values = np.array([profile(t) for t in theta])
    assert np.all(np.diff(values) > 0.0)


# -- the table -------------------------------------------------------------


def test_the_table_is_two_scalars_per_row():
    engine = _engine()
    bins = [SimpleNamespace(lam_min=20.0, lam_max=30.0,
                            zob_min=0.2, zob_max=0.35),
            SimpleNamespace(lam_min=30.0, lam_max=45.0,
                            zob_min=0.2, zob_max=0.35)]
    table = engine.build_table(bins)
    assert table.n_rows == 2
    assert table.b_small.shape == (2,) and table.b_large.shape == (2,)
    # and a row reconstructs a callable profile
    row = table.row(0)
    assert isinstance(row, SigmoidBias)
    assert row(0.5 * row.theta_lambda) == pytest.approx(
        0.5 * (row.b_small + row.b_large))


def test_the_table_round_trips_through_a_file(tmp_path):
    engine = _engine()
    bins = [SimpleNamespace(lam_min=20.0, lam_max=30.0,
                            zob_min=0.2, zob_max=0.35)]
    table = engine.build_table(bins)
    path = tmp_path / "bsel.npz"
    table.to_file(path)
    back = SelectionBiasTable.from_file(path)
    np.testing.assert_allclose(back.b_small, table.b_small)
    np.testing.assert_allclose(back.b_large, table.b_large)


# -- the h boundary --------------------------------------------------------


def test_the_mor_adapter_converts_mass_in_the_right_direction():
    r""":math:`M[h^{-1}M_\odot] = M[M_\odot]\times h`, once."""
    mor = HodMor.des_y1()
    adapter = PhysicalMassMor(mor, H)
    mass_physical = 3e14
    np.testing.assert_allclose(
        adapter.pdf(30.0, mass_physical, 0.4),
        mor.pdf(30.0, np.log(mass_physical * H), 0.4), rtol=1e-14,
    )
    # and the wrong direction is a different number, by h^2
    assert not np.allclose(adapter.pdf(30.0, mass_physical, 0.4),
                           mor.pdf(30.0, np.log(mass_physical / H), 0.4))


def test_the_default_mass_range_is_the_richness_selection_one():
    r""":math:`10^{13}` to :math:`10^{15.5}\,h^{-1}M_\odot`, converted."""
    engine = _engine()
    assert engine.min_mass == pytest.approx(1e13 / H)
    assert 10.0**engine.log10_M_max == pytest.approx(10.0**15.5 / H)
