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
    XiNL,
)
from clenspy.selection.geometry import r_lambda, sigmoid_theta
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


# -- XiNL: the cached, FFTLog-tabulated correlation function ---------------


class _FakePkGrid:
    """Minimal pkgrid stand-in: a `.k` array and a power-law `P(k)`."""

    def __init__(self, k):
        self.k = k
        self.n_calls = 0

    def __call__(self, k, z):
        self.n_calls += 1
        return np.asarray(k, dtype=float) ** -1.5


def _fake_pkgrid():
    return _FakePkGrid(np.logspace(-4, 3, 2048))


def test_xinl_output_shape_matches_r_and_is_non_negative():
    xi_nl = XiNL(_fake_pkgrid())
    r = np.array([0.3, 1.0, 5.0, 20.0, 100.0])
    out = xi_nl(r, zob=0.3)
    assert out.shape == r.shape
    assert np.all(out >= 0.0)


def test_xinl_caches_per_redshift_and_recomputes_on_a_new_one():
    pkgrid = _fake_pkgrid()
    xi_nl = XiNL(pkgrid)
    r = np.array([1.0, 10.0])

    out1 = xi_nl(r, zob=0.3)
    assert pkgrid.n_calls == 1

    # same zob (bit-identical) -> cache hit, no recompute
    out2 = xi_nl(r, zob=0.3)
    assert pkgrid.n_calls == 1
    np.testing.assert_allclose(out1, out2, rtol=0.0)

    # a zob that rounds to the same 8-decimal key -> still a cache hit
    out3 = xi_nl(r, zob=0.300000001)
    assert pkgrid.n_calls == 1
    np.testing.assert_allclose(out1, out3, rtol=0.0)

    # a genuinely different zob -> a new cache entry, pkgrid called again
    xi_nl(r, zob=0.5)
    assert pkgrid.n_calls == 2


def test_xinl_second_call_does_not_touch_the_pkgrid_at_all():
    r"""Proves the cache hit by making a second pkgrid call an error."""
    k = np.logspace(-4, 3, 2048)

    class _RaiseOnSecondCall:
        def __init__(self, k):
            self.k = k
            self.n_calls = 0

        def __call__(self, k, z):
            self.n_calls += 1
            if self.n_calls > 1:
                raise AssertionError(
                    "pkgrid called again for an already-cached z")
            return np.asarray(k, dtype=float) ** -1.5

    pkgrid = _RaiseOnSecondCall(k)
    xi_nl = XiNL(pkgrid)
    r = np.array([1.0, 5.0])
    xi_nl(r, zob=0.4)
    xi_nl(r, zob=0.4)          # would raise if it recomputed
    assert pkgrid.n_calls == 1


def test_xinl_left_clamps_below_the_r_grid():
    xi_nl = XiNL(_fake_pkgrid())
    r_below = np.array([xi_nl.rvals[0] * 1e-3])
    out = xi_nl(r_below, zob=0.3)
    xi_tab = xi_nl._cache[round(0.3, 8)]
    assert out[0] == pytest.approx(max(float(xi_tab[0]), 0.0))


def test_xinl_right_clamps_to_zero_above_the_r_grid():
    xi_nl = XiNL(_fake_pkgrid())
    r_above = np.array([xi_nl.rvals[-1] * 10.0])
    out = xi_nl(r_above, zob=0.3)
    assert out[0] == 0.0


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


def test_z_grid_drops_the_outer_fg_side_when_the_exclusion_ball_engulfs_it():
    r"""``_outer``'s ``R_excl >= dis_max`` guard.

    Shrinking the foreground window until its whole comoving span sits
    inside the exclusion ball leaves nothing for the log-spaced outer
    quadrature to sample, so that side must come back empty rather than
    erroring on ``log(dis_max) < log(R_excl)``.
    """
    engine = _engine()
    lob, zob = 200.0, 0.4
    R_excl = float(r_lambda(lob, engine.h) * (1.0 + zob))
    chi_o = float(engine.chi(zob))

    # a foreground boundary whose distance from chi_o is half of R_excl:
    # dis_fg_max < R_excl, so the fg branch must return empty arrays.
    from scipy.optimize import brentq

    target = 0.5 * R_excl
    z_fg_lo = brentq(lambda z: (chi_o - float(engine.chi(z))) - target,
                     1e-4, zob)
    z_bg_hi = zob + 0.4          # generous background side: not triggered

    zs_narrow, wzs_narrow = engine._z_grid(lob, zob, z_fg_lo, z_bg_hi)
    zs_wide, wzs_wide = engine._z_grid(lob, zob, 1e-4, z_bg_hi)

    # the wide call keeps its fg outer nodes, the narrow one drops them
    assert zs_narrow.size < zs_wide.size
    assert np.all(np.isfinite(zs_narrow)) and np.all(np.isfinite(wzs_narrow))
    assert zs_narrow.min() >= z_fg_lo - 1e-9


def test_operators_skip_z_nodes_whose_exclusion_angle_exceeds_theta_max():
    r"""``if th_lo >= theta_max or wz_kern[iz] == 0.0: continue``.

    Shrinking ``theta_lob`` (independent of the exclusion geometry, which
    is driven by ``R_excl``) pushes ``theta_max`` below the exclusion
    angle at some line-of-sight nodes, so the loop must skip them without
    crashing and still return a finite, non-negative result built from the
    surviving ones.
    """
    engine = _engine()
    natural_theta_lob = engine._theta_lob(LOB, ZOB)
    engine._theta_lob = lambda lob, zob: 0.05 * natural_theta_lob
    p1, i1, i2 = engine.operators(LOB, ZOB)
    assert np.all(np.isfinite([p1, i1, i2]))
    assert p1 >= 0.0 and i1 >= 0.0 and i2 >= 0.0
    # some z-nodes did contribute -- this is a partial skip, not a wipeout
    assert p1 > 0.0


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


def test_plateaus_accept_an_external_b_eff():
    r"""The ``b_eff=`` override: `ClusterCounts.average` computes the
    bin-averaged :math:`N[b]/N[1]` and feeds it here in place of the
    engine's own fixed-:math:`\lambda^{\rm ob}` average. ``None``
    reproduces the internal value exactly; a passed value threads into
    both plateaus (:math:`b_{\rm large} = b_{\rm eff}(1 +
    0.13\delta^{\rm prj})` directly, :math:`b_{\rm small}` through the
    closure)."""
    engine = _engine()
    internal = engine.plateaus(LOB, ZOB)
    explicit = engine.plateaus(LOB, ZOB, b_eff=engine.b_eff(LOB, ZOB))
    assert explicit == pytest.approx(internal, rel=1e-14)

    b_small_2, b_large_2 = engine.plateaus(LOB, ZOB, b_eff=2.0)
    b_small_3, b_large_3 = engine.plateaus(LOB, ZOB, b_eff=3.0)
    assert b_small_2 != b_small_3 and b_large_2 != b_large_3
    # b_large follows the closure formula at the passed b_eff exactly
    # (delta_prj itself depends on b_eff through Delta_RND = P1 + b_eff I2)
    p1, i1, i2 = engine.operators(LOB, ZOB)
    ltr, w = engine._ltr_weights(LOB, ZOB)
    delta = (LOB - ltr) / (p1 + 2.0 * i2) - 1.0
    expected = float(np.sum(w * 2.0 * (1.0 + engine.boost_slope * delta)))
    assert b_large_2 == pytest.approx(expected, rel=1e-12)
    profile = engine.marginalised_bias(LOB, ZOB, b_eff=2.0)
    assert profile.b_large == pytest.approx(b_large_2, rel=1e-14)


def test_the_ltr_weights_are_normalised():
    _, weights = _engine()._ltr_weights(LOB, ZOB)
    assert np.sum(weights) == pytest.approx(1.0)
    assert np.all(weights >= 0.0)


def test_ltr_weights_without_plob_use_the_hmf_prior_alone():
    r"""``use_plob_ltr=False`` takes the ``else: p_ltr = prior`` branch.

    Dropping :math:`P(\lambda^{\rm ob}\mid\lambda^{\rm tr})` still leaves a
    normalised, non-negative weight -- and a genuinely different one from
    the default, since the branch changes what enters the weight.
    """
    engine = _engine()
    ltr_with, w_with = engine._ltr_weights(LOB, ZOB, use_plob_ltr=True)
    ltr_without, w_without = engine._ltr_weights(LOB, ZOB, use_plob_ltr=False)
    np.testing.assert_allclose(ltr_with, ltr_without)
    assert np.sum(w_without) == pytest.approx(1.0)
    assert np.all(w_without >= 0.0)
    assert not np.allclose(w_with, w_without)


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


def test_physical_mass_mor_repr_names_itself():
    adapter = PhysicalMassMor(HodMor.des_y1(), H)
    text = repr(adapter)
    assert isinstance(text, str)
    assert "PhysicalMassMor" in text


def test_the_default_mass_range_is_the_richness_selection_one():
    r""":math:`10^{13}` to :math:`10^{15.5}\,h^{-1}M_\odot`, converted."""
    engine = _engine()
    assert engine.min_mass == pytest.approx(1e13 / H)
    assert 10.0**engine.log10_M_max == pytest.approx(10.0**15.5 / H)
