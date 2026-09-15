r"""The selection-affected halo bias, :math:`b_{\rm sel}(\theta)`.

The paper's Section 4.1. Most of these are structural rather than
numerical, because the physical amplitude needs a self-consistent halo
model and the point of the tests is the *machinery*: that the three
operators are positive and ordered, that the closure inverts, that both
plateaus are affine in :math:`\lambda^{\rm tr}` (which is what makes a
two-scalar table exact -- see docs/plan-bsel-stable-closure.md), and that
the sigmoid has the shape the paper specifies.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from clenspy.cosmology import TinkerMassFunction
from clenspy.cosmology.bias import BiasModel
from clenspy.cosmology.fiducial import fiducial_cosmology
from clenspy.cosmology.pkgrid import PkGrid
from clenspy.lensing import SigmaPrj
from clenspy.selection import (
    PhysicalMassMor,
    SelBiasEngine,
    SelectionBiasTable,
    SigmoidBias,
    XiNL,
)
from clenspy.selection.scaling_relation import HodMor

COSMO = fiducial_cosmology()
H = COSMO.h
LOB, ZOB = 40.0, 0.4

# the real halo model, once -- CAMB is disk-cached, so this is one real
# build shared by every test, not a toy stand-in (matches the "no toy
# stand-ins" convention of tests/test_projection.py)
_MGRID = np.geomspace(1.0e12, 1.0e16, 128)
_ZGRID = np.linspace(0.0, 1.0, 21)
_HMF = TinkerMassFunction(cosmo=COSMO, mvec=_MGRID, zvec=_ZGRID)
_BIAS_MODEL = BiasModel(cosmo=COSMO, mvec=_MGRID, zvec=_ZGRID)
_XI_NL = XiNL(PkGrid(cosmo=COSMO, nonlinear=True), clip=False)
_SIGMA_PRJ = SigmaPrj(cosmology=COSMO, hmf=_HMF, bias=_BIAS_MODEL,
                      xi_nl=_XI_NL).build()


def _engine(**kw):
    """SelBiasEngine sharing the module's built SigmaPrj; ``xi_nl=``
    (only) rebuilds a fresh SigmaPrj with that override -- hmf/bias stay
    the already-built grid models, so no CAMB re-run."""
    xi_override = kw.pop("xi_nl", None)
    sigma_prj = (_SIGMA_PRJ if xi_override is None else
                 SigmaPrj(cosmology=COSMO, hmf=_HMF, bias=_BIAS_MODEL,
                          xi_nl=xi_override).build())
    kwargs = dict(sigma_prj=sigma_prj, mor=HodMor.des_y1(),
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
    b_vals = engine.bias(masses, ZOB)
    assert b_vals.min() < b_eff < b_vals.max()


def test_operators_stay_finite_when_exclusion_dominates_the_aperture():
    r"""The old mechanism was a per-`z` Python skip when
    ``theta_excl(z)`` exceeded ``theta_max``; the new one is
    `clenspy.utils.los_integrals.LosGeometry`'s ``u_split`` collapsing
    the "outside" interval smoothly to (near-)empty. Shrinking
    ``theta_lob`` (independent of the exclusion geometry, driven by
    ``R_excl``) pushes the exclusion angle above ``theta_max`` at some
    line-of-sight nodes; `operators` must still return finite,
    non-negative numbers built from the surviving ones, not NaN or a
    crash.
    """
    engine = _engine()
    natural_theta_lob = engine._theta_lob(LOB, ZOB)
    engine._theta_lob = lambda lob, zob: 0.05 * natural_theta_lob
    p1, i1, i2 = engine.operators(LOB, ZOB)
    assert np.all(np.isfinite([p1, i1, i2]))
    assert p1 >= 0.0 and i1 >= 0.0 and i2 >= 0.0
    # some nodes did contribute -- this is a partial skip, not a wipeout
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


def test_the_stable_D_matches_the_subtraction_away_from_degeneracy():
    r"""``D`` (`_operators`'s directly-quadratured :math:`I_2-I_1`) and the
    plain float subtraction agree wherever there is no cancellation to
    protect against -- i.e. this is the same number, computed two ways."""
    engine = _engine()
    p1, i1, i2 = engine.operators(LOB, ZOB)
    D = engine._d_cache[("ops", float(LOB), float(ZOB))]
    assert D == pytest.approx(i2 - i1, rel=1e-6)


# -- the marginalisation, and why the table is two columns wide ------------


def test_b_small_large_matches_the_closure_at_the_injected_delta():
    r"""`b_small_large` is `_closure`'s own algebra, evaluated at a single
    ``ltr`` equivalent to ``delta`` -- both plateaus are affine in
    :math:`\lambda^{\rm tr}`, so a posterior would contribute only its
    mean (docs/plan-bsel-stable-closure.md), and this is that mean
    evaluated directly rather than quadratured and averaged.
    """
    engine = _engine()
    P1, I1, I2 = engine.operators(LOB, ZOB)
    D = engine._d_cache[("ops", float(LOB), float(ZOB))]
    b_eff = engine.b_eff(LOB, ZOB)
    delta = 0.3
    D_RND = P1 + b_eff * I2
    ltr_equiv = np.array([LOB - D_RND * (1.0 + delta)])
    _, bs_expected, bl_expected = engine._closure(
        LOB, P1, I1, I2, b_eff, ltr_equiv, D=D)

    bs, bl = engine.b_small_large(LOB, ZOB, b_eff=b_eff, delta=delta)
    assert bs == pytest.approx(float(bs_expected[0]), rel=1e-12)
    assert bl == pytest.approx(float(bl_expected[0]), rel=1e-12)


def test_b_small_large_at_zero_delta_is_the_fixed_point():
    """delta=0 (an average line of sight) collapses both plateaus to
    b_eff exactly -- the closure's own fixed point."""
    engine = _engine()
    b_eff = engine.b_eff(LOB, ZOB)
    bs, bl = engine.b_small_large(LOB, ZOB, b_eff=b_eff, delta=0.0)
    assert bs == pytest.approx(b_eff, rel=1e-10)
    assert bl == pytest.approx(b_eff, rel=1e-14)


def test_b_small_large_accept_an_external_b_eff():
    r"""The ``b_eff=`` override: `ClusterCounts.average` computes the
    bin-averaged :math:`N[b]/N[1]` and feeds it here in place of the
    engine's own fixed-:math:`\lambda^{\rm ob}` average. ``None``
    reproduces the internal value exactly; a passed value threads into
    both plateaus (:math:`b_{\rm large} = b_{\rm eff}(1 +
    0.13\delta^{\rm prj})` directly, :math:`b_{\rm small}` through the
    closure)."""
    engine = _engine()
    internal = engine.b_small_large(LOB, ZOB)
    explicit = engine.b_small_large(LOB, ZOB, b_eff=engine.b_eff(LOB, ZOB))
    assert explicit == pytest.approx(internal, rel=1e-14)

    b_small_2, b_large_2 = engine.b_small_large(LOB, ZOB, b_eff=2.0)
    b_small_3, b_large_3 = engine.b_small_large(LOB, ZOB, b_eff=3.0)
    assert b_small_2 != b_small_3 and b_large_2 != b_large_3
    # b_large follows the closure formula at the passed b_eff exactly
    # (delta itself depends on b_eff through Delta_RND = P1 + b_eff I2,
    # and through excess_delta's own b_eff factor)
    delta = engine.excess_delta(LOB, ZOB, 2.0)
    expected = 2.0 * (1.0 + engine.boost_slope * delta)
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


def test_engine_wraps_the_raw_mor_itself():
    r"""``SelBiasEngine(mor=...)`` takes the raw h-scaled MOR and wraps it
    in `PhysicalMassMor` internally -- the caller never constructs that
    adapter by hand."""
    engine = _engine()
    assert isinstance(engine.mor, PhysicalMassMor)


def test_the_default_mass_range_is_the_richness_selection_one():
    r""":math:`10^{13}` to :math:`10^{15.5}\,h^{-1}M_\odot`, converted."""
    engine = _engine()
    assert engine.min_mass == pytest.approx(1e13 / H)
    assert 10.0**engine.log10_M_max == pytest.approx(10.0**15.5 / H)
