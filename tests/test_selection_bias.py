"""Tests for clenspy.clusters.selection_bias and geometry."""

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from clenspy.cosmology import PkGrid
from clenspy.halo import SigmaGrid, Tinker08MassFunction, Tinker10Bias
from clenspy.clusters import (
    BinDefinition,
    HodMOR,
    HodParams,
    SelBiasEngine,
    SelectionBiasTable,
    SigmoidBias,
    XiNL,
)
from clenspy.clusters.geometry import area_overlap, r_lambda, sigmoid_theta

COSMO = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.045)
H = 0.7


# ---------------- geometry ------------------------------------------------
def test_r_lambda_physical_units():
    # (lam/100)^0.2 / h: lam=100 -> 1/h Mpc
    assert np.isclose(r_lambda(100.0, H), 1.0 / H)


def test_area_overlap_limits():
    t_lob = 1e-3
    t_ltr = np.array([5e-4])
    # concentric small projector: fully inside the target
    assert np.isclose(area_overlap(np.array([0.0]), t_lob, t_ltr).ravel()[0], 1.0)
    # disjoint
    assert area_overlap(np.array([1e-2]), t_lob, t_ltr).ravel()[0] == 0.0
    # projector bigger than target, concentric: A = lob^2/ltr^2
    big = np.array([4e-3])
    assert np.isclose(
        area_overlap(np.array([0.0]), t_lob, big).ravel()[0],
        (t_lob / big[0]) ** 2,
    )
    # equal disks at separation d: analytic lens area
    d = 1e-3
    got = area_overlap(np.array([d]), t_lob, np.array([t_lob])).ravel()[0]
    x = d / (2 * t_lob)
    lens = 2 * t_lob**2 * np.arccos(x) - 0.5 * d * np.sqrt(
        4 * t_lob**2 - d**2
    )
    assert np.isclose(got, lens / (np.pi * t_lob**2), rtol=1e-10)


def test_sigmoid_theta_limits():
    t_lam = 2e-3
    assert sigmoid_theta(0.0, t_lam) < 0.25
    assert np.isclose(sigmoid_theta(0.5 * t_lam, t_lam), 0.5, rtol=1e-12)
    assert sigmoid_theta(20 * t_lam, t_lam) > 0.999999


# ---------------- SigmoidBias / table -------------------------------------
def test_sigmoid_bias_limits():
    sb = SigmoidBias(
        lob=25.0, zob=0.3, theta_lambda=2e-3, b_small=0.5, b_large=3.0
    )
    # sigma(0) = 1/(1 + e^{k*theta0}) = 1/(1 + e^{1.25}) — finite pedestal
    s0 = 1.0 / (1.0 + np.exp(2.5 * 0.5))
    assert np.isclose(sb(0.0), 0.5 + 2.5 * s0, rtol=1e-10)
    assert np.isclose(sb(1.0), 3.0, rtol=1e-6)  # theta >> theta_lam: b_large
    assert np.isclose(sb(1e-3), 1.75, rtol=1e-10)  # midpoint at theta0


def test_table_roundtrip(tmp_path):
    tab = SelectionBiasTable(
        lam_min=np.array([20.0]), lam_max=np.array([30.0]),
        zo_low=np.array([0.2]), zo_high=np.array([0.35]),
        lob=np.array([24.5]), zob=np.array([0.275]),
        theta_lambda=np.array([2e-3]),
        b_small=np.array([0.7]), b_large=np.array([3.1]),
    )
    p = tmp_path / "bsel.npz"
    tab.to_file(p)
    tab2 = SelectionBiasTable.from_file(p)
    assert tab2.n_rows == 1
    row = tab2.row(0)
    assert row.b_small == 0.7 and row.b_large == 3.1
    assert np.isclose(row(1.0), 3.1, rtol=1e-6)


# ---------------- engine ---------------------------------------------------
@pytest.fixture(scope="module")
def engine():
    pk_nl = PkGrid(
        backend="camb", cosmo=COSMO, nonlinear=True,
        k_range=(1e-4, 100.0), z_range=(0.0, 1.2), nk=600, nz=25,
    )
    pk_lin = PkGrid(
        backend="camb", cosmo=COSMO, nonlinear=False,
        k_range=(1e-4, 100.0), z_range=(0.0, 1.2), nk=600, nz=25,
    )
    sg = SigmaGrid(pk_lin, cosmo=COSMO)
    hmf = Tinker08MassFunction(sg)
    bias = Tinker10Bias(sg)
    return SelBiasEngine(
        cosmology=COSMO,
        xi_nl=XiNL(pk_nl),
        hmf=hmf,
        bias=bias,
        mor=HodMOR(HodParams.des_y1(H)),
    )


def test_xi_nl_sane(engine):
    xi = engine.xi_nl(np.array([0.5, 1.0, 10.0, 50.0, 400.0]), 0.3)
    assert np.all(xi >= 0)
    assert xi[0] > xi[2] > xi[3]  # decreasing
    assert xi[1] > 1.0  # nonlinear xi(1 Mpc) >> 1


def test_operators_positive_and_ordered(engine):
    P1, I1, I2 = engine.operators(25.0, 0.3)
    assert P1 > 0 and I2 > 0 and I1 > 0
    # sigmoid <= 1 pointwise -> I1 <= I2
    assert I1 < I2
    # background contamination of a lam<=25 cluster: a few galaxies at most
    assert 0.01 < P1 < 25.0


def test_b_eff_reasonable(engine):
    beff = engine.b_eff(25.0, 0.3)
    assert 1.0 < beff < 6.0
    # richer clusters are more biased
    assert engine.b_eff(80.0, 0.3) > beff


def test_closure_identity(engine):
    """(lob - ltr) = P1 + b_large*I1 + b_small*(I2 - I1) per ltr node."""
    lob, zob = 25.0, 0.3
    P1, I1, I2 = engine.operators(lob, zob)
    beff = engine.b_eff(lob, zob)
    ltr = np.linspace(5.0, 24.0, 10)
    _, b_small, b_large = engine._closure(lob, P1, I1, I2, beff, ltr)
    lhs = lob - ltr
    rhs = P1 + b_large * I1 + b_small * (I2 - I1)
    assert np.allclose(lhs, rhs, rtol=1e-10)


def test_marginalised_bias_and_table(engine):
    sb = engine.marginalised_bias(25.0, 0.3)
    assert np.isfinite(sb.b_small) and np.isfinite(sb.b_large)
    assert sb.b_large > 0
    assert sb.theta_lambda > 0
    # profile interpolates between plateaus
    th = np.array([1e-6, sb.theta_lambda / 2, 10 * sb.theta_lambda])
    prof = sb(th)
    assert np.isclose(prof[1], 0.5 * (sb.b_small + sb.b_large), rtol=1e-6)
    assert np.isclose(prof[2], sb.b_large, rtol=1e-3)

    tab = engine.build_table(
        (BinDefinition(20.0, 30.0, 0.2, 0.35),
         BinDefinition(30.0, 45.0, 0.2, 0.35)),
    )
    assert tab.n_rows == 2
    assert np.all(np.isfinite(tab.b_large))
