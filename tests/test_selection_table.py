"""Tests for clenspy.clusters.selection (S_ij tables)."""

import numpy as np
import pytest
from scipy.special import erfc

from clenspy.clusters import (
    AnalyticLogNormalKernel,
    BinDefinition,
    EmgRichnessKernel,
    HodMOR,
    HodParams,
    LogNormalMOR,
    LogNormalParams,
    SelectionFunctionBuilder,
)

H = 0.7

BINS = (
    BinDefinition(20.0, 30.0, 0.2, 0.35),
    BinDefinition(30.0, 45.0, 0.2, 0.35),
    BinDefinition(45.0, 60.0, 0.2, 0.35, sigma_z=0.03),
    BinDefinition(60.0, 200.0, 0.35, 0.5),
)


@pytest.fixture(scope="module")
def hod_builder():
    return SelectionFunctionBuilder(
        HodMOR(HodParams.des_y1(H)), EmgRichnessKernel(), n_lnm=96, n_z=48
    )


@pytest.fixture(scope="module")
def hod_table(hod_builder):
    return hod_builder.build(BINS)


def test_resonance_guard():
    with pytest.raises(ValueError, match="resonance"):
        SelectionFunctionBuilder(
            HodMOR(HodParams.des_y1(H)), EmgRichnessKernel(), n_lnm=64
        )


def test_table_shapes_and_range(hod_table):
    assert hod_table.S.shape == (4, 96, 48)
    assert np.all((hod_table.S >= 0.0) & (hod_table.S <= 1.0))
    # photo-z kernel: bins with zob in [0.2, 0.35] vanish at z=0.8
    assert hod_table(0, np.log(3e14), 0.79) < 1e-8


def test_table_vs_pointwise(hod_builder, hod_table):
    """Interpolated table matches the direct pointwise quadrature off-grid."""
    rng = np.random.default_rng(42)
    lnM = rng.uniform(np.log(5e13), np.log(2e15), 20)
    z = rng.uniform(0.22, 0.33, 20)
    for b in (0, 1, 3):
        bd = BINS[b]
        direct = np.array(
            [
                hod_builder.s_i_pointwise(np.exp(lm), zz, bd.lam_min, bd.lam_max)
                for lm, zz in zip(lnM, z)
            ]
        )
        # sigma_z = 0 and z inside [zob_min, zob_max) -> K_j = 1
        inside = (z >= bd.zob_min) & (z < bd.zob_max)
        got = np.array([hod_table(b, lm, zz) for lm, zz in zip(lnM, z)])
        ref = direct * inside
        # bilinear-interp error dominates: ~3% on the steep low-S tail,
        # sub-percent where S is appreciable
        keep = ref > 1e-4
        assert np.allclose(got[keep], ref[keep], rtol=4e-2), b
        strong = ref > 0.05
        assert np.allclose(got[strong], ref[strong], rtol=2e-2), b


def test_grid_resonance_regression():
    """Contracted mass integral must agree between n_lnm=96 and 192 (<0.2%)."""
    mor = HodMOR(HodParams.des_y1(H))
    kern = EmgRichnessKernel()
    results = []
    for n_lnm in (96, 192):
        table = SelectionFunctionBuilder(mor, kern, n_lnm=n_lnm, n_z=16).build(
            BINS[:2]
        )
        # proxy for N_ij: trapz of S over lnM at fixed z, weighted by a
        # falling power law standing in for the mass function
        w = np.exp(-1.5 * (table.lnM - table.lnM[0]))
        contracted = np.trapezoid(table.S[:, :, 8] * w[None, :], table.lnM, axis=1)
        results.append(contracted)
    assert np.allclose(results[0], results[1], rtol=2e-3)


def test_lognormal_erfc_closed_form():
    """AnalyticLogNormalKernel + LogNormalMOR reproduces the closed-form
    erfc lognormal bin probability (clens/util/scaling_relation.py path)."""
    params = LogNormalParams.costanzi21(H)
    mor = LogNormalMOR(params)
    builder = SelectionFunctionBuilder(
        mor, AnalyticLogNormalKernel(), n_lnm=96, n_z=16, n_q=128, L=8.0
    )
    lam_min, lam_max = 20.0, 45.0
    M = np.logspace(13.8, 15.2, 25)
    z = 0.3
    got = builder.s_i_pointwise(M, z, lam_min, lam_max)

    mu = mor.ln_lambda_mean(M, z)
    sig = mor.sigma_ln_lambda(M, z)
    x_lo = (np.log(lam_min) - mu) / (np.sqrt(2.0) * sig)
    x_hi = (np.log(lam_max) - mu) / (np.sqrt(2.0) * sig)
    expected = 0.5 * erfc(x_lo) - 0.5 * erfc(x_hi)

    keep = expected > 1e-3
    assert np.allclose(got[keep], expected[keep], rtol=5e-3)


def test_partition_of_unity(hod_builder):
    """Contiguous richness bins spanning (0, inf) sum to ~1 where the MOR
    support is well inside the covered range."""
    edges = [1e-2, 20.0, 30.0, 45.0, 60.0, 1e4]
    M, z = 5e14, 0.3
    total = sum(
        hod_builder.s_i_pointwise(M, z, lo, hi)
        for lo, hi in zip(edges[:-1], edges[1:])
    )
    assert np.isclose(total, 1.0, atol=5e-3)
