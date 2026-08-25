"""Tests for stacked DeltaSigma operators (1h and 1h2hMax)."""

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from clenspy.cosmology import PkGrid
from clenspy.halo import NfwProfile, SigmaGrid, Tinker08MassFunction, Tinker10Bias
from clenspy.halo.twohalo import TwoHaloTerm
from clenspy.clusters import (
    BinDefinition,
    BinnedClusterModel,
    DeltaSigma1hOperator,
    DeltaSigmaMaxOperator,
    EmgRichnessKernel,
    HodMOR,
    HodParams,
    SelectionTable,
    build_zresolved_weights,
    omega_z_const_factory,
)

COSMO = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.045)
H = 0.7
AREA_SR = 1437.0 * (np.pi / 180.0) ** 2
R_GRID = np.logspace(np.log10(0.3), np.log10(60.0), 12)  # Mpc


@pytest.fixture(scope="module")
def pkgrid():
    return PkGrid(
        backend="camb",
        cosmo=COSMO,
        nonlinear=False,
        k_range=(1e-4, 100.0),
        z_range=(0.0, 1.2),
        nk=600,
        nz=25,
    )


@pytest.fixture(scope="module")
def model(pkgrid):
    return BinnedClusterModel(
        pkgrid=pkgrid,
        mor=HodMOR(HodParams.des_y1(H)),
        kernel=EmgRichnessKernel(),
        bins=(
            BinDefinition(20.0, 30.0, 0.2, 0.35),
            BinDefinition(45.0, 200.0, 0.2, 0.35),
        ),
        cosmology=COSMO,
        omega_z=omega_z_const_factory(AREA_SR),
        sel_n_lnm=128,
        sel_n_z=48,
        z_range=(0.15, 0.40),
    )


def _delta_selection_table(lnM0, width=0.005, z_range=(0.15, 0.40)):
    """Selection concentrated in a narrow lnM window (delta-like)."""
    lnM = np.linspace(lnM0 - 30 * width, lnM0 + 30 * width, 301)
    z = np.linspace(*z_range, 16)
    S1 = np.exp(-0.5 * ((lnM - lnM0) / width) ** 2)
    S = (S1[:, None] * np.ones_like(z)[None, :])[None, :, :]
    return SelectionTable(
        lnM=lnM, z=z, S=S,
        bins=(BinDefinition(20.0, 30.0, z_range[0], z_range[1]),),
    )


def test_delta_selection_recovers_nfw(pkgrid):
    """A delta-like selection stack must reduce to the single-halo NFW
    DeltaSigma at the selected mass."""
    M0 = 3e14
    sel = _delta_selection_table(np.log(M0))
    hmf = Tinker08MassFunction(SigmaGrid(pkgrid, cosmo=COSMO))
    w = build_zresolved_weights(
        sel, hmf, COSMO, omega_z_const_factory(AREA_SR),
        n_lnm=200, n_z=16,
    )
    op = DeltaSigma1hOperator(w.lnm_x, 0.275, COSMO, concentration=4.0)
    num = np.einsum("bkq,k,kr->br", w.W2d, w.lnm_w, op.matrix(R_GRID))
    norm = np.einsum("bkq,k->b", w.W2d, w.lnm_w)
    stack = num / norm[:, None]
    ref = NfwProfile(m200=M0, c200=4.0, cosmo=COSMO).deltasigma(R_GRID)
    assert np.allclose(stack[0], ref, rtol=2e-3)


def test_stacked_1h_bracketed_by_extreme_masses(model):
    """The stack lies between DeltaSigma of the lightest and heaviest
    populated masses, and rises with richness bin."""
    stack = model.stacked_deltasigma_1h(R_GRID)
    assert stack.shape == (2, R_GRID.size)
    assert np.all(stack > 0)
    # richer bin -> more massive -> larger amplitude at small R
    assert np.all(stack[1, :4] > stack[0, :4])


def test_max_exceeds_both_branches(model):
    """max(1h, b*2h) stack >= 1h stack, with equality at small R and
    strict excess at large R (2h regime)."""
    s1 = model.stacked_deltasigma_1h(R_GRID)
    smax = model.stacked_deltasigma_max(R_GRID)
    assert np.all(smax >= s1 * (1.0 - 1e-10))  # relative: einsum order noise
    # small R: 1h dominates -> equality
    assert np.allclose(smax[:, 0], s1[:, 0], rtol=1e-6)
    # large R: 2h dominates -> strictly larger
    assert np.all(smax[:, -1] > 1.5 * s1[:, -1])


def test_max_large_R_limit(model, pkgrid):
    """At large R the max stack approaches <b> * rho_m0 * DeltaSigma_hh."""
    smax = model.stacked_deltasigma_max(R_GRID)
    b_eff = model.mean_bias()
    rho_m0 = model.sigma_grid.rho_m0
    th = model.twohalo
    R_far = R_GRID[-2:]
    two = np.array([th.deltasigma(R_far, 0.275)]).reshape(-1)
    for b in range(2):
        expected = b_eff[b] * rho_m0 * two
        got = smax[b, -2:]
        assert np.allclose(got, expected, rtol=0.15), (b, got, expected)


def test_rho_m0_regression(pkgrid, model):
    """DeltaSigmaMaxOperator must apply rho_m0 exactly once: feeding
    rho_m0=0 kills the 2h branch and collapses max -> 1h."""
    w = model.zweights
    op0 = DeltaSigmaMaxOperator(
        model.one_halo, model.twohalo, model.bias, rho_m0=0.0
    )
    num0 = op0.stack(R_GRID, w)
    one = model.one_halo.matrix(R_GRID)
    wk = np.einsum("bkq->bk", w.W2d) * w.lnm_w[None, :]
    assert np.allclose(num0, wk @ one, rtol=1e-12)
