"""Tests for clenspy.clusters.weights and the counts side of observables."""

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from clenspy.cosmology import PkGrid
from clenspy.cosmology.utils import comoving_volume_element
from clenspy.halo import SigmaGrid, Tinker08MassFunction
from clenspy.clusters import (
    BinDefinition,
    BinnedClusterModel,
    EmgRichnessKernel,
    HodMOR,
    HodParams,
    SelectionFunctionBuilder,
    build_mass_weights,
    build_zresolved_weights,
    omega_z_const_factory,
)

COSMO = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.045)
H = 0.7
AREA_SR = 1437.0 * (np.pi / 180.0) ** 2

BINS = (
    BinDefinition(20.0, 30.0, 0.2, 0.35),
    BinDefinition(30.0, 45.0, 0.2, 0.35),
    BinDefinition(45.0, 200.0, 0.2, 0.35),
)


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
def hmf(pkgrid):
    return Tinker08MassFunction(SigmaGrid(pkgrid, cosmo=COSMO))


@pytest.fixture(scope="module")
def sel_table():
    return SelectionFunctionBuilder(
        HodMOR(HodParams.des_y1(H)),
        EmgRichnessKernel(),
        n_lnm=128,
        n_z=48,
        z_range=(0.15, 0.40),
    ).build(BINS)


@pytest.fixture(scope="module")
def weights(sel_table, hmf):
    return build_mass_weights(
        sel_table, hmf, COSMO, omega_z_const_factory(AREA_SR)
    )


@pytest.fixture(scope="module")
def zweights(sel_table, hmf):
    return build_zresolved_weights(
        sel_table, hmf, COSMO, omega_z_const_factory(AREA_SR)
    )


def test_w2d_contraction_invariant(weights, zweights):
    """W2d.sum(axis=2) must equal the z-contracted W."""
    assert np.allclose(zweights.W2d.sum(axis=2), weights.W, rtol=1e-12)


def test_counts_vs_brute_force(sel_table, hmf, weights):
    """GL-engine N_ij vs a dense trapz triple integral (<0.3%)."""
    lnm = np.linspace(sel_table.lnM[0], sel_table.lnM[-1], 800)
    z = np.linspace(sel_table.z[0], sel_table.z[-1], 400)
    dv = comoving_volume_element(z, COSMO) * AREA_SR  # (nz,)
    n_block = hmf.at_lnM(lnm[:, None], z[None, :])  # (nlnm, nz)
    for b in range(sel_table.n_bins):
        S = sel_table(b, lnm[:, None], z[None, :])
        inner = np.trapezoid(n_block * S, lnm, axis=0)  # (nz,)
        N_ref = np.trapezoid(dv * inner, z)
        assert np.isclose(weights.norm[b], N_ref, rtol=3e-3), b


def test_counts_magnitude_and_ordering(weights):
    """DES-Y1-like configuration: hundreds-to-thousands of clusters,
    falling with richness."""
    N = weights.norm
    assert np.all(N > 0)
    assert N[0] > N[1]  # 20-30 bin richer than 30-45
    assert 10 < N.sum() < 1e5


def test_mean_lnM_ordering_and_value(weights):
    """Mean mass increases with richness; sits in the cluster regime."""
    lnm_eff = weights.lnm_eff
    assert np.all(np.diff(lnm_eff) > 0)
    M_eff = np.exp(lnm_eff)
    assert np.all((M_eff > 5e13) & (M_eff < 1e15))
    assert np.all(weights.mu2 > 0)


def test_contract_expectation_unit(weights):
    """N_i[1]/N_i = 1 exactly."""
    ones = np.ones_like(weights.lnm_x)
    assert np.allclose(weights.expectation(ones), 1.0, rtol=1e-12)
    # contract with lnM reproduces lnm_eff
    assert np.allclose(
        weights.expectation(weights.lnm_x), weights.lnm_eff, rtol=1e-12
    )


def test_resonance_guard(sel_table, hmf):
    with pytest.raises(ValueError):
        build_mass_weights(
            sel_table, hmf, COSMO, omega_z_const_factory(AREA_SR), n_lnm=64
        )


def test_binned_cluster_model_counts(pkgrid):
    """Facade produces the same counts as the manual pipeline."""
    model = BinnedClusterModel(
        pkgrid=pkgrid,
        mor=HodMOR(HodParams.des_y1(H)),
        kernel=EmgRichnessKernel(),
        bins=BINS,
        cosmology=COSMO,
        omega_z=omega_z_const_factory(AREA_SR),
        sel_n_lnm=128,
        sel_n_z=48,
        z_range=(0.15, 0.40),
    )
    N = model.counts()
    assert N.shape == (3,)
    assert np.all(N > 0)
    b_eff = model.mean_bias()
    assert np.all((b_eff > 1.5) & (b_eff < 8.0))
    assert np.all(np.diff(b_eff) > 0)  # bias rises with richness
    lnm = model.mean_lnM()
    assert np.all(np.diff(lnm) > 0)


def test_pkgrid_envelope_guard(pkgrid):
    with pytest.raises(ValueError, match="does not cover"):
        BinnedClusterModel(
            pkgrid=pkgrid,
            mor=HodMOR(HodParams.des_y1(H)),
            kernel=EmgRichnessKernel(),
            bins=(BinDefinition(20.0, 30.0, 1.5, 2.0),),
            cosmology=COSMO,
            omega_z=omega_z_const_factory(AREA_SR),
        )
