"""Tests for the intrinsic (halo-to-halo) profile-variance term."""

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from clenspy.cosmology import PkGrid
from clenspy.halo import SigmaGrid, Tinker08MassFunction, Tinker10Bias
from clenspy.halo.twohalo import TwoHaloTerm
from clenspy.clusters import (
    BinDefinition,
    EmgRichnessKernel,
    HodMOR,
    HodParams,
    IntrinsicProfileVariance,
    SelectionFunctionBuilder,
    build_mass_weights,
    omega_z_const_factory,
)

COSMO = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.045)
H = 0.7
AREA_SR = 0.44
R_GRID = np.logspace(np.log10(0.3), np.log10(30.0), 12)


@pytest.fixture(scope="module")
def setup():
    pk = PkGrid(
        backend="camb", cosmo=COSMO, nonlinear=False,
        k_range=(1e-4, 100.0), z_range=(0.0, 1.2), nk=600, nz=25,
    )
    sel = SelectionFunctionBuilder(
        HodMOR(HodParams.des_y1(H)), EmgRichnessKernel(),
        n_lnm=96, n_z=32, z_range=(0.15, 0.40),
    ).build((BinDefinition(20.0, 30.0, 0.2, 0.35),
             BinDefinition(60.0, 1000.0, 0.2, 0.35)))
    sg = SigmaGrid(pk, cosmo=COSMO)
    hmf = Tinker08MassFunction(sg)
    w = build_mass_weights(sel, hmf, COSMO, omega_z_const_factory(AREA_SR))
    th = TwoHaloTerm(pk.k, pk.pk, zvec=pk.z)
    ipv = IntrinsicProfileVariance(
        w, th, Tinker10Bias(sg), sg.rho_m0, z_eff=0.275, cosmology=COSMO
    )
    return w, ipv


def test_cov_symmetric_psd(setup):
    w, ipv = setup
    C = ipv.cov(R_GRID, 0)
    assert np.allclose(C, C.T, rtol=1e-12)
    eig = np.linalg.eigvalsh(C)
    assert np.all(eig > -1e-8 * eig.max())
    assert np.all(np.diag(C) > 0)


def test_scales_inversely_with_counts(setup):
    """C_intr = Cov_pop / N_cl: scaling norm by 10 scales cov by 1/10."""
    import copy

    w, ipv = setup
    C = ipv.cov(R_GRID, 0)
    ipv10 = copy.copy(ipv)
    ipv10.w = copy.deepcopy(w)
    ipv10.w.norm[:] = w.norm * 10.0
    C10 = ipv10.cov(R_GRID, 0)
    assert np.allclose(C10, C / 10.0, rtol=1e-12)


def test_broad_correlations(setup):
    """The intrinsic term is broadly correlated across radii (unlike the
    Gaussian terms): first-offdiagonal correlation > 0.7, d=4 > 0.3."""
    _, ipv = setup
    C = ipv.cov(R_GRID, 1)
    d = np.sqrt(np.diag(C))
    Rc = C / np.outer(d, d)
    off1 = np.mean([Rc[i, i + 1] for i in range(len(d) - 1)])
    off4 = np.mean([Rc[i, i + 4] for i in range(len(d) - 4)])
    assert off1 > 0.7, off1
    assert off4 > 0.3, off4


def test_richness_scaling(setup):
    """Richer bin: larger per-cluster profile variance AND fewer clusters
    -> much larger intrinsic term."""
    w, ipv = setup
    C_lo = ipv.cov(R_GRID, 0)
    C_hi = ipv.cov(R_GRID, 1)
    assert np.all(np.diag(C_hi) > np.diag(C_lo))


def test_small_scatter_limit(setup):
    """sigma_lnc -> 0 with a delta-like mass population -> cov -> 0."""
    w, ipv = setup
    import copy
    w0 = copy.deepcopy(w)
    # concentrate the mass weight on a single node
    w0.W[0, :] = 0.0
    k0 = np.argmax(w.W[0])
    w0.W[0, k0] = 1.0
    ipv0 = IntrinsicProfileVariance(
        w0, ipv.twohalo, _BiasShim(ipv._b_of_M, w.lnm_x), ipv.rho_m0,
        ipv.z_eff, cosmology=COSMO, sigma_lnc=1e-6,
    )
    C0 = ipv0.cov(R_GRID, 0)
    C1 = ipv.cov(R_GRID, 0)
    assert np.max(np.diag(C0)) < 1e-6 * np.max(np.diag(C1))


class _BiasShim:
    def __init__(self, b_of_M, lnm_x):
        self._b = b_of_M
        self._lnm = lnm_x

    def at_lnM(self, lnM, z):
        return np.interp(np.asarray(lnM, dtype=float), self._lnm, self._b)
