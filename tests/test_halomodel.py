"""Tests for clenspy.clusters.halomodel (per-bin halo-model spectra)."""

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from clenspy.cosmology import PkGrid
from clenspy.halo import ConstantBias, SigmaGrid, Tinker08MassFunction
from clenspy.clusters import (
    BinDefinition,
    BinHaloModelSpectra,
    EmgRichnessKernel,
    HodMOR,
    HodParams,
    SelectionFunctionBuilder,
    build_zresolved_weights,
    omega_z_const_factory,
)

COSMO = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.045)
H = 0.7
AREA_SR = 0.44


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
             BinDefinition(45.0, 200.0, 0.2, 0.35)))
    hmf = Tinker08MassFunction(SigmaGrid(pk, cosmo=COSMO))
    w = build_zresolved_weights(
        sel, hmf, COSMO, omega_z_const_factory(AREA_SR)
    )
    return pk, w


def test_pk_hh_constant_bias_identity(setup):
    """With ConstantBias(b0), pk_hh must equal exactly b0^2 P_lin."""
    pk, w = setup
    b0 = 2.75
    spec = BinHaloModelSpectra(w, ConstantBias(b0), pk, COSMO, cross_model="additive")
    k = np.logspace(-3, 1, 20)
    got = spec.pk_hh(0)(k, 0.3)
    assert np.allclose(got, b0**2 * pk(k, 0.3), rtol=1e-10)


def test_pk_hm_low_k_limit(setup):
    """k -> 0: 1h term -> <M>_S / rho_m (u -> 1), 2h -> b P_lin."""
    pk, w = setup
    from clenspy.halo import Tinker10Bias
    sg = SigmaGrid(pk, cosmo=COSMO)
    bias = Tinker10Bias(sg)
    spec = BinHaloModelSpectra(w, bias, pk, COSMO, cross_model="additive")

    k_lo = 1e-3
    for b in (0, 1):
        pk_hm = spec.pk_hm(b)(k_lo, 0.3)
        b_eff = np.interp(0.3, w.z_x, spec.b_eff[b])
        two_h = b_eff * pk(k_lo, 0.3)
        one_h = pk_hm - two_h
        # 1h plateau = selection-weighted <M>/rho_m
        wkq = w.W2d[b] * w.lnm_w[:, None]
        m_mean = np.sum(wkq * np.exp(w.lnm_x)[:, None]) / np.sum(wkq)
        expected = m_mean / spec.rho_m0
        assert np.isclose(one_h, expected, rtol=0.05), (b, one_h, expected)
        # richer bin -> heavier halos -> bigger plateau
    p0 = spec.pk_hm(0)(k_lo, 0.3) - np.interp(0.3, w.z_x, spec.b_eff[0]) * pk(k_lo, 0.3)
    p1 = spec.pk_hm(1)(k_lo, 0.3) - np.interp(0.3, w.z_x, spec.b_eff[1]) * pk(k_lo, 0.3)
    assert p1 > p0


def test_pk_hm_high_k_one_halo_dominates(setup):
    """At k ~ 20/Mpc the 1h term dominates over the linear 2h term."""
    pk, w = setup
    from clenspy.halo import Tinker10Bias
    bias = Tinker10Bias(SigmaGrid(pk, cosmo=COSMO))
    spec = BinHaloModelSpectra(w, bias, pk, COSMO, cross_model="additive")
    k = 20.0
    pk_hm = spec.pk_hm(1)(k, 0.3)
    b_eff = np.interp(0.3, w.z_x, spec.b_eff[1])
    assert pk_hm > 3.0 * b_eff * pk(k, 0.3)


def test_b_eff_ordering(setup):
    pk, w = setup
    from clenspy.halo import Tinker10Bias
    bias = Tinker10Bias(SigmaGrid(pk, cosmo=COSMO))
    spec = BinHaloModelSpectra(w, bias, pk, COSMO, cross_model="additive")
    assert np.all(spec.b_eff[1] > spec.b_eff[0])  # richer -> more biased


def test_max_cross_model(setup):
    """Hayashi-White max composition: low-k -> 2h (b P_lin), high-k ->
    1h-dominated, always >= the pure 2h spectrum."""
    pk, w = setup
    from clenspy.halo import Tinker10Bias
    bias = Tinker10Bias(SigmaGrid(pk, cosmo=COSMO))
    spec = BinHaloModelSpectra(w, bias, pk, COSMO, cross_model="max")
    b_eff = np.interp(0.3, w.z_x, spec.b_eff[1])
    # k_min ~ 0.02: scales safely inside the xi-space table (r <= 500 Mpc)
    k = np.logspace(-1.7, 1.3, 30)
    p_max = spec.pk_hm(1)(k, 0.3)
    p_2h = b_eff * pk(k, 0.3)
    # low k: within ~20% of the pure 2h spectrum
    assert np.allclose(p_max[:5], p_2h[:5], rtol=0.2)
    # high k: strongly 1h-enhanced over linear 2h
    assert p_max[-1] > 3.0 * p_2h[-1]
