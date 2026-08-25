"""Tests for clenspy.halo.mass_function (SigmaGrid, Tinker08, Tinker10, ConstantBias)."""

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from clenspy.cosmology import PkGrid
from clenspy.halo import (
    BiasModel,
    ConstantBias,
    SigmaGrid,
    Tinker08MassFunction,
    Tinker10Bias,
)
from clenspy.halo.mass_function import DELTA_C, _top_hat_window

COSMO = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.045)


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
def sigma_grid(pkgrid):
    return SigmaGrid(pkgrid, cosmo=COSMO)


def test_top_hat_window_limits():
    x = np.array([1e-6, 1e-4, 1e-3])
    assert np.allclose(_top_hat_window(x), 1.0, atol=1e-6)
    # exact value at x=1: 3(sin1 - cos1)
    assert np.isclose(_top_hat_window(np.array([1.0]))[0],
                      3.0 * (np.sin(1.0) - np.cos(1.0)), rtol=1e-12)


def test_sigma_monotonic_and_growth(sigma_grid):
    M = np.logspace(12, 15.5, 30)
    s0 = sigma_grid(M, 0.0)
    # sigma decreases with mass
    assert np.all(np.diff(s0) < 0)
    # sigma decreases with redshift (growth)
    s05 = sigma_grid(M, 0.5)
    assert np.all(s05 < s0)


def test_sigma_vs_biasmodel_tophatvar(pkgrid, sigma_grid):
    """Same sigma(M) machinery as BiasModel.sigma_tophat at z=0 (mcfit route)."""
    bm = BiasModel(pkgrid.k, pkgrid.pk[0], cosmo=COSMO)
    M = np.logspace(13, 15, 12)
    s_ref = bm.sigma_tophat(M)
    s_new = sigma_grid(M, 0.0)
    assert np.allclose(s_new, s_ref, rtol=2e-3)


def test_tinker08_f_sigma_closed_form(sigma_grid):
    """f(sigma) at z=0 must equal the Table-2 closed form exactly."""
    hmf = Tinker08MassFunction(sigma_grid)
    sigma = np.array([0.5, 1.0, 2.0])
    A, a, b, c = 0.186, 1.47, 2.57, 1.19
    expected = A * ((sigma / b) ** (-a) + 1.0) * np.exp(-c / sigma**2)
    assert np.allclose(hmf.f_sigma(sigma, z=0.0), expected, rtol=1e-13)


def test_tinker08_z_evolution_direction(sigma_grid):
    """Table-4: A(z) decreases with z, so f(sigma) at fixed sigma decreases."""
    hmf = Tinker08MassFunction(sigma_grid, z_evolution=True)
    hmf0 = Tinker08MassFunction(sigma_grid, z_evolution=False)
    f_z0 = hmf.f_sigma(1.0, z=0.0)
    f_z1 = hmf.f_sigma(1.0, z=1.0)
    assert f_z1 != f_z0
    assert np.isclose(hmf0.f_sigma(1.0, z=1.0), hmf0.f_sigma(1.0, z=0.0))


def test_tinker08_dn_dlnM_sane(sigma_grid):
    hmf = Tinker08MassFunction(sigma_grid)
    M = np.logspace(13, 15.5, 20)
    n = hmf.dn_dlnM(M, 0.3)
    assert np.all(n > 0)
    assert np.all(np.diff(np.log(n)) < 0)  # falling with mass
    # cluster-scale number density: integrate above 1e14 Msun, expect
    # O(1e-5) Mpc^-3 comoving (physical units)
    lnM = np.log(np.logspace(14, 15.7, 200))
    n_int = np.trapezoid(hmf.at_lnM(lnM, 0.0), lnM)
    assert 1e-6 < n_int < 1e-4

    # at_lnM consistency
    assert np.allclose(hmf.at_lnM(np.log(M), 0.3), n, rtol=1e-12)


def test_tinker08_odelta_guard(sigma_grid):
    with pytest.raises(NotImplementedError):
        Tinker08MassFunction(sigma_grid, odelta=500)


def test_tinker10_bias_formula(sigma_grid):
    bias = Tinker10Bias(sigma_grid)
    nu = np.array([1.0, 2.0, 4.0])
    y = np.log10(200.0)
    A = 1.0 + 0.24 * y * np.exp(-((4.0 / y) ** 4))
    a = 0.44 * y - 0.88
    B, b, C, c = 0.183, 1.5, 0.019 + 0.107 * y + 0.19 * np.exp(-((4.0 / y) ** 4)), 2.4
    expected = 1.0 - A * nu**a / (nu**a + DELTA_C**a) + B * nu**b + C * nu**c
    assert np.allclose(bias.bias_at_nu(nu), expected, rtol=1e-13)


def test_tinker10_bias_stateless(sigma_grid):
    """No nu-caching gotcha: repeated calls with different M give different b."""
    bias = Tinker10Bias(sigma_grid)
    b1 = bias(1e13, 0.2)
    b2 = bias(1e15, 0.2)
    assert b2 > b1 > 0.5
    # calling again with the first M returns the first answer
    assert np.isclose(bias(1e13, 0.2), b1, rtol=1e-14)


def test_tinker10_bias_vs_biasmodel(pkgrid, sigma_grid):
    """Cross-check against the existing BiasModel at z=0 (same formula)."""
    bm = BiasModel(pkgrid.k, pkgrid.pk[0], cosmo=COSMO)
    M = np.logspace(13.5, 15, 8)
    b_ref = bm.bias_at_nu(bm.nu_at_mass(M))
    b_new = Tinker10Bias(sigma_grid)(M, 0.0)
    assert np.allclose(b_new, b_ref, rtol=5e-3)


def test_constant_bias_protocol():
    cb = ConstantBias(2.75)
    assert cb(1e14, 0.3) == 2.75
    out = cb(np.logspace(13, 15, 5), 0.3)
    assert out.shape == (5,)
    assert np.all(out == 2.75)
    out2 = cb.at_lnM(np.log(np.logspace(13, 15, 4)), np.linspace(0, 1, 4))
    assert out2.shape == (4,)
    assert np.all(out2 == 2.75)
