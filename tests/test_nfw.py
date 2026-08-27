# tests/test_nfw.py

import numpy as np
import pytest

from clenspy.halo.nfw import NfwProfile

# NOTE: the pyccl comparisons that used to live here are in
# validation/validate_nfw_pyccl.py. They compare against another library
# rather than checking that this module runs, which is the tests/ vs
# validation/ split -- and their tolerances were 5e-3 against a measured
# agreement of 1e-10, so as unit tests they asserted almost nothing.

class TestNfwProfile:
    """Test NFW profile implementation."""
    
    def test_nfw_initialization(self):
        """Test NFW profile initialization."""
        M200 = 1e14  # Msun
        c200 = 5.0
        
        nfw = NfwProfile(m200=M200, c200=c200)
        
        # Check that basic parameters are set
        assert nfw.m200 == M200
        assert nfw.c200 == c200
        
        # Check that derived parameters are calculated
        assert hasattr(nfw, 'r200')
        assert hasattr(nfw, 'rs')
        assert hasattr(nfw, 'rho_s')
        
        # Basic sanity checks
        assert nfw.r200 > 0
        assert nfw.rs > 0
        assert nfw.rho_s > 0
        assert nfw.rs < nfw.r200  # Scale radius < virial radius
    
    def test_density_3d(self):
        """Test 3D density profile."""
        nfw = NfwProfile(m200=1e14, c200=5.0)
        
        r = np.array([0.1, 0.5, 1.0, 2.0])  # Mpc
        rho = nfw.density(r)
        
        # Should return array of same length
        assert len(rho) == len(r)
        # All values should be positive and finite
        assert np.all(rho > 0)
        assert np.all(np.isfinite(rho))
        # Density should decrease with radius
        assert np.all(np.diff(rho) < 0)
    
    def test_surface_density(self):
        """Test surface density profile."""
        nfw = NfwProfile(m200=1e14, c200=5.0)
        
        R = np.array([0.1, 0.5, 1.0, 2.0])  # Mpc
        sigma = nfw.sigma(R)
        
        # Should return array of same length
        assert len(sigma) == len(R)
        # All values should be positive and finite
        assert np.all(sigma > 0)
        assert np.all(np.isfinite(sigma))
    
    def test_deltasigma(self):
        """Test mean surface density."""
        nfw = NfwProfile(m200=1e14, c200=5.0)
        
        R = np.array([0.5, 1.0, 2.0])  # Mpc
        deltasigma = nfw.deltasigma(R)

        # Should return array of same length
        assert len(deltasigma) == len(R)
        # All values should be positive and finite
        assert np.all(deltasigma > 0)
        assert np.all(np.isfinite(deltasigma))

    def test_surface_vs_mean_density(self):
        """Test relationship between surface and mean surface density."""
        nfw = NfwProfile(m200=1e14, c200=5.0)
        
        R = 1.0  # Mpc
        sigma = nfw.sigma(R)
        deltasigma = nfw.deltasigma(R)
        sigma_mean = deltasigma + sigma
        
        # For NFW profiles, mean surface density is typically larger
        # than surface density at most radii
        assert sigma_mean > 0
        assert sigma > 0
    
    def test_scalar_input(self):
        """Test that scalar inputs work correctly."""
        nfw = NfwProfile(m200=1e14, c200=5.0)
        
        r_scalar = 1.0
        R_scalar = 1.0
        
        # These should return scalars, not arrays
        rho = nfw.density(r_scalar)
        sigma = nfw.sigma(R_scalar)
        deltasigma = nfw.deltasigma(R_scalar)
        
        assert np.isscalar(rho)
        assert np.isscalar(sigma)
        assert np.isscalar(deltasigma)

# --- small-x stability of the projected kernels ----------------------------
#
# The miscentering integrand samples x = R/r_s -> 0 whenever the azimuthal
# ring passes through the halo centre (R = R_mis), so these kernels have to
# stay accurate far below any physically interesting radius.


@pytest.mark.parametrize("x", [1e-300, 1e-100, 1e-20, 1e-12, 1e-8, 1e-4, 1e-2])
def test_fNfw_small_x(x):
    """f(x) -> ln(2/x) - 1; the old arctanh form returned inf below ~1e-17."""
    f = float(np.ravel(NfwProfile._fNfw(np.array([x])))[0])
    assert np.isfinite(f)
    assert f == pytest.approx(np.log(2.0 / x) - 1.0, rel=2e-3)


@pytest.mark.parametrize("x", [1e-300, 1e-100, 1e-20, 1e-12, 1e-8, 1e-4, 1e-2])
def test_gbarNfw_small_x(x):
    """gbar(x) -> ln(2/x) - 1/2, and stays above f(x) by 1/2."""
    gb = float(np.ravel(NfwProfile._gbarNfw(np.array([x])))[0])
    f = float(np.ravel(NfwProfile._fNfw(np.array([x])))[0])
    assert np.isfinite(gb)
    assert gb == pytest.approx(np.log(2.0 / x) - 0.5, rel=2e-3)
    # DeltaSigma_hat = gbar - f -> 1/2 exactly
    assert gb - f == pytest.approx(0.5, rel=1e-3)


@pytest.mark.parametrize("x", [1e-300, 1e-100, 1e-20, 1e-12, 1e-8, 1e-4, 1e-2])
def test_gNfw_small_x_is_positive_and_tends_to_one(x):
    """g(x) -> 1. The old form went negative below x ~ 1e-9."""
    g = float(np.ravel(NfwProfile._gNfw(np.array([x])))[0])
    assert np.isfinite(g)
    assert g > 0.0, "g(x) must stay positive"
    assert g == pytest.approx(1.0, rel=2e-3)


def test_gbar_consistent_with_f_and_g():
    """gbar == f + g/2 wherever the reconstruction is still trustworthy."""
    x = np.logspace(-3, 3, 60)
    gb = NfwProfile._gbarNfw(x)
    recon = NfwProfile._fNfw(x) + 0.5 * NfwProfile._gNfw(x)
    np.testing.assert_allclose(gb, recon, rtol=1e-8)


def test_mean_sigma_equals_sigma_plus_deltasigma():
    """The public closed form agrees with the sum over the fitted range."""
    nfw = NfwProfile(m200=1e14, c200=4.0)
    R = np.logspace(-2, 1.5, 40)
    np.testing.assert_allclose(
        np.ravel(nfw.mean_sigma(R)),
        np.ravel(nfw.sigma(R)) + np.ravel(nfw.deltasigma(R)),
        rtol=1e-8,
    )
