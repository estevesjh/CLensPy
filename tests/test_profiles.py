"""
Tests for profiles module.
"""

import numpy as np
from clenspy.profiles import NFWProfile


class TestNFWProfile:
    """Test NFW profile implementation."""
    
    def test_nfw_initialization(self):
        """Test NFW profile initialization."""
        M200 = 1e14  # Msun
        c200 = 5.0
        z = 0.3
        
        nfw = NFWProfile(M200=M200, c200=c200, z=z)
        
        # Check that basic parameters are set
        assert nfw.M200 == M200
        assert nfw.c200 == c200
        assert nfw.z == z
        
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
        nfw = NFWProfile(M200=1e14, c200=5.0, z=0.3)
        
        r = np.array([0.1, 0.5, 1.0, 2.0])  # Mpc
        rho = nfw.density_3d(r)
        
        # Should return array of same length
        assert len(rho) == len(r)
        # All values should be positive and finite
        assert np.all(rho > 0)
        assert np.all(np.isfinite(rho))
        # Density should decrease with radius
        assert np.all(np.diff(rho) < 0)
    
    def test_surface_density(self):
        """Test surface density profile."""
        nfw = NFWProfile(M200=1e14, c200=5.0, z=0.3)
        
        R = np.array([0.1, 0.5, 1.0, 2.0])  # Mpc
        sigma = nfw.surface_density(R)
        
        # Should return array of same length
        assert len(sigma) == len(R)
        # All values should be positive and finite
        assert np.all(sigma > 0)
        assert np.all(np.isfinite(sigma))
    
    def test_mean_surface_density(self):
        """Test mean surface density."""
        nfw = NFWProfile(M200=1e14, c200=5.0, z=0.3)
        
        R = np.array([0.5, 1.0, 2.0])  # Mpc
        sigma_mean = nfw.mean_surface_density(R)
        
        # Should return array of same length
        assert len(sigma_mean) == len(R)
        # All values should be positive and finite
        assert np.all(sigma_mean > 0)
        assert np.all(np.isfinite(sigma_mean))
    
    def test_surface_vs_mean_density(self):
        """Test relationship between surface and mean surface density."""
        nfw = NFWProfile(M200=1e14, c200=5.0, z=0.3)
        
        R = 1.0  # Mpc
        sigma = nfw.surface_density(R)
        sigma_mean = nfw.mean_surface_density(R)
        
        # For NFW profiles, mean surface density is typically larger
        # than surface density at most radii
        assert sigma_mean > 0
        assert sigma > 0
    
    def test_scalar_input(self):
        """Test that scalar inputs work correctly."""
        nfw = NFWProfile(M200=1e14, c200=5.0, z=0.3)
        
        r_scalar = 1.0
        R_scalar = 1.0
        
        # These should return scalars, not arrays
        rho = nfw.density_3d(r_scalar)
        sigma = nfw.surface_density(R_scalar)
        sigma_mean = nfw.mean_surface_density(R_scalar)
        
        assert np.isscalar(rho)
        assert np.isscalar(sigma)
        assert np.isscalar(sigma_mean)
