import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from clenspy.cosmology.bias import BiasModel
from clenspy.cosmology.growth import growth_factor


def test_bias_model_basic():
    # Setup: simple power spectrum and cosmology
    k = np.logspace(-3, 1, 50)  # h/Mpc
    P = np.ones_like(k) * 1e4   # Flat power spectrum for test
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    model = BiasModel(k, P, cosmo)

    # Test bias returns finite, positive values for typical mass
    M = 1e14  # Msun/h
    bias = model.bias(M)
    assert np.all(np.isfinite(bias))
    assert np.all(bias > 0)

    # Test nu_at_mass and sigma_tophat are consistent
    nu = model.nu_at_mass(M)
    sigma = model.sigma_tophat(M)
    assert np.isclose(nu, 1.686 / sigma)
    assert sigma > 0

    # Test bias_at_nu returns finite, positive values
    bias_nu = model.bias_at_nu(nu)
    assert np.all(np.isfinite(bias_nu))
    assert np.all(bias_nu > 0)

    # Test get_tinker_params returns 6 parameters
    params = model.get_tinker_params()
    assert len(params) == 6

    # Test _bias_at_nu (private, but check for coverage)
    A, a, B, b, C, c = params
    bias_direct = model._bias_at_nu(nu, A, a, B, b, C, c)
    assert np.isfinite(bias_direct)
    assert bias_direct > 0


def test_sigma_tophat_applies_the_growth_factor():
    """sigma(M,z) = D(z) sigma(M,0): z > 0 must scale sigma_tophat(M, z=0)
    by exactly D(z), the same relation TinkerMassFunction's dndlnm_grid
    applies to sigma^2 (there, D(z)^2, since that is sigma^2 not sigma).
    """
    k = np.logspace(-3, 1, 50)
    P = np.ones_like(k) * 1e4
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    model = BiasModel(k, P, cosmo)

    z = 0.6
    d_z = growth_factor(z, cosmo)
    sigma_z0 = model.sigma_tophat(1e14, z=0.0)
    sigma_z = model.sigma_tophat(1e14, z=z)
    assert sigma_z == pytest.approx(sigma_z0 * d_z, rel=1e-12)

    # and bias must rise with z at fixed mass: a smaller, less-grown sigma
    # makes a fixed mass a rarer, more strongly biased peak
    assert model.bias(1e14, z=z) > model.bias(1e14, z=0.0)


def test_bias_model_builds_its_own_pk_from_cosmo():
    """k/P omitted: sigma_grid must build a PkGrid from cosmo lazily,
    rather than requiring the caller to supply a spectrum by hand.
    """
    from clenspy.cosmology import fiducial_cosmology

    model = BiasModel(cosmo=fiducial_cosmology())
    b = model.bias(1e14)
    assert np.isfinite(b) and b > 0
