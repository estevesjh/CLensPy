import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from clenspy.halo.bias import BiasModel

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
