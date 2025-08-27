import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from clenspy.halo.bias import biasModel

def test_bias_model_basic():
    # Setup: simple power spectrum and cosmology
    k = np.logspace(-3, 1, 50)  # h/Mpc
    P = np.ones_like(k) * 1e4   # Flat power spectrum for test
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    model = biasModel(k, P, cosmo)

    # Test biasAtM returns finite, positive values for typical mass
    M = 1e14  # Msun/h
    bias = model.biasAtM(M)
    assert np.all(np.isfinite(bias))
    assert np.all(bias > 0)

    # Test nuAtM and sigmaAtM are consistent
    nu = model.nuAtM(M)
    sigma = model.sigmaAtM(M)
    assert np.isclose(nu, 1.686 / sigma)
    assert sigma > 0

    # Test biasAtNu returns finite, positive values
    bias_nu = model.biasAtNu(nu)
    assert np.all(np.isfinite(bias_nu))
    assert np.all(bias_nu > 0)

    # Test getTinkerParams returns 6 parameters
    params = model.getTinkerParams()
    assert len(params) == 6

    # Test _biasAtNu (private, but check for coverage)
    A, a, B, b, C, c = params
    bias_direct = model._biasAtNu(nu, A, a, B, b, C, c)
    assert np.isfinite(bias_direct)
    assert bias_direct > 0
