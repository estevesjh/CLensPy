"""Angular <-> comoving distance conversions and the comoving volume element.

`comoving_to_theta` and `theta_to_comoving` are exact algebraic inverses of
each other at fixed z and cosmology (theta = D_c / D_A, D_c = theta * D_A),
so the round trip is tested as an identity across every supported unit, for
both scalar and array inputs. `comoving_volume_element` is a thin wrapper
around `astropy`'s `differential_comoving_volume`, so it is cross-checked
directly against that call rather than against an independent formula.
"""

import numpy as np
import pytest

from clenspy.cosmology.distances import (
    comoving_to_theta,
    comoving_volume_element,
    theta_to_comoving,
)
from clenspy.cosmology.fiducial import fiducial_cosmology

UNITS = ["arcsec", "arcmin", "deg", "rad"]


class TestRoundTrip:
    @pytest.mark.parametrize("unit", UNITS)
    def test_scalar_D_c(self, unit):
        cosmo = fiducial_cosmology()
        z = 0.35
        D_c = 1.5
        theta = comoving_to_theta(D_c, z, cosmo, unit=unit)
        D_c_back = theta_to_comoving(theta, z, cosmo, unit=unit)
        assert np.isclose(D_c_back, D_c)

    @pytest.mark.parametrize("unit", UNITS)
    def test_array_D_c(self, unit):
        cosmo = fiducial_cosmology()
        z = 0.35
        D_c = np.array([0.1, 1.0, 10.0])
        theta = comoving_to_theta(D_c, z, cosmo, unit=unit)
        D_c_back = theta_to_comoving(theta, z, cosmo, unit=unit)
        np.testing.assert_allclose(D_c_back, D_c)


class TestUnitValidation:
    def test_comoving_to_theta_invalid_unit(self):
        cosmo = fiducial_cosmology()
        with pytest.raises(ValueError):
            comoving_to_theta(1.5, 0.35, cosmo, unit="parsec")

    def test_theta_to_comoving_invalid_unit(self):
        cosmo = fiducial_cosmology()
        with pytest.raises(ValueError):
            theta_to_comoving(1.5, 0.35, cosmo, unit="parsec")


class TestComovingVolumeElement:
    def test_scalar_returns_float(self):
        result = comoving_volume_element(0.5)
        assert isinstance(result, float)
        assert not isinstance(result, np.ndarray)

    def test_array_returns_ndarray_same_shape(self):
        z = np.array([0.1, 0.5, 1.0])
        result = comoving_volume_element(z)
        assert isinstance(result, np.ndarray)
        assert result.shape == z.shape

    def test_matches_astropy_directly(self):
        cosmo = fiducial_cosmology()
        z = np.array([0.1, 0.5, 1.0])
        expected = cosmo.differential_comoving_volume(
            np.atleast_1d(z)
        ).to_value("Mpc3/sr")
        result = comoving_volume_element(z, cosmo)
        np.testing.assert_array_equal(result, expected)

    def test_matches_astropy_directly_scalar(self):
        cosmo = fiducial_cosmology()
        z = 0.5
        expected = cosmo.differential_comoving_volume(
            np.atleast_1d(z)
        ).to_value("Mpc3/sr")[0]
        result = comoving_volume_element(z, cosmo)
        assert result == expected

    def test_default_cosmology_matches_fiducial(self):
        result_default = comoving_volume_element(0.5)
        result_explicit = comoving_volume_element(0.5, fiducial_cosmology())
        assert result_default == result_explicit
