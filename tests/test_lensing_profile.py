"""`LensingProfile` builds nothing until asked, and validates immediately.

These are structural tests, not physics tests: they exist so that a future
edit cannot quietly put the Boltzmann solver back into the constructor. The
physics is covered by ``test_nfw.py`` and ``test_twohalo.py``.
"""

import sys

import numpy as np
import pytest

from clenspy.halo import NfwProfile
from clenspy.lensing import LensingProfile

CHEAP = dict(z_cluster=0.3, m200=1e14)


def test_construction_builds_nothing():
    """No cached_property is populated by __init__."""
    lp = LensingProfile(**CHEAP)
    lazy = ("halo_profile", "Pkvec", "two_halo_profile", "bias_model",
            "bias", "sigma_crit")
    built = [name for name in lazy if name in lp.__dict__]
    assert built == [], f"constructor eagerly built {built}"


def test_construction_does_not_import_a_boltzmann_solver():
    """The expensive dependency is not even imported until it is needed.

    Skipped if something earlier in the session already imported camb --
    the assertion is about what construction does, and it cannot un-import.
    """
    if "camb" in sys.modules:
        pytest.skip("camb already imported by another test")
    LensingProfile(**CHEAP)
    assert "camb" not in sys.modules


def test_validation_is_still_eager():
    """A bad input raises at construction, not pages later."""
    with pytest.raises(ValueError, match="redshift"):
        LensingProfile(z_cluster=-1.0, m200=1e14)
    with pytest.raises(ValueError, match="Mass"):
        LensingProfile(z_cluster=0.3, m200=-1e14)
    with pytest.raises(ValueError, match="Concentration"):
        LensingProfile(**CHEAP, concentration=0.0)
    with pytest.raises(ValueError, match="not supported"):
        LensingProfile(**CHEAP, model="Einasto")
    with pytest.raises(ValueError, match="Source redshift"):
        LensingProfile(z_cluster=1.5, m200=1e14, z_source=1.0)


def test_one_halo_only_needs_no_power_spectrum():
    """include_2halo=False must never touch P(k)."""
    lp = LensingProfile(**CHEAP, include_2halo=False)
    assert np.isfinite(lp.deltasigma(1.0))
    assert "Pkvec" not in lp.__dict__


def test_supplied_collaborators_are_stored_verbatim():
    """halo_profile= and bias= are used as given, not rebuilt."""
    nfw = NfwProfile(m200=2e14, c200=6.0)
    lp = LensingProfile(**CHEAP, include_2halo=False, halo_profile=nfw, bias=2.5)
    assert lp.halo_profile is nfw
    assert lp.bias == 2.5
    # and the injected profile is what the observable actually uses
    np.testing.assert_allclose(lp.deltasigma(1.0), nfw.deltasigma(1.0))


def test_supplied_bias_skips_the_power_spectrum():
    """A fixed b(M) removes the only other consumer of P(k)."""
    lp = LensingProfile(**CHEAP, include_2halo=False, bias=2.0)
    assert lp.bias == 2.0
    assert "Pkvec" not in lp.__dict__
    assert "bias_model" not in lp.__dict__


def test_k_grid_is_an_argument_not_a_hidden_default():
    lp = LensingProfile(**CHEAP, k_grid=np.logspace(-2, 0, 20))
    assert lp.kvec.shape == (20,)


def test_sigma_crit_matches_the_kernel():
    """`sigma_crit` is the kernels-layer function, not a second copy."""
    from clenspy.kernels import sigma_critical

    lp = LensingProfile(**CHEAP, include_2halo=False, z_source=1.2)
    assert lp.sigma_crit == sigma_critical(0.3, 1.2, lp.cosmo)


def test_info_reports_without_the_two_halo_term():
    """`info` may touch the cheap properties but must not need P(k)."""
    lp = LensingProfile(**CHEAP, include_2halo=False)
    info = lp.info
    assert info.model == "nfw"
    assert info.r200 > 0 and info.rs > 0
    assert "Pkvec" not in lp.__dict__
