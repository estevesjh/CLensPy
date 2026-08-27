"""Structural conformance: do the sibling classes present the same surface?

This is what the deleted ``check_structure.py`` was reaching for -- but
checked against the objects rather than against a hardcoded file listing, so
it runs anywhere and fails for a reason that matters.
"""

import numpy as np
import pytest

from clenspy.cosmology import fiducial_cosmology
from clenspy.halo import EinastoProfile, NfwProfile
from clenspy.protocols import Cosmology, Profile

PROFILE_METHODS = ("density", "sigma", "mean_sigma", "deltasigma", "fourier")


def profiles():
    """One instance of each class claiming to be a Profile."""
    return [
        NfwProfile(m200=1e14, c200=4.0),
        EinastoProfile(alpha=0.2, rho_0=1e15, r_s=0.3),
    ]


@pytest.mark.parametrize("profile", profiles(), ids=lambda p: type(p).__name__)
def test_profiles_conform(profile):
    assert isinstance(profile, Profile)


@pytest.mark.parametrize("profile", profiles(), ids=lambda p: type(p).__name__)
@pytest.mark.parametrize("method", PROFILE_METHODS)
def test_profile_methods_are_vectorised(profile, method):
    """Each method broadcasts over its radius/wavenumber argument."""
    arg = np.array([0.1, 0.5, 2.0])
    out = np.ravel(getattr(profile, method)(arg))
    assert out.shape == arg.shape
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("profile", profiles(), ids=lambda p: type(p).__name__)
def test_profile_methods_accept_scalars(profile):
    """A scalar in gives a scalar-like out, for every method."""
    for method in PROFILE_METHODS:
        out = getattr(profile, method)(1.0)
        assert np.ndim(out) == 0 or np.size(out) == 1


@pytest.mark.parametrize("profile", profiles(), ids=lambda p: type(p).__name__)
def test_sigmabar_identity(profile):
    """The three projections are consistent: Sigmabar = Sigma + DeltaSigma.

    Each is evaluated from its own closed form, so agreement is a real check
    on all three rather than a tautology.
    """
    R = np.logspace(-2, 1, 25)
    np.testing.assert_allclose(
        np.ravel(profile.mean_sigma(R)),
        np.ravel(profile.sigma(R)) + np.ravel(profile.deltasigma(R)),
        rtol=1e-8,
    )


@pytest.mark.parametrize("profile", profiles(), ids=lambda p: type(p).__name__)
def test_profiles_are_positive_and_decreasing(profile):
    """Sigma and the density fall monotonically; both stay positive."""
    r = np.logspace(-2, 1, 30)
    for method in ("density", "sigma", "mean_sigma"):
        v = np.ravel(getattr(profile, method)(r))
        assert np.all(v > 0), f"{method} must be positive"
        assert np.all(np.diff(v) < 0), f"{method} must decrease outward"


def test_the_two_profiles_expose_the_same_surface():
    """No sibling quietly grows or loses a Profile method."""
    nfw, ein = profiles()
    for method in PROFILE_METHODS:
        assert hasattr(nfw, method), f"NfwProfile lacks {method}"
        assert hasattr(ein, method), f"EinastoProfile lacks {method}"


def test_fiducial_cosmology_conforms():
    assert isinstance(fiducial_cosmology(), Cosmology)


def test_protocols_reject_a_class_that_forgot_a_method():
    """The check has teeth: dropping one method breaks conformance."""

    class AlmostAProfile:
        def density(self, r):
            return r

        def sigma(self, R):
            return R

        def mean_sigma(self, R):
            return R

        def deltasigma(self, R):
            return R
        # no fourier

    assert not isinstance(AlmostAProfile(), Profile)


def test_protocols_are_not_imported_by_the_science_modules():
    """protocols.py is for checking and documentation, not for runtime.

    Inheriting from a Protocol, or importing it in a science module, is the
    thing these contracts exist to avoid -- conformance is structural.
    """
    import clenspy.halo.einasto as einasto
    import clenspy.halo.nfw as nfw

    for module in (nfw, einasto):
        src = open(module.__file__).read()
        assert "from ..protocols" not in src
        assert "from clenspy.protocols" not in src
    # and conformance is by shape, not by inheritance
    assert Profile not in NfwProfile.__mro__
    assert Profile not in EinastoProfile.__mro__
