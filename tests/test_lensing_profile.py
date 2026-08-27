"""`LensingProfile` builds nothing until asked, and validates immediately.

These are structural tests, not physics tests: they exist so that a future
edit cannot quietly put the Boltzmann solver back into the constructor. The
physics is covered by ``test_nfw.py`` and ``test_twohalo.py``.
"""

import sys

import numpy as np
import pytest

from clenspy.cosmology import BiasModel
from clenspy.halo import NfwProfile, TwoHaloTerm
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


def test_supplied_two_halo_skips_the_power_spectrum():
    """two_halo= is used as given, the other collaborator P(k) could build."""
    stub_two_halo = object()
    lp = LensingProfile(**CHEAP, include_2halo=False, two_halo=stub_two_halo)
    assert lp.two_halo_profile is stub_two_halo
    assert "Pkvec" not in lp.__dict__


def test_density_one_halo_only_matches_nfw():
    """include_2halo=False must reduce `density` to the bare NFW profile."""
    lp = LensingProfile(**CHEAP, include_2halo=False)
    r = np.array([0.5, 2.0])
    np.testing.assert_allclose(lp.density(r), lp.halo_profile.density(r))


def test_halo_profile_defensive_branch_for_unimplemented_model():
    """Unreachable through the constructor (`_validate_inputs` rejects an
    unsupported ``model`` first), but the guard inside `halo_profile`
    should still raise its documented error if ever hit directly."""
    lp = LensingProfile(**CHEAP, include_2halo=False)
    lp.model = "made_up_model_name"
    with pytest.raises(NotImplementedError, match="not implemented"):
        lp.halo_profile


def test_reduced_shear_raises_when_convergence_saturates():
    """kappa >= 1 anywhere must raise, not silently divide by <= 0."""
    lp = LensingProfile(**CHEAP, include_2halo=False)
    lp.convergence = lambda R: np.array([1.2, 0.3])
    with pytest.raises(ValueError, match="less than 1"):
        lp.reduced_shear(np.array([0.1, 1.0]))


def test_fourier_profile_scalar_input_returns_length_one_array():
    """`fourier_profile` runs ``k = np.atleast_1d(k)`` before the
    scalar/array check, so a bare python float never reaches the
    "was it scalar" branch as 0-d -- it always comes back as shape (1,)."""
    lp = LensingProfile(**CHEAP, include_2halo=False)
    val = lp.fourier_profile(0.5)
    assert isinstance(val, np.ndarray)
    assert val.shape == (1,)


def test_repr_reports_key_parameters():
    lp = LensingProfile(**CHEAP, include_2halo=False)
    r = repr(lp)
    assert "nfw" in r
    assert "0.300" in r
    assert "1.00e+14" in r


# --- the real 2-halo chain: Pkvec -> two_halo_profile -> bias_model -> bias -


@pytest.fixture(scope="module")
def real_profile():
    """One real, PyCCL-backed `LensingProfile`, built once for this module.

    No `bias=`/`two_halo=` is supplied, so `deltasigma`/`sigma`/etc. below
    force the full chain: `Pkvec` (runs PyCCL) -> `two_halo_profile` ->
    `bias_model` -> `bias`. PyCCL with `PkGrid`'s default grid is
    sub-second, so this is cheap enough to share across the module.
    """
    return LensingProfile(z_cluster=0.3, m200=1e14, backend_2halo="pyccl")


def test_pkvec_two_halo_bias_are_built_from_a_real_pk(real_profile):
    """The untested chain: a real Boltzmann/halofit P(k) all the way to
    b(M), with nothing pre-supplied to short-circuit it."""
    lp = real_profile

    pk = lp.Pkvec
    assert isinstance(pk, np.ndarray)
    assert pk.shape == lp.kvec.shape
    assert np.all(np.isfinite(pk))
    assert np.all(pk > 0)

    two_halo = lp.two_halo_profile
    assert isinstance(two_halo, TwoHaloTerm)
    # built from exactly that Pkvec, on the single z_cluster slice
    np.testing.assert_allclose(two_halo.Pk_grid[:, 0], pk)
    np.testing.assert_allclose(two_halo.zvec, [lp.z_cluster])

    bias_model = lp.bias_model
    assert isinstance(bias_model, BiasModel)

    bias = lp.bias
    assert np.isfinite(bias)
    assert 0.5 < bias < 20  # generous range for a cluster-mass halo

    # cached_property: repeated access is stable and returns cached objects
    assert lp.Pkvec is pk
    assert lp.two_halo_profile is two_halo
    assert lp.bias_model is bias_model
    assert lp.bias == bias


def test_bias_cached_property_does_not_recompute():
    """A second access must not touch `bias_model` again -- poison it after
    the first access and confirm the cached value survives untouched."""
    lp = LensingProfile(z_cluster=0.3, m200=1e14, backend_2halo="pyccl")
    b1 = lp.bias
    assert "bias" in lp.__dict__
    lp.__dict__["bias_model"] = None  # would raise if bias recomputed
    b2 = lp.bias
    assert b2 == b1


def test_deltasigma_adds_the_two_halo_term(real_profile):
    lp = real_profile
    R = np.array([0.5, 1.0, 3.0])
    ds_total = lp.deltasigma(R)
    ds_1h = lp.halo_profile.deltasigma(R)
    ds_2h = lp.rho_m * lp.two_halo_profile.deltasigma(R, lp.z_cluster)
    np.testing.assert_allclose(ds_total, ds_1h + lp.bias * ds_2h)
    assert np.all(ds_2h > 0)  # a genuine addition at these radii


def test_sigma_adds_the_two_halo_term(real_profile):
    lp = real_profile
    R = np.array([0.5, 1.0, 3.0])
    sig_total = lp.sigma(R)
    sig_1h = lp.halo_profile.sigma(R)
    sig_2h = lp.rho_m * lp.two_halo_profile.sigma(R, lp.z_cluster)
    np.testing.assert_allclose(sig_total, sig_1h + lp.bias * sig_2h)


def test_density_adds_two_halo_mean_field(real_profile):
    lp = real_profile
    r = np.array([1.0, 5.0, 20.0])
    rho_total = lp.density(r)
    rho_1h = lp.halo_profile.density(r)
    xi = lp.two_halo_profile.xi(r, lp.z_cluster)
    expected = rho_1h + lp.rho_m * (1 + lp.bias * xi)
    np.testing.assert_allclose(rho_total, expected)


def test_shear_is_deltasigma_over_sigma_crit(real_profile):
    lp = real_profile
    R = np.array([0.5, 1.0])
    np.testing.assert_allclose(lp.shear(R), lp.deltasigma(R) / lp.sigma_crit)


def test_convergence_is_sigma_over_sigma_crit(real_profile):
    lp = real_profile
    R = np.array([0.5, 1.0])
    np.testing.assert_allclose(lp.convergence(R), lp.sigma(R) / lp.sigma_crit)


def test_reduced_shear_formula(real_profile):
    lp = real_profile
    R = np.array([0.5, 1.0, 3.0])
    kappa = lp.convergence(R)
    gamma = lp.shear(R)
    np.testing.assert_allclose(lp.reduced_shear(R), gamma / (1.0 - kappa))


def test_fourier_profile_adds_two_halo_term(real_profile):
    lp = real_profile
    k = np.array([0.1, 1.0, 5.0])
    u_total = lp.fourier_profile(k)
    u_1h = lp.halo_profile.fourier(k)
    u_2h = lp.bias * lp.two_halo_profile.p_kz(k, lp.z_cluster) / lp.m200
    np.testing.assert_allclose(u_total, u_1h + u_2h)
    assert u_total.shape == k.shape
