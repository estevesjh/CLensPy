"""
Tests for the single-offset miscentering kernel.

Reference values were computed with mpmath (dps=30) on the dimensionless
NFW kernel Sigma(x) = f(x) (i.e. r_s = 1, 2 r_s rho_s = 1), where the
Wright & Brainerd (2000) closed forms give

    Sigmabar(<x) = (2/x^2) * [ln(x/2) + arccosh(1/x)/sqrt(1-x^2)]   (x < 1)

with the usual analytic continuations at x >= 1. Sigma_mis is the exact
azimuthal average and Sigmabar_mis the exact aperture mean, both evaluated
by adaptive quadrature at high precision.
"""

import numpy as np
import pytest

from clenspy.halo import NfwProfile
from clenspy.lensing.miscentering import MiscenteringProfile
from clenspy.selection.miscentering_kernel import (
    miscentered_deltasigma,
    miscentered_mean_sigma,
    miscentered_sigma,
)

# --- dimensionless NFW closed forms (r_s = 1, 2 r_s rho_s = 1) ---


def sigma_kernel(x):
    """Sigma(x) = f(x), Wright & Brainerd (2000)."""
    return NfwProfile._fNfw(np.asarray(x, dtype=float))


def mean_sigma_kernel(x):
    """Sigmabar(<x) = f(x) + g(x)/2 with WB2000 kernels f, g."""
    x = np.asarray(x, dtype=float)
    return NfwProfile._fNfw(x) + 0.5 * NfwProfile._gNfw(x)


# (R, R_mis, Sigma_mis, Sigmabar_mis) from mpmath, dps=30
MPMATH_CASES = [
    (0.5, 0.3, 0.72459514039203388, 0.95797043911056462),
    (1.0, 0.3, 0.34122700031159266, 0.59550990147905215),
    (1.0, 0.8, 0.39875460398360935, 0.47479635247706416),
    (1.0, 1.0, 0.44795141046236581, 0.38511064318987631),
    (1.0, 1.2, 0.33713622190889416, 0.29881392060589614),
    (1.0, 2.0, 0.14854842247232078, 0.13984388083942959),
    (2.0, 2.0, 0.23926062442747441, 0.17507538029157525),
    (3.0, 1.0, 0.075358634591457397, 0.18032249042896084),
    (3.0, 8.0, 0.014612558882477502, 0.013765422533642415),
    (0.5, 2.0, 0.13562639450390134, 0.13369517052700128),
]


@pytest.mark.parametrize("R,r_mis,smis_ref,sbar_ref", MPMATH_CASES)
def test_sigma_mis_vs_mpmath(R, r_mis, smis_ref, sbar_ref):
    val = miscentered_sigma(sigma_kernel, R, r_mis, n_nodes=256)
    assert val == pytest.approx(smis_ref, rel=5e-9)


@pytest.mark.parametrize("R,r_mis,smis_ref,sbar_ref", MPMATH_CASES)
def test_mean_sigma_mis_vs_mpmath(R, r_mis, smis_ref, sbar_ref):
    val = miscentered_mean_sigma(mean_sigma_kernel, R, r_mis, n_nodes=256)
    assert val == pytest.approx(sbar_ref, rel=1e-10)


def test_zero_offset_recovers_centered():
    R = np.logspace(-2, 1, 20)
    np.testing.assert_allclose(
        miscentered_sigma(sigma_kernel, R, 0.0), sigma_kernel(R), rtol=1e-14
    )
    np.testing.assert_allclose(
        miscentered_mean_sigma(mean_sigma_kernel, R, 0.0),
        mean_sigma_kernel(R),
        rtol=1e-14,
    )


def test_small_offset_limit():
    """r_mis -> 0 must converge continuously to the centered profile."""
    R = np.array([0.5, 1.0, 3.0])
    val = miscentered_deltasigma(sigma_kernel, mean_sigma_kernel, R, 1e-6)
    centered = mean_sigma_kernel(R) - sigma_kernel(R)
    np.testing.assert_allclose(val, centered, rtol=1e-6)


def test_uniform_sheet_invariance():
    """A constant Sigma is invariant under miscentering: DeltaSigma_mis = 0."""
    const = lambda u: np.full_like(np.asarray(u, dtype=float), 7.5)
    for R, r_mis in [(1.0, 0.5), (1.0, 3.0), (0.2, 0.2)]:
        assert miscentered_sigma(const, R, r_mis) == pytest.approx(7.5, rel=1e-12)
        assert miscentered_mean_sigma(const, R, r_mis) == pytest.approx(
            7.5, rel=1e-12
        )


def test_point_mass_limits():
    """
    Point mass M at the true center: Sigmabar(<u) = M / (pi u^2), so
    Sigmabar_mis(<R) = M/(pi R^2) if r_mis < R (center inside aperture)
    and exactly 0 if r_mis > R (center outside). The exact cancellation
    outside is the reason the negative lobe of DeltaSigma_mis for
    extended profiles is a finite-profile effect.
    """
    M = 2.0
    point_bar = lambda u: M / (np.pi * np.asarray(u, dtype=float) ** 2)
    inside = miscentered_mean_sigma(point_bar, 1.0, 0.5, n_nodes=256)
    outside = miscentered_mean_sigma(point_bar, 1.0, 2.0, n_nodes=256)
    assert inside == pytest.approx(M / np.pi, rel=1e-12)
    assert abs(outside) < 1e-14 * (M / np.pi)


def test_deltasigma_mis_is_signed():
    """Positive for r_mis << R, negative for r_mis >~ R; never clamped."""
    for r_mis, sign in [(0.3, +1), (0.8, +1), (1.0, -1), (1.2, -1), (2.0, -1)]:
        d = miscentered_deltasigma(sigma_kernel, mean_sigma_kernel, 1.0, r_mis)
        assert np.sign(d) == sign


def test_mean_field_cancellation():
    """
    Integral of DeltaSigma_mis over a uniform offset population,
    I = int_0^Rmax DeltaSigma_mis(R|Rm) 2 pi Rm dRm, must converge to 0
    (a uniform halo population is a uniform sheet). Requires the signed
    negative lobe: clamping to >= 0 pins I at ~+50% of its L1 norm.
    """
    R = 1.0
    r_mis_grid = np.linspace(0.02, 60.0, 600)
    d = np.array(
        [
            miscentered_deltasigma(sigma_kernel, mean_sigma_kernel, R, rm)
            for rm in r_mis_grid
        ]
    )
    w = 2 * np.pi * r_mis_grid
    total = np.trapezoid(d * w, r_mis_grid)
    l1 = np.trapezoid(np.abs(d * w), r_mis_grid)
    assert abs(total) < 0.005 * l1  # truncation residual only


def test_vector_matches_scalar():
    R = np.array([0.3, 1.0, 2.5])
    vec = miscentered_deltasigma(sigma_kernel, mean_sigma_kernel, R, 0.7)
    scal = [
        miscentered_deltasigma(sigma_kernel, mean_sigma_kernel, r, 0.7) for r in R
    ]
    np.testing.assert_allclose(vec, scal, rtol=1e-12)
    assert vec.shape == R.shape


def test_nfw_kernel_stability_at_unity():
    """f, g are analytic at x=1; the series window must be seamless."""
    x = 1.0 + np.array([-2e-2, -1e-2, -1e-3, -1e-6, 0.0, 1e-6, 1e-3, 1e-2, 2e-2])
    f = NfwProfile._fNfw(x)
    g = NfwProfile._gNfw(x)
    # exact values at x=1
    assert f[4] == pytest.approx(1.0 / 3.0, rel=1e-15)
    assert g[4] == pytest.approx(10.0 / 3.0 - 4 * np.log(2.0), rel=1e-15)
    # monotone through the window (no jumps at the branch switches)
    assert np.all(np.diff(f) < 0)
    assert np.all(np.diff(g) < 0)
    # second differences stay small (no kink at the window edges)
    assert np.all(np.abs(np.diff(f, 2)) < 1e-2)


def test_profile_class_wiring():
    """MiscenteringProfile routes through the same kernels (1-halo only)."""
    prof = MiscenteringProfile(
        z_cluster=0.3, m200=1e14, include_2halo=False, r_mis=0.4
    )
    R = np.array([0.5, 1.0, 2.0])
    sig = prof.sigma_mis(R)
    dsig = prof.deltasigma_mis(R)
    # the profile now INTERPOLATES; the quadrature is the generator, so
    # they agree to the table's accuracy rather than to round-off
    expected = miscentered_sigma(prof.sigma, R, 0.4)
    np.testing.assert_allclose(sig, expected, rtol=5e-3)
    np.testing.assert_allclose(
        dsig, prof.mean_sigma_mis(R) - prof.sigma_mis(R), rtol=1e-12
    )
    # zero offset degenerates to the centered profile
    prof0 = MiscenteringProfile(
        z_cluster=0.3, m200=1e14, include_2halo=False, r_mis=0.0
    )
    np.testing.assert_allclose(
        prof0.deltasigma_mis(R), prof0.deltasigma(R), rtol=1e-14
    )
    with pytest.raises(ValueError):
        MiscenteringProfile(z_cluster=0.3, m200=1e14, include_2halo=False, r_mis=-1)


# --- the table is the only runtime path -------------------------------------


def test_mean_sigma_passthrough_to_halo_profile():
    """`mean_sigma` is a pure passthrough to the wrapped 1-halo profile's
    closed form -- it does not touch the miscentering table at all."""
    prof = MiscenteringProfile(
        z_cluster=0.3, m200=1e14, include_2halo=False, r_mis=0.4
    )
    R = np.array([0.2, 0.5, 1.0, 3.0])
    np.testing.assert_allclose(prof.mean_sigma(R), prof.halo_profile.mean_sigma(R))


def test_einasto_has_no_table_and_says_so():
    """Untabulated profiles refuse rather than falling back to quadrature."""
    from clenspy.halo import EinastoProfile
    from clenspy.selection.miscentering import (
        MiscenteringTableError,
        require_tabulated_profile,
    )

    ein = EinastoProfile(alpha=0.2, rho_0=1e15, r_s=0.3)
    with pytest.raises(MiscenteringTableError) as exc:
        require_tabulated_profile(ein)
    msg = str(exc.value)
    assert "EinastoProfile" in msg
    assert "NfwProfile" in msg
    assert "make_miscentering_table" in msg
    # it is a NotImplementedError, so `except NotImplementedError` still works
    assert isinstance(exc.value, NotImplementedError)

    require_tabulated_profile(NfwProfile(m200=1e14))  # NFW is fine


def test_table_matches_its_generator():
    """The packaged table reproduces the quadrature that built it."""
    from clenspy.selection.miscentering import load_nfw_miscentering_table
    from clenspy.selection.miscentering_kernel import nfw_mean_sigma_hat, nfw_sigma_hat

    table = load_nfw_miscentering_table()
    x = np.array([0.05, 0.2, 1.0, 5.0, 40.0])
    for x_mis in (0.03, 0.3, 1.0, 4.0):
        ref_s = miscentered_sigma(nfw_sigma_hat, x, x_mis, n_nodes=1024)
        ref_d = miscentered_deltasigma(
            nfw_sigma_hat, nfw_mean_sigma_hat, x, x_mis, n_nodes=1024
        )
        np.testing.assert_allclose(table.sigma_hat(x, x_mis), ref_s, rtol=5e-3)
        np.testing.assert_allclose(table.ds_hat(x, x_mis), ref_d, rtol=5e-3)


def test_table_keeps_the_negative_lobe_on_the_cusp():
    """No sign flips at x = x_mis -- the reason for the ratio axes."""
    from clenspy.selection.miscentering import load_nfw_miscentering_table
    from clenspy.selection.miscentering_kernel import nfw_mean_sigma_hat, nfw_sigma_hat

    table = load_nfw_miscentering_table()
    for x_mis in (0.01, 0.05, 0.2, 1.0, 3.0, 20.0):
        ref = float(
            miscentered_deltasigma(
                nfw_sigma_hat, nfw_mean_sigma_hat, np.array([x_mis]), x_mis,
                n_nodes=1024,
            )[0]
        )
        got = float(table.ds_hat(np.array([x_mis]), x_mis)[0])
        assert ref < 0.0, f"reference should be negative at x = x_mis = {x_mis}"
        assert got < 0.0, f"table lost the negative lobe at x = x_mis = {x_mis}"
        assert got == pytest.approx(ref, rel=5e-3)


def test_zero_offset_is_exact_not_interpolated():
    """r_mis = 0 short-circuits to the analytic centred profile."""
    from clenspy.selection.miscentering import load_nfw_miscentering_table
    from clenspy.selection.miscentering_kernel import nfw_mean_sigma_hat, nfw_sigma_hat

    table = load_nfw_miscentering_table()
    x = np.array([0.1, 1.0, 10.0])
    np.testing.assert_allclose(table.sigma_hat(x, 0.0), nfw_sigma_hat(x), rtol=1e-15)
    np.testing.assert_allclose(
        table.ds_hat(x, 0.0), nfw_mean_sigma_hat(x) - nfw_sigma_hat(x), rtol=1e-15
    )


def test_out_of_range_x_uses_clamp_below_and_centred_above():
    """Below the table: clamp. Above it: the centred closed form."""
    from clenspy.selection.miscentering import load_nfw_miscentering_table

    table = load_nfw_miscentering_table()
    x_lo, x_hi = table.x_range
    x_mis = 0.4

    # below: the value freezes at the left edge
    edge_s = float(np.ravel(table.sigma_hat(np.array([x_lo]), x_mis))[0])
    edge_d = float(np.ravel(table.ds_hat(np.array([x_lo]), x_mis))[0])
    for x in (x_lo * 0.5, x_lo * 1e-3, x_lo * 1e-8):
        assert float(np.ravel(table.sigma_hat(np.array([x]), x_mis))[0]) == edge_s
        assert float(np.ravel(table.ds_hat(np.array([x]), x_mis))[0]) == edge_d

    # above: the centred profile, and it must keep FALLING, not freeze
    prev_s = float(np.ravel(table.sigma_hat(np.array([x_hi]), x_mis))[0])
    prev_d = float(np.ravel(table.ds_hat(np.array([x_hi]), x_mis))[0])
    for x in (x_hi * 2.0, x_hi * 10.0, x_hi * 1e3):
        s = float(np.ravel(table.sigma_hat(np.array([x]), x_mis))[0])
        d = float(np.ravel(table.ds_hat(np.array([x]), x_mis))[0])
        assert s < prev_s, "Sigma must keep decreasing past the table"
        assert d < prev_d, "DeltaSigma must keep decreasing past the table"
        # and it must equal the centred closed form there
        assert s == pytest.approx(
            float(np.ravel(NfwProfile._fNfw(np.array([x])))[0]), rel=1e-12
        )
        prev_s, prev_d = s, d


def test_centred_extrapolation_is_accurate_at_the_right_edge():
    """Past the right edge, centred == true miscentered to ~1/q^2."""
    from clenspy.selection.miscentering import load_nfw_miscentering_table
    from clenspy.selection.miscentering_kernel import (
        miscentered_deltasigma,
        miscentered_sigma,
        nfw_mean_sigma_hat,
        nfw_sigma_hat,
    )

    table = load_nfw_miscentering_table()
    x_hi = table.x_range[1]
    for x_mis, tol in [(0.01, 1e-5), (1.0, 1e-5), (100.0, 5e-3)]:
        x = np.array([x_hi * 2.0])
        got_s = float(np.ravel(table.sigma_hat(x, x_mis))[0])
        got_d = float(np.ravel(table.ds_hat(x, x_mis))[0])
        ref_s = float(np.ravel(miscentered_sigma(nfw_sigma_hat, x, x_mis, 2048))[0])
        ref_d = float(
            np.ravel(
                miscentered_deltasigma(
                    nfw_sigma_hat, nfw_mean_sigma_hat, x, x_mis, 2048
                )
            )[0]
        )
        assert got_s == pytest.approx(ref_s, rel=tol)
        assert got_d == pytest.approx(ref_d, rel=tol)


def test_missing_table_file_raises(tmp_path):
    """Constructing directly from a nonexistent path refuses cleanly."""
    from clenspy.selection.miscentering import (
        MiscenteringTableError,
        NfwMiscenteringTable,
    )

    with pytest.raises(MiscenteringTableError, match="not found"):
        NfwMiscenteringTable(tmp_path / "missing.npz")


def test_x_mis_range_and_q_range_are_exp_of_the_log_grids():
    from clenspy.selection.miscentering import load_nfw_miscentering_table

    table = load_nfw_miscentering_table()
    expected_x_mis = (float(np.exp(table._ln_x_mis[0])),
                       float(np.exp(table._ln_x_mis[-1])))
    expected_q = (float(np.exp(table._ln_q[0])),
                  float(np.exp(table._ln_q[-1])))
    assert table.x_mis_range == pytest.approx(expected_x_mis)
    assert table.q_range == pytest.approx(expected_q)


def test_table_repr_contains_class_name():
    from clenspy.selection.miscentering import load_nfw_miscentering_table

    table = load_nfw_miscentering_table()
    assert "NfwMiscenteringTable" in repr(table)


def test_mixed_in_and_out_of_range_vector():
    """A single call spanning both bounds routes each element correctly."""
    from clenspy.selection.miscentering import load_nfw_miscentering_table

    table = load_nfw_miscentering_table()
    x_lo, x_hi = table.x_range
    x = np.array([x_lo * 1e-3, x_lo, 1.0, x_hi, x_hi * 100.0])
    got = table.ds_hat(x, 0.4)
    assert got.shape == x.shape
    assert np.all(np.isfinite(got))
    one_by_one = [float(np.ravel(table.ds_hat(np.array([v]), 0.4))[0]) for v in x]
    np.testing.assert_allclose(got, one_by_one, rtol=0, atol=0)
