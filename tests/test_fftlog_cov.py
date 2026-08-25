"""Tests for the FFTLog bin-averaged double-Bessel covariance engine."""

import numpy as np
import pytest

from clenspy.utils.fftlog_cov import (
    BinAveragedJ2DoubleBessel,
    GaussianCovFFTLog,
    j2_bin_averaged,
    white_noise_diagonal,
)

RHO = (30.0 / 0.3) ** (1.0 / 15.0)  # 15 log bins over 2 decades
EDGES = 1e-3 * RHO ** np.arange(16)  # radians-ish scale
ELL = np.logspace(-1, 6.5, 2048)


def test_j2bar_small_ell_limit():
    """J2bar = O(ell^2) down to arbitrarily small ell (series branch
    protects against the float cancellation floor of the closed form)."""
    a, b = 2e-3, 2e-3 * RHO
    ell = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0])
    got = j2_bin_averaged(ell, a, b)
    expected = (a**2 + b**2) * ell**2 / 16.0
    assert np.allclose(got, expected, rtol=1e-6)


def test_j2bar_series_direct_continuity():
    """Series and direct branches agree across the x = 1e-2 switch."""
    a, b = 2e-3, 2e-3 * RHO
    ell_switch = 1e-2 / b
    ell = np.linspace(0.5 * ell_switch, 2.0 * ell_switch, 101)
    # divide out the leading ell^2 scaling: the residual must be smooth
    # through the branch switch (no jump beyond the O(x^4) curvature)
    vals = j2_bin_averaged(ell, a, b) / ell**2
    rel_step = np.abs(np.diff(vals) / vals[:-1])
    assert np.all(rel_step < 1e-3)


def test_j2bar_matches_numerical_annulus_average():
    """Closed form vs numerical (2/(b^2-a^2)) int_a^b theta J2 dtheta."""
    from scipy.special import jv

    a, b = 1e-3, 1e-3 * RHO
    for ell in (300.0, 3000.0, 30000.0):
        th = np.linspace(a, b, 20001)
        num = 2.0 / (b**2 - a**2) * np.trapezoid(th * jv(2, ell * th), th)
        assert np.isclose(j2_bin_averaged(ell, a, b), num, rtol=1e-6, atol=1e-9)


def test_powerlaw_selfconsistency():
    """For F = ell^{2-p} (power law), FFTLog must reproduce
    G(y) = y^{p-2} U(2-p) with U from the summed Mellin kernel."""
    from clenspy.utils.fftlog_cov import _mellin_binavg_j2j2

    p = 1.4
    for d in (0, 1, 3):
        alpha = RHO**d
        tr = BinAveragedJ2DoubleBessel(ELL, RHO, alpha, q=1.0)
        y, G = tr(ELL ** (2.0 - p), extrap=True)
        U = _mellin_binavg_j2j2(RHO, alpha)(np.array([2.0 - p + 0j]))[0]
        expected = y ** (p - 2.0) * U.real
        mid = slice(len(y) // 4, 3 * len(y) // 4)
        # measured: max rel 2e-7 (d=0), 8e-7 (d=1), 8e-6 (d=3)
        assert np.allclose(G[mid], expected[mid], rtol=3e-5), d


def test_constant_C_orthogonality():
    """Constant C: diagonal -> 2/(b^2-a^2) per bin, off-diagonals ~ 0."""
    engine = GaussianCovFFTLog(ELL, EDGES, f_sky=1.0)
    # trapz reference with constant C (converges thanks to J2bar^2 ~ 1/l^3)
    C_const = np.ones_like(ELL)
    cov_ref = engine.covariance_trapz_reference(C_const, dlnell=2e-3)
    expected_diag = white_noise_diagonal(EDGES, 1.0, 1.0)
    assert np.allclose(np.diag(cov_ref), expected_diag, rtol=1e-3)
    # off-diagonals vanish relative to their own row/column scale
    # (finite ell_max truncation leaves oscillatory residue, so compare
    # against sqrt(d_i d_j), not the global minimum diagonal)
    d = np.diag(cov_ref)
    corr = cov_ref / np.sqrt(np.outer(d, d))
    off = corr[np.triu_indices_from(corr, k=2)]
    assert np.all(np.abs(off) < 1e-3)


def _smooth_C(ell):
    """Limber-like smooth spectrum: flat at low ell, falling at high ell."""
    return 1e-6 / (1.0 + (ell / 3000.0) ** 2) ** 1.5


def test_fftlog_vs_trapz_smooth():
    """The production gate: FFTLog covariance vs legacy trapz <= 1e-3."""
    engine = GaussianCovFFTLog(ELL, EDGES, f_sky=0.1)
    C = _smooth_C(ELL)
    cov_fft = engine.covariance(C)
    cov_ref = engine.covariance_trapz_reference(C, dlnell=1e-3)
    # relative on the diagonal
    d_fft, d_ref = np.diag(cov_fft), np.diag(cov_ref)
    assert np.allclose(d_fft, d_ref, rtol=1e-3)
    # correlation-matrix elements to atol 1e-3
    corr_fft = cov_fft / np.sqrt(np.outer(d_fft, d_fft))
    corr_ref = cov_ref / np.sqrt(np.outer(d_ref, d_ref))
    assert np.allclose(corr_fft, corr_ref, atol=1e-3)


def test_N_doubling_stability():
    """Doubling the ell sampling changes the result by <= 1e-4."""
    C = _smooth_C
    engine1 = GaussianCovFFTLog(ELL, EDGES, f_sky=0.1)
    ell2 = np.logspace(-1, 6.5, 4096)
    engine2 = GaussianCovFFTLog(ell2, EDGES, f_sky=0.1)
    cov1 = engine1.covariance(C(ELL))
    cov2 = engine2.covariance(C(ell2))
    d1, d2 = np.diag(cov1), np.diag(cov2)
    assert np.allclose(d1, d2, rtol=1e-4)


def test_white_noise_added_on_diagonal():
    engine = GaussianCovFFTLog(ELL, EDGES, f_sky=0.1)
    C = _smooth_C(ELL)
    cov0 = engine.covariance(C)
    covN = engine.covariance(C, noise_const=1e-5)
    dN = white_noise_diagonal(EDGES, 1e-5, 0.1)
    assert np.allclose(np.diag(covN) - np.diag(cov0), dN, rtol=1e-12)
    off = ~np.eye(len(EDGES) - 1, dtype=bool)
    assert np.allclose(covN[off], cov0[off], rtol=1e-12)


def test_non_geometric_edges_rejected():
    bad = np.concatenate([EDGES[:-1], [EDGES[-1] * 1.5]])
    with pytest.raises(ValueError, match="geometric"):
        GaussianCovFFTLog(ELL, bad, f_sky=0.1)
