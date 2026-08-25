"""Tests for clenspy.clusters.mor."""

import numpy as np
import pytest

from clenspy.clusters import HodMOR, HodParams, LogNormalMOR, LogNormalParams

H = 0.7


@pytest.fixture(scope="module")
def hod():
    return HodMOR(HodParams.des_y1(H))


@pytest.fixture(scope="module")
def logn():
    return LogNormalMOR(LogNormalParams.costanzi21(H))


def test_h_conversion():
    p = HodParams.des_y1(H)
    assert np.isclose(p.M_min, 10.0**11.3852818 / H)
    assert np.isclose(p.M1, 10.0**12.6964410 / H)
    lp = LogNormalParams.costanzi21(H)
    assert np.isclose(lp.M_pivot, 3.0e14 / H)


@pytest.mark.parametrize("which", ["hod", "logn"])
def test_pdf_normalized(which, hod, logn):
    mor = {"hod": hod, "logn": logn}[which]
    for M in (5e13, 2e14, 1e15):
        a, b = mor.bracket(M, 0.3, L=10.0)
        ltr = np.linspace(max(a, 1e-6), b, 4000)
        norm = np.trapezoid(mor.pdf(ltr, M, 0.3), ltr)
        assert np.isclose(norm, 1.0, atol=5e-3), (which, M, norm)


@pytest.mark.parametrize("which", ["hod", "logn"])
def test_moments_match_pdf(which, hod, logn):
    mor = {"hod": hod, "logn": logn}[which]
    M, z = 3e14, 0.4
    a, b = mor.bracket(M, z, L=12.0)
    ltr = np.linspace(max(a, 1e-6), b, 8000)
    p = mor.pdf(ltr, M, z)
    mean = np.trapezoid(ltr * p, ltr)
    var = np.trapezoid((ltr - mean) ** 2 * p, ltr)
    assert np.isclose(mean, mor.ltr_mean(M, z), rtol=0.05)
    assert np.isclose(np.sqrt(var), mor.ltr_sigma(M, z), rtol=0.10)


def test_hod_lsat_monotonic(hod):
    M = np.logspace(13, 15.5, 40)
    ls = hod.l_sat(M, 0.3)
    assert np.all(np.diff(ls) > 0)
    assert hod.l_sat(hod.M_min * 0.5, 0.3) < 1e-20  # clipped below M_min


def test_hod_pdf_negative_ltr_zero(hod):
    assert hod.pdf(-1.0, 1e14, 0.3) == 0.0


def test_lognormal_mean_scaling(logn):
    """<ln lambda> scales as B ln M."""
    p = logn.params
    r = logn.ln_lambda_mean(2.0 * p.M_pivot, p.z_pivot) - logn.ln_lambda_mean(
        p.M_pivot, p.z_pivot
    )
    assert np.isclose(r, p.B_lambda * np.log(2.0), rtol=1e-12)
    assert np.isclose(
        logn.ln_lambda_mean(p.M_pivot, p.z_pivot), np.log(p.A_lambda), rtol=1e-12
    )


def test_bracket_shapes(hod):
    M = np.logspace(13, 15, 7)
    a, b = hod.bracket(M, 0.3)
    assert a.shape == b.shape == (7,)
    assert np.all(a >= 0) and np.all(b > a)


def test_lambda_mean_below(hod):
    """Truncated mean below a huge lob equals the full mean."""
    M = 2e14
    full = hod.lambda_mean_below(M, 0.3, 1e4)
    # true mean of the shifted-Poisson pdf is l_tr = 1 + l_sat
    assert np.isclose(full, hod.l_tr(M, 0.3), rtol=0.02)
    # truncation reduces the mean
    trunc = hod.lambda_mean_below(M, 0.3, hod.ltr_mean(M, 0.3))
    assert trunc < full
