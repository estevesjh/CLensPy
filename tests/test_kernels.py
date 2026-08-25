"""Tests for clenspy.clusters.kernels (EMG CDF, K_i, K_j)."""

import mpmath
import numpy as np
import pytest

from clenspy.clusters import (
    AnalyticLogNormalKernel,
    EmgRichnessKernel,
    K_j,
    PlobLtrParams,
    emg_cdf,
)


def _emg_cdf_mpmath(x, mu, sigma, tau):
    """High-precision EMG CDF: Phi(z) - exp(A) Phi(z - tau sigma)."""
    mpmath.mp.dps = 40
    z = (x - mu) / sigma
    Phi = lambda t: 0.5 * (1 + mpmath.erf(t / mpmath.sqrt(2)))  # noqa: E731
    A = -tau * (x - mu) + 0.5 * (tau * sigma) ** 2
    return float(Phi(z) - mpmath.exp(A) * Phi(z - tau * sigma))


@pytest.mark.parametrize(
    "x,mu,sigma,tau",
    [
        (25.0, 20.0, 5.0, 0.1),
        (5.0, 20.0, 5.0, 0.1),  # deep lower tail
        (120.0, 20.0, 5.0, 0.1),  # deep upper tail
        (30.0, 30.0, 2.0, 0.5),
        (200.0, 30.0, 2.0, 0.5),  # exp*erfc overflow territory
        (30.0, 100.0, 10.0, 0.02),
    ],
)
def test_emg_cdf_vs_mpmath(x, mu, sigma, tau):
    ref = _emg_cdf_mpmath(x, mu, sigma, tau)
    got = float(emg_cdf(x, mu, sigma, tau))
    assert np.isclose(got, np.clip(ref, 0.0, 1.0), atol=1e-12)


def test_emg_cdf_limits():
    # tau*sigma -> large x: CDF -> 1; x -> -inf: -> 0
    assert emg_cdf(1e4, 20.0, 5.0, 0.1) == pytest.approx(1.0, abs=1e-10)
    assert emg_cdf(-1e3, 20.0, 5.0, 0.1) == pytest.approx(0.0, abs=1e-12)
    # monotonic in x
    xs = np.linspace(0, 300, 500)
    cdf = emg_cdf(xs, 40.0, 8.0, 0.05)
    assert np.all(np.diff(cdf) >= -1e-12)


def test_plob_params_shapes():
    plob = PlobLtrParams.from_file()
    ltr = np.linspace(5, 200, 50)
    mu, sigma, tau, fprj = plob.at(ltr, 0.35)
    for arr in (mu, sigma, tau, fprj):
        assert np.asarray(arr).shape == (50,)
    assert np.all(sigma > 0) and np.all(tau > 0)
    assert np.all((fprj >= 0) & (fprj <= 1))
    # mu ~ ltr for the Y3 fit (b_mu ~ 1)
    assert np.all(np.abs(mu - ltr) < 0.3 * ltr + 10)


def test_plob_params_z_broadcast():
    plob = PlobLtrParams.from_file()
    ltr = np.linspace(5, 200, 12)[:, None]
    z = np.linspace(0.1, 0.8, 7)[None, :]
    mu, sigma, tau, fprj = plob.at(ltr, z)
    assert mu.shape == (12, 7)


def test_emg_kernel_bin_probability():
    kern = EmgRichnessKernel()
    ltr = np.array([5.0, 25.0, 50.0, 120.0])
    ki = kern.K_i(ltr, 0.3, 20.0, 30.0)
    assert ki.shape == (4,)
    assert np.all((ki >= 0) & (ki <= 1))
    # ltr=25 in the middle of the [20,30] bin: highest probability
    assert ki[1] == max(ki)
    # partition: sum over contiguous bins spanning everything ~ 1
    edges = [0.0, 20.0, 30.0, 45.0, 60.0, 1e5]
    total = sum(
        kern.K_i(25.0, 0.3, lo, hi) for lo, hi in zip(edges[:-1], edges[1:])
    )
    assert np.isclose(total, 1.0, atol=1e-8)


def test_emg_kernel_pdf_consistent_with_cdf():
    """d/dx CDF == pdf_lob numerically."""
    kern = EmgRichnessKernel()
    ltr, z = 40.0, 0.4
    x = np.linspace(10, 120, 2001)
    cdf = kern.cdf(x, ltr, z)
    pdf_num = np.gradient(cdf, x)
    pdf = kern.pdf_lob(x, ltr, z)
    keep = pdf > 1e-5
    assert np.allclose(pdf[keep], pdf_num[keep], rtol=2e-2)


def test_analytic_lognormal_kernel_indicator():
    kern = AnalyticLogNormalKernel()
    ltr = np.array([10.0, 25.0, 35.0])
    ki = kern.K_i(ltr, 0.3, 20.0, 30.0)
    assert np.array_equal(ki, [0.0, 1.0, 0.0])


def test_K_j_limits():
    z = np.linspace(0.0, 1.0, 201)
    # sigma_z -> 0: top-hat
    kj0 = K_j(z, 0.2, 0.4, 0.0)
    assert np.array_equal(kj0, ((z >= 0.2) & (z < 0.4)).astype(float))
    # finite sigma_z: smooth, in [0, 1], ~1 in the middle of a wide bin
    kj = K_j(z, 0.2, 0.4, 0.01)
    assert np.all((kj >= 0) & (kj <= 1))
    assert kj[np.argmin(np.abs(z - 0.3))] > 0.999
