"""`compute_sigma_grid`'s method dispatch: the ValueError branch, and that
the two supported quadrature schemes agree with each other on the same
Abel integral for a simple synthetic correlation function.
"""

import numpy as np
import pytest

from clenspy.utils.integrate import compute_sigma_grid


def xi_func(r, z):
    """A simple, positive, decaying correlation function, xi(r, z)."""
    r = np.asarray(r, dtype=float)
    return np.exp(-r)


def test_compute_sigma_grid_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        compute_sigma_grid(xi_func, np.array([1.0]), np.array([0.0]),
                            method="bogus")


def test_leggauss_and_trapz_give_finite_positive_results():
    Rvec = np.array([0.3, 1.0, 3.0])
    zvec = np.array([0.0])
    for method in ("leggauss", "trapz"):
        sigma = compute_sigma_grid(xi_func, Rvec, zvec, method=method,
                                    rmax_integral=50.0, n_points=80)
        assert sigma.shape == (Rvec.size, zvec.size)
        assert np.all(np.isfinite(sigma))
        assert np.all(sigma > 0.0)


def test_leggauss_and_trapz_agree_with_each_other():
    """Two different quadrature schemes for the same Abel integral."""
    Rvec = np.array([0.3, 1.0, 3.0])
    zvec = np.array([0.0])
    leggauss = compute_sigma_grid(xi_func, Rvec, zvec, method="leggauss",
                                   rmax_integral=50.0, n_points=80)
    trapz = compute_sigma_grid(xi_func, Rvec, zvec, method="trapz",
                                rmax_integral=50.0, n_points=800)
    np.testing.assert_allclose(leggauss, trapz, rtol=0.05)
