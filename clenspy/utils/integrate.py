"""
A class that holds the integration methods for cluster lensing observables.
"""

import mcfit
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad_vec

try:
    from scipy.integrate import cumtrapz, trapz
except ImportError:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
    from scipy.integrate import trapezoid as trapz

from ..utils.interpolate import make_log_interpolation


def compute_sigma_grid(
    xi_func,
    Rvec,
    zvec,
    method="trapz",
    rmax_integral=1000.0,
    n_points=100,
) -> np.ndarray:
    """
    Dispatch and run the chosen integration method for Sigma(R, z).
    Returns grid of shape (nR, nz).
    """
    method = method.lower()
    if method == "leggauss":
        return compute_sigma_leggauss(xi_func, Rvec, zvec, rmax_integral, n_points).T
    elif method == "trapz":
        return compute_sigma_trapz_vectorized(
            xi_func, Rvec, zvec, rmax_integral, n_points
        ).T
    elif method == "quad_vec":
        return compute_sigma_quadvec(xi_func, Rvec, zvec, rmax_integral)
    else:
        raise ValueError(f"Unknown method '{method}' for Sigma(R) grid computation.")


def sigma_to_deltasigma_cumtrapz(
    Rvec: np.ndarray, sigma_grid: np.ndarray
) -> np.ndarray:
    """
    Compute ΔΣ(R) = mean_Σ(<R) - Σ(R) from a grid of Σ(R).

    Parameters
    ----------
    Rvec : np.ndarray
        Radii (nR), must be strictly increasing.
    sigma_grid : np.ndarray
        Σ(R) (nz, nR) or (nR,) for single z.

    Returns
    -------
    deltasigma_grid : np.ndarray
        ΔΣ(R) (nz, nR) or (nR,) (same shape as input).
    """
    logR = np.log(Rvec)
    shape = sigma_grid.shape
    # Ensure 2D: (nz, nR)
    if sigma_grid.ndim == 1:
        sigma_grid = sigma_grid[None, :]
    integrand = sigma_grid * Rvec[None, :] ** 2
    res = cumtrapz(integrand, logR, axis=1, initial=0)
    mean_sigma = 2 * res / (Rvec**2)[None, :]
    deltasigma = np.clip(mean_sigma - sigma_grid, 0, None)
    return deltasigma if shape == sigma_grid.shape else deltasigma.squeeze()


def pk_to_xi_fftlog(
    kvec: np.ndarray,
    Pk: np.ndarray,
    rvals: np.ndarray,
    *,
    lowring: bool = True,
    **mcfit_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute xi(r) from P(k) using FFTLog via mcfit.

    Parameters
    ----------
    kvec : np.ndarray
        Wavenumber grid.
    Pk : np.ndarray
        Power spectrum (same shape as kvec).
    rvals : np.ndarray
        Output r values.
    lowring : bool, optional
        Use low-ring extrapolation (recommended, default True).
    **mcfit_kwargs : dict
        Additional options passed to mcfit.P2xi.

    Returns
    -------
    r_fftlog : np.ndarray
        Radial grid output by mcfit (may differ from rvals).
    xi_r : np.ndarray
        xi(r) evaluated at r_fftlog.
    """
    r_fftlog, xi_r = mcfit.P2xi(kvec, lowring=lowring, **mcfit_kwargs)(Pk)
    interp = make_log_interpolation(r_fftlog, xi_r)
    xi_r_interp = interp(rvals)
    return xi_r_interp


def compute_sigma_trapz_vectorized(
    xi_func,
    Rvec: np.ndarray,
    zvec: np.ndarray,
    r_max: float = 1000.0,
    n_grid: int = 100,
) -> np.ndarray:
    """
    Compute Σ(R, z) using the Abel integral and the trapezoidal rule (vectorized).

    Parameters
    ----------
    xi_func : callable
        Function xi(r, z), must accept array inputs.
    Rvec : np.ndarray
        Projected radii (shape nR).
    zvec : np.ndarray
        Redshifts (shape nz).
    r_max : float
        Maximum 3D radius for integration. Default 1000.
    n_grid : int
        Number of points for the internal integration grid. Default 100.

    Returns
    -------
    sigma : np.ndarray
        Surface density Σ(R, z) with shape (nR, nz).
    """
    # Setup integration limits for each R (u ∈ [0, umax(R)])
    # ---- define limits in t ----
    u_max = max(np.arccosh(r_max / Rvec))  # finite thanks to r_max
    u_max = np.clip(u_max, None, 40)  # cosh(40) ~ 1.1e17, still in float64 range
    t_max = u_max / (1.0 + u_max)  # < 1
    assert 0.0 < t_max < 1.0

    # Create a grid for t ∈ [0, t_max] with n_grid points
    t_grid = np.linspace(0.0, t_max, n_grid)  # Integration grid for u

    # Create a meshgrid for R, z, u
    zA, RA, tA = np.meshgrid(zvec, Rvec, t_grid, indexing="ij")  # (nz, nR, nu)
    uA = tA / (1.0 - tA)  # u(t)
    rA = RA * np.cosh(uA)  # r = R * cosh(u)
    xiA = xi_func(rA.ravel(), zA.ravel()).reshape(rA.shape)
    integrand = xiA * np.cosh(uA) / (1.0 - tA) ** 2
    sigma = trapz(integrand, t_grid, axis=2)
    return 2 * Rvec * sigma


def compute_sigma_leggauss(
    xi_func, Rvec: np.ndarray, zvec: np.ndarray, r_max: float = 1000.0, N: int = 32
) -> np.ndarray:
    """
    Compute Σ(R, z) using the Abel integral and Gauss-Legendre quadrature.

    Parameters
    ----------
    xi_func : callable
        Function xi(r, z), must accept array inputs.
    Rvec : np.ndarray
        Projected radii (shape nR).
    zvec : np.ndarray
        Redshifts (shape nz).
    r_max : float
        Maximum 3D radius for integration. Default 1000.
    N : int
        Number of Legendre nodes. Default 32.

    Returns
    -------
    sigma : np.ndarray
        Surface density Σ(R, z) with shape (nR, nz).
    """

    def integrand(t: np.array, R: np.array, z: np.array) -> np.array:
        """Vectorised integrand in t ∈ [0,1)."""
        u = t / (1.0 - t)  # u(t)
        r = R * np.cosh(u)  # argument for ρ
        prefac = np.cosh(u) / (1.0 - t) ** 2  # cosh(u) / (1-t)^2
        return prefac * xi_func(r, z)

    # set integration limits
    tmin, tmax = 0, 1 - 1 / r_max

    # setup leggaus weights and nodes
    t_nodes, t_weights = leggauss(N)
    tvec = 0.5 * (tmax - tmin) * t_nodes + 0.5 * (tmax + tmin)
    dt = 0.5 * (tmax - tmin)  # Half-width of the l interval

    # make grid for z, R, t
    zz, RR, tt = np.meshgrid(zvec, Rvec, tvec, indexing="ij")
    fx = integrand(tt.ravel(), RR.ravel(), zz.ravel()).reshape(tt.shape)
    weighted = fx * t_weights
    sigma = 2 * Rvec * np.nansum(weighted, axis=2) * dt  # sum over l-axis
    return sigma


def compute_sigma_quadvec(
    xi_func, Rvec: np.ndarray, zvec: np.ndarray, r_max: float = 1000.0
) -> np.ndarray:
    """
    Compute Σ(R, z) using quad_vec adaptive quadrature.

    Parameters
    ----------
    xi_func : callable
        Function xi(r, z), must accept array inputs.
    Rvec : np.ndarray
        Projected radii (shape nR).
    zvec : np.ndarray
        Redshifts (shape nz).
    r_max : float
        Maximum 3D radius for integration. Default 1000.

    Returns
    -------
    sigma : np.ndarray
        Surface density Σ(R, z) with shape (nR, nz).
    """
    R_grid, z_grid = np.meshgrid(Rvec, zvec, indexing="ij")
    R_flat = R_grid.ravel()
    z_flat = z_grid.ravel()

    # ---- define limits in t ----
    u_max = max(np.arccosh(r_max / Rvec))  # finite thanks to r_max
    u_max = np.clip(u_max, None, 40)  # cosh(40) ~ 1.1e17, still in float64 range
    t_max = u_max / (1.0 + u_max)  # < 1
    assert 0.0 < t_max < 1.0

    def integrand(t: float, R: np.ndarray, z: np.ndarray) -> np.ndarray:
        u = t / (1.0 - t)
        r = R * np.cosh(u)
        return xi_func(r, z) * np.cosh(u) / (1.0 - t) ** 2

    sigma_flat, _ = quad_vec(integrand, 0, t_max, args=(R_flat, z_flat))
    sigma = sigma_flat.reshape(R_grid.shape)
    return 2 * sigma * Rvec[:, None]  # shape (len(Rvec), len(zvec))
