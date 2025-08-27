"""
A class for 2-halo term modeling (from P(k) to xi, Sigma, etc.)
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from ..utils.decorators import default_rvals_z, time_method
from ..utils.integrate import (
    compute_sigma_grid,
    pk_to_xi_fftlog,
    sigma_to_deltasigma_cumtrapz,
)
from ..utils.interpolate import LogGridInterpolator


class TwoHaloTerm:
    """
    Compute 2-halo correlation functions and lensing profiles from gridded P(k, z).

    Provides:
      - Correlation function ξ(r, z)
      - Projected surface density Σ(R, z)
      - Excess surface density ΔΣ(R, z)

    Parameters
    ----------
    kvec : np.ndarray
        Wavenumber grid (size nk)
    Pk : np.ndarray
        Power spectrum (size nk or [nz, nk])
    zvec : np.ndarray, optional
        Redshift grid for P(k, z). If None, assumed single spectrum.
    n_grid : int, optional
        Size of internal fine grid for r and R. Default 400.
    r_min : float, optional
        Minimum radius for calculations. Default 5e-2.
    r_max : float, optional
        Maximum radius for calculations. Default 120.
    method : {'leggauss', 'trapz', 'quad_vec'}, optional
        Numerical integration method for Σ(R, z). Default 'trapz'.
    n_points : int, optional
        Number of points for numerical integration of Σ(R), Rvec. Default 75.
    rmax_integral : float, optional
        Max radius for Abel integration. Default 300.

    Attributes
    ----------
    reval : np.ndarray
        Default projected radius grid.
    xi_rz_interp : LogGridInterpolator
        Interpolator for ξ(r, z).
    sigma_rz_interp : LogGridInterpolator
        Interpolator for Σ(R, z).
    deltasigma_rz_interp : LogGridInterpolator
        Interpolator for ΔΣ(R, z).

    Methods
    -------
    buildAll(R_vals=None, z=None, **sigma_kwargs):
        Compute and cache all interpolators for ξ, Σ, ΔΣ.
    xi(R_vals=None, z=None):
        Compute or interpolate ξ(r, z) at given radii and redshifts.
    sigma_R(R_vals=None, z=None, **kwargs):
        Compute/interpolate Σ(R, z) on the current grid and integration method.
    deltasigma_R(R_vals=None, z=None, **kwargs):
        Compute/interpolate ΔΣ(R, z) for the grid.

    Example
    -------
    >>> from clenspy.halo.twohalo import TwoHaloTerm
    >>> kvec = np.logspace(-3, 1, 100)  # Wavenumber grid
    >>> Pk = np.random.rand(100)  # Example power spectrum
    >>> zvec = np.array([0.0, 0.5, 1.0])  # Redshift grid
    >>> two_halo = TwoHaloTerm(kvec, Pk, zvec=zvec).buildAll()
    >>> R_vals = np.logspace(-2, 1, 50)  # Projected radius grid
    >>> sigma = two_halo.sigma(R_vals, z=0.5)  # Compute Σ(R, z=0.5)
    >>> deltasigma = two_halo.deltasigma(R_vals, z=0.5)  # Compute ΔΣ(R, z=0.5)
    >>> xi = two_halo.xi(R_vals, z=0.5)
    """

    def __init__(
        self,
        kvec: np.ndarray,
        Pk: np.ndarray,
        zvec: Optional[np.ndarray] = None,
        n_grid: int = 400,
        r_max: float = 120.0,
        r_min: float = 5e-2,
        method: str = "trapz",
        n_points: int = 150,
        rmax_integral: float = 300,
    ) -> None:
        self.kvec, self.Pk_grid, self.zvec = prepare_pk_grid(kvec, Pk, zvec)
        self._kfine = np.logspace(-3.0, 5, n_grid)
        self._rfine = np.logspace(-3.0, np.log10(r_max), n_grid)
        self.p_kz = LogGridInterpolator(self.kvec, self.zvec, self.Pk_grid)
        self.reval = np.logspace(np.log10(r_min), np.log10(r_max), 100)
        self.method = method
        self.n_points = n_points
        self.rmax_integral = rmax_integral

    @time_method
    def build_all(self, R_vals=None, z=None, **sigma_kwargs):
        """
        Compute and cache ξ(r, z), Σ(R, z), ΔΣ(R, z) interpolators.
        """
        self.xi(R_vals, z)
        self.sigma(R_vals, z, **sigma_kwargs)
        self.deltasigma(R_vals, z)
        return self

    @default_rvals_z
    @time_method
    def xi(self, R_vals=None, z=None) -> np.ndarray:
        """
        Compute or interpolate ξ(r, z) at given radii and redshifts.
        """
        if hasattr(self, "xi_rz_interp"):
            return self.xi_rz_interp(R_vals, z)

        def xi_at_z(args):
            iz, zval = args
            Pk_at_z = self.p_kz.at_z(zval)
            xi_res = pk_to_xi_fftlog(self._kfine, Pk_at_z(self._kfine), self._rfine)
            return iz, xi_res

        n_z = len(self.zvec)
        n_r = len(self._rfine)
        xi_grid = np.zeros((n_z, n_r), dtype=float)
        with ThreadPoolExecutor() as executor:
            for iz, xi_tmp in executor.map(
                xi_at_z, [(iz, z) for iz, z in enumerate(self.zvec)]
            ):
                xi_grid[iz, :] = xi_tmp
        self.xi_grid = xi_grid
        self.xi_rz_interp = LogGridInterpolator(self._rfine, self.zvec, xi_grid.T)
        return self.xi_rz_interp(R_vals, z)

    @default_rvals_z
    @time_method
    def sigma(self, R_vals=None, z=None, **kwargs) -> np.ndarray:
        """
        Compute/interpolate Σ(R, z) on the current grid and integration method.

        Parameters
        ----------
        R_vals : np.ndarray, optional
            Projected radius values to evaluate Σ(R, z). If None, uses internal
            grid.
        z : float or np.ndarray, optional
            Redshift(s) to evaluate Σ(R, z). If None, uses internal zvec.
        method : {'leggauss', 'trapz', 'quad_vec'}, optional
            Numerical integration method for Σ(R, z). Default is the instance's
            method.
        n_points : int, optional
            Number of points for numerical integration of Σ(R). Default is the 
            instance's n_points.
        rmax_integral : float, optional
            Max radius for Abel integration. Default is the instance's 
            `rmax_integral`.
        """
        if not hasattr(self, "xi_rz_interp"):
            self.xi()

        if not hasattr(self, "sigma_rz_interp"):
            xi_func = lambda r, z_: self.xi_rz_interp(r, z_)
            sigma_grid = compute_sigma_grid(
                xi_func,
                self._rfine,
                self.zvec,
                method=kwargs.pop("method", self.method),
                rmax_integral=kwargs.pop("rmax_integral", self.rmax_integral),
                n_points=kwargs.pop("n_points", self.n_points),
            )
            self.sigma_rz_interp = LogGridInterpolator(
                self._rfine, self.zvec, sigma_grid
            )

        return self.sigma_rz_interp(R_vals, z)

    @default_rvals_z
    @time_method
    def deltasigma(self, R_vals=None, z=None, **kwargs) -> np.ndarray:
        """
        Compute/interpolate ΔΣ(R, z) for the grid.
        """
        if not hasattr(self, "sigma_rz_interp"):
            _ = self.sigma(**kwargs)

        if hasattr(self, "deltasigma_rz_interp"):
            return self.deltasigma_rz_interp(R_vals, z)

        Rvec = self._rfine
        sigma_grid = self.sigma_rz_interp(Rvec, self.zvec).T  # (nz, nR)
        deltasigma = sigma_to_deltasigma_cumtrapz(Rvec, sigma_grid)
        self.deltasigma_rz_interp = LogGridInterpolator(Rvec, self.zvec, deltasigma.T)
        return self.deltasigma_rz_interp(R_vals, z)


def prepare_pk_grid(
    kvec: np.ndarray, Pk: np.ndarray, zvec: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare (kvec, Pk_grid, zvec) with consistent shapes.
    """
    kvec = np.asarray(kvec)
    nk = len(kvec)
    Pk = np.asarray(Pk)
    if zvec is None:
        zvec = np.array([0.0])
        if Pk.ndim == 1:
            Pk_grid = Pk[:, None]
        elif Pk.ndim == 2 and Pk.shape[1] == 1:
            Pk_grid = Pk
        else:
            raise ValueError("If zvec is None, Pk must be 1D or shape (nk, 1).")
    else:
        if np.isscalar(zvec):
            zvec = np.array([zvec])
        zvec = np.asarray(zvec)
        if Pk.ndim == 1:
            Pk_grid = np.tile(Pk[:, None], (1, len(zvec)))
        elif Pk.shape == (len(zvec), nk):
            Pk_grid = Pk.T
        elif Pk.shape == (nk, len(zvec)):
            Pk_grid = Pk
        else:
            raise ValueError("Pk shape must be (nk,), (nk, nz), or (nz, nk).")
    assert np.all(np.diff(zvec) > 0), "zvec must be strictly increasing!"
    if not np.all(np.diff(kvec) > 0):
        sort_idx = np.argsort(zvec)
        zvec = zvec[sort_idx]
        if Pk_grid.shape == (nk, len(zvec)):
            Pk_grid = Pk_grid[:, sort_idx]
        elif Pk_grid.shape == (len(zvec), nk):
            Pk_grid = Pk_grid[sort_idx, :].T
        else:
            raise ValueError("Shape of Pk_grid does not match kvec and zvec")
    return kvec, Pk_grid, zvec


__all__ = ["TwoHaloTerm"]
