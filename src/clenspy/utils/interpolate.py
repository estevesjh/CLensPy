from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d


class LogGridInterpolator:
    r"""
    Log-linear grid interpolator.

    - Interpolates log(values) over log(x) (axis 0) and z (axis 1, linear,
      or a single value).
    - Handles the case where zvec is None (scalar mode).
    - Masks out bad (<=0 or nan/inf) values.
    - On __call__, clips output to [minval, maxval].

    NOTE: **unit-agnostic** -- this is machinery, not physics. ``xvec`` and
    ``values`` carry whatever units the caller supplies and come back in
    the same ones. It is listed here only so that no class in the package
    is silent about units.

    NOTE: interpolating in log(values) means the interpolant is **strictly
    positive**, and non-positive inputs are masked rather than
    interpolated. A signed quantity -- the miscentered
    :math:`\Delta\Sigma`, for instance -- must not be passed through
    this class; its negative lobe would be discarded silently. See
    ``docs/miscentering_math.md`` section 9.2.
    """

    def __init__(
        self,
        xvec: np.ndarray,
        zvec: np.ndarray = None,
        values: np.ndarray = None,
        minval: float = 1e-128,
        maxval: float = 1e128,
    ) -> None:
        """
        Parameters
        ----------
        xvec : np.ndarray
            Grid values along the interpolated (log) axis; must be > 0.
        zvec : np.ndarray, optional
            Grid values along the second (linear) axis. If None or a
            single value, the interpolator operates in scalar-z mode.
        values : np.ndarray
            Function values on the grid: shape ``(len(xvec),)`` or
            ``(len(xvec), 1)`` in scalar-z mode, else ``(len(xvec),
            len(zvec))`` or ``(len(zvec), len(xvec))`` (auto-transposed).
        minval, maxval : float, optional
            Values below ``minval`` are clipped to 0; above ``maxval`` to
            `inf` (default: 1e-128, 1e128).
        """
        x = np.asarray(xvec)
        if zvec is None or (
            np.ndim(zvec) == 0
            or (isinstance(zvec, (np.ndarray, list)) and np.size(zvec) == 1)
        ):
            y = np.array([0.0]) if zvec is None else np.atleast_1d(zvec)
            values = np.asarray(values)
            # If values is 1D, convert to 2D (nk, 1)
            if values.ndim == 1:
                values = values[:, None]
        else:
            y = np.asarray(zvec)
            values = np.asarray(values)
            # If values is (nz, nk), transpose to (nk, nz)
            if values.shape == (len(y), len(x)):
                values = values.T
            elif values.shape != (len(x), len(y)):
                raise ValueError(
                    f"Shape of values {values.shape} is incompatible"
                    + f"with xvec ({len(x)}) and zvec ({len(y)})"
                )

        mask = valid_mask_2d(values)
        logx = np.log(x)
        logvalues = np.full_like(values, np.nan, dtype=float)
        logvalues[mask] = np.log(values[mask])

        self._interp = RegularGridInterpolator(
            (logx, y), logvalues, bounds_error=False, fill_value=None, method="linear"
        )
        self.minval = minval
        self.maxval = maxval
        self.xvec = x
        self.zvec = y

    def __call__(
        self, x: float | np.ndarray, z: float | np.ndarray = None
    ) -> float | np.ndarray:
        xarr = np.atleast_1d(x)
        # If z is None, assume scalar mode
        if z is None:
            zarr = self.zvec
        else:
            zarr = np.atleast_1d(z)
        scalar_input = np.isscalar(x) and (z is None or np.isscalar(z))
        # Pairwise evaluation (x[i], z[i])
        if xarr.shape == zarr.shape and xarr.ndim == 1 and xarr.size > 1:
            pts = np.column_stack((np.log(xarr), zarr))
            logvals = self._interp(pts)
            xi_eval = np.exp(logvals)
            return xi_eval if not scalar_input else float(xi_eval.squeeze())
        # Otherwise, full meshgrid
        logx = np.log(xarr)
        pts = np.array(np.meshgrid(logx, zarr, indexing="ij")).reshape(2, -1).T
        logvals = self._interp(pts)
        xi_eval = np.exp(logvals)
        xi_eval = np.where(np.isnan(xi_eval), 0.0, xi_eval)
        xi_eval = np.where(xi_eval > self.maxval, np.inf, xi_eval)
        xi_eval = xi_eval.reshape(logx.size, zarr.size)
        if scalar_input:
            return float(xi_eval.squeeze())
        return xi_eval.squeeze()

    def at_z(self, z0):
        """Return a 1D function of x at fixed z0."""
        return lambda x: self(x, z0)


def make_log_interpolation(
    xgrid: np.ndarray, ygrid: np.ndarray, minval: float = 1e-128, maxval: float = 1e128
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a log-log interpolation function of a single variable.

    The returned function's output is not logarithmic - only the
    interpolation itself is done in log space (linear interpolation of
    log(y) vs. log(x)), extrapolating linearly in log space beyond the
    input range. Non-finite or non-positive ``ygrid`` points are dropped
    before fitting.

    Parameters
    ----------
    xgrid, ygrid : np.ndarray
        Grid to interpolate; ``xgrid`` must be > 0.
    minval, maxval : float, optional
        Output below ``minval`` is clipped to 0; above ``maxval`` to `inf`
        (default: 1e-128, 1e128).

    Returns
    -------
    callable
        Function ``f(r)`` returning the interpolated value(s) at ``r``.
    """
    # Only use valid, positive values
    mask = np.isfinite(ygrid) & (ygrid > 0)
    log_x = np.log(xgrid[mask])
    log_y = np.log(ygrid[mask])
    loginterp = interp1d(log_x, log_y, kind="linear", fill_value="extrapolate")

    def myInterpFunction(r):
        log_r_eval = np.log(r)
        log_xi_eval = loginterp(log_r_eval)
        xi_eval = np.exp(log_xi_eval)
        # Apply min/max logic for extrapolation
        xi_eval = np.where(xi_eval < minval, 0.0, xi_eval)
        xi_eval = np.where(xi_eval > maxval, np.inf, xi_eval)
        if np.isscalar(r):
            return float(xi_eval)
        return xi_eval

    return myInterpFunction


def valid_mask_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    mask = np.isfinite(values) & (values > 0)
    return mask


__all__ = ["LogGridInterpolator", "make_log_interpolation"]


if __name__ == "__main__":
    import numpy as np

    x = np.logspace(-2, 1, 40)
    z = np.array([0.0, 0.5, 1.0])
    # a separable power law, so the log-log interpolant is exact
    values = np.outer(x**-1.5, 1.0 / (1.0 + z))

    interp = LogGridInterpolator(xvec=x, zvec=z, values=values)
    xq = np.array([0.05, 0.5, 5.0])
    print("LogGridInterpolator on a separable power law (exact case)")
    for zq in (0.0, 0.25, 1.0):
        got = np.ravel(interp(xq, zq))
        exact = xq**-1.5 / (1.0 + zq)
        print(f"  z={zq:.2f}  max rel err = {np.max(np.abs(got / exact - 1)):.2e}")

    print("\nscalar-z mode (zvec=None):")
    flat = LogGridInterpolator(xvec=x, values=x**-1.5)
    print("  ", np.ravel(flat(xq)))

    print("\nNOTE: interpolation is in log(values), so the result is")
    print("      strictly positive and non-positive input is masked. A")
    print("      signed quantity must not be passed through this class.")
