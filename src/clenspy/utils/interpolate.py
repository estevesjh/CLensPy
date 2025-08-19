from typing import Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d


class LogGridInterpolator:
    """
    Log-linear grid interpolator:
      - Interpolates log(values) over
      log(x) (axis 0),
        z (axis 1, linear, or single value).
      - Handles the case where zvec is None (scalar mode).
      - Masks out bad (<=0 or nan/inf) values.
      - On __call__, clips output to [minval, maxval].
    """

    def __init__(
        self,
        xvec: np.ndarray,
        zvec: np.ndarray = None,
        values: np.ndarray = None,
        minval: float = 1e-128,
        maxval: float = 1e128,
    ) -> None:
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
    Create a log-log interpolation.

    The output is not logarithmic, but the interpolation is done in log space.

    Extrapolates linearly in log space, ensuring that the interpolation
    behaves well for large and small values of x.
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


__all__ = ["LogGridInterpolator"]
