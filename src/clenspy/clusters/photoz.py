"""Photo-z kernel from the DES Y3 z-kernel table.

Loads the vendored ``z_kernel_5perc_ext_z01.txt`` and defines

.. math::

    \\sigma_z(z) = \\frac{1}{100\\sqrt{{\\rm sig}(z)}}, \\qquad
    w_z(z, z^{\\rm ob}) = \\max\\!\\left[1 -
        \\left(\\frac{z - z^{\\rm ob}}{\\sigma_z(z)}\\right)^2, 0\\right]

— the parabolic line-of-sight weight of the Costanzi 2026 projection
model (support :math:`|z - z^{\\rm ob}| < \\sigma_z`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import bisect

__all__ = ["sigma_z", "w_z", "z_support"]

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_Z_KERNEL_FILE = "z_kernel_5perc_ext_z01.txt"

_cache: dict = {}


def _kernel_spline():
    if "spl" not in _cache:
        z_red, sig = np.loadtxt(_DATA_DIR / _Z_KERNEL_FILE, unpack=True)
        _cache["spl"] = InterpolatedUnivariateSpline(
            z_red, 1.0 / 100.0 / np.sqrt(sig), k=1, ext=3
        )
    return _cache["spl"]


def sigma_z(z):
    """Photo-z kernel half-width in z-space."""
    return _kernel_spline()(np.asarray(z, dtype=float))


def w_z(z, zob):
    """Parabolic kernel with support ``|z - zob| < sigma_z(z)``."""
    z = np.asarray(z, dtype=float)
    u = (z - zob) / sigma_z(z)
    return np.where(np.abs(u) < 1.0, 1.0 - u * u, 0.0)


def z_support(zob: float) -> tuple[float, float]:
    """(z_lo, z_hi) bounds of the w_z support around ``zob`` (bisect on
    the table; symmetric fallback near the table edges)."""
    try:
        z_lo = float(bisect(lambda zz: zz + sigma_z(zz) - zob, -2.0, 2.0))
        z_hi = float(bisect(lambda zz: zz - sigma_z(zz) - zob, -2.0, 2.0))
    except ValueError:
        sig = float(sigma_z(zob))
        z_lo, z_hi = max(0.01, zob - sig), zob + sig
    return z_lo, z_hi
