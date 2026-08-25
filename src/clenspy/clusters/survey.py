"""Survey-footprint models Omega(z) in steradians.

The effective solid angle of an optical cluster catalogue drops at high z
as the red-sequence photometric contrast degrades.  Ported from
``y3_cluster_cpp/src/models/omega_z_{sdss,des}.hh`` (via the
RichnessSelection transcription):

- ``omega_z_sdss(z)``: SDSS redMaPPer-v5.10 volume-limited polynomial fit
  (Costanzi 2019b / 2021), ~3.13 sr plateau over z in [0.1, 0.4].
- ``omega_z_des(z)``: DES Y1 three-piece fit, ~0.45 sr plateau below
  z ~ 0.5 with a sharp cutoff above z ~ 0.7.
- ``omega_z_const_factory(area)``: constant-Omega callable.

All return steradians and accept scalar or ndarray ``z`` — drop-in
callables for the ``omega_z=`` argument of the weight builders.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEG2_TO_STER",
    "omega_z_sdss",
    "omega_z_des",
    "omega_z_const_factory",
]

DEG2_TO_STER = (np.pi / 180.0) ** 2

_SDSS_COEFFS = np.array([
    -1.14293122e+05, 5.96846869e+04, 9.24239180e+03, -2.23118813e+03,
    -4.52580713e+03, 1.18404878e+03, 1.27951911e+02, -5.05716847e+01,
    1.01744577e+00, -3.11253383e-01, 5.48481084e-03, 3.12629987e+00,
])  # highest degree first, applied to (z - 0.2)

_DES_FIT1 = np.array([0.0, 0.0, 0.0, -0.00262353, 0.01940118, 0.45133063])
_DES_FIT2 = np.array([
    1.33647377e4, 1.35291046e3, -1.26204891e2, -2.83454918e1,
    -2.26465905, 3.84958753e-1,
])  # input (z - 0.6)
_DES_FIT3 = np.array([0.0, 0.0, -1.88101967, 4.8071839, -4.11424324, 1.18196785])


def omega_z_sdss(z):
    """SDSS polynomial fit, Omega(z) [sr]; valid z in ~[0.05, 0.60]."""
    z = np.asarray(z, dtype=float)
    return np.polyval(_SDSS_COEFFS, z - 0.2)


def omega_z_des(z):
    """DES Y1 three-piece polynomial fit, Omega(z) [sr]."""
    z = np.asarray(z, dtype=float)
    v1 = np.polyval(_DES_FIT1, z)
    v2 = np.polyval(_DES_FIT2, z - 0.6)
    v3 = np.polyval(_DES_FIT3, z)
    return np.where(z < 0.504, v1, np.where(z < 0.7, v2, v3))


def omega_z_const_factory(area_sr: float):
    """Return a constant-Omega callable: ``omega(z) = area_sr`` [sr]."""

    def omega_z(z):
        return np.full_like(np.asarray(z, dtype=float), float(area_sr))

    return omega_z
