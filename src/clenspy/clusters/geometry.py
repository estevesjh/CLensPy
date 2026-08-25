"""redMaPPer aperture geometry (Costanzi 2026 projection model).

Physical units: the richness radius is

.. math::

    R_\\lambda = (\\lambda / 100)^{0.2}\\; h^{-1}\\,{\\rm Mpc}
    \\quad (\\text{physical}),

converted to physical Mpc at construction (pass ``h``).  Angles are
computed from comoving quantities:
:math:`\\theta_\\lambda = R_\\lambda (1+z) / \\chi(z)`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["r_lambda", "theta_lambda", "area_overlap", "sigmoid_theta"]


def r_lambda(lam, h: float):
    r"""Richness radius :math:`R_\lambda` in physical Mpc."""
    lam = np.asarray(lam, dtype=float)
    return (lam / 100.0) ** 0.2 / h


def theta_lambda(lam, z, chi, h: float):
    r"""Angular size of the :math:`\lambda`-disk [rad].

    ``chi`` is the comoving distance to ``z`` in Mpc (scalar or callable).
    """
    chi_val = chi(z) if callable(chi) else chi
    return r_lambda(lam, h) * (1.0 + np.asarray(z, dtype=float)) / chi_val


def area_overlap(theta, theta_lob, theta_ltr):
    """Fractional overlap of two circular apertures, normalized by the
    projector's disk area pi*theta_ltr^2 (Costanzi 2026 convention).

    ``theta``: separations [rad], shape (Nth,) or (..., N_ltr);
    ``theta_lob``: scalar target radius; ``theta_ltr``: projector radii,
    shape (N_ltr,).  Returns shape broadcast(theta[..., None], theta_ltr).
    """
    theta = np.asarray(theta, dtype=float)
    theta_ltr = np.atleast_1d(np.asarray(theta_ltr, dtype=float))
    if theta.ndim >= 1 and theta.shape[-1] == theta_ltr.shape[-1]:
        theta_b = np.array(theta, dtype=float)
        ltr_b = np.broadcast_to(theta_ltr, theta_b.shape).copy()
    else:
        theta_b, ltr_b = np.broadcast_arrays(theta[..., None], theta_ltr)
        theta_b = np.array(theta_b, dtype=float)
        ltr_b = np.array(ltr_b, dtype=float)
    A = np.ones_like(theta_b)

    # no overlap
    A[theta_b > theta_lob + ltr_b] = 0.0
    # full containment of the target inside a bigger projector
    mask_full = ltr_b > theta_lob
    A[mask_full] = theta_lob**2 / ltr_b[mask_full] ** 2
    # partial overlap (lens formula)
    cond = theta_b > np.abs(theta_lob - ltr_b)
    if np.any(cond):
        tt = theta_b[cond]
        ll = ltr_b[cond]
        arg1 = np.clip(
            (tt**2 + ll**2 - theta_lob**2) / (2.0 * tt * ll), -1.0, 1.0
        )
        arg2 = np.clip(
            (tt**2 + theta_lob**2 - ll**2) / (2.0 * tt * theta_lob), -1.0, 1.0
        )
        argsqrt = np.clip(
            (-tt + ll + theta_lob)
            * (tt + ll - theta_lob)
            * (tt - ll + theta_lob)
            * (tt + ll + theta_lob),
            0.0,
            None,
        )
        A[cond] = (
            ll**2 * np.arccos(arg1)
            + theta_lob**2 * np.arccos(arg2)
            - 0.5 * np.sqrt(argsqrt)
        ) / (np.pi * ll**2)
    return A


def sigmoid_theta(theta, theta_lob, damping: float = 2.5,
                  theta0_frac: float = 0.5):
    r"""Sigmoid transition :math:`\sigma(\theta) = [1 + e^{-k(\theta -
    \theta_0)}]^{-1}` with :math:`k = {\rm damping}/\theta_\lambda`,
    :math:`\theta_0 = \theta_\lambda/2` (Costanzi 2026 eq. 6)."""
    k = damping / theta_lob
    theta0 = theta0_frac * theta_lob
    return 1.0 / (1.0 + np.exp(-k * (np.asarray(theta, dtype=float) - theta0)))
