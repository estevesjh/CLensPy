r"""redMaPPer aperture geometry: the disk, its angle, and their overlap.

The geometric ingredients of the selection-affected bias. Three of them,
and all three appear in the paper's Section 4.1:

.. math::
    R_\lambda(\lambda) = \left(\frac{\lambda}{100}\right)^{0.2}
        h^{-1}\,{\rm Mpc},
    \qquad
    \theta_\lambda = \frac{R_\lambda\,(1+z)}{\chi(z)}

.. math::
    \sigma(\theta) = \frac{1}{1 + e^{-k(\theta - \theta_0)}},
    \qquad k = \frac{2.5}{\theta_\lambda},
    \qquad \theta_0 = \frac{\theta_\lambda}{2}

plus `area_overlap`, the closed-form fractional overlap of two circular
apertures -- the :math:`f_A` of the projection kernel
:math:`\rho^{\rm prj}`.

NOTE: **units, and the one place they turn over.** :math:`R_\lambda` is
returned in **physical Mpc**: the fit is calibrated in
:math:`h^{-1}{\rm Mpc}`, so `r_lambda` divides by ``h`` once, visibly, and
that is the only h in this module. `theta_lambda` then multiplies by
:math:`(1+z)` to get a *comoving* length and divides by the **comoving**
:math:`\chi` -- the two must match or the angle is wrong by
:math:`(1+z)`. Angles are radians; `area_overlap` and `sigmoid_theta` are
dimensionless.

NOTE: :math:`R_\lambda \propto \lambda^{0.2}` is a redMaPPer *definition*,
not a fit to be varied -- it is the aperture the richness was measured in.
The 100 and the 0.2 are part of the catalogue.

NOTE: `sigmoid_theta`'s :math:`k = 2.5/\theta_\lambda` and
:math:`\theta_0 = \theta_\lambda/2` are **named choices with no
derivation**: they parametrise a transition whose shape is not predicted,
only assumed to be smooth and to turn over near the aperture scale. Both
are exposed as ``damping`` and ``theta0_frac`` so a systematic study can
move them; the defaults are the production values.

NOTE: `area_overlap` is normalised by the **projector's** disk area
:math:`\pi\theta_{\lambda^{\rm tr}}^2`, not the target's. That asymmetry is
the Costanzi convention and it matters: swapping the normalisation changes
:math:`\rho^{\rm prj}` for every pair where the projector is the larger
disk, which is most of them at low target richness.
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
