r"""Line-of-sight quadrature for the projection integrals.

The exact chord is :math:`r = R_\perp\cosh u` with :math:`|d\chi| = r\,du`;
an exclusion radius is an interval boundary between two smooth batched
intervals, never a mask sampled at quadrature nodes. The split of labour:
`LosGeometry` owns the chord and the interval limits, `integrate_los` owns
the Gauss--Legendre rule and the Jacobian, and the *integrand* — all of the
physics — is a plain callable supplied by the caller::

    geom = LosGeometry(thetas, chi_o, chi_min, chi_max, r_excl)
    W = integrate_los(geom, integrand, n_u, interval="outside")

NOTE: unit-agnostic machinery -- ``chi``, ``R_perp`` and bounds carry the
caller's units (comoving Mpc in package use); physics enters only through
the integrand callable.
"""

from __future__ import annotations

import numpy as np

from .integrate import gl_nodes, gl_nodes_batched

__all__ = ["LosGeometry", "integrate_los", "field_integrand",
           "theta_breakpoint_grid"]


def theta_breakpoint_grid(chi_o, theta_min, theta_max_factor, R, n_per_seg,
                          r_excl=0.0):
    r""":math:`\theta` quadrature nodes and weights [rad] for
    :math:`\int f(\theta)\,d\theta`, log-Gauss-Legendre on segments split
    at feature breakpoints rather than one uniform log grid.

    :math:`\Sigma_{\rm mis}(R, s\mid M)` (:math:`s=\theta\chi_o`) peaks
    sharply at :math:`s\approx R` -- a ring of width :math:`\sim r_s` --
    and :math:`\xi_{\rm NL}` couples :math:`s` and the line-of-sight
    position through the exact chord, so a naive uniform grid needs many
    more nodes than a grid that simply puts one exactly where the
    feature is. Lower bound ``theta_min`` is a fixed numerical floor;
    the upper bound is ``theta_max_factor`` times the largest requested
    :math:`\theta_R=R/\chi_o`, not a fixed angle or a fixed physical
    radius -- it adapts to whatever ``R`` batch is actually being
    queried, rather than blowing up the RND channel's untruncated-NFW
    tail (unconverged all the way to a fixed cutoff, let alone to
    :math:`\pi`) or clipping the CL channel's ring feature at small R.
    Breakpoints: ``theta_min``, that upper bound, the exclusion tangency
    :math:`\arcsin(r_{\rm excl}/\chi_o)` and twice that (the ``b_sel``
    sigmoid's own transition scale), and :math:`\theta_R` for *every*
    requested ``R`` -- one grid shared across the whole ``R`` batch, not
    rebuilt per point. ``n_per_seg`` Gauss-Legendre nodes are laid on
    each segment in :math:`\ln\theta` (so ``d\theta = \theta\,d\ln
    \theta`` is folded into the returned weights already)."""
    R = np.atleast_1d(np.asarray(R, dtype=float))
    theta_R = R / chi_o
    theta_lo, theta_hi = theta_min, theta_max_factor * float(np.max(theta_R))
    breakpoints = {theta_lo, theta_hi}
    breakpoints.update(theta_R.tolist())
    if 0.0 < r_excl < chi_o:
        theta_tan = np.arcsin(r_excl / chi_o)
        if theta_lo < theta_tan < theta_hi:
            breakpoints.add(theta_tan)
            breakpoints.add(min(2.0 * theta_tan, theta_hi))
    breakpoints = sorted(b for b in breakpoints if theta_lo <= b <= theta_hi)
    # dedupe breakpoints too close together for a well-conditioned segment
    clean = [breakpoints[0]]
    for b in breakpoints[1:]:
        if b > clean[-1] * (1.0 + 1e-6):
            clean.append(b)

    thetas, weights = [], []
    for a, b in zip(clean[:-1], clean[1:]):
        u, w_u = gl_nodes(np.log(a), np.log(b), n_per_seg)
        th = np.exp(u)
        thetas.append(th)
        weights.append(w_u * th)
    return np.concatenate(thetas), np.concatenate(weights)


class LosGeometry:
    r"""Cosh--Abel chord and interval limits for every :math:`\theta` cell.

    Foreground and background line-of-sight branches are stacked into
    ``2 n_theta`` rows; on each, the exact law-of-cosines chord is
    :math:`r = R_\perp\cosh u`, :math:`\chi = \chi_0 + {\rm sign}\,
    R_\perp\sinh u`. An exclusion sphere of radius ``r_excl`` around the
    cluster is the boundary ``u_split`` between two smooth intervals —
    ``"inside"`` :math:`(u_{\min}, u_{\rm split})` and ``"outside"``
    :math:`(u_{\rm split}, u_{\max})` — so no quadrature node ever samples
    the discontinuity (``r_excl = 0`` leaves the inside interval empty).

    Attributes are ``(2 n_theta,)`` arrays: ``R_perp``, ``chi_0``,
    ``sign``, ``theta_index``, ``u_min``, ``u_split``, ``u_max``.
    """

    def __init__(self, thetas, chi_o, chi_min, chi_max, r_excl=0.0):
        thetas = np.asarray(thetas, dtype=float)
        R_perp = chi_o * np.sin(thetas)
        chi_0 = chi_o * np.cos(thetas)

        u_min_plus = np.arcsinh(np.maximum(chi_min - chi_0, 0.0) / R_perp)
        u_max_plus = np.arcsinh(np.maximum(chi_max - chi_0, 0.0) / R_perp)
        u_min_minus = np.arcsinh(np.maximum(chi_0 - chi_max, 0.0) / R_perp)
        u_max_minus = np.arcsinh(np.maximum(chi_0 - chi_min, 0.0) / R_perp)

        u_ex = np.zeros_like(R_perp)
        if r_excl > 0.0:
            intersects = R_perp < r_excl
            u_ex[intersects] = np.arccosh(r_excl / R_perp[intersects])

        self.n_theta = thetas.size
        self.R_perp = np.concatenate([R_perp, R_perp])
        self.chi_0 = np.concatenate([chi_0, chi_0])
        self.sign = np.concatenate([np.ones(self.n_theta),
                                    -np.ones(self.n_theta)])
        self.theta_index = np.concatenate([np.arange(self.n_theta),
                                           np.arange(self.n_theta)])
        self.u_min = np.concatenate([u_min_plus, u_min_minus])
        self.u_max = np.concatenate([u_max_plus, u_max_minus])
        self.u_split = np.clip(np.concatenate([u_ex, u_ex]),
                               self.u_min, self.u_max)

    def bounds(self, interval: str):
        """(u_lo, u_hi) of one smooth interval, ``"inside"`` or
        ``"outside"`` the exclusion sphere."""
        if interval == "inside":
            return self.u_min, self.u_split
        if interval == "outside":
            return self.u_split, self.u_max
        raise ValueError(
            f"interval must be 'inside' or 'outside', got {interval!r}")

    def fold(self, W):
        """Sum the foreground and background branches onto the theta axis
        (last axis ``2 n_theta`` -> ``n_theta``)."""
        n = self.n_theta
        return W[..., :n] + W[..., n:]


def integrate_los(geometry: LosGeometry, integrand, n_u: int,
                  interval: str = "outside"):
    r"""One smooth interval of the cosh--Abel projection, Gauss--Legendre
    in :math:`u` per branch row.

    ``integrand(r, chi, theta_index)`` receives the ``(n_branch, n_u)``
    node arrays on the exact chord and returns the full physical integrand,
    shape ``(..., n_branch, n_u)`` — any leading axes (e.g. mass) broadcast
    through. The exact Jacobian :math:`|d\chi| = r\,du` is applied here and
    the foreground and background branches are summed, so the result is the
    per-theta-cell line-of-sight integral, shape ``(n_theta, ...)``.
    """
    u_lo, u_hi = geometry.bounds(interval)
    u, w_u = gl_nodes_batched(u_lo, u_hi, n_u)
    R_perp = geometry.R_perp[:, None]
    r = R_perp * np.cosh(u)
    chi = (geometry.chi_0[:, None]
           + geometry.sign[:, None] * R_perp * np.sinh(u))
    W = np.einsum("...bk,bk->...b", integrand(r, chi, geometry.theta_index),
                  r * w_u)
    return np.moveaxis(geometry.fold(W), -1, 0)


def field_integrand(distance, hmf, common, Ms, M_weight):
    r"""The bias/xi-free line-of-sight weight, as an ``integrand(r, chi,
    theta_index)`` for `integrate_los`: :math:`{\rm common}(z)/(d\chi/dz)\,
    n(M,z)\,M\,d\ln M`, shape ``(n_M, n_branch, n_u)``.

    ``common(z)`` is the caller's own :math:`z`-density (comoving volume
    element times whatever photo-z weight applies); ``hmf`` is
    :math:`n(M,z)` [mass^-1 length^-3]; ``Ms``/``M_weight`` are
    `clenspy.utils.integrate.mass_nodes`. Ignores ``theta_index`` -- this
    weight has no :math:`\theta` dependence on its own. Shared by
    `clenspy.lensing.projection.SigmaPrj` (the background channel) and
    `clenspy.selection.bsel.SelBiasEngine` (:math:`\mathcal P[1]`'s
    :math:`z`-density) -- the one piece of the two modules' line-of-sight
    integrals that is identical, not merely analogous.
    """
    def integrand(r, chi, theta_index):
        z = distance.z_of_chi(chi)
        w = (common(z.ravel()) / distance.dchi_dz(z.ravel())).reshape(z.shape)
        n_M = hmf(Ms, z.ravel()).reshape(Ms.size, *z.shape)
        return w * n_M * M_weight[:, None, None]
    return integrand


