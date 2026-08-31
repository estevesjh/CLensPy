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

from .integrate import gl_nodes_batched

__all__ = ["LosGeometry", "integrate_los", "shell_masses", "tail_masses",
           "theta_edges", "theta_grid"]


def theta_edges(chi_o, theta_range, n_theta, r_excl=0.0):
    r""":math:`\theta` cell edges [rad] -- log-spaced, ``n_theta + 1``
    without exclusion. With ``r_excl > 0`` the tangency angle
    :math:`\arcsin(r_{\rm excl}/\chi_o)` is inserted as one extra edge (the
    point where the inside interval shrinks to zero); the exclusion curve
    itself is handled exactly by the cosh--Abel interval split."""
    edges = np.geomspace(*theta_range, n_theta + 1) / chi_o
    if 0.0 < r_excl < chi_o:
        theta_tan = np.arcsin(r_excl / chi_o)
        if edges[0] < theta_tan < edges[-1]:
            edges = np.sort(np.append(edges, theta_tan))
    return edges


def theta_grid(edges):
    r"""Cell centres (log-mean of the `theta_edges`) and the per-cell
    spherical-measure correction :math:`\sin\bar\theta/\bar\theta`."""
    centres = np.sqrt(edges[:-1] * edges[1:])
    return centres, np.sin(centres) / centres


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


def shell_masses(R, s_edges, rs, sigma0, mean_sigma, which, n_gl: int = 4):
    r"""Mass of the offset profile in each shell of ``s_edges``, shape
    ``(n_shell, n_halo, n_R)``.

    Exact integration by parts: the enclosed mass of the halo offset
    by ``R`` gives :math:`\int_{s_1}^{s_2} 2\pi s\,\Sigma_{\rm mis}\,ds =
    \pi\Sigma_0[s^2\hat m]_{s_1}^{s_2}` (``which="sigma"``); for the
    signed ``"ds"`` case the smooth aperture-mean term takes per-shell
    Gauss--Legendre nodes and the same exact shell mass is subtracted.
    ``mean_sigma(x, x_mis)`` is the dimensionless mean enclosed surface
    density :math:`\bar\Sigma_{\rm mis}/\Sigma_0`, broadcasting over both
    arguments.
    """
    from .integrate import gl_nodes

    R = np.atleast_1d(np.asarray(R, dtype=float))
    s_edges = np.asarray(s_edges, dtype=float)
    n_t, n_m = s_edges.size - 1, rs.size
    masses = np.empty((n_t, n_m, R.size))
    x_gl, w_gl = gl_nodes(0.0, 1.0, n_gl)
    width = np.diff(s_edges)
    s_j = s_edges[:-1, None] + width[:, None] * x_gl[None, :]   # (n_t, n_gl)
    for im in range(n_m):
        # exact shell masses of the halo offset by R (symmetry)
        m_edges = (s_edges[:, None] ** 2) * mean_sigma(
            s_edges[:, None] / rs[im], R[None, :] / rs[im]
        )
        shell = np.pi * sigma0[im] * np.diff(m_edges, axis=0)
        if which == "sigma":
            masses[:, im, :] = shell
        else:
            # smooth term 2 pi s SigmaBar_mis(<R | s): curved on the r_s
            # scale near s ~ R, so per-shell GL nodes rather than an edge
            # trapezoid (-5% in DeltaSigma at R = 8 at test resolution)
            mh = mean_sigma(R[None, None, :] / rs[im],
                            s_j[:, :, None] / rs[im])           # (t, j, r)
            smooth = (2.0 * np.pi * sigma0[im] * width[:, None]
                      * np.einsum("tj,j,tjr->tr", s_j, w_gl, mh))
            masses[:, im, :] = smooth - shell
    return masses


def tail_masses(R, s_edges, rs, sigma0, r_trunc, fhat, n_phi: int = 16):
    r"""Per-cell mass of the profile **beyond** a halo-centric truncation
    radius, to subtract from the untruncated exact cells.

    The removed tail carries no cusp (:math:`r_{\rm trunc} \gg r_s`), so
    its azimuthal average is smooth and ordinary quadrature converges:
    with :math:`u^2 = R^2 + s^2 - 2Rs\cos\varphi` monotone in
    :math:`\varphi`, the tail sits at :math:`\varphi > \varphi_t`,
    :math:`\cos\varphi_t = (R^2 + s^2 - r_{\rm trunc}^2)/(2Rs)`.
    ``fhat(x)`` is the dimensionless surface-density shape (e.g. the NFW
    :math:`\hat f`). Returns ``(n_cell, n_halo, n_R)``.
    """
    from .integrate import gl_nodes

    x_gl, w_gl = gl_nodes(0.0, 1.0, n_phi)
    # phi_t at every (cell edge, R): 0 = all tail, pi = no tail
    cos_t = ((R[None, :] ** 2 + s_edges[:, None] ** 2 - r_trunc**2)
             / (2.0 * R[None, :] * s_edges[:, None]))
    phi_t = np.arccos(np.clip(cos_t, -1.0, 1.0))           # (n_e, n_R)
    span = np.pi - phi_t
    phi = phi_t[..., None] + span[..., None] * x_gl        # (n_e,n_R,n_phi)
    u = np.sqrt(np.maximum(
        R[None, :, None] ** 2 + s_edges[:, None, None] ** 2
        - 2.0 * R[None, :, None] * s_edges[:, None, None]
        * np.cos(phi), 0.0,
    ))
    tail = np.zeros((s_edges.size - 1, rs.size, R.size))
    for im in range(rs.size):
        # azimuth-averaged tail: (1/pi) int_{phi_t}^pi. Weighting the arc
        # length instead discards the cusp-concentrated ring mass.
        sig_tail = (sigma0[im] / np.pi) * np.einsum(
            "erp,p,er->er", fhat(u / rs[im]), w_gl, span)
        g = 2.0 * np.pi * s_edges[:, None] * sig_tail      # (n_e, n_R)
        tail[:, im, :] = (0.5 * (g[1:] + g[:-1])
                          * np.diff(s_edges)[:, None])
    return tail
