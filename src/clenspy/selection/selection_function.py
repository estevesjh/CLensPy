r"""The selection function :math:`\mathcal S_{ij}(M, z^{\rm tr})`.

The probability that a halo of mass :math:`M` at true redshift
:math:`z^{\rm tr}` is *observed* inside richness bin :math:`i` and
redshift bin :math:`j`. This is the factor that turns a halo mass function
into a prediction for a catalogue.

The five-dimensional integral of the forward model,

.. math::
    \langle N_{ij}\rangle = \int dM \int dz^{\rm tr}\int d\lambda^{\rm tr}
      \int d\lambda^{\rm ob}\int dz^{\rm ob}\;(\cdots)

collapses to two dimensions because three of the five are analytic:

- the :math:`z^{\rm ob}` integral is a Gaussian CDF difference
  (`clenspy.kernels.photoz.photoz_counts`);
- the :math:`\lambda^{\rm ob}` integral is the EMG CDF difference
  (`clenspy.selection.richness_kernel`);
- only :math:`\lambda^{\rm tr}` is left, and it is one Gauss--Legendre
  rule.

.. math::
    \boxed{\;
    \mathcal S_{ij}(M, z^{\rm tr}) = S_i(M, z^{\rm tr})\,
                                     \mathcal S_j(z^{\rm tr})\;}

.. math::
    S_i(M,z) = \int_0^\infty d\lambda^{\rm tr}\,
               \mathcal S_i(\lambda^{\rm tr},z)\,
               P(\lambda^{\rm tr}\mid M,z)
             \approx \sum_{k=1}^{N_q}\frac{b-a}{2}\,w_k\,
               \mathcal S_i(\lambda_k,z)\,P(\lambda_k\mid M,z)

NOTE: **the factorisation is exact, and it is a property of the photo-z
kernel, not an approximation.** :math:`\mathcal S_{ij}` separates because
:math:`P(z^{\rm ob}\mid z^{\rm tr})` is taken independent of
:math:`\lambda^{\rm tr}` -- the redshift kernel depends on the richness bin
only through :math:`\sigma_z(\Delta\lambda_i)`, a per-bin constant. If
:math:`\sigma_z` ever depends on :math:`\lambda^{\rm tr}` itself the
product form fails and the two integrals no longer commute.

NOTE: **the bracket is the approximation, and it is the one that bites.**
:math:`\lambda^{\rm tr}` runs over :math:`(0,\infty)` but the quadrature
covers :math:`[a, b] = [\max(0, \mu_{\rm eff} - L\sigma_{\rm eff}),\;
\mu_{\rm eff} + L\sigma_{\rm eff}]`. Too narrow and probability is
silently discarded; the y3 pipeline widened its own bracket from
:math:`[\lambda/4,\,2\lambda_{\max}]` to
:math:`[\lambda/8,\,4\lambda_{\max}]` precisely because the
:math:`{\rm HMF}\times S_i` tail matters even where :math:`S_i` does not
peak. `bracket_width` defaults to 8 and `SelectionFunction.residual`
exists so a caller can *measure* what the bracket misses rather than
trusting it.

NOTE: **units.** Dimensionless throughout -- both :math:`S_i` and
:math:`\mathcal S_j` are probabilities in :math:`[0,1]`, and so is their
product. ``ln_mass`` is :math:`\ln M` with :math:`M` in
:math:`h^{-1}M_\odot`, matching the mass--observable relations; richness is
a dimensionless count.

NOTE: the Gauss--Legendre nodes are computed once at construction and
reused for every :math:`(M, z)` cell. The *bracket* is per-cell, the
*rule* is shared -- which is the only reason this is fast enough to sit
inside a two-dimensional integral inside a sampler.
"""

from __future__ import annotations

import numpy as np

from ..kernels.photoz import photoz_counts
from .richness_kernel import richness_bin_first_moment, richness_bin_probability

__all__ = ["SelectionFunction"]

#: Default half-width of the quadrature bracket, in units of the
#: mass--observable relation's own standard deviation. 8, not 6: see the
#: module NOTE on what the y3 pipeline learned about the tail.
BRACKET_WIDTH = 8.0

#: Default Gauss--Legendre order for the lambda_true integral.
N_QUAD = 64


class SelectionFunction:
    r""":math:`\mathcal S_{ij}(M, z^{\rm tr})`, and its two factors.

    NOTE: units are dimensionless; ``ln_mass`` is :math:`\ln M` with M in
    :math:`h^{-1}M_\odot`. See the module NOTE.

    Parameters
    ----------
    lambda_edges : array-like, shape (n_lambda + 1,)
        Observed-richness bin edges, ascending and contiguous.
    z_edges : array-like, shape (n_z + 1,)
        Observed-redshift bin edges, ascending and contiguous.
    mor : LogNormalMor or HodMor
        The mass--observable relation. Stored verbatim; supplies ``pdf``,
        ``mean`` and ``std``.
    emg_params : EmgParams
        The observed-richness kernel parameters.
    sigma_z : float or array-like
        Photo-z **scatter** (not the 3-sigma window -- see
        `clenspy.kernels.photoz`). A scalar, or one value per redshift bin.
    bracket_width : float, optional
        :math:`L` in :math:`\mu_{\rm eff} \pm L\sigma_{\rm eff}`
        (default: 8).
    n_quad : int, optional
        Gauss--Legendre order (default: 64).
    """

    def __init__(self, lambda_edges, z_edges, mor, emg_params, sigma_z,
                 bracket_width: float = BRACKET_WIDTH,
                 n_quad: int = N_QUAD):
        self.lambda_edges = np.asarray(lambda_edges, dtype=float)
        self.z_edges = np.asarray(z_edges, dtype=float)
        for name, edges in (("lambda_edges", self.lambda_edges),
                            ("z_edges", self.z_edges)):
            if edges.ndim != 1 or edges.size < 2:
                raise ValueError(f"{name} must be 1-D with >= 2 entries")
            if np.any(np.diff(edges) <= 0.0):
                raise ValueError(f"{name} must be strictly ascending")

        self.mor = mor
        self.emg_params = emg_params
        self.bracket_width = float(bracket_width)
        self.n_quad = int(n_quad)

        sigma_z = np.atleast_1d(np.asarray(sigma_z, dtype=float))
        if sigma_z.size not in (1, self.n_z_bins):
            raise ValueError(
                f"sigma_z must be a scalar or one value per redshift bin "
                f"({self.n_z_bins}), got {sigma_z.size}"
            )
        if np.any(sigma_z <= 0.0):
            raise ValueError("sigma_z must be positive")
        self.sigma_z = (np.full(self.n_z_bins, sigma_z[0])
                        if sigma_z.size == 1 else sigma_z)

        # the rule is shared across cells; only the bracket is per-cell
        self._nodes, self._weights = np.polynomial.legendre.leggauss(
            self.n_quad
        )

    @property
    def n_lambda_bins(self):
        return self.lambda_edges.size - 1

    @property
    def n_z_bins(self):
        return self.z_edges.size - 1

    def bracket(self, ln_mass, z):
        r"""``(a, b)``: the :math:`\lambda^{\rm tr}` quadrature bracket.

        .. math::
            a = \max(0,\;\mu_{\rm eff} - L\sigma_{\rm eff}),
            \qquad b = \mu_{\rm eff} + L\sigma_{\rm eff}

        NOTE: clipped at zero, never at the lowest bin edge. Clipping at
        :math:`\lambda_i^{\min}` would be wrong: the projection tail
        scatters low-richness haloes *up* into the bin, so the integrand is
        non-zero well below the bin's own edge. That upscatter is the entire
        physical effect this package exists to model.
        """
        mu = np.asarray(self.mor.mean(ln_mass, z), dtype=float)
        sd = np.asarray(self.mor.std(ln_mass, z), dtype=float)
        half = self.bracket_width * sd
        return np.maximum(mu - half, 0.0), mu + half

    def _nodes_for(self, ln_mass, z):
        """Mapped nodes and weights on this cell's bracket."""
        a, b = self.bracket(ln_mass, z)
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        # shape (..., n_quad)
        lam = mid[..., None] + half[..., None] * self._nodes
        wts = half[..., None] * self._weights
        return lam, wts

    def S_i(self, ln_mass, z):
        r""":math:`S_i(M, z)` for every richness bin.

        Returns shape ``(*broadcast(ln_mass, z).shape, n_lambda_bins)``.
        """
        ln_mass, z = np.broadcast_arrays(
            np.asarray(ln_mass, dtype=float), np.asarray(z, dtype=float)
        )
        lam, wts = self._nodes_for(ln_mass, z)
        # S_i(lambda_k, z) -> (..., n_quad, n_bins)
        kernel = richness_bin_probability(
            self.lambda_edges, lam, z[..., None], self.emg_params
        )
        # P(lambda_k | M, z) -> (..., n_quad)
        pdf = self.mor.pdf(lam, ln_mass[..., None], z[..., None])
        # contract the quadrature axis
        return np.einsum("...q,...q,...qb->...b", wts, pdf, kernel)

    def first_moment_i(self, ln_mass, z):
        r"""``\int d\lambda^{\rm tr}\,P(\lambda^{\rm tr}\mid M,z)\,M_i``,
        for every richness bin.

        The same :math:`\lambda^{\rm tr}` quadrature `S_i` uses, contracted
        against `~clenspy.selection.richness_kernel.richness_bin_first_moment`
        instead of `richness_bin_probability`. Dividing element-wise by
        `S_i` gives the observed-richness mean of haloes of mass :math:`M`
        at redshift :math:`z` that land in bin :math:`i`.

        Returns shape ``(*broadcast(ln_mass, z).shape, n_lambda_bins)``.
        """
        ln_mass, z = np.broadcast_arrays(
            np.asarray(ln_mass, dtype=float), np.asarray(z, dtype=float)
        )
        lam, wts = self._nodes_for(ln_mass, z)
        # M_i(lambda_k, z) -> (..., n_quad, n_bins)
        kernel = richness_bin_first_moment(
            self.lambda_edges, lam, z[..., None], self.emg_params
        )
        # P(lambda_k | M, z) -> (..., n_quad)
        pdf = self.mor.pdf(lam, ln_mass[..., None], z[..., None])
        return np.einsum("...q,...q,...qb->...b", wts, pdf, kernel)

    def S_j(self, z):
        r""":math:`\mathcal S_j(z^{\rm tr})` for every redshift bin.

        Returns shape ``(*np.shape(z), n_z_bins)``.
        """
        z = np.asarray(z, dtype=float)
        return np.stack(
            [photoz_counts(z, self.z_edges[j], self.z_edges[j + 1],
                           self.sigma_z[j])
             for j in range(self.n_z_bins)],
            axis=-1,
        )

    def S_ij(self, ln_mass, z):
        r""":math:`\mathcal S_{ij} = S_i\,\mathcal S_j`.

        Returns shape ``(..., n_lambda_bins, n_z_bins)``.
        """
        return self.S_i(ln_mass, z)[..., :, None] * (
            self.S_j(z)[..., None, :]
        )

    def first_moment_ij(self, ln_mass, z):
        r"""``first_moment_i * S_j``, the :math:`\lambda^{\rm ob}` first
        moment's share of :math:`\mathcal S_{ij}`.

        Factorises exactly like `S_ij`: the redshift factor
        :math:`\mathcal S_j` carries no :math:`\lambda^{\rm ob}`
        dependence, so it multiplies through unchanged. This is the
        numerator `clenspy.observables.ClusterCounts.mean_richness` needs
        -- dividing its :math:`(\ln M, z)` integral by `counts` gives
        :math:`\langle\lambda^{\rm ob}\rangle_{ij}`.

        Returns shape ``(..., n_lambda_bins, n_z_bins)``.
        """
        return self.first_moment_i(ln_mass, z)[..., :, None] * (
            self.S_j(z)[..., None, :]
        )

    def residual(self, ln_mass, z):
        r"""How much :math:`\lambda^{\rm tr}` probability the bracket misses.

        :math:`1 - \int_a^b P(\lambda^{\rm tr}\mid M,z)\,
        d\lambda^{\rm tr}`, computed on the same nodes the quadrature uses.

        NOTE: this is the module's named approximation, made measurable.
        A value of :math:`10^{-6}` says the bracket is fine; :math:`10^{-2}`
        says :math:`S_i` is low by about a percent and `bracket_width`
        needs raising. There is no way to know without evaluating it, which
        is why it is a method and not a comment.
        """
        ln_mass, z = np.broadcast_arrays(
            np.asarray(ln_mass, dtype=float), np.asarray(z, dtype=float)
        )
        lam, wts = self._nodes_for(ln_mass, z)
        captured = np.einsum(
            "...q,...q->...", wts,
            self.mor.pdf(lam, ln_mass[..., None], z[..., None])
        )
        return 1.0 - captured

    def __repr__(self):
        return (f"SelectionFunction({self.n_lambda_bins} richness x "
                f"{self.n_z_bins} redshift bins, {self.mor!r}, "
                f"L={self.bracket_width:g}, n_quad={self.n_quad})")


if __name__ == "__main__":
    from .richness_kernel import EmgParams
    from .scaling_relation import HodMor, LogNormalMor

    lam_edges = np.array([20.0, 30.0, 45.0, 60.0, 200.0])   # DES Y1
    z_edges = np.array([0.20, 0.35, 0.50, 0.65])
    params = EmgParams(delta_mu=-1.5, sigma=3.0, f_prj=0.3, tau=0.12)

    sel = SelectionFunction(lam_edges, z_edges, LogNormalMor(), params,
                            sigma_z=0.01)
    print(sel, "\n")

    print("S_i(M, z=0.3) -- probability of landing in each richness bin:")
    print(f"{'M [h^-1Msun]':>13s}  " + "  ".join(
        f"{f'[{a:.0f},{b:.0f})':>10s}"
        for a, b in zip(lam_edges[:-1], lam_edges[1:]))
        + f"  {'sum':>8s}  {'bracket miss':>13s}")
    for m in (1e13, 5e13, 1e14, 3e14, 1e15):
        lm = np.log(m)
        s = sel.S_i(lm, 0.3)
        print(f"{m:13.1e}  " + "  ".join(f"{v:10.6f}" for v in s)
              + f"  {s.sum():8.6f}  {sel.residual(lm, 0.3):13.2e}")
    print("  <- the sum is the probability of being observed anywhere in")
    print("     [20, 200), and it is NOT monotonic in mass. It peaks near")
    print("     3e14 and falls again by 1e15, because this MOR puts")
    print(f"     <lambda> = {LogNormalMor().mean(np.log(1e15), 0.3).item():.0f}"
          " there -- above the last edge, 200. The")
    print("     binning, not the physics, is what runs out.")

    print("\nS_j(z_tr) -- the redshift factor, sigma_z = 0.01:")
    print(f"{'z_tr':>6s}  " + "  ".join(
        f"{f'[{a:.2f},{b:.2f})':>14s}"
        for a, b in zip(z_edges[:-1], z_edges[1:])))
    for z in (0.20, 0.28, 0.35, 0.42, 0.65):
        print(f"{z:6.2f}  " + "  ".join(
            f"{v:14.6f}" for v in sel.S_j(z)))
    print("  <- 0.5 exactly at a shared edge, and the pair straddling it")
    print("     sums to 1: no cluster is lost between contiguous bins.")

    print("\nS_ij factorises exactly:")
    lm = np.log(3e14)
    sij = sel.S_ij(lm, 0.3)
    outer = np.outer(sel.S_i(lm, 0.3), sel.S_j(0.3))
    print(f"  max|S_ij - S_i x S_j| = {np.max(np.abs(sij - outer)):.2e}")
    print(f"  shape = {sij.shape}  (n_lambda, n_z)")

    print("\nthe bracket is the approximation. What L costs:")
    print(f"{'L':>5s}  {'S_i sum at 1e14':>16s}  {'missed':>10s}")
    for L in (2.0, 4.0, 6.0, 8.0, 12.0):
        s = SelectionFunction(lam_edges, z_edges, LogNormalMor(), params,
                              sigma_z=0.01, bracket_width=L)
        lm = np.log(1e14)
        print(f"{L:5.1f}  {s.S_i(lm, 0.3).sum():16.8f}  "
              f"{s.residual(lm, 0.3):10.2e}")
    print("  <- L = 2 discards percent-level probability silently; the")
    print("     residual is what makes that visible.")

    print("\nthe same bins with the DES Y1 HOD relation instead:")
    sel_hod = SelectionFunction(lam_edges, z_edges, HodMor(), params,
                                sigma_z=0.01)
    for m in (1e14, 3e14, 1e15):
        lm = np.log(m)
        a = sel.S_i(lm, 0.3)
        b = sel_hod.S_i(lm, 0.3)
        print(f"  M = {m:.0e}:  log-normal sum {a.sum():.6f}   "
              f"HOD sum {b.sum():.6f}")
    print("  <- the MOR is a swappable collaborator, not a hard-coded law.")
