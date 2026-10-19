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
- only :math:`\lambda^{\rm tr}` is left, and it is a composite
  Gauss--Legendre rule.

.. math::
    \boxed{\;
    \mathcal S_{ij}(M, z^{\rm tr}) = S_i(M, z^{\rm tr})\,
                                     \mathcal S_j(z^{\rm tr})\;}

.. math::
    S_i(M,z) = \int_0^\infty d\lambda^{\rm tr}\,
               \mathcal S_i(\lambda^{\rm tr},z)\,
               P(\lambda^{\rm tr}\mid M,z)
             \approx \sum_p\sum_{k=1}^{N_q}\frac{b_p-a_p}{2}\,w_k\,
               \mathcal S_i(\lambda_{pk},z)\,P(\lambda_{pk}\mid M,z)

where the panels :math:`[a_p, b_p]` tile the bracket :math:`[a, b]`
(NOTE below).

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

NOTE: **the** :math:`\lambda^{\rm tr}` **rule is composite: it restarts
wherever the integrand changes character -- measured, not chosen by
taste.** Gauss--Legendre converges exponentially only on an integrand that
is smooth on the scale of its nodes, and this one is not.

- Every richness edge :math:`\lambda_e` makes
  :math:`\mathcal S_i(\lambda^{\rm tr})` step, centred where
  :math:`\mu(\lambda^{\rm tr}) = \lambda_e`, over the kernel width
  :math:`w_e = \sigma/b_\mu` (`richness_edge_steps`): 3 richness units for
  this module's example kernel, while the bracket of a
  :math:`10^{15}\,h^{-1}M_\odot` halo spans 900. One 64-node rule across
  the whole bracket misses :math:`S_i(M)` of the :math:`[60, 200)` bin by
  up to 1.3% of its peak at :math:`z = 0.4` (:math:`7\times10^{-4}` at
  128 nodes, :math:`7\times10^{-7}` at 256), in a ripple of period
  :math:`\sim 0.3` in :math:`\ln M` as the nodes slide past the step at
  :math:`\lambda^{\rm tr} = 201.5`. No :math:`\ln M` rule averages that
  out: it is what held `clenspy.observables.ClusterCounts` at
  :math:`10^{-3}`.
- `HodMor`'s density jumps from zero at :math:`\lambda^{\rm cen} = 1`
  (`HodMor.support_min`), and a rule across a jump converges only as
  :math:`O(1/n)`: :math:`5\times10^{-4}` of the :math:`[20, 30)` peak at
  64 nodes, still :math:`10^{-4}` at 256, with the same kernel.

So the bracket is cut into panels, each with its own ``n_quad`` nodes
(`panel_edges`): at :math:`\lambda^{\rm tr}_e \pm L_e w_e` around every
edge (:math:`L_e` = ``edge_width``, 5), so each step sits mid-panel and
hands its neighbours a remnant of only :math:`\Phi(-5) = 3\times10^{-7}`;
at the relation's mean, so :math:`P` is integrated in two halves with its
peak at a panel end, where the nodes cluster; and at ``support_min``,
where the relation declares one. Breaking at the steps themselves is not
enough: it leaves a half-step at the end of a panel that spans hundreds of
richness units for a massive halo, :math:`2\times10^{-5}` of the peak at
32 nodes per panel. Against adaptive quadrature -- both relations, the
fixed and the Y3 kernel, three binnings, :math:`z = 0.2, 0.4, 0.6`, 201
masses over :math:`[10^{13}, 10^{15.5}]\,h^{-1}M_\odot` -- the default 20
nodes per panel is within :math:`3\times10^{-9}` of every bin's peak and
:math:`5\times10^{-8}` of :math:`S_i(M)` itself wherever
:math:`S_i > 10^{-5}`, at 145 nodes per cell on average.

NOTE: breaks outside the bracket are clipped to it and leave empty
panels. Their nodes carry zero weight, and the integrand is evaluated only
where the rule has weight -- which also keeps it off
:math:`\lambda^{\rm tr} = 0` exactly, where the Y3 kernel's
:math:`\sigma \propto (\lambda^{\rm tr})^{a_\sigma}` vanishes and
`~clenspy.selection.richness_kernel.emg_cdf` refuses it.

NOTE: **units.** Dimensionless throughout -- both :math:`S_i` and
:math:`\mathcal S_j` are probabilities in :math:`[0,1]`, and so is their
product. ``ln_mass`` is :math:`\ln M` with :math:`M` in
:math:`h^{-1}M_\odot`, matching the mass--observable relations; richness is
a dimensionless count.

NOTE: the Gauss--Legendre rule on :math:`[-1, 1]` is computed once and
cached (`clenspy.utils.integrate.gl_nodes_batched`); every panel of every
:math:`(M, z)` cell is an affine map of it. The *bracket* and its *panels*
are per-cell, the *rule* is shared -- which is the only reason this is
fast enough to sit inside a two-dimensional integral inside a sampler.
"""

from __future__ import annotations

import numpy as np

from ..kernels.photoz import photoz_counts
from ..utils.integrate import gl_nodes_batched
from .richness_kernel import (
    richness_bin_first_moment,
    richness_bin_probability,
    richness_edge_steps,
)

__all__ = ["SelectionFunction"]

#: Default half-width of the quadrature bracket, in units of the
#: mass--observable relation's own standard deviation. 8, not 6: see the
#: module NOTE on what the y3 pipeline learned about the tail.
BRACKET_WIDTH = 8.0

#: Default half-width of the panel around each richness edge's step, in
#: units of the kernel's own width there. 5, not 4 or 6: at 4 the step
#: hands its neighbour a :math:`\Phi(-4) = 3\times10^{-5}` remnant and the
#: error floors near :math:`10^{-6}`; at 6 the step panel is wider and
#: needs more nodes for the same accuracy. See the module NOTE.
EDGE_WIDTH = 5.0

#: Default Gauss--Legendre order **per panel** of the lambda_true rule.
#: Measured, not chosen: within 3e-9 of every bin's peak in the module
#: NOTE's sweep; 16 per panel reaches only 3e-7, 24 reaches 2e-10.
N_QUAD = 20


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
        Gauss--Legendre order **per panel** (default: 20). See the module
        NOTE; the total per cell is ``n_quad`` times the number of
        non-empty panels.
    edge_width : float, optional
        :math:`L_e` in the panel :math:`\lambda^{\rm tr}_e \pm L_e w_e`
        around each richness edge (default: 5).
    """

    def __init__(self, lambda_edges, z_edges, mor, emg_params, sigma_z,
                 bracket_width: float = BRACKET_WIDTH,
                 n_quad: int = N_QUAD, edge_width: float = EDGE_WIDTH):
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
        self.edge_width = float(edge_width)
        if int(n_quad) != n_quad or n_quad < 1:
            raise ValueError(f"n_quad must be a positive integer, "
                             f"got {n_quad!r}")
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

    def panel_edges(self, ln_mass, z):
        r"""The composite rule's panel boundaries, shape
        ``(*broadcast(ln_mass, z).shape, n_panels + 1)``.

        The bracket :math:`[a, b]` cut at every break that falls inside it:
        :math:`\lambda^{\rm tr}_e \pm L_e w_e` around each richness edge
        (`~clenspy.selection.richness_kernel.richness_edge_steps`), the
        relation's mean, and its ``support_min`` if it has one. See the
        module NOTE for why each is there.

        NOTE: the same number of panels in every cell, so every cell is the
        same array shape. A break outside the bracket is clipped to it and
        leaves an empty panel -- ascending, never inverted, zero weight.
        """
        ln_mass, z = np.broadcast_arrays(
            np.asarray(ln_mass, dtype=float), np.asarray(z, dtype=float)
        )
        a, b = (np.broadcast_to(v, ln_mass.shape)
                for v in self.bracket(ln_mass, z))
        centre, width = richness_edge_steps(self.lambda_edges, z,
                                            self.emg_params)
        half = self.edge_width * width
        breaks = [centre - half, centre + half,
                  np.asarray(self.mor.mean(ln_mass, z), dtype=float)[..., None]]
        # optional: only a relation whose density jumps declares one
        support_min = getattr(self.mor, "support_min", None)
        if support_min is not None:
            breaks.append(
                np.asarray(support_min(ln_mass, z), dtype=float)[..., None]
            )
        breaks = np.concatenate(
            [np.broadcast_to(x, ln_mass.shape + x.shape[-1:]) for x in breaks],
            axis=-1,
        )
        inside = np.sort(np.clip(breaks, a[..., None], b[..., None]), axis=-1)
        return np.concatenate([a[..., None], inside, b[..., None]], axis=-1)

    def _nodes_for(self, ln_mass, z):
        """Nodes and weights of this cell's composite rule, shape
        ``(..., n_panels * n_quad)``; zero weight on empty panels."""
        edges = self.panel_edges(ln_mass, z)
        lower, upper = edges[..., :-1], edges[..., 1:]
        lam, wts = gl_nodes_batched(lower.ravel(), upper.ravel(), self.n_quad)
        shape = lower.shape[:-1] + (lower.shape[-1] * self.n_quad,)
        return lam.reshape(shape), wts.reshape(shape)

    def _integrate(self, ln_mass, z, kernel, n_out):
        r""":math:`\sum_k w_k\,P(\lambda_k\mid M,z)\,K(\lambda_k,z)` on this
        cell's composite rule, for ``kernel(lambda_true, z) -> (n, n_out)``.

        NOTE: the integrand is evaluated only on nodes that carry weight.
        Empty panels cost nothing, and their nodes -- piled on the
        bracket's clipped ends, possibly at :math:`\lambda^{\rm tr} = 0` --
        never reach the kernel (module NOTE). A cell with no weight at all
        (`HodMor` below :math:`M_{\min}`, whose bracket is the single
        point zero) integrates to zero.

        Returns shape ``(*broadcast(ln_mass, z).shape, n_out)``.
        """
        ln_mass, z = np.broadcast_arrays(
            np.asarray(ln_mass, dtype=float), np.asarray(z, dtype=float)
        )
        lam, wts = self._nodes_for(ln_mass, z)
        live = wts > 0.0
        integrand = np.zeros(lam.shape + (n_out,))
        if live.any():
            lam_live = lam[live]
            z_live = np.broadcast_to(z[..., None], lam.shape)[live]
            pdf = self.mor.pdf(
                lam_live, np.broadcast_to(ln_mass[..., None], lam.shape)[live],
                z_live,
            )
            # (n_live,) x (n_live, n_out)
            integrand[live] = ((wts[live] * pdf)[:, None]
                               * kernel(lam_live, z_live))
        # contract the quadrature axis
        return integrand.sum(axis=-2)

    def S_i(self, ln_mass, z):
        r""":math:`S_i(M, z)` for every richness bin.

        Returns shape ``(*broadcast(ln_mass, z).shape, n_lambda_bins)``.
        """
        return self._integrate(
            ln_mass, z,
            lambda lam, z_tr: richness_bin_probability(
                self.lambda_edges, lam, z_tr, self.emg_params),
            self.n_lambda_bins,
        )

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
        return self._integrate(
            ln_mass, z,
            lambda lam, z_tr: richness_bin_first_moment(
                self.lambda_edges, lam, z_tr, self.emg_params),
            self.n_lambda_bins,
        )

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
        captured = self._integrate(
            ln_mass, z, lambda lam, z_tr: np.ones(lam.shape + (1,)), 1
        )
        return 1.0 - captured[..., 0]

    def __repr__(self):
        return (f"SelectionFunction({self.n_lambda_bins} richness x "
                f"{self.n_z_bins} redshift bins, {self.mor!r}, "
                f"L={self.bracket_width:g}, L_e={self.edge_width:g}, "
                f"n_quad={self.n_quad} per panel)")


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
