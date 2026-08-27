r"""Covariance of the binned cluster counts.

Two contributions, stored separately and summed at the end:

.. math::
    {\rm Cov}\left[N_{ij}, N_{i'j'}\right] =
      \underbrace{\delta_{ii'}\delta_{jj'}\,N_{ij}}_{\rm Poisson}
      + \underbrace{\delta_{jj'}\,
        \bar b_{ij}\bar b_{i'j}\,N_{ij}N_{i'j}\,\sigma_W^2(z_j)
        }_{\rm sample\ variance}

The Poisson term is the shot noise of a finite catalogue and is diagonal.
The sample-variance term is the response of the counts to the large-scale
density fluctuation inside the survey window: clusters in the *same*
redshift slice all sit in the same realisation of that mode, so richness
bins at equal :math:`z` are **fully correlated** while different redshift
slices are independent.

NOTE: **the off-diagonal structure is the physics, not a detail.** The
sample-variance term is a rank-one outer product within each redshift
block, :math:`(\bar b N)(\bar b N)^{\rm T}\sigma_W^2`. Treating the counts
covariance as diagonal -- the common shortcut -- discards precisely the
correlation that limits how much a richness-binned abundance can say about
:math:`\sigma_8`, and it makes the errors look smaller than they are.

NOTE: **different redshift bins are independent here, and that is an
approximation.** It holds when the bins are wider than the correlation
length of the window mode, which for :math:`\Delta z \sim 0.15` and
:math:`R_{\rm eff}` of order 100 Mpc is marginal. It is the same
assumption the exemplar makes; it is stated rather than hidden, and
`sigma_window` exposes the scale it rests on.

NOTE: **units.** :math:`N_{ij}` is a dimensionless count and so is the
covariance. :math:`\sigma_W` is the dimensionless r.m.s. of the linear
density field smoothed on the window scale, and :math:`R_{\rm eff}` is in
:math:`{\rm Mpc}/h`, matching `clenspy.cosmology.SigmaGrid`.

NOTE: the window r.m.s. is
:math:`\sigma_W(z) = \sigma_R(R_{\rm eff})\,D(z)` -- the **linear** growth
factor times the :math:`z = 0` variance, which is exactly the
factorisation `clenspy.cosmology.sigma` provides. Using a nonlinear
:math:`\sigma` here would be wrong: the response of the counts to a
long-wavelength mode is a linear-theory statement.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CountsCovariance"]


class CountsCovariance:
    r"""Poisson plus sample variance for :math:`N_{ij}`.

    NOTE: units -- dimensionless throughout; ``r_eff_hinv`` in Mpc/h.

    NOTE: the matrix is returned in the flattened ordering
    ``(i, j) -> i * n_z_bins + j``, i.e. **richness-major**. The redshift blocks
    are therefore *not* contiguous, which matters when reading the matrix
    by eye; `block` returns one redshift slice at a time for that reason.

    Parameters
    ----------
    counts : array-like
        :math:`N_{ij}` with shape ``(n_lambda, n_z)``, from
        `clenspy.observables.ClusterAbundance.counts`.
    bias : array-like
        The count-weighted effective halo bias :math:`\bar b_{ij}`, same
        shape as ``counts``.
    sigma_window : array-like
        :math:`\sigma_W(z_j)`, the r.m.s. of the linear density field in
        the survey window at each bin's mean redshift. One entry per
        redshift bin.
    """

    def __init__(self, counts, bias, sigma_window):
        self.counts = np.asarray(counts, dtype=float)
        self.bias = np.asarray(bias, dtype=float)
        self.sigma_window = np.atleast_1d(
            np.asarray(sigma_window, dtype=float)
        )
        if self.counts.ndim != 2:
            raise ValueError(
                f"counts must be 2-D (n_lambda_bins, n_z_bins), got {self.counts.shape}"
            )
        if self.bias.shape != self.counts.shape:
            raise ValueError(
                f"bias must match counts {self.counts.shape}, got "
                f"{self.bias.shape}"
            )
        if self.sigma_window.shape != (self.counts.shape[1],):
            raise ValueError(
                f"sigma_window must have one entry per redshift bin "
                f"({self.counts.shape[1]}), got {self.sigma_window.shape}"
            )
        if np.any(self.counts < 0.0):
            raise ValueError("counts must be non-negative")
        if np.any(self.sigma_window < 0.0):
            raise ValueError("sigma_window must be non-negative")

    @property
    def n_lambda_bins(self):
        return self.counts.shape[0]

    @property
    def n_z_bins(self):
        return self.counts.shape[1]

    @property
    def size(self):
        """Dimension of the flattened data vector."""
        return self.counts.size

    # -- the two components, separately --------------------------------

    def cov_poisson(self):
        r"""The shot-noise term, :math:`\delta_{ii'}\delta_{jj'}N_{ij}`."""
        return np.diag(self.counts.ravel())

    def cov_sample_variance(self):
        r"""The window term,
        :math:`\delta_{jj'}\bar b\bar b' N N'\sigma_W^2`.

        NOTE: rank one within each redshift block, and exactly zero between
        blocks. Its trace-to-total ratio is what tells you whether an
        analysis is shot-noise or sample-variance limited.
        """
        out = np.zeros((self.size, self.size))
        # b_ij * N_ij, one vector per redshift slice
        weighted = self.bias * self.counts
        for j in range(self.n_z_bins):
            idx = np.arange(self.n_lambda_bins) * self.n_z_bins + j
            vector = weighted[:, j]
            block = np.outer(vector, vector) * self.sigma_window[j] ** 2
            out[np.ix_(idx, idx)] += block
        return out

    def components(self):
        """``{name: matrix}`` for every stored term."""
        return {
            "poisson": self.cov_poisson(),
            "sample_variance": self.cov_sample_variance(),
        }

    def cov(self, poisson=True, sample_variance=True):
        r"""The total, with switches to isolate either term.

        NOTE: both default to on. ``sample_variance=False`` gives the
        diagonal-only matrix that is easy to invert and wrong.
        """
        total = np.zeros((self.size, self.size))
        if poisson:
            total += self.cov_poisson()
        if sample_variance:
            total += self.cov_sample_variance()
        return total

    def block(self, j, **kw):
        r"""The :math:`n_\lambda \times n_\lambda` sub-matrix at redshift
        bin ``j``."""
        idx = np.arange(self.n_lambda_bins) * self.n_z_bins + j
        return self.cov(**kw)[np.ix_(idx, idx)]

    def correlation(self, **kw):
        """The correlation matrix, for reading structure off by eye."""
        c = self.cov(**kw)
        d = np.sqrt(np.diag(c))
        return c / np.outer(d, d)

    def __repr__(self):
        return (f"CountsCovariance({self.n_lambda_bins} x {self.n_z_bins} bins, "
                f"size {self.size})")


if __name__ == "__main__":
    from ..cosmology.growth import growth_factor
    from ..cosmology.sigma import LinearPk, SigmaGrid

    # a DES-Y1-like set of counts and biases
    counts = np.array([[2500.0, 3100.0, 2700.0],
                       [900.0, 1150.0, 1000.0],
                       [300.0, 380.0, 330.0],
                       [110.0, 140.0, 120.0]])
    bias = np.array([[2.1, 2.2, 2.3],
                     [2.6, 2.7, 2.8],
                     [3.2, 3.3, 3.5],
                     [4.3, 4.5, 4.8]])
    z_mid = np.array([0.28, 0.43, 0.57])

    # sigma_W = sigma_R(R_eff) * D(z): the linear factorisation.
    #
    # The toy spectrum must be NORMALISED first. An arbitrary amplitude
    # gives an arbitrary sigma_W, and since sigma_W enters the answer
    # linearly the fractional errors below would be meaningless. Fix it by
    # sigma_8: rescale so sigma(8 Mpc/h) = 0.8.
    k = np.logspace(-4.0, 2.0, 500)
    shape = k**-1.5 * np.exp(-((k / 30.0) ** 2))
    unnormalised = SigmaGrid(LinearPk(k, shape))
    amplitude = (0.8 / unnormalised.sigma(8.0, truncate=False)) ** 2
    grid = SigmaGrid(LinearPk(k, amplitude * shape))
    print(f"toy P(k) normalised to sigma_8 = "
          f"{grid.sigma(8.0, truncate=False):.4f}")

    r_eff_hinv = 120.0
    sigma_r0 = grid.sigma(r_eff_hinv, truncate=False)
    sigma_w = sigma_r0 * growth_factor(z_mid)
    print(f"R_eff = {r_eff_hinv:.0f} Mpc/h,  sigma_R(z=0) = {sigma_r0:.5f}")
    print(f"sigma_W(z) = {np.array2string(sigma_w, precision=5)}\n")

    cc = CountsCovariance(counts, bias, sigma_w)
    print(cc, "\n")

    print("fractional error on each N_ij, by component:")
    print(f"{'bin':>12s}  {'N':>8s}  {'Poisson':>9s}  {'sample var':>11s}  "
          f"{'total':>8s}")
    diag_p = np.sqrt(np.diag(cc.cov_poisson()))
    diag_s = np.sqrt(np.diag(cc.cov_sample_variance()))
    diag_t = np.sqrt(np.diag(cc.cov()))
    flat = counts.ravel()
    for a in range(cc.size):
        i, j = divmod(a, cc.n_z_bins)
        print(f"{f'lam{i} z{j}':>12s}  {flat[a]:8.0f}  "
              f"{diag_p[a] / flat[a]:9.4f}  {diag_s[a] / flat[a]:11.4f}  "
              f"{diag_t[a] / flat[a]:8.4f}")
    print("  <- Poisson falls as 1/sqrt(N) with richness; sample variance")
    print("     does not fall at all, because it is a coherent mode.")

    print("\nthe correlation structure -- redshift bin 0 only:")
    corr = cc.correlation()
    idx = np.arange(cc.n_lambda_bins) * cc.n_z_bins
    print("        " + "  ".join(f"{f'lam{i}':>7s}"
                                 for i in range(cc.n_lambda_bins)))
    for i in range(cc.n_lambda_bins):
        print(f"  lam{i}  " + "  ".join(
            f"{corr[idx[i], idx[k]]:7.4f}" for k in range(cc.n_lambda_bins)))
    print("  <- strongly correlated ACROSS richness at fixed z: every")
    print("     cluster in the slice sees the same window mode.")

    print("\nand between redshift bins, exactly zero by construction:")
    print(f"  Cov[(lam0,z0), (lam0,z1)] = {cc.cov()[0, 1]:.3e}")
    print(f"  Cov[(lam0,z0), (lam3,z2)] = "
          f"{cc.cov()[0, 3 * cc.n_z_bins + 2]:.3e}")

    print("\nwhat dropping sample variance would claim:")
    poisson_only = np.sqrt(np.diag(cc.cov(sample_variance=False)))
    print(f"{'bin':>12s}  {'sigma(N)/N total':>17s}  "
          f"{'Poisson only':>13s}  {'understated by':>15s}")
    for a in (0, 3, 6, 9):
        i, j = divmod(a, cc.n_z_bins)
        print(f"{f'lam{i} z{j}':>12s}  {diag_t[a] / flat[a]:17.4f}  "
              f"{poisson_only[a] / flat[a]:13.4f}  "
              f"{diag_t[a] / poisson_only[a]:14.2f}x")

    # the matrix must be a valid covariance
    eigenvalues = np.linalg.eigvalsh(cc.cov())
    print(f"\nsmallest eigenvalue: {eigenvalues.min():.4e}  "
          f"(positive definite: {eigenvalues.min() > 0})")
    print(f"condition number: {eigenvalues.max() / eigenvalues.min():.3e}")
