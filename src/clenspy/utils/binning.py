r"""Richness--redshift bins for stacked cluster observables.

A cluster analysis is indexed by joint bins of observed richness and observed
photometric redshift. Following E26 / RSF (docs/notation.md), the $i$-th
richness bin and $j$-th redshift bin are

.. math::
    \Delta\lambda_i \equiv [\lambda_i^{\min}, \lambda_i^{\max}],
    \qquad
    \Delta z_j \equiv [z_j^{\min}, z_j^{\max}],

and every binned quantity -- counts :math:`\langle N_{ij}\rangle`, the
selection function :math:`\mathcal S_{ij}`, the stacked profile
:math:`\langle\Sigma(R\mid\lambda^{\rm ob}, z^{\rm ob})\rangle` -- is labelled
by the pair :math:`(i, j)`.

`RichnessBin` carries one such pair: its four edges, its two indices, and the
photo-z scatter :math:`\sigma_z(\Delta\lambda_i)`, which the papers make
richness-bin dependent. Bundling them is what lets a kernel take ``bin`` in
its signature instead of six loose floats that can be passed in the wrong
order.

NOTE: richness is dimensionless and redshift is dimensionless; the only
dimensional quantity here is none. ``sigma_z`` is in redshift units.

The evaluation-bar notation of the papers,

.. math::
    \left. g(x) \right|_{\Delta\lambda_i}
    \equiv g(\lambda_i^{\max}) - g(\lambda_i^{\min}),

is `RichnessBin.diff` / `RichnessBin.diff_z`: pass a CDF, get the probability
mass in the bin. Both the Gaussian and EMG richness kernels reduce to exactly
this, so the bin object owns the differencing rather than each kernel
re-implementing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

__all__ = ["RichnessBin", "BinCollection"]


@dataclass(frozen=True)
class RichnessBin:
    r"""One joint bin :math:`(\Delta\lambda_i, \Delta z_j)`.

    Parameters
    ----------
    lam_min, lam_max : float
        Observed-richness bin edges :math:`\lambda_i^{\min}`,
        :math:`\lambda_i^{\max}` (dimensionless).
    z_min, z_max : float
        Observed-redshift bin edges :math:`z_j^{\min}`, :math:`z_j^{\max}`.
    i_lam, i_z : int
        Richness and redshift bin indices. These are the :math:`i` and
        :math:`j` of :math:`\langle N_{ij}\rangle`, and they are what a
        result array is addressed by, so they travel with the edges.
    sigma_z : float
        Photo-z scatter :math:`\sigma_z(\Delta\lambda_i)` for this bin
        (redshift units). Richness-bin dependent in the papers.
    """

    lam_min: float
    lam_max: float
    z_min: float
    z_max: float
    i_lam: int = 0
    i_z: int = 0
    sigma_z: float = 0.0

    def __post_init__(self) -> None:
        if not self.lam_max > self.lam_min:
            raise ValueError(
                f"richness edges must increase: got "
                f"[{self.lam_min}, {self.lam_max}]"
            )
        if not self.z_max > self.z_min:
            raise ValueError(
                f"redshift edges must increase: got [{self.z_min}, {self.z_max}]"
            )
        if self.lam_min < 0.0:
            raise ValueError(f"richness must be non-negative: {self.lam_min}")
        if self.z_min < 0.0:
            raise ValueError(f"redshift must be non-negative: {self.z_min}")
        if self.sigma_z < 0.0:
            raise ValueError(f"sigma_z must be non-negative: {self.sigma_z}")

    # -- geometry -------------------------------------------------------

    @property
    def lam_edges(self) -> tuple[float, float]:
        r""":math:`(\lambda_i^{\min}, \lambda_i^{\max})`."""
        return (self.lam_min, self.lam_max)

    @property
    def z_edges(self) -> tuple[float, float]:
        r""":math:`(z_j^{\min}, z_j^{\max})`."""
        return (self.z_min, self.z_max)

    @property
    def index(self) -> tuple[int, int]:
        r"""The :math:`(i, j)` that addresses this bin."""
        return (self.i_lam, self.i_z)

    @property
    def lam_mid(self) -> float:
        """Richness bin centre."""
        return 0.5 * (self.lam_min + self.lam_max)

    @property
    def z_mid(self) -> float:
        r"""Redshift bin centre -- the :math:`z^{\rm ob}` of a stacked slice."""
        return 0.5 * (self.z_min + self.z_max)

    # -- the evaluation bar ---------------------------------------------

    def diff(self, cdf: Callable) -> np.ndarray:
        r"""Definite-integral bar over richness: :math:`g|_{\Delta\lambda_i}`.

        Returns ``cdf(lam_max) - cdf(lam_min)``, i.e. the probability mass
        the distribution places inside this richness bin. ``cdf`` may be
        vectorised over trailing axes; the subtraction broadcasts.
        """
        return np.asarray(cdf(self.lam_max)) - np.asarray(cdf(self.lam_min))

    def diff_z(self, cdf: Callable) -> np.ndarray:
        r"""Definite-integral bar over redshift: :math:`g|_{\Delta z_j}`."""
        return np.asarray(cdf(self.z_max)) - np.asarray(cdf(self.z_min))

    def contains(self, lam, z) -> np.ndarray:
        """Half-open membership test, ``[min, max)`` on both axes."""
        lam = np.asarray(lam, dtype=float)
        z = np.asarray(z, dtype=float)
        return (
            (lam >= self.lam_min)
            & (lam < self.lam_max)
            & (z >= self.z_min)
            & (z < self.z_max)
        )

    def __repr__(self) -> str:
        return (
            f"RichnessBin(lam=[{self.lam_min:g}, {self.lam_max:g}), "
            f"z=[{self.z_min:g}, {self.z_max:g}), "
            f"index={self.index}, sigma_z={self.sigma_z:g})"
        )


class BinCollection(Sequence):
    """An ordered set of `RichnessBin`, addressable by ``(i_lam, i_z)``.

    Behaves as a sequence, so ``for b in bins`` and ``bins[k]`` work in the
    flat order the bins were supplied. ``bins.at(i, j)`` addresses a bin by
    its paper indices, which is how results are labelled.
    """

    def __init__(self, bins: Iterable[RichnessBin]) -> None:
        self._bins = tuple(bins)
        self._by_index = {}
        for b in self._bins:
            if b.index in self._by_index:
                raise ValueError(f"duplicate bin index {b.index}")
            self._by_index[b.index] = b

    @classmethod
    def from_edges(cls, lam_edges, z_edges, sigma_z=None) -> "BinCollection":
        r"""Build the full outer product of richness and redshift edges.

        Parameters
        ----------
        lam_edges : sequence of float, length ``n_lam + 1``
            Monotonic richness bin edges.
        z_edges : sequence of float, length ``n_z + 1``
            Monotonic redshift bin edges.
        sigma_z : float or sequence of float, optional
            Photo-z scatter. A scalar applies to every bin; a sequence must
            have length ``n_lam``, since :math:`\sigma_z` depends on the
            richness bin.

        Returns
        -------
        BinCollection
            ``n_lam * n_z`` bins in row-major (richness-outer) order.
        """
        lam_edges = np.asarray(lam_edges, dtype=float)
        z_edges = np.asarray(z_edges, dtype=float)
        n_lam, n_z = lam_edges.size - 1, z_edges.size - 1
        if n_lam < 1 or n_z < 1:
            raise ValueError("need at least two edges on each axis")

        if sigma_z is None:
            sig = np.zeros(n_lam)
        else:
            sig = np.atleast_1d(np.asarray(sigma_z, dtype=float))
            if sig.size == 1:
                sig = np.full(n_lam, sig.item())
            elif sig.size != n_lam:
                raise ValueError(
                    f"sigma_z has {sig.size} values for {n_lam} richness bins"
                )

        return cls(
            RichnessBin(
                lam_min=float(lam_edges[i]), lam_max=float(lam_edges[i + 1]),
                z_min=float(z_edges[j]), z_max=float(z_edges[j + 1]),
                i_lam=i, i_z=j, sigma_z=float(sig[i]),
            )
            for i in range(n_lam)
            for j in range(n_z)
        )

    def at(self, i_lam: int, i_z: int) -> RichnessBin:
        r"""The bin with paper indices :math:`(i, j)`."""
        try:
            return self._by_index[(i_lam, i_z)]
        except KeyError:
            raise KeyError(
                f"no bin with index ({i_lam}, {i_z}); "
                f"have {sorted(self._by_index)}"
            ) from None

    @property
    def n_lam(self) -> int:
        """Number of distinct richness bins."""
        return len({b.i_lam for b in self._bins})

    @property
    def n_z(self) -> int:
        """Number of distinct redshift bins."""
        return len({b.i_z for b in self._bins})

    def reshape(self, flat) -> np.ndarray:
        r"""Fold a flat per-bin result into ``(n_lam, n_z, ...)``.

        The inverse of iterating this collection: a stage that returns one
        value per bin in sequence order gets its :math:`N_{ij}` matrix back.
        """
        flat = np.asarray(flat)
        if flat.shape[0] != len(self._bins):
            raise ValueError(
                f"got {flat.shape[0]} values for {len(self._bins)} bins"
            )
        return flat.reshape((self.n_lam, self.n_z) + flat.shape[1:])

    def __len__(self) -> int:
        return len(self._bins)

    def __getitem__(self, k):
        return self._bins[k]

    def __repr__(self) -> str:
        return f"BinCollection({self.n_lam} richness x {self.n_z} redshift)"


if __name__ == "__main__":
    from scipy.stats import norm

    # The DES Y1 richness bins, with the Y3 photo-z bins.
    bins = BinCollection.from_edges(
        lam_edges=[20, 30, 45, 60, 200],
        z_edges=[0.2, 0.35, 0.5, 0.65],
        sigma_z=[0.015, 0.014, 0.013, 0.012],
    )
    print(bins, "->", len(bins), "bins")
    print(bins.at(0, 0))
    print(bins.at(3, 2))

    # The evaluation bar: Gaussian probability mass in each richness bin,
    # for a halo whose kernel is centred at lambda = 40 with width 5.
    #     K_i^G = Phi((lam_max - mu)/sigma) - Phi((lam_min - mu)/sigma)
    mu, sigma = 40.0, 5.0
    mass = [b.diff(lambda x: norm.cdf((x - mu) / sigma)) for b in bins if b.i_z == 0]
    print("\nGaussian mass per richness bin (mu=40, sigma=5):")
    for b, m in zip([b for b in bins if b.i_z == 0], mass):
        print(f"  lam in [{b.lam_min:5g}, {b.lam_max:5g}) -> {float(m):.4f}")
    print(f"  total = {float(np.sum(mass)):.4f}")

    counts = np.arange(len(bins), dtype=float)
    print("\nreshape to N_ij:", bins.reshape(counts).shape)
