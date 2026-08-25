"""Cached Gauss-Legendre nodes and weights.

Fixed-order Gauss-Legendre quadrature is the workhorse of the binned
cluster-observable engines (see :mod:`clenspy.clusters`): the canonical
nodes/weights on ``[-1, 1]`` are computed once per order and affine-mapped
to the requested interval.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss

__all__ = ["gl_nodes"]


@lru_cache(maxsize=64)
def _leggauss_cached(n: int):
    return leggauss(n)


def gl_nodes(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Nodes and weights for :math:`\int_a^b f(x)\,dx \approx \sum_i w_i f(x_i)`.

    Parameters
    ----------
    a, b : float
        Integration limits.
    n : int
        Quadrature order.

    Returns
    -------
    x, w : ndarray, shape (n,)
        Nodes and weights on ``[a, b]``.
    """
    t, w = _leggauss_cached(int(n))
    return 0.5 * (b - a) * t + 0.5 * (a + b), 0.5 * (b - a) * w
