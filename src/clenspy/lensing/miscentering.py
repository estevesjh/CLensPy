r"""
Miscentering corrections for cluster lensing profiles (single offset).

For a halo whose true center is offset by :math:`R_{\rm mis}` from the
assumed center, the observed surface density is the azimuthal average of
the centered profile :math:`\Sigma` over the circle of radius :math:`R`
about the assumed center (Johnston et al. 2007; Yang et al. 2006):

.. math::
    \Sigma_{\rm mis}(R \mid R_{\rm mis})
    = \frac{1}{\pi} \int_0^{\pi}
      \Sigma\!\big(u(t)\big)\, dt,
    \qquad
    u(t) = \sqrt{R^2 + R_{\rm mis}^2 - 2 R R_{\rm mis} \cos t}.

The excess surface density requires the miscentered aperture mean
:math:`\bar\Sigma_{\rm mis}(<R)`. Instead of the usual nested cumulative
integral over :math:`\Sigma_{\rm mis}` (slow, and singular at
:math:`R' = R_{\rm mis}`), this module uses a closed-form reduction: the
aperture-mass identity written through the halo-centric overlap angle
:math:`\Lambda(u)`, integrated by parts onto the *known* centered mean
:math:`\bar\Sigma(<u)`, followed by the law-of-cosines substitution
:math:`u(t)^2 = R^2 + R_{\rm mis}^2 - 2 R R_{\rm mis}\cos t` which absorbs
the inverse-square-root endpoint singularity of :math:`\Lambda'(u)`
exactly. The result is a single smooth integral,

.. math::
    \bar\Sigma_{\rm mis}(<R \mid R_{\rm mis})
    = \frac{1}{2\pi R^2} \int_0^{\pi}
      \Big[u(t)^2 + R^2 - R_{\rm mis}^2\Big]\,
      \bar\Sigma\!\big(<u(t)\big)\, dt,

with the same nodes :math:`u(t)` as the azimuthal average. Both integrals
are evaluated with fixed Gauss-Legendre quadrature after the node-clustering
map :math:`t = \pi s^2`, which resolves the integrable
:math:`\log|R - R_{\rm mis}|` behavior at :math:`t \to 0` when
:math:`R \approx R_{\rm mis}`. See ``docs/miscentering_math.md`` for the
full derivation and validation.

The miscentered excess surface density

.. math::
    \Delta\Sigma_{\rm mis}(R \mid R_{\rm mis})
    = \bar\Sigma_{\rm mis}(<R \mid R_{\rm mis})
    - \Sigma_{\rm mis}(R \mid R_{\rm mis})

is **signed**: it is negative for :math:`R_{\rm mis} \gtrsim R`. This is a
genuine finite-profile effect (a point mass at :math:`R_{\rm mis} > R`
gives exactly zero), and the population average
:math:`\int_0^\infty \Delta\Sigma_{\rm mis}\, 2\pi R_{\rm mis}\,
dR_{\rm mis} \to 0` relies on the negative lobe -- do not clamp it to zero.
"""

from functools import lru_cache
from typing import Callable, Union

import numpy as np
from astropy.cosmology import Cosmology
from numpy.polynomial.legendre import leggauss

from ..lensing.profile import LensingProfile

__all__ = [
    "MiscenteringProfile",
    "miscentered_sigma",
    "miscentered_mean_sigma",
    "miscentered_deltasigma",
]


@lru_cache(maxsize=8)
def _clustered_nodes(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Gauss-Legendre nodes for :math:`\int_0^\pi h(t)\,dt` after the
    clustering map :math:`t = \pi s^2`, :math:`s \in [0, 1]`.

    Returns ``(t, w)`` such that :math:`\int_0^\pi h(t)\,dt \approx
    \sum_i w_i\, h(t_i)`; the weights already include the Jacobian
    :math:`dt = 2\pi s\, ds`. Clustering the nodes at :math:`t = 0`
    resolves the integrable logarithmic singularity of the miscentering
    integrands at :math:`u \to |R - R_{\rm mis}|`.
    """
    s, w = leggauss(n_nodes)
    s = 0.5 * (s + 1.0)  # map [-1, 1] -> [0, 1]
    w = 0.5 * w
    t = np.pi * s * s
    return t, w * 2.0 * np.pi * s


def _halo_centric_radii(
    R: np.ndarray, r_mis: float, t: np.ndarray
) -> np.ndarray:
    r"""
    Law-of-cosines radii :math:`u(t) = \sqrt{R^2 + R_{\rm mis}^2
    - 2 R R_{\rm mis} \cos t}`, evaluated in the cancellation-free form
    :math:`u^2 = (R - R_{\rm mis})^2 + 4 R R_{\rm mis} \sin^2(t/2)`.

    Shapes: ``R`` is ``(n,)``, ``t`` is ``(m,)``; returns ``(n, m)``.
    """
    R = R[:, None]
    return np.sqrt((R - r_mis) ** 2 + 4.0 * R * r_mis * np.sin(t / 2.0) ** 2)


def _eval_on_grid(func: Callable, u: np.ndarray) -> np.ndarray:
    """Evaluate a 1-D profile function on a 2-D node grid."""
    return np.asarray(func(u.ravel()), dtype=float).reshape(u.shape)


def miscentered_sigma(
    sigma_func: Callable,
    R: Union[float, np.ndarray],
    r_mis: float,
    n_nodes: int = 128,
) -> Union[float, np.ndarray]:
    r"""
    Miscentered surface density :math:`\Sigma_{\rm mis}(R \mid R_{\rm mis})`.

    .. math::
        \Sigma_{\rm mis}(R \mid R_{\rm mis})
        = \frac{1}{\pi} \int_0^{\pi} \Sigma\!\big(u(t)\big)\, dt,
        \qquad
        u(t) = \sqrt{R^2 + R_{\rm mis}^2 - 2 R R_{\rm mis} \cos t}

    Parameters
    ----------
    sigma_func : callable
        Centered surface density ``Sigma(R)``; must accept a 1-D array of
        radii [Mpc] and return the same shape.
    R : float or np.ndarray
        Projected radius from the assumed center [Mpc].
    r_mis : float
        Offset of the true halo center [Mpc].
    n_nodes : int, optional
        Number of Gauss-Legendre nodes (default 128; accuracy ~1e-9 of
        the local value for NFW, including the worst case R = r_mis).

    Returns
    -------
    float or np.ndarray
        :math:`\Sigma_{\rm mis}`, same shape as ``R`` (scalar in, scalar
        out). Same units as ``sigma_func``.
    """
    scalar_in = np.ndim(R) == 0
    R = np.atleast_1d(np.asarray(R, dtype=float))
    if r_mis == 0.0:
        result = np.asarray(sigma_func(R), dtype=float)
        return result.item() if scalar_in else result
    t, w = _clustered_nodes(n_nodes)
    u = _halo_centric_radii(R, r_mis, t)
    result = _eval_on_grid(sigma_func, u) @ w / np.pi
    return result.item() if scalar_in else result


def miscentered_mean_sigma(
    mean_sigma_func: Callable,
    R: Union[float, np.ndarray],
    r_mis: float,
    n_nodes: int = 128,
) -> Union[float, np.ndarray]:
    r"""
    Miscentered aperture mean
    :math:`\bar\Sigma_{\rm mis}(<R \mid R_{\rm mis})`.

    Uses the by-parts reduction of the aperture-mass identity onto the
    known centered mean :math:`\bar\Sigma(<u)` (see the module docstring):

    .. math::
        \bar\Sigma_{\rm mis}(<R \mid R_{\rm mis})
        = \frac{1}{2\pi R^2} \int_0^{\pi}
          \Big[u(t)^2 + R^2 - R_{\rm mis}^2\Big]\,
          \bar\Sigma\!\big(<u(t)\big)\, dt

    which is smooth (no cusp at :math:`R' = R_{\rm mis}`, no endpoint
    singularity) and exact for any axisymmetric profile.

    Parameters
    ----------
    mean_sigma_func : callable
        Centered aperture mean ``Sigmabar(<R)``; must accept a 1-D array
        of radii [Mpc] and return the same shape. For profiles with
        closed-form :math:`\Delta\Sigma`, this is
        ``sigma(R) + deltasigma(R)``.
    R : float or np.ndarray
        Aperture radius about the assumed center [Mpc]; must be positive.
    r_mis : float
        Offset of the true halo center [Mpc].
    n_nodes : int, optional
        Number of Gauss-Legendre nodes (default 128).

    Returns
    -------
    float or np.ndarray
        :math:`\bar\Sigma_{\rm mis}`, same shape as ``R`` (scalar in,
        scalar out). Same units as ``mean_sigma_func``.
    """
    scalar_in = np.ndim(R) == 0
    R = np.atleast_1d(np.asarray(R, dtype=float))
    if r_mis == 0.0:
        result = np.asarray(mean_sigma_func(R), dtype=float)
        return result.item() if scalar_in else result
    t, w = _clustered_nodes(n_nodes)
    u = _halo_centric_radii(R, r_mis, t)
    kernel = u * u + (R * R - r_mis * r_mis)[:, None]
    result = (kernel * _eval_on_grid(mean_sigma_func, u)) @ w
    result /= 2.0 * np.pi * R * R
    return result.item() if scalar_in else result


def miscentered_deltasigma(
    sigma_func: Callable,
    mean_sigma_func: Callable,
    R: Union[float, np.ndarray],
    r_mis: float,
    n_nodes: int = 128,
) -> Union[float, np.ndarray]:
    r"""
    Miscentered excess surface density
    :math:`\Delta\Sigma_{\rm mis}(R \mid R_{\rm mis})
    = \bar\Sigma_{\rm mis}(<R) - \Sigma_{\rm mis}(R)`.

    The result is signed (negative for :math:`R_{\rm mis} \gtrsim R`);
    see the module docstring for why it must not be clamped.

    Parameters
    ----------
    sigma_func, mean_sigma_func : callable
        Centered ``Sigma(R)`` and ``Sigmabar(<R)``, each accepting a 1-D
        array of radii [Mpc].
    R : float or np.ndarray
        Projected radius from the assumed center [Mpc]; must be positive.
    r_mis : float
        Offset of the true halo center [Mpc].
    n_nodes : int, optional
        Number of Gauss-Legendre nodes (default 128).

    Returns
    -------
    float or np.ndarray
        :math:`\Delta\Sigma_{\rm mis}`, same shape as ``R`` (scalar in,
        scalar out). Same units as the input callables.
    """
    return miscentered_mean_sigma(
        mean_sigma_func, R, r_mis, n_nodes
    ) - miscentered_sigma(sigma_func, R, r_mis, n_nodes)


class MiscenteringProfile(LensingProfile):
    r"""
    A `LensingProfile` with a single (delta-function) miscentering offset.

    All centered observables (`sigma`, `deltasigma`, ...) are inherited
    unchanged; the miscentered counterparts `sigma_mis`, `mean_sigma_mis`
    and `deltasigma_mis` convolve the centered profile with a fixed offset
    ``r_mis`` using the smooth by-parts kernel (module docstring). For a
    population-averaged correction, integrate these over the offset
    distribution, e.g. Rayleigh-distributed :math:`R_{\rm mis}` (Johnston
    et al. 2007; Simet et al. 2017).

    Attributes
    ----------
    r_mis : float
        Offset of the true halo center from the assumed center [Mpc].
    n_nodes : int
        Gauss-Legendre nodes per miscentering integral.
    """

    def __init__(
        self,
        z_cluster: float,
        m200: float,
        cosmology: Cosmology | None = None,
        concentration: float = 4.0,
        model: str = "NFW",
        include_2halo: bool = True,
        backend_2halo: str = "camb",
        z_source: float = 1.0,
        r_mis: float = 0.0,
        n_nodes: int = 128,
    ) -> None:
        super().__init__(
            z_cluster,
            m200,
            cosmology,
            concentration,
            model,
            include_2halo,
            backend_2halo,
            z_source,
        )
        if r_mis < 0:
            raise ValueError("Miscentering offset r_mis must be non-negative")
        self.r_mis = r_mis
        self.n_nodes = n_nodes

    def mean_sigma(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Centered aperture mean, in Msun/Mpc^2.

        .. math::
            \bar\Sigma(<R) = \Delta\Sigma(R) + \Sigma(R)
        """
        return self.sigma(R) + self.deltasigma(R)

    def sigma_mis(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Miscentered surface density
        :math:`\Sigma_{\rm mis}(R \mid r_{\rm mis})`, in Msun/Mpc^2.

        See `miscentered_sigma`.
        """
        return miscentered_sigma(self.sigma, R, self.r_mis, self.n_nodes)

    def mean_sigma_mis(
        self, R: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        r"""
        Miscentered aperture mean
        :math:`\bar\Sigma_{\rm mis}(<R \mid r_{\rm mis})`, in Msun/Mpc^2.

        See `miscentered_mean_sigma`.
        """
        return miscentered_mean_sigma(self.mean_sigma, R, self.r_mis, self.n_nodes)

    def deltasigma_mis(
        self, R: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        r"""
        Miscentered excess surface density
        :math:`\Delta\Sigma_{\rm mis}(R \mid r_{\rm mis})
        = \bar\Sigma_{\rm mis}(<R) - \Sigma_{\rm mis}(R)`, in Msun/Mpc^2.

        Signed -- negative for :math:`r_{\rm mis} \gtrsim R` (see the
        module docstring).
        """
        return miscentered_deltasigma(
            self.sigma, self.mean_sigma, R, self.r_mis, self.n_nodes
        )
