r"""Lensing profile with a single miscentering offset.

A `LensingProfile` whose assumed centre is displaced from the true halo
centre by a fixed projected distance :math:`R_{\rm mis}`. The centred
observables are inherited unchanged; the miscentered ones come from the
packaged lookup table.

NOTE: the miscentered profiles are **interpolated, never integrated**.
`clenspy` does not solve the offset integrals at evaluation time -- it reads
`clenspy.selection.miscentering`. The quadrature that built that table
lives in `clenspy.selection.miscentering_kernel` and is an offline generator, not
a runtime fallback. A profile with no table raises `MiscenteringTableError`
rather than quietly switching to quadrature; only NFW is tabulated today.

NOTE: units follow `LensingProfile` -- radii in Mpc, surface densities in
Msun/Mpc^2.

NOTE: :math:`\Delta\Sigma_{\rm mis}` is **signed**, negative for
:math:`R_{\rm mis} \gtrsim R`. A point mass outside the aperture gives
exactly zero, and the population average
:math:`\int_0^\infty \Delta\Sigma_{\rm mis}\,2\pi R_{\rm mis}\,dR_{\rm mis}`
vanishes only because of that lobe -- do not clamp it. See
``docs/miscentering_math.md`` for the derivation, the table design, and the
validation.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from astropy.cosmology import Cosmology

from ..selection.miscentering import (
    MiscenteringTableError,
    load_nfw_miscentering_table,
    require_tabulated_profile,
)
from .profile import LensingProfile

__all__ = ["MiscenteringProfile", "MiscenteringTableError"]


class MiscenteringProfile(LensingProfile):
    r"""
    A `LensingProfile` with a single (delta-function) miscentering offset.

    All centered observables (`sigma`, `deltasigma`, ...) are inherited
    unchanged; the miscentered counterparts `sigma_mis`, `mean_sigma_mis`
    and `deltasigma_mis` are read from the packaged table. For a
    population-averaged correction, integrate these over the offset
    distribution, e.g. the Gamma law of McClintock et al. (2019).

    Parameters
    ----------
    r_mis : float
        Offset of the true halo center from the assumed center [Mpc].
        ``0.0`` returns the centred profile exactly.

    Other parameters are those of `LensingProfile`.

    Raises
    ------
    MiscenteringTableError
        If the underlying halo profile has no miscentering table.

    Attributes
    ----------
    r_mis : float
        The offset, in Mpc.
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
        # Fail at construction, not on the first evaluation: a profile with
        # no table can never produce a miscentered observable.
        require_tabulated_profile(self.halo_profile)
        self.r_mis = float(r_mis)
        self._table = load_nfw_miscentering_table()

    def mean_sigma(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Centered aperture mean, in Msun/Mpc^2.

        .. math::
            \bar\Sigma(<R) = \Delta\Sigma(R) + \Sigma(R)

        Evaluated from `NfwProfile.mean_sigma`'s closed form rather than
        that sum, which cancels at small :math:`R / r_s`.
        """
        return self.halo_profile.mean_sigma(R)

    def sigma_mis(self, R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Miscentered surface density
        :math:`\Sigma_{\rm mis}(R \mid r_{\rm mis})`, in Msun/Mpc^2.
        """
        return self._table.sigma_mis(self.halo_profile, R, self.r_mis)

    def mean_sigma_mis(
        self, R: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        r"""
        Miscentered aperture mean
        :math:`\bar\Sigma_{\rm mis}(<R \mid r_{\rm mis})`, in Msun/Mpc^2.
        """
        return self._table.mean_sigma_mis(self.halo_profile, R, self.r_mis)

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
        return self._table.deltasigma_mis(self.halo_profile, R, self.r_mis)


if __name__ == "__main__":
    R = np.array([0.1, 0.3, 1.0, 3.0])
    for r_mis in (0.0, 0.2, 1.0):
        p = MiscenteringProfile(z_cluster=0.25, m200=2e14, r_mis=r_mis,
                                include_2halo=False)
        ds = np.ravel(p.deltasigma_mis(R))
        print(f"r_mis={r_mis:.1f} Mpc  DeltaSigma_mis = "
              + "  ".join(f"{v:+.4e}" for v in ds))
