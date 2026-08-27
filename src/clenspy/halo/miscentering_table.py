r"""Miscentered NFW profiles by table lookup -- the runtime path.

`clenspy` does not solve the miscentering integrals at evaluation time. It
interpolates the packaged dimensionless grid
``clenspy/data/nfw_miscentering.npz``, built offline by
``tools/make_miscentering_table.py`` from
`clenspy.halo.miscentering_kernel`.

The lookup is possible because the single-offset profiles are universal in
units of the scale radius. With :math:`\Sigma_0 = 2 r_s \rho_s`,
:math:`x = R/r_s` and :math:`x_{\rm mis} = R_{\rm mis}/r_s`,

.. math::
    \Sigma_{\rm mis}(R \mid R_{\rm mis}, M, c)
      = \Sigma_0 \, \hat\Sigma_{\rm mis}(x, x_{\rm mis}),
    \qquad
    \Delta\Sigma_{\rm mis} = \Sigma_0 \, \widehat{\Delta\Sigma}_{\rm mis}
      (x, x_{\rm mis}),

so mass, concentration and cosmology enter only the prefactor and one grid
serves every halo (docs/miscentering_math.md section 9.1).

NOTE: units follow `NfwProfile` -- lengths in Mpc, densities in
Msun/Mpc^3, and the returned surface densities in Msun/Mpc^2.

NOTE: :math:`\widehat{\Delta\Sigma}_{\rm mis}` is **signed** and negative
for :math:`x_{\rm mis} \gtrsim x`. That lobe is physical and is what makes
the mean-field term cancel -- do not clamp it (section 7).

The table is stored on axes :math:`(\ln x_{\rm mis}, \ln q)` with
:math:`q = x/x_{\rm mis}`, because the cusp at :math:`x = x_{\rm mis}` then
lies along a grid line instead of cutting diagonally across the cells
(section 9.3).

Only NFW is tabulated. Other profiles raise `MiscenteringTableError` rather
than silently falling back to quadrature.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .nfw import NfwProfile

__all__ = [
    "MiscenteringTableError",
    "NfwMiscenteringTable",
    "load_nfw_miscentering_table",
    "require_tabulated_profile",
]

TABLE_PATH = Path(__file__).resolve().parents[1] / "data" / "nfw_miscentering.npz"


class MiscenteringTableError(NotImplementedError):
    """No miscentering table exists for the requested profile."""


def require_tabulated_profile(profile) -> None:
    """Raise unless ``profile`` is one the packaged table covers.

    `clenspy` evaluates miscentering by interpolation only, so a profile
    with no table has no runtime path. Einasto is the case in practice:
    its miscentered profile is not universal in a single shape parameter
    the way NFW's is -- it carries the index ``n`` as a third axis -- so
    the NFW grid cannot be reused and no Einasto grid has been generated.

    Parameters
    ----------
    profile : object
        The centred halo profile, e.g. `NfwProfile`.

    Raises
    ------
    MiscenteringTableError
        If no table covers ``profile``.
    """
    if isinstance(profile, NfwProfile):
        return
    name = type(profile).__name__
    raise MiscenteringTableError(
        f"no miscentering table for {name}; only NfwProfile is tabulated. "
        "clenspy evaluates miscentering by interpolating "
        "clenspy/data/nfw_miscentering.npz and does not solve the offset "
        "integrals at runtime, so there is no fallback. To add support, "
        "generate a table with tools/make_miscentering_table.py -- note an "
        "Einasto grid needs a third axis in the index n."
    )


@lru_cache(maxsize=1)
def load_nfw_miscentering_table() -> "NfwMiscenteringTable":
    """The packaged NFW table, read once and cached for the process."""
    return NfwMiscenteringTable(TABLE_PATH)


class NfwMiscenteringTable:
    r"""Interpolator over the dimensionless miscentered NFW grid.

    Parameters
    ----------
    path : str or Path, optional
        Location of the ``.npz``. Defaults to the packaged table.

    Attributes
    ----------
    x_mis_range, q_range : tuple of float
        Covered ranges in :math:`x_{\rm mis}` and :math:`q = x/x_{\rm mis}`.
        Queries outside are clamped to the edge, as the profile is smooth
        and monotonic there.
    """

    def __init__(self, path=TABLE_PATH) -> None:
        path = Path(path)
        if not path.is_file():
            raise MiscenteringTableError(
                f"miscentering table not found at {path}. Build it with "
                "`python tools/make_miscentering_table.py`."
            )
        with np.load(path) as data:
            self._ln_x_mis = data["ln_x_mis"]
            self._ln_q = data["ln_q"]
            sigma_hat = data["sigma_hat_mis"]
            ds_hat = data["ds_hat_mis"]
        grid = (self._ln_x_mis, self._ln_q)
        opts = dict(method="linear", bounds_error=False, fill_value=None)
        self._sigma = RegularGridInterpolator(grid, sigma_hat, **opts)
        self._ds = RegularGridInterpolator(grid, ds_hat, **opts)
        self.path = path

    @property
    def x_mis_range(self) -> tuple[float, float]:
        return (float(np.exp(self._ln_x_mis[0])), float(np.exp(self._ln_x_mis[-1])))

    @property
    def q_range(self) -> tuple[float, float]:
        return (float(np.exp(self._ln_q[0])), float(np.exp(self._ln_q[-1])))

    def _query(self, interp, x, x_mis):
        """Clamped lookup at (x, x_mis), broadcasting over x."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        ln_xm = np.clip(np.log(x_mis), self._ln_x_mis[0], self._ln_x_mis[-1])
        ln_q = np.clip(np.log(x / x_mis), self._ln_q[0], self._ln_q[-1])
        pts = np.stack([np.full_like(ln_q, ln_xm), ln_q], axis=-1)
        return interp(pts)

    def sigma_hat(self, x, x_mis) -> np.ndarray:
        r""":math:`\hat\Sigma_{\rm mis} = \Sigma_{\rm mis}/\Sigma_0`."""
        if x_mis == 0.0:
            from .miscentering_kernel import nfw_sigma_hat

            return nfw_sigma_hat(np.atleast_1d(x))
        return self._query(self._sigma, x, x_mis)

    def ds_hat(self, x, x_mis) -> np.ndarray:
        r""":math:`\widehat{\Delta\Sigma}_{\rm mis}`, signed. See the module docstring."""
        if x_mis == 0.0:
            from .miscentering_kernel import nfw_mean_sigma_hat, nfw_sigma_hat

            x = np.atleast_1d(x)
            return nfw_mean_sigma_hat(x) - nfw_sigma_hat(x)
        return self._query(self._ds, x, x_mis)

    # -- physical, for a given halo --------------------------------------

    @staticmethod
    def _scale(profile: NfwProfile) -> tuple[np.ndarray, np.ndarray]:
        r""":math:`(r_s, \Sigma_0 = 2 r_s\rho_s)` for ``profile``."""
        rs = np.asarray(profile.rs, dtype=float)
        return rs, 2.0 * rs * np.asarray(profile.rho_s, dtype=float)

    def sigma_mis(self, profile: NfwProfile, R, r_mis) -> np.ndarray:
        r"""Miscentered :math:`\Sigma_{\rm mis}(R \mid r_{\rm mis})` [Msun/Mpc^2]."""
        require_tabulated_profile(profile)
        rs, sigma0 = self._scale(profile)
        return sigma0 * self.sigma_hat(np.asarray(R, dtype=float) / rs, r_mis / rs)

    def deltasigma_mis(self, profile: NfwProfile, R, r_mis) -> np.ndarray:
        r"""Miscentered :math:`\Delta\Sigma_{\rm mis}` [Msun/Mpc^2], signed."""
        require_tabulated_profile(profile)
        rs, sigma0 = self._scale(profile)
        return sigma0 * self.ds_hat(np.asarray(R, dtype=float) / rs, r_mis / rs)

    def mean_sigma_mis(self, profile: NfwProfile, R, r_mis) -> np.ndarray:
        r"""Miscentered aperture mean :math:`\bar\Sigma_{\rm mis}(<R)`.

        :math:`\bar\Sigma_{\rm mis} = \Sigma_{\rm mis} + \Delta\Sigma_{\rm mis}`
        by definition, so it costs no extra table.
        """
        return (self.sigma_mis(profile, R, r_mis)
                + self.deltasigma_mis(profile, R, r_mis))

    def __repr__(self) -> str:
        return (f"NfwMiscenteringTable(x_mis={self.x_mis_range}, "
                f"q={self.q_range[0]:.1e}..{self.q_range[1]:.1e})")


if __name__ == "__main__":
    table = load_nfw_miscentering_table()
    print(table)
    nfw = NfwProfile(m200=1e14, c200=4.0)
    R = np.array([0.1, 0.3, 1.0, 3.0])
    print(f"r_s = {float(nfw.rs):.4f} Mpc")
    for r_mis in (0.0, 0.1, 0.5):
        ds = table.deltasigma_mis(nfw, R, r_mis)
        print(f"  r_mis={r_mis:.2f}  DeltaSigma_mis = "
              + "  ".join(f"{v:+.4e}" for v in np.ravel(ds)))
