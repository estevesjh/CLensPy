#!/usr/bin/env python3
"""
Halo bias models for relating halo abundance to matter density.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from astropy.cosmology import Cosmology

from ..utils.decorators import default_mvals_z, scalar_array_output
from ..utils.interpolate import LogGridInterpolator
from .concentration import DELTA_COLLAPSE
from .fiducial import fiducial_cosmology, mean_matter_density
from .growth import growth_factor
from .pkgrid import PkGrid, _astropy_to_dict, _hash
from .sigma import SigmaGrid, lnr_grid

# ------------------------------------------------------------------
# Disk cache -- same shape as pkgrid._data_dir/_hash, own subdir -----
# ------------------------------------------------------------------
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # cosmology/ -> clenspy/
_DEFAULT_DATA = _PACKAGE_ROOT / "data"


def _cache_dir(subdir: str) -> Path:
    """`pkgrid._data_dir`, parameterized by cache subdir name."""
    root = os.environ.get("CLENSPY_DATA", str(_DEFAULT_DATA))
    path = Path(root).expanduser() / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _hash_arrays(*arrays) -> str:
    """MD5 of the concatenated raw bytes of one or more float arrays."""
    h = hashlib.md5()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=float).tobytes())
    return h.hexdigest()


class BiasModel:
    r"""Compute the linear halo bias b(M,z) from the Tinker et al. (2010)
    fit, for a given linear power spectrum.

    The calculation is based on the peak height of a top-hat sphere
    of lagrangian radius R corresponding to a mass M of linear
    power-spectrum:

    .. math::
        \nu(M,z) = \frac{\delta_c}{\sigma(M,z)}, \qquad
        \sigma^2(M,z=0) = \int \frac{dk}{2\pi^2} k^2 P(k)\, W^2(kR),
        \qquad \sigma(M,z) = D(z)\,\sigma(M,0),

    where :math:`W` is the top-hat window function, :math:`R = (3M /
    4\pi\bar\rho_m)^{1/3}` is the Lagrangian radius, :math:`D(z)` is the
    linear growth factor ({doc}`../cosmology`), and :math:`\delta_c =
    1.686`. The bias is then (Tinker et al. 2010, eq. 6)

    .. math::
        b(\nu) = 1 - A \frac{\nu^a}{\nu^a + \delta_c^a} + B \nu^b + C \nu^c,

    with :math:`A, a, B, b, C, c` fit as functions of the spherical
    overdensity :math:`\Delta` (``odelta``).

    NOTE: physical units, h-free -- mass in Msun, k in 1/Mpc, P in Mpc^3.

    Calibrated for :math:`\Delta = 200`--:math:`1600` and
    :math:`\nu \lesssim 4`. The constructor stores; `sigma_grid` builds
    lazily on first use.

    Parameters
    ----------
    k : array, optional
        Wavenumbers [1/Mpc], physical (not h-scaled). Give this **and**
        ``P`` to override the `PkGrid` step with a custom spectrum; give
        neither to build one from ``cosmo``.
    P : array, optional
        Linear power spectrum **at z=0** [Mpc^3], physical (not h-scaled);
        redshift enters through :math:`\sigma(M,z) = D(z)\,\sigma(M,0)`.
    cosmo : astropy.cosmology instance, optional
        Cosmology to use (default: `fiducial_cosmology()`). Builds the
        z=0 `PkGrid` this instance's spectrum comes from if ``k``/``P``
        are not given.
    odelta : int, optional
        Spherical overdensity :math:`\Delta` defining the halo mass, e.g.
        200 for :math:`M_{200m}` (default: 200).
    mvec : array, optional
        Mass grid [Msun] `bias_grid` is built and cached on. Defaults to
        the production Lagrangian-radius grid.
    zvec : array, optional
        Redshift grid, default ``linspace(0, 1.5, 31)``.
    sigma_grid : object, optional
        Prebuilt :math:`\sigma^2` evaluator; share one `SigmaGrid` with
        `~clenspy.cosmology.TinkerMassFunction` (same peak height).
    cache : bool, optional
        If True (default), store / reuse ``*.npz`` files of `bias_grid`
        in ``clenspy-data/bias_cache``, keyed on cosmology, ``odelta``,
        and the ``mval``/``zvec`` grids (plus a content hash of ``k``/``P``
        if given). NOTE: an injected ``sigma_grid`` is an opaque object
        that cannot be hashed by content, so caching is skipped entirely
        in that case -- the grid is always rebuilt, never read from or
        written to disk.

    Examples
    --------
    Build the (M, z) grid once, interp thereafter::

        model = BiasModel(k, P, zvec=np.linspace(0.0, 1.0, 21))
        model.bias(1e14, z=0.3)     # float
        model.bias(Mvec, z=0.3)     # (nM,)
        model.bias(Mvec, zvec)      # outer grid (nM, nz)
    """

    def __init__(
        self,
        k: np.ndarray | None = None,
        P: np.ndarray | None = None,
        cosmo: Cosmology | None = None,
        odelta: int = 200,
        mvec: np.ndarray | None = None,
        zvec: np.ndarray | None = None,
        sigma_grid=None,
        cache: bool = True,
    ):
        self.k = k
        self.P = P
        self.cosmo = fiducial_cosmology() if cosmo is None else cosmo
        self.omega_m = self.cosmo.Om0
        self.odelta = odelta
        self.rhom = mean_matter_density(self.cosmo)
        self._sigma_grid = sigma_grid
        self._sigma_grid_injected = sigma_grid is not None
        self.cache = cache
        # default z grid; a single-z grid would silently return that z
        # for every query
        self.zvec = (np.linspace(0.0, 1.5, 31) if zvec is None
                     else np.atleast_1d(np.asarray(zvec, dtype=float)))
        # production Lagrangian radii (Mpc/h -> Mpc at this boundary only)
        self.mval = ((4.0 * np.pi / 3.0) * self.rhom
                     * (np.exp(lnr_grid()) / self.cosmo.h) ** 3
                     if mvec is None
                     else np.atleast_1d(np.asarray(mvec, dtype=float)))

    def _compute_cache_path(self) -> Path:
        # Recomputed inside build(), not cached at construction: unlike
        # TinkerMassFunction, build(mvec=, zvec=) can replace the grids
        # after __init__.
        spec = dict(
            cosmo=_astropy_to_dict(self.cosmo),
            odelta=self.odelta,
            mval=_hash_arrays(self.mval),
            zvec=_hash_arrays(self.zvec),
        )
        if self.k is not None and self.P is not None:
            spec["kP"] = _hash_arrays(self.k, self.P)
        return _cache_dir("bias_cache") / f"{_hash(spec)}.npz"

    @property
    def bias_grid(self):
        r"""b(M, z) on ``mval`` x ``zvec``, built once on first use from
        `sigma_grid`, with :math:`\sigma(M,z) = D(z)\,\sigma(M,0)`."""
        if getattr(self, "_bias_grid", None) is None:
            sigma0 = np.atleast_1d(self.sigma_tophat(self.mval, z=0.0))
            growth = np.atleast_1d(growth_factor(self.zvec, self.cosmo))
            nu = DELTA_COLLAPSE / (sigma0[:, None] * growth[None, :])
            self._bias_grid = np.asarray(self.bias_at_nu(nu))
            self._interp = LogGridInterpolator(self.mval, self.zvec,
                                               self._bias_grid)
        return self._bias_grid

    def build(self, mvec=None, zvec=None):
        """Materialize the bias grid (optionally on new grids); return self."""
        if mvec is not None:
            self.mval = np.atleast_1d(np.asarray(mvec, dtype=float))
            self._bias_grid = None
        if zvec is not None:
            self.zvec = np.atleast_1d(np.asarray(zvec, dtype=float))
            self._bias_grid = None
        if getattr(self, "_bias_grid", None) is not None:
            return self
        if self.cache and not self._sigma_grid_injected:
            cache_file = self._compute_cache_path()
            if cache_file.exists():
                grid = np.load(cache_file)["grid"]
                if grid.shape != (len(self.mval), len(self.zvec)):
                    raise RuntimeError(
                        f"cache shape mismatch at {cache_file}: "
                        f"{grid.shape} vs expected "
                        f"{(len(self.mval), len(self.zvec))}"
                    )
                self._bias_grid = grid
                self._interp = LogGridInterpolator(self.mval, self.zvec, grid)
                print(f"BiasModel loaded cache file: {cache_file}")
                return self
            _ = self.bias_grid  # fills self._bias_grid, self._interp
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_file, grid=self._bias_grid)
            print(f"BiasModel saved cache file: {cache_file}")
            return self
        _ = self.bias_grid
        return self

    build_all = build  # alias, one release

    @property
    def is_built(self) -> bool:
        """Whether the (M, z) grid has been materialized."""
        return getattr(self, "_bias_grid", None) is not None

    @default_mvals_z
    def bias(self, M=None, z=None):
        r"""b(M, z), interpolated from `bias_grid`.

        ``bias(1e14, z=0.3)`` is a float; ``bias(Mvec, z=0.3)`` has shape
        ``(nM,)``; ``bias(Mvec, zvec)`` returns the outer grid ``(nM, nz)``.

        Parameters
        ----------
        M : float or array, optional
            Halo mass [Msun]. Defaults to `self.mval`.
        z : float or array, optional
            Redshift(s). Defaults to `self.zvec`.
        """
        if not self.is_built:
            self.build()
        return self._interp(M, z)

    @scalar_array_output
    def nu_at_mass(self, M, z=0.0, deltac=DELTA_COLLAPSE):
        r"""
        Compute peak-height :math:`\nu(M,z) = \delta_c / \sigma(M,z)`.

        Parameters
        ----------
        M : float or array
            Halo mass [Msun].
        z : float, optional
            Redshift (default: 0.0).
        deltac : float, optional
            Critical linear overdensity for collapse (default: 1.686).

        Returns
        -------
        float or array
            Peak height ν(M,z), same shape as M.
        """
        sigma = self.sigma_tophat(M, z=z)
        return deltac / sigma

    @property
    def sigma_grid(self):
        r"""The :math:`\sigma(M)` grid, built once on first use (Tinker
        2008 MF and Tinker 2010 bias fit the same peak height, so both
        read one table)."""
        if getattr(self, "_sigma_grid", None) is None:
            k, P = self.k, self.P
            if k is None or P is None:
                pk_grid = PkGrid(cosmo=self.cosmo, nonlinear=False)
                k, P = pk_grid.k, pk_grid(pk_grid.k, z=0.0)
            self._sigma_grid = SigmaGrid(k, P)
        return self._sigma_grid

    @scalar_array_output
    def sigma_tophat(self, M, z=0.0):
        r"""
        Calculate σ(M,z), the top-hat variance amplitude at the Lagrangian
        radius of mass M.

        .. math::
            \sigma^2(M,z=0) = \int \frac{dk}{2\pi^2} k^2 P(k)\, W^2(kR),
            \qquad R = \left(\frac{3M}{4\pi\bar\rho_m}\right)^{1/3},
            \qquad \sigma(M,z) = D(z)\,\sigma(M,0)

        where :math:`W` is the Fourier transform of the real-space top-hat
        window, :math:`\bar\rho_m` is the comoving mean matter density, and
        :math:`D(z)` is the linear growth factor ({doc}`../cosmology`).
        Untruncated: the :math:`kR \le 20` convention belongs to the mass
        function, not the bias.

        Parameters
        ----------
        M : float or array
            Halo mass [Msun].
        z : float, optional
            Redshift (default: 0.0).

        Returns
        -------
        sigma : float or array
            σ(M,z), same shape as M.
        """
        # Lagrangian radius R [Mpc], comoving
        R = (3 * np.asarray(M, dtype=float)
             / (4 * np.pi * self.rhom)) ** (1 / 3)
        ln_sigma2, _ = self.sigma_grid.sigma2_fftlog(np.log(R))
        return np.exp(0.5 * ln_sigma2) * growth_factor(z, self.cosmo)

    @scalar_array_output
    def bias_at_nu(self, nu):
        """
        Evaluate the Tinker et al. (2010) bias function at peak height ν.

        Parameters
        ----------
        nu : float or array
            Peak height ν, e.g. from `nu_at_mass`.

        Returns
        -------
        float or array
            Linear bias b(ν), same shape as nu.
        """
        A, a, B, b, C, c = self.get_tinker_params()
        return self._bias_at_nu(nu, A, a, B, b, C, c, deltac=DELTA_COLLAPSE)

    def get_tinker_params(self):
        r"""
        Get the Tinker et al. (2010) bias fit parameters for ``self.odelta``.

        .. math::
            A = 1 + 0.24\, y\, e^{-(4/y)^4}, \qquad a = 0.44 y - 0.88, \qquad
            B = 0.183, \qquad b = 1.5,

        .. math::
            C = 0.019 + 0.107 y + 0.19\, e^{-(4/y)^4}, \qquad c = 2.4,
            \qquad y = \log_{10}\Delta

        with :math:`\Delta` = ``self.odelta`` the spherical overdensity.

        Returns
        -------
        list of float
            ``[A, a, B, b, C, c]``.
        """
        y = np.log10(self.odelta)
        tinker_best_fit = {
            "A": 1.0 + 0.24 * y * np.exp(-((4 / y) ** 4)),
            "a": 0.44 * y - 0.88,
            "B": 0.183,
            "b": 1.5,
            "C": 0.019 + 0.107 * y + 0.19 * np.exp(-((4 / y) ** 4)),
            "c": 2.4,
        }
        return [tinker_best_fit[col] for col in ["A", "a", "B", "b", "C", "c"]]

    def _bias_at_nu(self, nu, A, a, B, b, C, c, deltac=DELTA_COLLAPSE):
        r"""
        Tinker et al. (2010) eq. 6:
        :math:`b(\nu) = 1 - A \nu^a / (\nu^a + \delta_c^a) + B \nu^b + C \nu^c`.
        """
        res = 1.0 - A * nu**a / (nu**a + deltac**a)
        res += B * nu**b
        res += C * nu**c
        return res


if __name__ == "__main__":
    import numpy as np

    # a smooth power-law P(k), so no Boltzmann solver is needed
    k = np.logspace(-4, 3, 800)
    P = 2e4 * k**-1.5 / (1.0 + (k / 0.2) ** 2)

    model = BiasModel(k, P)
    M = np.array([1e13, 1e14, 5e14, 1e15])
    print("Tinker et al. (2010) linear halo bias, Delta = 200m")
    print(f"{'M [Msun]':>11s}  {'sigma(M)':>9s}  {'nu':>7s}  {'b(M)':>7s}")
    for m in M:
        s, nu = model.sigma_tophat(m), model.nu_at_mass(m)
        b = model.bias(m, z=0.0)
        print(f"{m:11.2e}  {s:9.4f}  {nu:7.4f}  {b:7.4f}")

    print("\nb rises with M and nu, as it must: rarer haloes are more biased.")
    print("The fit is calibrated for nu <~ 4; beyond that b(M) extrapolates.")

    print("\nb(M, z) at M = 1e14 Msun, against z:")
    print("sigma(M,z) = D(z) sigma(M,0), so b rises with z at fixed mass --")
    print("a fixed-mass halo is rarer relative to a smaller, less-grown sigma.")
    for z in (0.0, 0.5, 1.0, 1.5):
        print(f"  z = {z:4.2f}:  sigma = {model.sigma_tophat(1e14, z=z):.4f}  "
              f"b = {model.bias(1e14, z=z):.4f}")
