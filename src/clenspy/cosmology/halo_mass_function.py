r"""The Tinker et al. (2008) halo mass function.

.. math::
    f(\sigma) = A\left[\left(\frac{\sigma}{b}\right)^{-a} + 1\right]
                \exp\!\left(-\frac{c}{\sigma^{2}}\right),
    \qquad
    \frac{dn}{d\ln M} = -\frac{\bar\rho_m}{6M}\,f(\sigma)\,
        \frac{d\ln\sigma^{2}}{d\ln R}

NOTE: physical units, h-free -- M in Msun, R in Mpc, k in 1/Mpc, P in
Mpc^3, dn/dlnM in Mpc^-3. Physics and provenance: ``docs/mass_function.md``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from ..utils.decorators import default_mvals_z, scalar_array_output
from ..utils.interpolate import LogGridInterpolator
from .fiducial import fiducial_cosmology, mean_matter_density
from .growth import growth_factor
from .pkgrid import PkGrid, _astropy_to_dict, _hash
from .sigma import SigmaGrid, lnr_grid

__all__ = [
    "TINKER08_TABLE2",
    "TinkerMassFunction",
]

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

#: Tinker et al. (2008) Table 2 -- ``Delta -> (A0, a0, b0, c)``, with
#: :math:`\Delta` referred to the mean matter density. Interpolated
#: linearly in :math:`\log_{10}\Delta`, as the paper prescribes.
TINKER08_TABLE2 = {
    "delta": (200.0, 300.0, 400.0, 600.0, 800.0, 1200.0, 1600.0, 2400.0,
              3200.0),
    "A0": (0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260),
    "a0": (1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66),
    "b0": (2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41),
    "c": (1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44),
}

#: :math:`\log_{10}75`, the pivot of the :math:`b(z)` exponent (Tinker
#: 2008 Eq. 8). Written as the logarithm it is, rather than as the magic
#: constant 1.8750612633 that appears in other implementations.
_LOG10_75 = np.log10(75.0)


class TinkerMassFunction:
    r"""Tinker et al. (2008) :math:`f(\sigma)` and :math:`dn/d\ln M`.

    The constructor stores; the chain cosmo -> `PkGrid` -> `SigmaGrid` ->
    `dndlnm_grid` runs lazily on first query.

    Parameters
    ----------
    cosmo : astropy.cosmology.Cosmology, optional
        Defaults to `fiducial_cosmology()`.
    k : array-like, optional
        Wavenumbers [1/Mpc], strictly ascending. Give this and ``pk`` to
        override the `PkGrid` step with a custom z=0 linear spectrum.
    pk : array-like, optional
        Linear power spectrum at z=0 [Mpc^3], same shape as ``k``.
    mvec : array-like, optional
        Mass grid [Msun] `dndlnm_grid` is built and cached on. Defaults
        to the production Lagrangian-radius grid.
    zvec : array-like, optional
        Redshift grid, default ``linspace(0, 1.5, 31)``.
    delta : float, optional
        Spherical overdensity w.r.t. mean matter (default: 200).
    truncate : bool, optional
        Apply the :math:`kR \le 20` truncation in :math:`\sigma^2`
        (default: True, the production quantity).
    sigma_grid : object, optional
        Prebuilt :math:`\sigma^2` evaluator with ``sigma2(r, truncate=)``
        and ``dlnsigma2_dlnr(r, truncate=)``; overrides ``k``/``pk``.
        Share one `SigmaGrid` with `~clenspy.cosmology.BiasModel` -- both
        fit the same peak height.
    cache : bool, optional
        If True (default), store / reuse ``*.npz`` files of
        `dndlnm_grid` in ``clenspy-data/hmf_cache``, keyed on cosmology,
        ``delta``, ``truncate``, and the ``mval``/``zvec`` grids (plus a
        content hash of ``k``/``pk`` if given). NOTE: an injected
        ``sigma_grid`` is an opaque object that cannot be hashed by
        content, so caching is skipped entirely in that case -- the grid
        is always rebuilt, never read from or written to disk.

    Examples
    --------
    Cosmology triggers all::

        hmf = TinkerMassFunction()       # instant: stores only
        dn = hmf.dndlnm(1e14, z=0.3)     # Mpc^-3; first call builds

    A custom physical spectrum::

        hmf = TinkerMassFunction(k=k, pk=pk)   # k [1/Mpc], P [Mpc^3]

    One :math:`\sigma(M)` shared with the bias::

        grid = SigmaGrid(k, pk)
        hmf = TinkerMassFunction(sigma_grid=grid)

    Query shapes: ``dndlnm(1e14, z=0.3)`` is a float;
    ``dndlnm(Mvec, z=0.3)`` has shape ``(nM,)``; ``dndlnm(Mvec, zvec)``
    always returns the outer grid ``(nM, nz)``.
    """

    def __init__(self, cosmo=None, k=None, pk=None, mvec=None,
                 zvec=None, delta: float = 200.0, truncate: bool = True,
                 sigma_grid=None, cache: bool = True):
        d = np.asarray(TINKER08_TABLE2["delta"], dtype=float)
        if not (d[0] <= delta <= d[-1]):
            raise ValueError(
                f"Tinker (2008) is calibrated for {d[0]:.0f} <= Delta <= "
                f"{d[-1]:.0f}, got {delta}"
            )
        if (k is None) != (pk is None):
            raise ValueError("k and pk must be supplied together")
        self.cosmo = fiducial_cosmology() if cosmo is None else cosmo
        self.rhom = mean_matter_density(self.cosmo)   # Msun/Mpc^3, comoving
        self.k = None if k is None else np.asarray(k, dtype=float)
        self.pk = None if pk is None else np.asarray(pk, dtype=float)
        self._sigma_grid = sigma_grid
        self._sigma_grid_injected = sigma_grid is not None
        self.cache = cache
        self.delta = float(delta)
        self.truncate = truncate
        # default z grid; a single-z grid would silently return that z
        # for every query
        self.zvec = (np.linspace(0.0, 1.5, 31) if zvec is None
                     else np.atleast_1d(np.asarray(zvec, dtype=float)))
        # default mass grid: the production Lagrangian radii (Mpc/h ->
        # Mpc is the one visible h conversion, at this boundary only)
        self.mval = (self.mass_of_radius(np.exp(lnr_grid()) / self.cosmo.h)
                     if mvec is None
                     else np.atleast_1d(np.asarray(mvec, dtype=float)))

        ld, target = np.log10(d), np.log10(self.delta)
        self.A0, self.a0, self.b0, self.c = (
            float(np.interp(target, ld,
                            np.asarray(TINKER08_TABLE2[key], dtype=float)))
            for key in ("A0", "a0", "b0", "c")
        )
        # eq. 8: the exponent of the b(z) evolution
        self.alpha = 10.0 ** (-((0.75 / (target - _LOG10_75)) ** 1.2))

        # cache path is fixed at construction: build() takes no grid
        # overrides, so (mval, zvec) here are the ones that will ever be
        # used. An injected sigma_grid cannot be hashed by content, so
        # caching is skipped for it (see class docstring).
        self._cache_file = (
            None if (not self.cache or self._sigma_grid_injected)
            else self._compute_cache_path()
        )

    def _compute_cache_path(self) -> Path:
        spec = dict(
            cosmo=_astropy_to_dict(self.cosmo),
            delta=self.delta,
            truncate=self.truncate,
            mval=_hash_arrays(self.mval),
            zvec=_hash_arrays(self.zvec),
        )
        if self.k is not None:
            spec["kpk"] = _hash_arrays(self.k, self.pk)
        return _cache_dir("hmf_cache") / f"{_hash(spec)}.npz"

    @property
    def sigma_grid(self):
        r"""The :math:`\sigma^2` evaluator: injected, or built on first use
        from ``(k, pk)``, else from a z=0 linear `PkGrid` of `cosmo`."""
        if self._sigma_grid is None:
            k, pk = self.k, self.pk
            if k is None:
                pk_grid = PkGrid(cosmo=self.cosmo, nonlinear=False)
                k, pk = pk_grid.k, pk_grid(pk_grid.k, z=0.0)
            self._sigma_grid = SigmaGrid(k, pk)
        return self._sigma_grid

    @property
    def dndlnm_grid(self):
        r""":math:`dn/d\ln M` on ``mval`` x ``zvec`` [Mpc^-3], built once
        on first use from `sigma_grid`, with
        :math:`\sigma(R,z) = D(z)\,\sigma(R,0)`."""
        if getattr(self, "_dndlnm_grid", None) is None:
            r = self.radius_of_mass(self.mval)
            sigma0 = np.array([
                np.sqrt(self.sigma_grid.sigma2(ri, truncate=self.truncate))
                for ri in r
            ])
            dln_sigma2 = np.array([
                self.sigma_grid.dlnsigma2_dlnr(ri, truncate=self.truncate)
                for ri in r
            ])
            prefactor = -self.rhom / (6.0 * self.mval) * dln_sigma2
            grid = np.empty((len(self.mval), len(self.zvec)))
            for iz, zi in enumerate(self.zvec):
                d_z = growth_factor(zi, self.cosmo)
                grid[:, iz] = prefactor * self.f_sigma(sigma0 * d_z, zi)
            self._dndlnm_grid = grid
            self._interp = LogGridInterpolator(self.mval, self.zvec, grid)
        return self._dndlnm_grid

    def coefficients(self, z):
        r"""``(A, a, b, c)`` at ``z`` -- Tinker (2008) Eqs. 5--8."""
        one_plus_z = 1.0 + np.asarray(z, dtype=float)
        return (
            self.A0 * one_plus_z**-0.14,        # eq. 5
            self.a0 * one_plus_z**-0.06,        # eq. 6
            self.b0 * one_plus_z**-self.alpha,  # eq. 7, exponent from eq. 8
            np.full_like(one_plus_z, self.c),   # c does not evolve
        )

    @scalar_array_output
    def f_sigma(self, sigma, z=0.0):
        r"""The multiplicity function :math:`f(\sigma)` -- Tinker (2008)
        eq. 3, dimensionless."""
        sigma = np.asarray(sigma, dtype=float)
        if np.any(sigma <= 0.0):
            raise ValueError("sigma must be positive")
        A, a, b, c = self.coefficients(z)
        return A * ((sigma / b) ** (-a) + 1.0) * np.exp(-c / sigma**2)

    @scalar_array_output
    def mass_of_radius(self, r):
        r""":math:`M(R) = \frac{4\pi}{3}\bar\rho_m R^3` [Msun], R in Mpc."""
        r = np.asarray(r, dtype=float)
        return (4.0 * np.pi / 3.0) * self.rhom * r**3

    @scalar_array_output
    def radius_of_mass(self, m):
        r"""The Lagrangian radius :math:`R(M) = (3M/4\pi\bar\rho_m)^{1/3}`
        [Mpc], the inverse of `mass_of_radius`."""
        m = np.asarray(m, dtype=float)
        return (3.0 * m / (4.0 * np.pi * self.rhom)) ** (1.0 / 3.0)

    @default_mvals_z
    def dndlnm(self, M_vals=None, z=None):
        r"""``dn/dlnM(M, z)`` [Mpc^-3], interpolated from `dndlnm_grid`.

        Parameters
        ----------
        M_vals : array-like, optional
            Masses [Msun]. Defaults to `self.mval`.
        z : float or array-like, optional
            Redshift(s). Defaults to `self.zvec`. Vector ``M_vals``
            and vector ``z`` always return the outer ``(nM, nz)`` grid.
        """
        if not self.is_built:
            self.build()
        return self._interp(M_vals, z)

    def build(self):
        """Materialize the cached mass-function grid and return ``self``."""
        if getattr(self, "_dndlnm_grid", None) is not None:
            return self
        if self._cache_file is not None and self._cache_file.exists():
            grid = np.load(self._cache_file)["grid"]
            if grid.shape != (len(self.mval), len(self.zvec)):
                raise RuntimeError(
                    f"cache shape mismatch at {self._cache_file}: "
                    f"{grid.shape} vs expected "
                    f"{(len(self.mval), len(self.zvec))}"
                )
            self._dndlnm_grid = grid
            self._interp = LogGridInterpolator(self.mval, self.zvec, grid)
            print(f"TinkerMassFunction loaded cache file: {self._cache_file}")
            return self
        _ = self.dndlnm_grid  # fills self._dndlnm_grid, self._interp
        if self._cache_file is not None:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self._cache_file, grid=self._dndlnm_grid)
            print(f"TinkerMassFunction saved cache file: {self._cache_file}")
        return self

    build_all = build  # alias, one release

    @property
    def is_built(self) -> bool:
        """Whether the (M, z) grid has been materialized."""
        return getattr(self, "_dndlnm_grid", None) is not None

    def __repr__(self):
        return (f"TinkerMassFunction(Delta={self.delta:.0f}m, "
                f"A0={self.A0:.4f}, a0={self.a0:.3f}, b0={self.b0:.3f}, "
                f"c={self.c:.3f}, alpha={self.alpha:.5f})")


if __name__ == "__main__":
    # toy sigma grid: low-order polynomial in ln sigma vs ln R (LCDM-ish),
    # no P(k), no Boltzmann solver, derivative analytic by construction.
    #   ln sigma(R) = ln 0.8 - 0.6 x - 0.07 x^2,   x = ln(R / 8 Mpc)
    class ToySigma:
        def sigma2(self, r, truncate=True):
            x = np.log(np.asarray(r, dtype=float) / 8.0)
            return np.exp(2.0 * (np.log(0.8) - 0.6 * x - 0.07 * x**2))

        def dlnsigma2_dlnr(self, r, truncate=True):
            x = np.log(np.asarray(r, dtype=float) / 8.0)
            return 2.0 * (-0.6 - 0.14 * x)

    hmf = TinkerMassFunction(sigma_grid=ToySigma(),   # the injection point
                             mvec=np.logspace(12.5, 16.0, 140),
                             zvec=np.array([0.0, 1.0]))
    print(hmf, "\n")

    M = np.logspace(13, 15.5, 6)                      # Msun
    dn0, dn1 = hmf.dndlnm(M, z=0.0), hmf.dndlnm(M, z=1.0)
    print(f"{'M [Msun]':>11s}  {'dn/dlnM z=0':>12s}  {'z=1':>12s}")
    for m, a, b in zip(M, np.ravel(dn0), np.ravel(dn1)):
        print(f"{m:11.2e}  {a:12.4e}  {b:12.4e}")

    # physics checks: falls with M; falls with z at cluster masses
    assert np.all(np.diff(np.ravel(dn0)) < 0)
    assert np.all(np.ravel(dn1) < np.ravel(dn0))
    # Lagrangian radius round trip
    r = hmf.radius_of_mass(1e14)
    assert abs(hmf.mass_of_radius(r) / 1e14 - 1.0) < 1e-12
    print("\ndemo OK")
