# clenspy/cosmology/pkgrid.py
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.interpolate import RectBivariateSpline

# ------------------------------------------------------------------
# Helpers -----------------------------------------------------------
# ------------------------------------------------------------------


# 1) absolute path to the *package* root
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # clenspy/ → ..

# 2) default data dir inside the project tree
_DEFAULT_DATA = _PACKAGE_ROOT / "data"


def _data_dir() -> Path:
    """
    Directory for auto-cached grids.

    Precedence:
      1. user sets env-var CLENSPY_DATA
      2. fallback to <package>/data
    """
    root = os.environ.get("CLENSPY_DATA", str(_DEFAULT_DATA))
    path = Path(root).expanduser()
    (path / "pk_cache").mkdir(parents=True, exist_ok=True)
    return path / "pk_cache"


def _hash(spec: dict) -> str:
    return hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def _astropy_to_dict(cosmo, *, sigma8=0.8, n_s=0.96) -> dict:
    """Translate an Astropy cosmology to the scalar dict CAMB / PyCCL expect."""
    ob0 = cosmo.Ob0
    return dict(
        h=cosmo.h,
        Omega_m=cosmo.Om0,
        Omega_b=ob0 if ob0 is not None else 0.05,  # default if not set
        Omega_k=cosmo.Ok0,
        sigma8=getattr(cosmo, "sigma8", sigma8),
        n_s=getattr(cosmo, "n_s", n_s),
    )


# ------------------------------------------------------------------
# Main class --------------------------------------------------------
# ------------------------------------------------------------------
class PkGrid:
    r"""
    Parameters
    ----------
    backend : {"camb", "pyccl"}
        Library used to compute the grid.
    cosmo : astropy.cosmology.Cosmology
        Cosmology object that defines (h, Ω_m, Ω_b, …).
    nonlinear : bool, default False
        Use the non-linear (Halofit) spectrum if supported.
    k_range  : Tuple[float, float], default (1e-4, 10.0)   [1/Mpc]
    z_range  : Tuple[float, float], default (0.0, 2.0)
    nk, nz   : int, default (200, 41)
    cache    : bool, default True
        If True, store / reuse *.npz files in ``clenspy-data/pk_cache``.
    """

    def __init__(
        self,
        *,
        backend: str = "camb",
        cosmo,
        nonlinear: bool = False,
        k_range: Tuple[float, float] = (1e-4, 10.0),
        z_range: Tuple[float, float] = (0.0, 1.0),
        nk: int = 512,
        nz: int = 100,
        cache: bool = True,
    ) -> None:

        self.backend = backend.lower()
        self.cosmo_dict = _astropy_to_dict(cosmo)
        self.nonlinear = nonlinear

        self.k = np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), nk)
        self.z = np.linspace(*z_range, nz)

        # ----------------------------------------------------------
        # 1) Try cache
        # ----------------------------------------------------------
        spec = dict(
            backend=self.backend,
            nonlinear=self.nonlinear,
            cosmo=self.cosmo_dict,
            k_range=k_range,
            z_range=z_range,
            nk=nk,
            nz=nz,
        )
        self._cache_file = _data_dir() / f"{_hash(spec)}.npz"

        if cache and self._cache_file.exists():
            self._load_from_file(self._cache_file)
            print(f"PkGrid loaded cache file ({self.backend}): {self._cache_file}")
        else:
            self._build_grid()  # fill self.pk
            if cache:
                self._dump_to_file(self._cache_file)
                print(f"PkGrid saved cache file {self.backend}: {self._cache_file}")

        # lazy spline (built on first call)
        self._spline: RectBivariateSpline | None = None

    # ------------------------------------------------------------------
    # Public call interface --------------------------------------------
    # ------------------------------------------------------------------
    def __call__(self, k, z):
        """
        Evaluate P(k, z) with broadcasting.

        Parameters
        ----------
        k : float or array_like
            Wave-number(s) [1/Mpc].
        z : float or array_like
            Redshift(s).

        Returns
        -------
        float or ndarray
            P(k, z) in the same unit system as the stored grid.
            * scalar in  ➜  scalar out
            * (N,) and scalar  ➜  (N,) out
            * scalar and (M,)  ➜  (M,) out
            * (N,M) & (M,) etc. obey NumPy broadcasting rules
        """
        # --- build spline once -------------------------------------------
        if self._spline is None:
            from scipy.interpolate import RectBivariateSpline

            self._spline = RectBivariateSpline(
                self.z,
                np.log(self.k),
                np.log(self.pk),
                kx=3,
                ky=1,  # cubic in z, linear in ln k
            )

        k_arr = np.asarray(k, dtype=float)
        z_arr = np.asarray(z, dtype=float)

        # Determine broadcast shape
        try:
            tgt_shape = np.broadcast_shapes(k_arr.shape, z_arr.shape)
        except AttributeError:  # numpy <1.20 fallback
            tgt_shape = np.broadcast(k_arr, z_arr).shape

        # Broadcast inputs to that shape
        kk = np.broadcast_to(k_arr, tgt_shape).ravel()
        zz = np.broadcast_to(z_arr, tgt_shape).ravel()

        # Evaluate log-spline on flattened grid, then reshape
        vals = np.exp(self._spline.ev(zz, np.log(kk))).reshape(tgt_shape)

        # Return scalar if both inputs were scalar
        if vals.size == 1:
            return float(vals)
        return vals

    # ------------------------------------------------------------------
    # Internal grid generation -----------------------------------------
    # ------------------------------------------------------------------
    def _build_grid(self):
        if self.backend == "camb":
            self.pk = self._grid_from_camb()
        elif self.backend == "pyccl":
            self.pk = self._grid_from_pyccl()
        else:
            raise ValueError(f"Unsupported backend '{self.backend}'")

    # -------- CAMB ----------------------------------------------------
    def _grid_from_camb(self) -> np.ndarray:
        try:
            import camb
        except ImportError as e:
            raise ImportError(
                "backend='camb' requested but CAMB is not installed"
            ) from e
        As_guess = 2.1e-9  # reasonable default for CAMB
        p = self.cosmo_dict
        pars = camb.set_params(
            H0=p["h"] * 100,
            ombh2=p["Omega_b"] * p["h"] ** 2,
            omch2=(p["Omega_m"] - p["Omega_b"]) * p["h"] ** 2,
            omk=p["Omega_k"],
            ns=p["n_s"],
            As=As_guess,  # reasonable default for CAMB
            # sigma8 = p["sigma8"],        # set_params supports sigma8
        )
        pars.set_matter_power(redshifts=[0], kmax=self.k[-1])
        res = camb.get_results(pars)
        sigma8_now = res.get_sigma8()  # σ8 at z=0
        target = p["sigma8"]
        As_new = pars.InitPower.As * (target / sigma8_now) ** 2

        pars.InitPower.set_params(ns=p["n_s"], As=As_new)
        pars.set_matter_power(redshifts=self.z.tolist(), kmax=self.k[-1])
        pars.NonLinear = (
            camb.model.NonLinear_both if self.nonlinear else camb.model.NonLinear_none
        )

        res = camb.get_results(pars)
        kh, zz, pk = res.get_matter_power_spectrum(
            minkh=self.k[0], maxkh=self.k[-1], npoints=len(self.k)
        )
        # Ensure order matches self.k (they should)
        return pk  # shape (nz, nk)

    # -------- PyCCL ---------------------------------------------------
    def _grid_from_pyccl(self) -> np.ndarray:
        try:
            import pyccl as ccl
        except ImportError as e:
            raise ImportError(
                "backend='pyccl' requested but PyCCL is not installed"
            ) from e

        p = self.cosmo_dict
        ccl_cosmo = ccl.Cosmology(
            Omega_c=p["Omega_m"] - p["Omega_b"],
            Omega_b=p["Omega_b"],
            h=p["h"],
            n_s=p["n_s"],
            sigma8=p["sigma8"],
        )

        fn = ccl.nonlin_matter_power if self.nonlinear else ccl.linear_matter_power

        pk = np.zeros((len(self.z), len(self.k)))
        for i, zi in enumerate(self.z):
            pk[i] = fn(ccl_cosmo, self.k, 1.0 / (1.0 + zi))
        return pk

    # ------------------------------------------------------------------
    # File helpers ------------------------------------------------------
    # ------------------------------------------------------------------
    def _dump_to_file(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, k=self.k, z=self.z, pk=self.pk, meta=self.cosmo_dict)

    def _load_from_file(self, path: Path) -> None:
        d = np.load(path, allow_pickle=True)
        self.k = d["k"]
        self.z = d["z"]
        self.pk = d["pk"]


if __name__ == "__main__":
    print("Running pkgrid.py as a script")
    from clenspy.config import DEFAULT_COSMOLOGY

    # Linear spectrum from CAMB
    pk_camb = PkGrid(backend="camb", cosmo=DEFAULT_COSMOLOGY, nonlinear=True)
    pk_camb = PkGrid(backend="camb", cosmo=DEFAULT_COSMOLOGY)

    # Non-linear spectrum from PyCCL
    pk_ccl_nl = PkGrid(backend="pyccl", cosmo=DEFAULT_COSMOLOGY, nonlinear=True)
