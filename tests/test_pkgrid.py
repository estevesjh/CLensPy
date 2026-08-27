# tests/test_pkgrid.py
"""PkGrid: cached 2-D P(k, z), its interpolation, and its cache round-trip.

Backend builds (`_grid_from_camb` / `_grid_from_pyccl`) are exercised with
tiny grids since both call out to a real Boltzmann/halofit solver. Cache
and dict/hash helpers are pure-Python and get exact-equality checks.
"""

import json
import types

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from clenspy.cosmology.fiducial import fiducial_cosmology
from clenspy.cosmology.pkgrid import PkGrid, _astropy_to_dict, _hash


def make_pkgrid(**overrides):
    """Small, fast PyCCL-backed grid, cache disabled by default."""
    kwargs = dict(
        backend="pyccl",
        cosmo=fiducial_cosmology(),
        nonlinear=False,
        k_range=(1e-3, 1.0),
        z_range=(0.0, 1.0),
        nk=8,
        nz=4,
        cache=False,
    )
    kwargs.update(overrides)
    return PkGrid(**kwargs)


class TestCallBroadcasting:
    """__call__(k, z) broadcasting and value sanity."""

    @pytest.fixture(scope="class")
    def grid(self):
        return make_pkgrid()

    def test_scalar_in_scalar_out(self, grid):
        val = grid(0.1, 0.5)
        assert isinstance(val, float)
        assert np.isfinite(val)
        assert val > 0

    def test_array_k_scalar_z(self, grid):
        k = np.logspace(-2, -0.1, 6)
        val = grid(k, 0.3)
        assert isinstance(val, np.ndarray)
        assert val.shape == k.shape
        assert np.all(np.isfinite(val))
        assert np.all(val > 0)

    def test_scalar_k_array_z(self, grid):
        z = np.linspace(0.0, 0.8, 5)
        val = grid(0.05, z)
        assert isinstance(val, np.ndarray)
        assert val.shape == z.shape
        assert np.all(np.isfinite(val))
        assert np.all(val > 0)

    def test_elementwise_matching_shapes(self, grid):
        k = np.logspace(-2, -0.1, 5)
        z = np.linspace(0.0, 0.8, 5)
        val = grid(k, z)
        assert val.shape == (5,)
        assert np.all(np.isfinite(val))
        assert np.all(val > 0)

    def test_full_grid_broadcast(self, grid):
        # (N,1) k against (M,) z broadcasts to (N,M), NumPy-style.
        k = np.logspace(-2, -0.1, 4).reshape(-1, 1)
        z = np.linspace(0.0, 0.8, 3)
        val = grid(k, z)
        assert val.shape == (4, 3)
        assert np.all(np.isfinite(val))
        assert np.all(val > 0)


class TestCacheRoundTrip:
    """cache=True: build once, reuse the .npz on the second construction."""

    def spec_kwargs(self):
        return dict(
            backend="pyccl",
            cosmo=fiducial_cosmology(),
            nonlinear=False,
            k_range=(1e-3, 1.0),
            z_range=(0.0, 1.0),
            nk=8,
            nz=4,
            cache=True,
        )

    def test_cache_file_written(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLENSPY_DATA", str(tmp_path))
        PkGrid(**self.spec_kwargs())

        cache_dir = tmp_path / "pk_cache"
        assert cache_dir.is_dir()
        npz_files = list(cache_dir.glob("*.npz"))
        assert len(npz_files) == 1

    def test_second_instance_loads_from_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLENSPY_DATA", str(tmp_path))
        first = PkGrid(**self.spec_kwargs())

        def _boom(self):
            raise AssertionError("_build_grid should not be called on cache hit")

        monkeypatch.setattr(PkGrid, "_build_grid", _boom)

        second = PkGrid(**self.spec_kwargs())
        assert np.array_equal(second.k, first.k)
        assert np.array_equal(second.z, first.z)
        assert np.array_equal(second.pk, first.pk)


class TestFileHelpers:
    """_dump_to_file / _load_from_file round-trip exactly."""

    def test_dump_and_load(self, tmp_path):
        grid = make_pkgrid()
        path = tmp_path / "x.npz"
        grid._dump_to_file(path)
        assert path.exists()

        loaded = PkGrid.__new__(PkGrid)
        loaded._load_from_file(path)

        assert np.array_equal(loaded.k, grid.k)
        assert np.array_equal(loaded.z, grid.z)
        assert np.array_equal(loaded.pk, grid.pk)


class TestBackendDispatch:
    """_build_grid raises for a backend it doesn't recognize."""

    def test_unsupported_backend_raises(self):
        with pytest.raises(ValueError):
            make_pkgrid(backend="bogus")


class TestRealBackendBuilds:
    """Tiny end-to-end grids from both real solvers."""

    def test_camb_grid_shape_and_values(self):
        grid = PkGrid(
            backend="camb",
            cosmo=fiducial_cosmology(),
            nonlinear=False,
            k_range=(1e-2, 1.0),
            z_range=(0.0, 1.0),
            nk=6,
            nz=2,
            cache=False,
        )
        assert grid.pk.shape == (2, 6)
        assert np.all(np.isfinite(grid.pk))
        assert np.all(grid.pk > 0)

    def test_pyccl_grid_shape_and_values(self):
        grid = PkGrid(
            backend="pyccl",
            cosmo=fiducial_cosmology(),
            nonlinear=False,
            k_range=(1e-2, 1.0),
            z_range=(0.0, 1.0),
            nk=6,
            nz=2,
            cache=False,
        )
        assert grid.pk.shape == (2, 6)
        assert np.all(np.isfinite(grid.pk))
        assert np.all(grid.pk > 0)


class TestAstropyToDict:
    """_astropy_to_dict: Omega_b fallback and sigma8/n_s defaults."""

    def test_ob0_zero_falls_back_to_default(self):
        # FlatLambdaCDM defaults Ob0 to 0.0 (not None) when unset.
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        assert cosmo.Ob0 == 0.0
        d = _astropy_to_dict(cosmo)
        assert d["Omega_b"] == 0.05

    def test_ob0_explicit_is_read_through(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        d = _astropy_to_dict(cosmo)
        assert d["Omega_b"] == 0.05

    def test_sigma8_ns_default_when_absent(self):
        fake = types.SimpleNamespace(h=0.7, Om0=0.3, Ob0=0.05, Ok0=0.0)
        d = _astropy_to_dict(fake)
        assert d["sigma8"] == 0.8
        assert d["n_s"] == 0.96

    def test_sigma8_ns_read_when_present(self):
        fake = types.SimpleNamespace(
            h=0.7, Om0=0.3, Ob0=0.05, Ok0=0.0, sigma8=0.75, n_s=0.9,
        )
        d = _astropy_to_dict(fake)
        assert d["sigma8"] == 0.75
        assert d["n_s"] == 0.9

    def test_remaining_fields_pass_through(self):
        fake = types.SimpleNamespace(h=0.65, Om0=0.31, Ob0=0.049, Ok0=0.01)
        d = _astropy_to_dict(fake)
        assert d["h"] == 0.65
        assert d["Omega_m"] == 0.31
        assert d["Omega_k"] == 0.01


class TestHash:
    """_hash: order-independent, value-sensitive md5 of the sorted JSON."""

    def test_key_order_does_not_change_hash(self):
        d1 = {"a": 1, "b": 2, "c": {"x": 1, "y": 2}}
        d2 = {"c": {"y": 2, "x": 1}, "b": 2, "a": 1}
        assert _hash(d1) == _hash(d2)

    def test_changed_value_changes_hash(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"a": 1, "b": 3}
        assert _hash(d1) != _hash(d2)

    def test_hash_is_deterministic_md5_hex(self):
        d = {"a": 1, "b": 2}
        expected = __import__("hashlib").md5(
            json.dumps(d, sort_keys=True).encode()
        ).hexdigest()
        assert _hash(d) == expected
