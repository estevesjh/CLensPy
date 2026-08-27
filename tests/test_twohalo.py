"""Does `TwoHaloTerm` run, and are its three outputs mutually consistent?

NOTE: the comparison against `cluster_toolkit` and CLMM that used to be the
only test in this file is now `validation/validate_twohalo_chain.py`, where
it checks each transform stage against a closed-form NFW instead of only the
end of the chain. See ``docs/validation.md``.

These are cheap invariants that need no external library: shapes, signs,
monotonicity, and the identity relating the three projections.
"""

import numpy as np
import pytest

from clenspy.halo.twohalo import TwoHaloTerm, prepare_pk_grid

#: A pure power law, so xi(r) is a pure power law and the projections are
#: strictly monotonic -- the invariants below are then exact statements
#: rather than statements about this particular input. A cored P(k) gives a
#: genuinely flat Sigma core and would make the monotonicity check false for
#: physical reasons, which is not what these tests are for.
K = np.logspace(-3, 1, 64)  # 1/Mpc
PK = 2e4 * K ** (-1.5)
Z = 0.2


@pytest.fixture
def twohalo():
    return TwoHaloTerm(K, PK, zvec=Z)


def test_xi_is_finite_and_decreasing(twohalo):
    r = np.logspace(-1, 1.5, 30)
    xi = np.ravel(twohalo.xi(r, Z))
    assert np.all(np.isfinite(xi))
    assert np.all(np.diff(xi) < 0)


def test_sigma_and_deltasigma_are_positive_and_decreasing(twohalo):
    R = np.logspace(-1, 1, 25)
    for name in ("sigma", "deltasigma"):
        v = np.ravel(getattr(twohalo, name)(R, Z))
        assert np.all(np.isfinite(v)), f"{name} must be finite"
        assert np.all(v > 0), f"{name} must be positive"
        assert np.all(np.diff(v) < 0), f"{name} must decrease outward"


def test_sigma_falls_off_faster_than_xi():
    """Projection is shallower than the 3D profile, never steeper."""
    th = TwoHaloTerm(K, PK, zvec=Z)
    r = np.array([1.0, 4.0])
    slope_xi = np.diff(np.log(np.ravel(th.xi(r, Z)))) / np.diff(np.log(r))
    slope_sig = np.diff(np.log(np.ravel(th.sigma(r, Z)))) / np.diff(np.log(r))
    assert slope_sig > slope_xi


def test_p_kz_reproduces_the_input_spectrum(twohalo):
    """The P(k, z) interpolator must pass its own input through."""
    k = K[5:-5]  # off the interpolation edges
    np.testing.assert_allclose(
        np.ravel(twohalo.p_kz(k, Z)), PK[5:-5], rtol=2e-2
    )


def test_scalar_and_array_radii_agree(twohalo):
    """A scalar R gives the same number as the length-1 array."""
    for name in ("xi", "sigma", "deltasigma"):
        method = getattr(twohalo, name)
        np.testing.assert_allclose(
            np.ravel(method(1.5, Z)), np.ravel(method(np.array([1.5]), Z))
        )


def test_unsorted_k_is_accepted():
    """A descending k grid must give the same answer as an ascending one."""
    order = np.argsort(-K)
    shuffled = TwoHaloTerm(K[order], PK[order], zvec=Z)
    R = np.logspace(-1, 0.5, 8)
    np.testing.assert_allclose(
        np.ravel(shuffled.sigma(R, Z)),
        np.ravel(TwoHaloTerm(K, PK, zvec=Z).sigma(R, Z)),
        rtol=1e-6,
    )


@pytest.mark.parametrize("method", ["trapz", "quad_vec"])
def test_quadrature_backends_agree(method):
    """The Abel backends must agree to better than their own accuracy."""
    R = np.logspace(-1, 0.5, 10)
    ref = np.ravel(TwoHaloTerm(K, PK, zvec=Z, method="quad_vec").sigma(R, Z))
    got = np.ravel(TwoHaloTerm(K, PK, zvec=Z, method=method).sigma(R, Z))
    np.testing.assert_allclose(got, ref, rtol=1e-3)


class TestBuildAll:
    """`build_all` is a side-effecting convenience wrapper around the three
    per-quantity methods -- it should chain (return `self`) and leave all
    three interpolators cached.
    """

    def test_returns_self(self, twohalo):
        result = twohalo.build_all()
        assert result is twohalo

    def test_caches_all_three_interpolators(self, twohalo):
        for attr in ("xi_rz_interp", "sigma_rz_interp", "deltasigma_rz_interp"):
            assert not hasattr(twohalo, attr)
        twohalo.build_all()
        for attr in ("xi_rz_interp", "sigma_rz_interp", "deltasigma_rz_interp"):
            assert hasattr(twohalo, attr)
            assert getattr(twohalo, attr) is not None

    def test_matches_calling_methods_individually(self):
        """`build_all()` then read vs. calling each method directly agree."""
        R = np.logspace(-1, 0.5, 8)
        built = TwoHaloTerm(K, PK, zvec=Z).build_all()
        direct = TwoHaloTerm(K, PK, zvec=Z)
        for name in ("xi", "sigma", "deltasigma"):
            np.testing.assert_allclose(
                np.ravel(getattr(built, name)(R, Z)),
                np.ravel(getattr(direct, name)(R, Z)),
            )


class TestPreparePkGrid:
    """Direct unit tests of `prepare_pk_grid`'s shape-normalization branches.

    Uses small synthetic grids where each z-column (or row) is tagged with
    its own z-value, so the returned `Pk_grid` values can be checked
    directly rather than only its shape.
    """

    kvec = np.logspace(-2, 1, 5)
    nk = len(kvec)
    zvec = np.array([0.1, 0.3, 0.6])
    nz = len(zvec)

    def test_zvec_none_pk_1d(self):
        """(a) zvec is None, Pk is 1D -> single z=0.0 column."""
        Pk = np.arange(self.nk, dtype=float) + 1.0
        k, Pk_grid, zvec = prepare_pk_grid(self.kvec, Pk, None)
        np.testing.assert_array_equal(k, self.kvec)
        assert Pk_grid.shape == (self.nk, 1)
        np.testing.assert_array_equal(zvec, [0.0])
        np.testing.assert_array_equal(Pk_grid[:, 0], Pk)

    def test_zvec_none_pk_column(self):
        """(b) zvec is None, Pk already shape (nk, 1) -> passed through."""
        Pk = (np.arange(self.nk, dtype=float) + 1.0)[:, None]
        k, Pk_grid, zvec = prepare_pk_grid(self.kvec, Pk, None)
        assert Pk_grid.shape == (self.nk, 1)
        np.testing.assert_array_equal(zvec, [0.0])
        np.testing.assert_array_equal(Pk_grid, Pk)

    def test_zvec_none_bad_pk_shape_raises(self):
        """(c) zvec is None, Pk is 2D but not (nk, 1) -> ValueError."""
        Pk = np.ones((self.nk, 3))
        with pytest.raises(ValueError, match="must be 1D or shape"):
            prepare_pk_grid(self.kvec, Pk, None)

    def test_zvec_given_pk_1d_is_tiled(self):
        """(d) zvec given, Pk is 1D -> tiled identically across all z."""
        Pk = np.arange(self.nk, dtype=float) + 1.0
        k, Pk_grid, zvec = prepare_pk_grid(self.kvec, Pk, self.zvec)
        assert Pk_grid.shape == (self.nk, self.nz)
        for j in range(self.nz):
            np.testing.assert_array_equal(Pk_grid[:, j], Pk)
        np.testing.assert_array_equal(zvec, self.zvec)

    def test_zvec_given_pk_nz_by_nk_is_transposed(self):
        """(e) zvec given, Pk shape (nz, nk) -> transposed to (nk, nz)."""
        Pk = np.zeros((self.nz, self.nk))
        for i, zval in enumerate(self.zvec):
            Pk[i, :] = zval
        k, Pk_grid, zvec = prepare_pk_grid(self.kvec, Pk, self.zvec)
        assert Pk_grid.shape == (self.nk, self.nz)
        for j, zval in enumerate(self.zvec):
            np.testing.assert_array_equal(Pk_grid[:, j], np.full(self.nk, zval))

    def test_zvec_given_pk_nk_by_nz_used_as_is(self):
        """(f) zvec given, Pk already shape (nk, nz) -> used as-is."""
        Pk = np.zeros((self.nk, self.nz))
        for j, zval in enumerate(self.zvec):
            Pk[:, j] = zval
        k, Pk_grid, zvec = prepare_pk_grid(self.kvec, Pk, self.zvec)
        assert Pk_grid.shape == (self.nk, self.nz)
        np.testing.assert_array_equal(Pk_grid, Pk)

    def test_zvec_given_bad_pk_shape_raises(self):
        """(g) zvec given, Pk shape matches neither (nk,nz) nor (nz,nk)."""
        Pk = np.ones((self.nz + 4, self.nk))
        with pytest.raises(ValueError, match="Pk shape must be"):
            prepare_pk_grid(self.kvec, Pk, self.zvec)

    def test_unsorted_zvec_raises(self):
        """zvec must be strictly increasing on input -- an out-of-order
        zvec fails the internal assertion rather than being silently
        re-sorted (the sort branch further down only re-orders columns to
        match an unsorted *kvec*, not an unsorted zvec)."""
        unsorted_zvec = np.array([0.6, 0.1, 0.3])
        Pk = np.zeros((self.nz, self.nk))
        for i, zval in enumerate(unsorted_zvec):
            Pk[i, :] = zval
        with pytest.raises(AssertionError, match="strictly increasing"):
            prepare_pk_grid(self.kvec, Pk, unsorted_zvec)

    def test_unsorted_kvec_with_pk_nk_by_nz_reorders_columns(self):
        """(h, first sort path) An unsorted kvec triggers the re-sort
        branch; with Pk_grid already in (nk, nz) form it re-orders columns
        by `argsort(zvec)`. Since zvec here is already sorted, argsort is
        the identity permutation and the result is unchanged."""
        unsorted_kvec = self.kvec[::-1]
        Pk = np.zeros((self.nk, self.nz))
        for j, zval in enumerate(self.zvec):
            Pk[:, j] = zval
        k, Pk_grid, zvec = prepare_pk_grid(unsorted_kvec, Pk, self.zvec)
        np.testing.assert_array_equal(zvec, self.zvec)
        assert Pk_grid.shape == (self.nk, self.nz)
        for j, zval in enumerate(self.zvec):
            np.testing.assert_array_equal(Pk_grid[:, j], np.full(self.nk, zval))

    def test_unsorted_kvec_with_pk_nz_by_nk_reorders_columns(self):
        """(h, second sort path) Same as above, but starting from Pk given
        as (nz, nk) -- it is transposed to (nk, nz) before the unsorted-k
        branch runs, so it takes the same `[:, sort_idx]` path (the
        `(len(zvec), nk)` branch of the re-sort code is unreachable in
        practice, since `Pk_grid` is always normalized to (nk, nz) by the
        time the re-sort runs)."""
        unsorted_kvec = self.kvec[::-1]
        Pk = np.zeros((self.nz, self.nk))
        for i, zval in enumerate(self.zvec):
            Pk[i, :] = zval
        k, Pk_grid, zvec = prepare_pk_grid(unsorted_kvec, Pk, self.zvec)
        np.testing.assert_array_equal(zvec, self.zvec)
        assert Pk_grid.shape == (self.nk, self.nz)
        for j, zval in enumerate(self.zvec):
            np.testing.assert_array_equal(Pk_grid[:, j], np.full(self.nk, zval))
