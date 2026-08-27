import numpy as np
import pytest

from clenspy.selection.boost import (
    BoostFactorCollection,
    BoostFactorData,
    boost_factor_nfw,
    load_boost_factor_collection,
    load_boost_factor_data,
    scale_cuts,
)


# ---------------------------------------------------------------------------
# boost_factor_nfw
# ---------------------------------------------------------------------------

def test_boost_factor_nfw_finite_and_greater_than_one():
    rs = 0.35
    B0 = 0.1
    R = np.array([0.05, 0.15, 0.3, rs, 0.5, 1.0, 3.0])  # spans x<1, x==1, x>1
    B = boost_factor_nfw(R, B0, rs)

    assert np.all(np.isfinite(B))
    # B >= 1 everywhere, with equality only exactly at x == 1
    assert np.all(B >= 1)
    assert np.sum(B > 1) == len(R) - 1


def test_boost_factor_nfw_at_x_equal_one_is_exactly_one():
    # x == 1 hits fx[x==1] = 1 and the denominator patch (1e-10), so
    # B = 1 + B0*(1-1)/1e-10 = 1 exactly -- not the (B0+3)/3 NaN fallback.
    rs = 0.35
    B0 = 0.1
    B = boost_factor_nfw(rs, B0, rs)
    assert B.shape == (1,)
    assert B[0] == pytest.approx(1.0)
    # sanity: this is NOT the NaN-fallback value
    assert B[0] != pytest.approx((B0 + 3) / 3)


def test_boost_factor_nfw_broad_range_no_nans():
    rs = 0.35
    B0 = 0.1
    R = np.linspace(0.01, 20.0, 2001)  # densely crosses x == 1
    B = boost_factor_nfw(R, B0, rs)

    assert np.all(np.isfinite(B))
    assert not np.any(np.isnan(B))
    # boost factor should be sane: bounded and > 1 everywhere on this range
    assert np.all(B > 1)
    assert np.all(B < 10)


def test_boost_factor_nfw_decreasing_towards_large_R():
    # dilution should vanish (B -> 1) at large R
    rs = 0.35
    B0 = 0.1
    R = np.array([0.1, 1.0, 10.0, 100.0])
    B = boost_factor_nfw(R, B0, rs)
    assert B[0] > B[1] > B[2] > B[3]
    assert B[-1] == pytest.approx(1.0, abs=1e-2)


def test_boost_factor_nfw_scalar_input_returns_array():
    rs = 0.35
    B0 = 0.1
    B = boost_factor_nfw(0.2, B0, rs)
    assert isinstance(B, np.ndarray)
    assert B.shape == (1,)
    assert np.isfinite(B[0])
    assert B[0] > 1


def test_boost_factor_nfw_x_less_than_one_matches_arctanh_formula():
    rs = 0.35
    B0 = 0.2
    R = np.array([0.05, 0.1, 0.2])
    x = R / rs
    fx = np.arctanh(np.sqrt(1 - x**2)) / np.sqrt(1 - x**2)
    expected = 1 + B0 * (1 - fx) / (x**2 - 1)
    B = boost_factor_nfw(R, B0, rs)
    np.testing.assert_allclose(B, expected)


def test_boost_factor_nfw_x_greater_than_one_matches_arctan_formula():
    rs = 0.35
    B0 = 0.2
    R = np.array([0.5, 1.0, 3.0])
    x = R / rs
    fx = np.arctan(np.sqrt(x**2 - 1)) / np.sqrt(x**2 - 1)
    expected = 1 + B0 * (1 - fx) / (x**2 - 1)
    B = boost_factor_nfw(R, B0, rs)
    np.testing.assert_allclose(B, expected)


# ---------------------------------------------------------------------------
# scale_cuts
# ---------------------------------------------------------------------------

def _make_synthetic_config():
    return BoostFactorData(
        R=np.array([0.05, 0.2, 0.5, 1.0, 6.0]),
        data_vector=np.arange(5.0, dtype=float),
        sigma_B=np.arange(5.0, dtype=float) * 0.1,
        covariance=np.eye(5) * np.arange(1, 6),
        inv_cov=None,
        i_lam=0,
        i_z=0,
    )


def test_scale_cuts_filters_arrays_and_covariance():
    config = _make_synthetic_config()
    original_cov = config.covariance.copy()

    result = scale_cuts(config, r_min=0.1, r_max=5.0)

    # keeps indices 1, 2, 3 (R = 0.2, 0.5, 1.0)
    np.testing.assert_array_equal(result.R, np.array([0.2, 0.5, 1.0]))
    np.testing.assert_array_equal(result.data_vector, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(result.sigma_B, np.array([0.1, 0.2, 0.3]))

    expected_cov = original_cov[np.ix_([1, 2, 3], [1, 2, 3])]
    np.testing.assert_array_equal(result.covariance, expected_cov)
    assert result.covariance.shape == (3, 3)


def test_scale_cuts_mutates_and_returns_same_object():
    config = _make_synthetic_config()
    result = scale_cuts(config, r_min=0.1, r_max=5.0)
    assert result is config


# ---------------------------------------------------------------------------
# load_boost_factor_data / load_boost_factor_collection
# ---------------------------------------------------------------------------

def _write_synthetic_files(tmp_path, lbin, zbin):
    R = np.array([0.05, 0.2, 0.5, 1.0, 6.0])
    data_vector = np.array([2.0, 1.8, 1.5, 1.2, 1.0])
    sigma_B = np.full(5, 0.1)
    covariance = np.eye(5) * 0.01 + 0.001

    stem = tmp_path / (
        f"full-unblind-v2-mcal-zmix_y1clust_l{lbin}_z{zbin}_zpdf_boost"
    )
    data_file = str(stem) + ".dat"
    cov_file = str(stem) + "_cov.dat"

    np.savetxt(data_file, np.column_stack([R, data_vector, sigma_B]))
    np.savetxt(cov_file, covariance)

    return R, data_vector, sigma_B, covariance


def test_load_boost_factor_data(tmp_path):
    R, data_vector, sigma_B, covariance = _write_synthetic_files(tmp_path, 0, 0)

    config = load_boost_factor_data(str(tmp_path), 0, 0, scale_cut=(0.1, 5.0))

    assert isinstance(config, BoostFactorData)
    # keeps in-range points: R = 0.2, 0.5, 1.0
    np.testing.assert_allclose(config.R, np.array([0.2, 0.5, 1.0]))
    np.testing.assert_allclose(config.data_vector, np.array([1.8, 1.5, 1.2]))
    np.testing.assert_allclose(config.sigma_B, np.array([0.1, 0.1, 0.1]))

    expected_cov = covariance[np.ix_([1, 2, 3], [1, 2, 3])]
    np.testing.assert_allclose(config.covariance, expected_cov)

    expected_inv_cov = np.linalg.pinv(expected_cov)
    np.testing.assert_allclose(config.inv_cov, expected_inv_cov)

    assert config.i_lam == 0
    assert config.i_z == 0


def test_load_boost_factor_collection(tmp_path):
    for l in (0, 1):
        for z in (0, 1):
            _write_synthetic_files(tmp_path, l, z)

    collection = load_boost_factor_collection(
        str(tmp_path), l0=0, le=2, z0=0, ze=2, scale_cut=(0.1, 5.0)
    )

    assert isinstance(collection, BoostFactorCollection)
    expected_keys = {"0l_0z", "0l_1z", "1l_0z", "1l_1z"}
    assert set(collection.datasets.keys()) == expected_keys

    for l in (0, 1):
        for z in (0, 1):
            key = f"{l}l_{z}z"
            dataset = collection.datasets[key]
            assert isinstance(dataset, BoostFactorData)
            assert dataset.i_lam == l
            assert dataset.i_z == z
