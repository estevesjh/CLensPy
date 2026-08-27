"""Does-it-run tests for the richness--redshift bin containers."""

import numpy as np
import pytest
from scipy.stats import norm

from clenspy.utils import BinCollection, RichnessBin


def test_edges_indices_and_midpoints():
    b = RichnessBin(20.0, 30.0, 0.2, 0.35, i_lam=1, i_z=2, sigma_z=0.015)
    assert b.lam_edges == (20.0, 30.0)
    assert b.z_edges == (0.2, 0.35)
    assert b.index == (1, 2)
    assert b.lam_mid == 25.0
    assert b.z_mid == pytest.approx(0.275)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(lam_min=30.0, lam_max=20.0, z_min=0.2, z_max=0.35),  # lam inverted
        dict(lam_min=20.0, lam_max=30.0, z_min=0.35, z_max=0.2),  # z inverted
        dict(lam_min=20.0, lam_max=20.0, z_min=0.2, z_max=0.35),  # empty lam
        dict(lam_min=-1.0, lam_max=30.0, z_min=0.2, z_max=0.35),  # negative lam
        dict(lam_min=20.0, lam_max=30.0, z_min=-0.1, z_max=0.35),  # negative z
        dict(lam_min=20.0, lam_max=30.0, z_min=0.2, z_max=0.35, sigma_z=-1.0),
    ],
)
def test_rejects_bad_edges(kwargs):
    with pytest.raises(ValueError):
        RichnessBin(**kwargs)


def test_diff_is_the_evaluation_bar():
    """b.diff(F) == F(lam_max) - F(lam_min), the papers' |_{Delta lambda_i}."""
    b = RichnessBin(30.0, 45.0, 0.2, 0.35)
    mu, sigma = 40.0, 5.0

    def cdf(x):
        return norm.cdf((x - mu) / sigma)

    assert b.diff(cdf) == pytest.approx(cdf(45.0) - cdf(30.0))
    assert b.diff_z(lambda z: z**2) == pytest.approx(0.35**2 - 0.2**2)


def test_richness_bins_partition_unit_probability():
    """Contiguous bins covering the support carry all the probability mass."""
    bins = BinCollection.from_edges([0.0, 30.0, 45.0, 60.0, 1000.0], [0.2, 0.35])
    mu, sigma = 40.0, 5.0
    total = sum(b.diff(lambda x: norm.cdf((x - mu) / sigma)) for b in bins)
    assert float(total) == pytest.approx(1.0, abs=1e-12)


def test_diff_broadcasts_over_trailing_axes():
    b = RichnessBin(30.0, 45.0, 0.2, 0.35)
    mu = np.array([35.0, 40.0, 50.0])
    out = b.diff(lambda x: norm.cdf(x - mu))
    assert out.shape == (3,)
    assert np.all(out > 0.0)


def test_contains_is_half_open():
    b = RichnessBin(20.0, 30.0, 0.2, 0.35)
    assert b.contains(20.0, 0.2)          # lower edge included
    assert not b.contains(30.0, 0.25)     # upper richness edge excluded
    assert not b.contains(25.0, 0.35)     # upper redshift edge excluded
    assert not b.contains(19.9, 0.25)


def test_from_edges_builds_the_outer_product():
    bins = BinCollection.from_edges([20, 30, 45], [0.2, 0.35, 0.5, 0.65])
    assert len(bins) == 6
    assert (bins.n_lam, bins.n_z) == (2, 3)
    assert {b.index for b in bins} == {(i, j) for i in range(2) for j in range(3)}


def test_sigma_z_is_per_richness_bin():
    bins = BinCollection.from_edges(
        [20, 30, 45], [0.2, 0.35, 0.5], sigma_z=[0.015, 0.011]
    )
    for b in bins:
        assert b.sigma_z == (0.015 if b.i_lam == 0 else 0.011)

    scalar = BinCollection.from_edges([20, 30, 45], [0.2, 0.35], sigma_z=0.02)
    assert all(b.sigma_z == 0.02 for b in scalar)

    with pytest.raises(ValueError):
        BinCollection.from_edges([20, 30, 45], [0.2, 0.35], sigma_z=[0.01, 0.02, 0.03])


def test_at_addresses_by_paper_index():
    bins = BinCollection.from_edges([20, 30, 45], [0.2, 0.35, 0.5, 0.65])
    assert (bins.n_lam, bins.n_z) == (2, 3)
    assert bins.at(1, 2).lam_edges == (30.0, 45.0)
    assert bins.at(1, 2).z_edges == (0.5, 0.65)
    with pytest.raises(KeyError):
        bins.at(9, 9)


def test_reshape_recovers_the_Nij_matrix():
    bins = BinCollection.from_edges([20, 30, 45], [0.2, 0.35, 0.5, 0.65])
    flat = np.arange(len(bins), dtype=float)
    N_ij = bins.reshape(flat)
    assert N_ij.shape == (2, 3)
    # row-major: sequence order is richness-outer, so reshape must agree
    for k, b in enumerate(bins):
        assert N_ij[b.i_lam, b.i_z] == flat[k]

    with pytest.raises(ValueError):
        bins.reshape(np.zeros(5))


def test_duplicate_indices_rejected():
    b = RichnessBin(20.0, 30.0, 0.2, 0.35, i_lam=0, i_z=0)
    with pytest.raises(ValueError):
        BinCollection([b, b])
