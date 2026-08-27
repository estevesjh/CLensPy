r"""Production bin definitions, per survey.

The :math:`(\Delta\lambda_i, \Delta z_j)` grid each analysis actually used.
These are the labels every result array is addressed by, so they belong with
the survey rather than being retyped in each driver.

NOTE: dimensionless throughout -- richness :math:`\lambda^{\rm ob}` and
observed redshift :math:`z^{\rm ob}` both carry no units, and
:math:`\sigma_z` is in redshift units.

NOTE: bins are **observed** richness and **observed** photometric redshift.
The photo-z kernel :math:`K_j(z)` that maps a true redshift into
:math:`\Delta z_j` uses :math:`\sigma_z`, which is why the scatter travels
on the bin (`~clenspy.utils.RichnessBin`) and not on the survey.
"""

from __future__ import annotations

from ..utils.binning import BinCollection

__all__ = ["des_y1_bins", "des_y3_bins", "sdss_bins", "survey_bins"]

#: DES Y1 production richness edges.
#:
#: NOTE: the top bin's upper edge is a stand-in for infinity and the two
#: sources disagree on it: the y3 production config
#: (``cosmosis-models/real_pipeline_extract_prj2h.ini``, transcribed in
#: ``test/make_hod_norm_impact.py``) uses **200**, while
#: ``cluster-lensing-cov/configs/des_y1.json`` uses 1000. 200 is used here
#: because the counts and the selection function were computed against it.
#: The choice is not free: it is the upper limit of a
#: :math:`\lambda^{\rm ob}` integral.
_DES_LAM_EDGES = (20.0, 30.0, 45.0, 60.0, 200.0)

#: DES Y1 and Y3 production redshift edges -- three bins over [0.20, 0.65],
#: matching `~clenspy.survey.area.DES_Y1_Z_RANGE`.
_DES_Z_EDGES = (0.20, 0.35, 0.50, 0.65)

#: Photo-z scatter, one value per richness bin. Flat 0.03 in production
#: (``generate_mock_dv.ini [sel_function]``); the papers allow it to depend
#: on the richness bin, which is why it is a per-bin sequence here.
_DES_SIGMA_Z = (0.03, 0.03, 0.03, 0.03)


def des_y1_bins() -> BinCollection:
    r"""The 4 x 3 DES Y1 :math:`(\lambda^{\rm ob}, z^{\rm ob})` grid.

    :math:`\lambda \in` [20, 30, 45, 60, 200] and
    :math:`z \in` [0.20, 0.35, 0.50, 0.65], with
    :math:`\sigma_z = 0.03`. Twelve bins, in richness-outer order.

    NOTE: transcribed from the y3 production config -- ``LAM_MIN``,
    ``LAM_MAX``, ``ZOB_MIN``, ``ZOB_MAX``, ``SIGMA_Z`` of
    ``y3_cluster_cpp/test/make_hod_norm_impact.py``, which quotes
    ``cosmosis-models/real_pipeline_extract_prj2h.ini``.
    """
    return BinCollection.from_edges(
        lam_edges=_DES_LAM_EDGES,
        z_edges=_DES_Z_EDGES,
        sigma_z=_DES_SIGMA_Z,
    )


def des_y3_bins() -> BinCollection:
    r"""The DES Y3 :math:`(\lambda^{\rm ob}, z^{\rm ob})` grid.

    NOTE: **identical to DES Y1's.** ``configs/des_y3.json`` carries the
    same edges, and no distinct Y3 binning is recorded anywhere this
    package tracks. It is a separate function so that a real Y3 binning is
    a one-function change rather than a search for call sites.
    """
    return des_y1_bins()


def sdss_bins() -> BinCollection:
    r"""Not available -- no SDSS binning to transcribe.

    Raises
    ------
    NotImplementedError
        Always. `~clenspy.survey.omega_sdss` exists because the SDSS
        :math:`\Omega(z)` fit is in ``y3_cluster_cpp``; the SDSS richness
        and redshift bin edges are not, and neither is the per-bin
        :math:`\sigma_z`. Since a bin edge is an integration limit, a
        guessed one is a wrong number that looks right, so it is refused.
        Build it with ``BinCollection.from_edges(...)`` once you have the
        analysis definition.
    """
    raise NotImplementedError(
        "no SDSS bin definition is available. clenspy transcribes analysis "
        "definitions rather than reconstructing them, and while omega_sdss "
        "comes from y3_cluster_cpp, the SDSS richness/redshift edges and "
        "per-bin sigma_z do not. Build them explicitly, e.g. "
        "BinCollection.from_edges(lam_edges=..., z_edges=..., sigma_z=...)."
    )


#: Addressed by the same names as `~clenspy.survey.area.survey_area`, so a
#: driver names its survey once.
_BINS = {
    "des_y1": des_y1_bins,
    "des_y3": des_y3_bins,
    "sdss": sdss_bins,
}


def survey_bins(name) -> BinCollection:
    r"""The `BinCollection` for ``name``.

    Parameters
    ----------
    name : {"des_y1", "des_y3", "sdss"}
        Survey identifier, case-insensitive. Matches the keys
        `~clenspy.survey.area.survey_area` accepts.

    Raises
    ------
    KeyError
        If ``name`` is unknown.
    NotImplementedError
        If the survey has no transcribed binning -- see `sdss_bins`.
    """
    key = str(name).lower()
    try:
        builder = _BINS[key]
    except KeyError:
        raise KeyError(
            f"unknown survey {name!r}; have {sorted(_BINS)}"
        ) from None
    return builder()


if __name__ == "__main__":
    bins = des_y1_bins()
    print(f"DES Y1: {len(bins)} bins, {bins.n_lam} richness x {bins.n_z} z\n")
    print(f"{'(i,j)':>7s}  {'lambda':>14s}  {'z':>14s}  {'sigma_z':>8s}")
    for b in bins:
        print(f"{str(b.index):>7s}  "
              f"[{b.lam_min:5.0f}, {b.lam_max:5.0f}]  "
              f"[{b.z_min:5.2f}, {b.z_max:5.2f}]  {b.sigma_z:8.2f}")
    print(f"\naddressing by paper index: bins.at(2, 1) = {bins.at(2, 1)}")

    try:
        sdss_bins()
    except NotImplementedError as exc:
        print(f"\nSDSS refused, as designed:\n  {str(exc)[:72]}...")
