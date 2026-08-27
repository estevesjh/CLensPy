r"""The effective survey area :math:`\Omega(z)`.

The solid angle a cluster search actually covers at redshift :math:`z`,
after masking and after whatever depth or volume limit the catalogue
imposes. It is a property of the **dataset**, so it lives here and not in
`clenspy.cosmology`.

.. math::
    \langle N_{ij}\rangle = \int\! dz \int\! d\ln M
      \int\! d\lambda^{\rm tr}\;
      n(M, z)\, \frac{dV}{d\Omega\, dz}\, \boldsymbol{\Omega(z)}\,
      K_j(z)\, \mathcal{S}_i(\lambda^{\rm tr}, z)\,
      P(\lambda^{\rm tr} \mid M, z)

NOTE: :math:`\Omega(z)` appears in the **counts** and **cancels** in the
shear projection -- it divides out of the surface density, and the exact
C++ core hard-excludes it there. Folding the footprint into a lensing
weight is a silent normalisation error, so any shared weight builder must
take :math:`\Omega(z)` as an explicit per-observable argument rather than
as an ambient survey property applied to both. See
``docs/refactor-plan.md`` errata E.2.

NOTE: units are **steradians** (rad^2), matching the C++ these fits are
transcribed from. `deg2` converts, as one visible multiplication.

Transcribed from ``y3_cluster_cpp/src/models/omega_z_{des,sdss}.hh`` and
``sptxdes/omega_z_y3xspt.hh``. The C++ ``polynomial<N>`` template takes the
**highest power first**, which is `numpy.polyval`'s convention, so the
coefficient arrays below are byte-for-byte the ones in those headers.

NOTE: the DES header names its coefficient arrays ``SDSS_fit``,
``SDSS_fit2``, ``SDSS_fit3``. That is a copy-paste artifact -- the C++
source flags it itself ("A+ naming SDSS_fit for DES =P") -- and the numbers
are DES, not SDSS. They are renamed here.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "deg2",
    "omega_des_y1",
    "omega_des_y3",
    "omega_sdss",
    "omega_y3xspt",
    "survey_area",
]

#: rad^2 per deg^2, for reporting a footprint in the units papers quote.
_RAD2_PER_DEG2 = (np.pi / 180.0) ** 2

# -- DES Y1: three pieces, breaking at z = 0.504 and z = 0.700 ------------
#
# NOTE: this is the fit the C++ calls ``OMEGA_Z_DES``, and it is **DES Y1**,
# not Y3: it gives 1494 deg^2 at z = 0.2, against the published Y1
# footprint of 1437 deg^2. The y3 repo's own python transcription
# (``test/make_hod_norm_impact.py``) names it ``omega_z_des_y1``, which
# settles it.
_DES_Y1_LOW = (0.0, 0.0, 0.0, -0.00262353, 0.01940118, 0.45133063)
_DES_Y1_MID = (1.33647377e4, 1.35291046e3, -1.26204891e2,
               -2.83454918e1, -2.26465905, 3.84958753e-1)
_DES_Y1_HIGH = (0.0, 0.0, -1.88101967, 4.8071839, -4.11424324, 1.18196785)

#: Where the DES Y1 pieces meet. The fit is **discontinuous** at both: by
#: -0.37% at 0.504 and by -30.6% at 0.700 (58.3 -> 40.5 deg^2). Both jumps
#: are in the C++ too; they sit outside the analysis range below.
_DES_Y1_BREAKS = (0.504, 0.700)

#: Redshift range the DES Y1 cluster analysis actually uses -- the bin
#: edges are 0.20/0.35/0.50/0.65. Outside it the polynomial is an
#: extrapolation: it crosses **zero at z = 0.9378** and is negative above,
#: which is why `omega_des_y1` clamps at zero.
DES_Y1_Z_RANGE = (0.20, 0.65)

# -- SDSS: one degree-11 fit in (z - 0.2) --------------------------------
_SDSS = (-1.14293122e05, 5.96846869e04, 9.24239180e03, -2.23118813e03,
         -4.52580713e03, 1.18404878e03, 1.27951911e02, -5.05716847e01,
         1.01744577e00, -3.11253383e-01, 5.48481084e-03, 3.12629987e00)

#: Redshift range the SDSS redMaPPer cluster analysis uses. The fit peaks
#: at 10263 deg^2 at z = 0.2 and is smooth across this range; a degree-11
#: polynomial diverges fast outside it, so `omega_sdss` clamps at zero.
SDSS_Z_RANGE = (0.10, 0.33)

#: DES Y3, as a flat effective area [rad^2]: the **gold** footprint,
#: 4143 deg^2. This is the area of the data.
#:
#: NOTE: **no z-dependent DES Y3 fit exists.** There is none in
#: ``y3_cluster_cpp``, so rather than invent one this is a constant. The
#: precedent is the repo's own ``OMEGA_Z_Y3XSPT``, which does exactly this
#: for Y3 x SPT ("These fits will need to be computed by Eli") with a
#: constant 2500 deg^2. Replace `omega_des_y3` the moment a real fit lands.
#:
#: NOTE: ``cluster-lensing-cov/configs/des_y3.json`` carries 5000 deg^2
#: instead. That file is a **forecast** -- its own provenance says the
#: counts are "DES Y1 counts scaled by 5000/1437" -- so the two numbers
#: describe different things and must not be reconciled. 4143 is the gold
#: footprint; use it for the data. A forecast that wants 5000 should say so
#: at the call site.
_DES_Y3_AREA_DEG2 = 4143.0

#: Y3 x SPT-SZ, as the C++ has it: a flat 2500 deg^2 placeholder.
_Y3XSPT_AREA_RAD2 = 0.7615435494667714


def deg2(omega_rad2):
    r"""Convert :math:`\Omega` from rad^2 to deg^2.

    The one unit conversion in this module, applied by the caller when it
    wants to compare a footprint against a published number.
    """
    return np.asarray(omega_rad2, dtype=float) / _RAD2_PER_DEG2


def omega_des_y1(z):
    r"""DES Y1 effective survey area :math:`\Omega(z)` [rad^2].

    Three polynomial pieces (``y3_cluster::OMEGA_Z_DES``):

    .. math::
        \Omega(z) = \begin{cases}
          p_1(z),        & z < 0.504 \\
          p_2(z - 0.6),  & 0.504 \le z < 0.700 \\
          p_3(z),        & z \ge 0.700
        \end{cases}

    NOTE: **valid on** :math:`z \in [0.20, 0.65]` (`DES_Y1_Z_RANGE`), the
    range the analysis bins span. Outside it the fit is an extrapolation
    with two known pathologies, both present in the C++: it is
    discontinuous at each break (-0.37% at 0.504, -30.6% at 0.700), and it
    crosses zero at :math:`z = 0.9378`. The result is clamped at zero so an
    integral that strays past that cannot pick up negative area -- the C++
    does not clamp, so beyond z = 0.94 the two differ deliberately.

    Parameters
    ----------
    z : float or array-like
        Cluster (true) redshift.

    Returns
    -------
    np.ndarray
        :math:`\Omega(z)` in rad^2, never negative.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    lo, hi = _DES_Y1_BREAKS
    out = np.where(
        z < lo,
        np.polyval(_DES_Y1_LOW, z),
        np.where(z < hi,
                 np.polyval(_DES_Y1_MID, z - 0.6),
                 np.polyval(_DES_Y1_HIGH, z)),
    )
    return np.maximum(out, 0.0)  # the fit goes negative above z = 0.9378


def omega_des_y3(z):
    r"""DES Y3 effective survey area :math:`\Omega(z)` [rad^2].

    NOTE: **flat in z** at the published 4143 deg^2. No redshift-dependent
    Y3 fit has been computed -- see the note on ``_DES_Y3_AREA_DEG2``. This
    is a stated approximation, not a transcription, and it is the one
    function in this module that is not taken from a source. For counts it
    biases the redshift *shape* of :math:`\langle N_{ij}\rangle`, not its
    normalisation; do not use it to compare bins at different z until a real
    fit replaces it.

    Parameters
    ----------
    z : float or array-like
        Cluster redshift. Used only for its shape.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return np.full(z.shape, _DES_Y3_AREA_DEG2 * _RAD2_PER_DEG2)


def omega_sdss(z):
    r"""SDSS effective survey area :math:`\Omega(z)` [rad^2].

    A single degree-11 polynomial in :math:`(z - 0.2)`
    (``y3_cluster::OMEGA_Z_SDSS``), peaking at 10263 deg^2 at
    :math:`z = 0.2`.

    NOTE: **valid on** :math:`z \in [0.10, 0.33]` (`SDSS_Z_RANGE`). A
    degree-11 fit diverges quickly outside the range it was fit on, so the
    result is clamped at zero.

    Parameters
    ----------
    z : float or array-like
        Cluster (true) redshift.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return np.maximum(np.polyval(_SDSS, z - 0.2), 0.0)


def omega_y3xspt(z):
    r"""DES Y3 x SPT-SZ area :math:`\Omega(z)` [rad^2]: flat 2500 deg^2.

    NOTE: a placeholder in the C++ as well
    (``y3_cluster::OMEGA_Z_Y3XSPT``), which carries the comment "These fits
    will need to be computed by Eli (+Lindsey)". Transcribed as-is,
    including the constant, so that swapping it for a real fit is a
    one-function change here.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return np.full(z.shape, _Y3XSPT_AREA_RAD2)


#: The registry the drivers address by name. Kept beside the functions so
#: that adding a survey means adding one entry, not editing a caller.
_OMEGA = {
    "des_y1": omega_des_y1,
    "des_y3": omega_des_y3,
    "sdss": omega_sdss,
    "y3xspt": omega_y3xspt,
}


def survey_area(name):
    r"""The :math:`\Omega(z)` callable for ``name``.

    Parameters
    ----------
    name : {"des_y1", "des_y3", "sdss", "y3xspt"}
        Survey identifier, case-insensitive.

    Returns
    -------
    callable
        ``omega(z) -> np.ndarray`` in rad^2.

    Raises
    ------
    KeyError
        If ``name`` is not a known survey. Listing the known ones in the
        message, because a typo here silently changes a normalisation.
    """
    key = str(name).lower()
    try:
        return _OMEGA[key]
    except KeyError:
        raise KeyError(
            f"unknown survey {name!r}; have {sorted(_OMEGA)}"
        ) from None


if __name__ == "__main__":
    print("effective survey area Omega(z), in deg^2\n")
    print(f"{'z':>6s}  {'DES Y1':>10s}  {'DES Y3':>10s}  {'SDSS':>10s}"
          f"  {'Y3xSPT':>10s}")
    for z in (0.05, 0.10, 0.20, 0.30, 0.35, 0.50, 0.55, 0.65, 0.70, 0.95):
        row = [float(deg2(f(z)).item()) for f in
               (omega_des_y1, omega_des_y3, omega_sdss, omega_y3xspt)]
        print(f"{z:6.2f}  " + "  ".join(f"{v:10.1f}" for v in row))

    print("\nthe DES Y1 fit's two seams and its zero crossing:")
    for zb in _DES_Y1_BREAKS:
        below = float(deg2(omega_des_y1(zb - 1e-6)).item())
        above = float(deg2(omega_des_y1(zb)).item())
        print(f"  z = {zb:.3f}: {below:9.1f} -> {above:9.1f} deg^2"
              f"   ({100 * (above / below - 1):+.2f}%)")
    print(f"  clamped to zero above z = 0.9378: "
          f"{float(deg2(omega_des_y1(1.2)).item()):.1f} deg^2")
    print(f"\nanalysis ranges: DES Y1 {DES_Y1_Z_RANGE}, SDSS {SDSS_Z_RANGE}")
