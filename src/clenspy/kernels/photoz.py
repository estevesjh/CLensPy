r"""The two photo-z kernels. They are **different functions**.

A cluster's true redshift is not its observed one, and the weight that maps
between them is *not the same* for the counts and for the shear projection.
Using one where the other belongs is a silent bias, so they live side by
side here with their difference stated.

**Counts** -- a Gaussian CDF difference over the observed-redshift bin
(``y3_cluster_cpp/src/pipelines/shared/sel_function.py::_S_j``):

.. math::
    \mathcal{S}_j(z^{\rm tr})
      = \Phi\!\left(\frac{z_j^{\max} - z^{\rm tr}}{\sigma_z}\right)
      - \Phi\!\left(\frac{z_j^{\min} - z^{\rm tr}}{\sigma_z}\right)

This is a **probability**: the chance that a cluster at true redshift
:math:`z^{\rm tr}` is observed inside :math:`\Delta z_j`. It is bounded by
1, it integrates over :math:`z^{\rm tr}` to the bin width, and it has
support everywhere.

**Projection** -- a parabolic weight about a single observed redshift
(``shear_prj_gl.py``, the exact C++ core ``sp_detail::ShearPrjCore``):

.. math::
    w_{pz}(z; z^{\rm ob}) =
      \begin{cases}
        1 - u^2, & |u| < 1 \\ 0, & \text{otherwise}
      \end{cases},
    \qquad u = \frac{z - z^{\rm ob}}{n_\sigma\,\sigma_z(z)}

This is an **unnormalised weight** with compact support: it vanishes
outside :math:`|z - z^{\rm ob}| < n_\sigma \sigma_z`, and
:math:`\int w_{pz}\,dz = \frac43 n_\sigma \sigma_z`, not 1. It is a window
on the line-of-sight integral, not a probability.

NOTE: :math:`n_\sigma = 3`. The parabola's half-width is the **3-sigma**
window, not :math:`\sigma_z` itself, which is where the 0.03 in the y3
configs comes from: :math:`\sigma_z = 0.01` and
:math:`3\sigma_z = 0.03`. Passing 0.03 as the scatter to `photoz_counts`
instead widens the counts kernel threefold; passing 0.01 as the half-width
here narrows the projection window threefold. Both are silent, and in
opposite directions, which is why the two kernels take their width
differently and say so.

Three ways they differ, each of which matters:

=================== ====================== ==============================
                    counts                 projection
=================== ====================== ==============================
shape               Gaussian CDF diff.     parabola
support             all :math:`z`          :math:`\pm 3\sigma_z` only
normalisation       :math:`\le 1`          :math:`\int = 4n_\sigma\sigma_z/3`
width used          :math:`\sigma_z`       :math:`n_\sigma \sigma_z`
keyed on            the bin **edges**      one :math:`z^{\rm ob}`
=================== ====================== ==============================

NOTE: units. Both are dimensionless functions of dimensionless
redshifts; :math:`\sigma_z` is in redshift units. :math:`w_{pz}` carries no
:math:`dz`, so a caller supplies the measure.

NOTE: :math:`\sigma_z` in the projection kernel is :math:`\sigma_z(z)`, a
120-node table (``y3_cluster_cpp/src/models/z_kernel_data.hh``, vendored
here as ``clenspy/data/z_kernel_5perc_ext_z01.txt``). `photoz_projection`
therefore accepts a scalar *or* a callable: pass a per-bin scalar for the
constant-width approximation, or `y3_photoz_window` for the exact
production width. The approximation is named, not hidden.

NOTE: **the tabulated quantity is the window, not the scatter**, and this
is the third face of the same 0.03 confusion. The table stores
:math:`{\rm sig}(z)` and the C++ forms
:math:`1/(100\sqrt{{\rm sig}})`, then uses *that* directly as the
parabola's half-width -- so its own :math:`\sigma_z` symbol already has
:math:`n_\sigma` folded in. The values run 0.040 to 0.148, i.e. 4 to 15
times 0.01, so they are *not* a constant :math:`3\sigma_z = 0.03`. Nor are
they monotonic: they rise from 0.040 at :math:`z = 0.1` to a peak of 0.148
near :math:`z = 0.73`, fall to about 0.10 by :math:`z = 0.9`, and dip
locally on the way (near :math:`z = 0.18` and :math:`z = 0.49`). It is a
calibrated curve, not a formula, which is why it is shipped as a table and
interpolated rather than fitted. Hence
`y3_photoz_window` returns a half-width to be passed with
``n_sigma=1.0``, and says so, rather than being handed to the default
``n_sigma=3`` and silently tripling.
"""

from __future__ import annotations

import pathlib

import numpy as np
from scipy.special import erf

__all__ = [
    "gaussian_cdf",
    "photoz_counts",
    "photoz_projection",
    "y3_photoz_window",
    "Y3_Z_KERNEL_FILE",
]

_SQRT2 = np.sqrt(2.0)


def gaussian_cdf(x):
    r"""The standard normal CDF :math:`\Phi(x) = \frac12[1 + {\rm erf}(x/\sqrt2)]`.

    NOTE: written out rather than taken from `scipy.stats.norm.cdf`, which
    is ~40x slower per call for array input and is the inner loop of the
    selection function. `scipy.special.erf` is the same computation without
    the distribution-object overhead.
    """
    return 0.5 * (1.0 + erf(np.asarray(x, dtype=float) / _SQRT2))


def photoz_counts(z_true, z_min, z_max, sigma_z):
    r"""The counts kernel :math:`\mathcal{S}_j(z^{\rm tr})`, dimensionless.

    .. math::
        \mathcal{S}_j = \Phi\!\left(\frac{z_j^{\max} - z^{\rm tr}}
                                        {\sigma_z}\right)
                      - \Phi\!\left(\frac{z_j^{\min} - z^{\rm tr}}
                                        {\sigma_z}\right)

    The probability that a cluster at true redshift :math:`z^{\rm tr}` is
    *observed* inside :math:`[z_j^{\min}, z_j^{\max}]`.

    NOTE: the arguments are the **bin edges minus the true redshift**, in
    that order. Reversing them flips the sign; using
    :math:`z^{\rm tr} - z^{\rm ob}` instead of the edges collapses the bin
    to a point.

    NOTE: the y3 pipeline additionally hard-zeroes this outside
    :math:`|z^{\rm tr} - z_j^{\rm mid}| > L_z \sigma_z` with
    :math:`L_z = 6`, purely to bound its shared grid
    (``sel_function.py``). That truncation is **not** applied here: it is a
    property of that grid, not of the kernel, and at :math:`6\sigma` it
    changes the value by :math:`O(10^{-9})`. Apply it at the grid if you
    need bit-compatibility.

    Parameters
    ----------
    z_true : float or array-like
        True cluster redshift.
    z_min, z_max : float or array-like
        Observed-redshift bin edges. Broadcast against ``z_true``.
    sigma_z : float or array-like
        Photo-z scatter, in redshift units.

    Returns
    -------
    np.ndarray
        :math:`\mathcal{S}_j \in [0, 1]`.
    """
    z_true = np.asarray(z_true, dtype=float)
    sigma_z = np.asarray(sigma_z, dtype=float)
    if np.any(sigma_z <= 0.0):
        raise ValueError(f"sigma_z must be positive, got {sigma_z}")
    return (gaussian_cdf((np.asarray(z_max, dtype=float) - z_true) / sigma_z)
            - gaussian_cdf((np.asarray(z_min, dtype=float) - z_true) / sigma_z))


def photoz_projection(z, z_ob, sigma_z, n_sigma: float = 3.0):
    r"""The projection weight :math:`w_{pz}(z; z^{\rm ob})`, dimensionless.

    .. math::
        w_{pz} = \max\!\left(0,\; 1 - u^2\right),
        \qquad u = \frac{z - z^{\rm ob}}{n_\sigma\,\sigma_z(z)}

    NOTE: the half-width is :math:`n_\sigma \sigma_z` with
    :math:`n_\sigma = 3`, **not** :math:`\sigma_z`. That is where the 0.03
    in the y3 configs comes from -- it is the 3-sigma window of
    :math:`\sigma_z = 0.01`, used as this parabola's half-width for the
    :math:`b_{\rm sel}` channel. Passing 0.03 as ``sigma_z`` here (with
    the default ``n_sigma``) makes the window three times too wide.

    NOTE: **unnormalised, with compact support.** It is exactly zero for
    :math:`|z - z^{\rm ob}| \ge n_\sigma \sigma_z`, and
    :math:`\int w_{pz}\,dz = \frac43 n_\sigma \sigma_z` for constant
    :math:`\sigma_z`. Do not use it as a probability, and do not substitute
    `photoz_counts` for it -- that one has infinite support and would put
    weight along the whole line of sight.

    NOTE: :math:`\sigma_z` is evaluated at :math:`z`, not at
    :math:`z^{\rm ob}`, when a callable is supplied. That is what the exact
    C++ core does (``sig_z = z_kernel.sigma_z(zs)``), and it makes the
    window's width vary along the line of sight.

    Parameters
    ----------
    z : float or array-like
        True redshift along the line of sight.
    z_ob : float
        The bin's observed redshift, the window's centre.
    sigma_z : float or callable
        Photo-z **scatter**, not the window. A scalar is the constant-width
        approximation; a callable ``sigma_z(z)`` is the y3 120-node table --
        see the module NOTE.
    n_sigma : float, optional
        Window half-width in units of ``sigma_z`` (default: 3, the
        production value). The support is
        :math:`|z - z^{\rm ob}| < n_\sigma \sigma_z`.

    Returns
    -------
    np.ndarray
        :math:`w_{pz} \in [0, 1]`, zero outside the support.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    scatter = np.asarray(sigma_z(z) if callable(sigma_z) else sigma_z,
                         dtype=float)
    if np.any(scatter <= 0.0):
        raise ValueError("sigma_z must be positive everywhere")
    if n_sigma <= 0.0:
        raise ValueError(f"n_sigma must be positive, got {n_sigma}")
    # the 3-sigma window, applied once and visibly
    u = (z - z_ob) / (n_sigma * scatter)
    # a parabola clipped at its roots, not a parabola plus a mask: the two
    # differ only in how obvious the compact support is
    return np.maximum(0.0, 1.0 - u * u)


#: The vendored DES Y3 z-kernel table, 120 nodes over z in [0.1, 0.9].
Y3_Z_KERNEL_FILE = (
    pathlib.Path(__file__).resolve().parents[1] / "data"
    / "z_kernel_5perc_ext_z01.txt"
)

_Y3_WINDOW_CACHE: dict = {}


def y3_photoz_window():
    r"""The exact DES Y3 projection-window half-width, as a callable.

    Returns ``half_width(z)`` interpolating

    .. math::
        n_\sigma\,\sigma_z(z) = \frac{1}{100\sqrt{{\rm sig}(z)}}

    from the vendored 120-node table.

    NOTE: **this is the window half-width, already including**
    :math:`n_\sigma`. Pass it with ``n_sigma=1.0``::

        w = photoz_projection(z, z_ob, y3_photoz_window(), n_sigma=1.0)

    Handing it to the default ``n_sigma=3`` widens the window by rather
    more than threefold, because the width itself varies across the
    enlarged support. The values are 0.040 to 0.148 across the table and
    are **not monotonic in z** -- see the module NOTE.

    NOTE: linear interpolation (``k=1``) with **constant** extrapolation
    outside :math:`[0.1, 0.9]` (``ext=3``), matching the production
    spline. A higher-order spline would ring on this table, and letting it
    extrapolate would give a negative width beyond the ends.

    Returns
    -------
    callable
        ``half_width(z) -> np.ndarray``, in redshift units. Cached, so
        repeated calls share one spline.
    """
    if "spline" not in _Y3_WINDOW_CACHE:
        from scipy.interpolate import InterpolatedUnivariateSpline

        z_nodes, sig = np.loadtxt(Y3_Z_KERNEL_FILE, unpack=True)
        _Y3_WINDOW_CACHE["spline"] = InterpolatedUnivariateSpline(
            z_nodes, 1.0 / 100.0 / np.sqrt(sig), k=1, ext=3
        )
    spline = _Y3_WINDOW_CACHE["spline"]

    def half_width(z):
        return spline(np.asarray(z, dtype=float))

    return half_width


if __name__ == "__main__":
    z_min, z_max, sigma_z = 0.35, 0.50, 0.01
    n_sigma = 3.0
    z_ob = 0.5 * (z_min + z_max)
    z = np.linspace(0.30, 0.55, 11)

    print(f"bin [{z_min}, {z_max}], sigma_z = {sigma_z}, "
          f"window = {n_sigma}*sigma_z = {n_sigma * sigma_z}, "
          f"z_ob = {z_ob}\n")
    print(f"{'z_tr':>7s}  {'S_j (counts)':>13s}  {'w_pz (proj)':>12s}")
    for zi, sj, wp in zip(z, photoz_counts(z, z_min, z_max, sigma_z),
                          photoz_projection(z, z_ob, sigma_z)):
        print(f"{zi:7.3f}  {sj:13.6f}  {wp:12.6f}")

    # the three differences, as numbers
    fine = np.linspace(0.0, 1.0, 200001)
    sj = photoz_counts(fine, z_min, z_max, sigma_z)
    wp = photoz_projection(fine, z_ob, sigma_z)
    print(f"\ncounts:      max = {sj.max():.6f}  "
          f"integral = {np.trapezoid(sj, x=fine):.6f}  "
          f"(bin width {z_max - z_min})")
    print(f"projection:  max = {wp.max():.6f}  "
          f"integral = {np.trapezoid(wp, x=fine):.6f}  "
          f"(4/3 * {n_sigma} sigma_z = {4 / 3 * n_sigma * sigma_z:.6f})")
    support = fine[wp > 0]
    print(f"projection support: [{support[0]:.4f}, {support[-1]:.4f}] "
          f"= z_ob +- {n_sigma * sigma_z}  <- the 3-sigma window, the 0.03")
    print(f"counts at 6 sigma below the bin: "
          f"{photoz_counts(z_min - 6 * sigma_z, z_min, z_max, sigma_z).item():.3e}"
          "   <- what the y3 L_z = 6 envelope discards")

    # a varying-width sigma_z(z), as the y3 table supplies
    print("\nwith a callable sigma_z(z) = 0.01 * (1 + z):")
    w = photoz_projection(np.array([0.42, 0.45, 0.48]), z_ob,
                          lambda zz: 0.01 * (1.0 + zz))
    print("  w_pz =", np.array2string(w, precision=6))
