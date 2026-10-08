r"""Verify `clenspy.selection.geometry.area_overlap` against the mock's
*actual* companion-richness selection rule, not a re-derivation of it.

The mock does not use a circle-overlap area at all. Its real rule
(``SelectionBias/make_mock_lob_sigma_catalog.py::run_lambda_ob``, faithful
to notebook cell 24):

1. a hard angular cut, ``ang < theta_lob`` -- only galaxies within the
   TARGET's own aperture radius are candidates at all;
2. among those, a soft radial weight ``theta_r(x) = 0.5*erfc(15*(x-1.2))``
   with ``x = R_i / R_lambda_target`` -- but the hard cut already confines
   ``x < 1``, where ``theta_r`` is >= 0.9999, i.e. effectively 1.

So the real selection is: what fraction of a companion halo's OWN galaxy
population (1 central at its own centre + `round(lambda_tr) - 1`
satellites drawn from the companion's own projected-NFW profile, per
``run_lambda_ob``'s inverse-CDF sampling of
``Mprj_com``/``fun`` -- notebook cell 15, transcribed verbatim below as
`nfw_projected_kernel`) lands inside the target's aperture disk.

`area_overlap(theta, theta_lob, theta_ltr)` answers a DIFFERENT question:
the geometric overlap area of two uniform disks (target radius theta_lob,
companion radius theta_ltr) -- i.e. it assumes the companion's richness is
smeared UNIFORMLY over its own disk, not NFW-concentrated at its centre.

This script computes the exact ring-overlap integral (deterministic
quadrature, no Monte Carlo noise) of the companion's true NFW radial
profile against the target's aperture disk, and compares it to
`area_overlap` directly, at fixed richness ratio and a bracket of
concentrations -- to measure how much the uniform-disk approximation
costs, not to guess it.

Units: everything is worked in target-aperture units (theta_lob = 1);
this is exact because at fixed redshift theta = R_lambda*(1+z)/chi(z), so
every ratio of angles equals the same ratio of physical R_lambda's, and
the (1+z)/chi(z) factors cancel identically between target and companion
(see the report section "Area overlap" for the derivation).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import quad

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed"

import sys  # noqa: E402
sys.path.insert(0, str(BASE.parents[1] / "src"))
from clenspy.selection.geometry import area_overlap  # noqa: E402

LOB_REF = 24.0  # richness of the reference bin used throughout this report


def fun(x):
    """Wright & Brainerd NFW projection kernel -- verbatim from the mock's
    `fun` (SelectionBias notebook cell 15)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(x)
    hi, lo = x > 1.0, x < 1.0
    out[hi] = (1.0 - 2.0 / np.sqrt(x[hi] ** 2 - 1.0)
              * np.arctan(np.sqrt((x[hi] - 1.0) / (x[hi] + 1.0)))) \
        / (x[hi] ** 2 - 1.0)
    out[lo] = (1.0 - 2.0 / np.sqrt(1.0 - x[lo] ** 2)
              * np.arctanh(np.sqrt((1.0 - x[lo]) / (1.0 + x[lo])))) \
        / (x[lo] ** 2 - 1.0)
    out[x == 1.0] = 1.0 / 3.0
    return out


def radial_pdf_norm(r_ltr, c):
    """int_0^r_ltr r*fun(r*c/r_ltr) dr -- computed once per (r_ltr, c),
    not per quadrature node (that recomputation is what made the naive
    version too slow to finish)."""
    return quad(lambda rr: rr * fun(np.array([rr * c / r_ltr]))[0],
               1e-8, r_ltr, limit=200)[0]


def radial_pdf(r, r_ltr, c, norm):
    """Normalised P(r) dr for a companion's satellites, r in target-
    aperture units: p(r) propto r * Sigma_NFW(r) propto r * fun(r*c/r_ltr),
    r_s = r_ltr / c, support [0, r_ltr]."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    return r * fun(r * c / r_ltr) / norm


def ring_frac(r, s):
    """Fraction of a ring of radius r centred at distance s from the
    target disk's centre (target radius 1) that lies inside the target."""
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    point = r < 1e-9
    out[point] = 1.0 if s < 1.0 else 0.0
    ring = ~point
    rr = r[ring]
    inside = rr + s <= 1.0
    outside = np.abs(s - rr) >= 1.0
    frac = np.zeros_like(rr)
    frac[inside] = 1.0
    frac[outside] = 0.0
    edge = ~inside & ~outside
    cos_arg = np.clip((s**2 + rr[edge] ** 2 - 1.0) / (2.0 * s * rr[edge]),
                      -1.0, 1.0)
    frac[edge] = np.arccos(cos_arg) / np.pi
    out[ring] = frac
    return out


def f_true(s, r_ltr, c, norm, lob=LOB_REF):
    """Exact fraction of a companion's own richness that lands inside the
    target aperture, given its true NFW-distributed satellites."""
    lam_tr = lob * r_ltr**5  # R_lambda(lam) = (lam/100)^0.2, so lam ~ R^5
    central = 1.0 if s < 1.0 else 0.0
    if lam_tr <= 1.0:
        return central / max(lam_tr, 1e-8)
    ring_integral = quad(
        lambda r: radial_pdf(np.array([r]), r_ltr, c, norm)[0]
        * ring_frac(np.array([r]), s)[0],
        1e-8, r_ltr, limit=200,
    )[0]
    return (central + (lam_tr - 1.0) * ring_integral) / lam_tr


def main():
    s_grid = np.geomspace(0.05, 3.0, 40)
    # r_ltr below ~0.5 gives lam_tr = LOB_REF * r_ltr**5 < 2: the mock's
    # own generator always adds a deterministic +1 central (create_l_mock),
    # so a clean central+satellites split needs lam_tr comfortably above 1
    r_ltr_values = [0.6, 0.8, 1.0]
    c_fid = 5.0

    rows = []
    for r_ltr in r_ltr_values:
        norm = radial_pdf_norm(r_ltr, c_fid)
        for s in s_grid:
            fa = float(area_overlap(np.array([s]), 1.0, np.array([r_ltr]))[0])
            ft = f_true(s, r_ltr, c_fid, norm)
            rows.append([r_ltr, s, fa, ft])
    rows = np.asarray(rows)

    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "area_overlap_check.csv", "w") as f:
        f.write(f"# r_ltr, s [theta/theta_lob], f_area (clenspy), "
                f"f_true (NFW ring quadrature, c={c_fid:.0f}, "
                f"lob={LOB_REF:.0f}); dimensionless\n")
        np.savetxt(f, rows, delimiter=",", fmt="%.6g")

    # concentration sensitivity at the reference bin's r_ltr = 0.75
    dev_by_c = {}
    for c in (3.0, 5.0, 8.0):
        norm_c = radial_pdf_norm(0.75, c)
        devs = [abs(f_true(s, 0.75, c, norm_c) -
                    float(area_overlap(np.array([s]), 1.0, np.array([0.75]))[0]))
                for s in s_grid]
        dev_by_c[c] = max(devs)

    with open(DATA / "area_overlap_summary.csv", "w") as f:
        f.write("# r_ltr, max|f_true - f_area| over s (c=5), "
                "mean|f_true - f_area| over s (c=5)\n")
        for r_ltr in r_ltr_values:
            sel = rows[:, 0] == r_ltr
            dev = np.abs(rows[sel, 3] - rows[sel, 2])
            f.write(f"{r_ltr},{dev.max():.6g},{dev.mean():.6g}\n")

    with open(DATA / "area_overlap_conc_sensitivity.csv", "w") as f:
        f.write("# concentration, max|f_true - f_area| over s, at r_ltr=0.75\n")
        for c, d in dev_by_c.items():
            f.write(f"{c},{d:.6g}\n")

    print("wrote", DATA / "area_overlap_check.csv",
          "and", DATA / "area_overlap_summary.csv")


if __name__ == "__main__":
    main()
