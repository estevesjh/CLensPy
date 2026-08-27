#!/usr/bin/env python
"""Cross-validate clenspy's miscentered DeltaSigma against the DES Y3 table.

Two fully independent implementations of the same dimensionless quantity:

  clenspy  `lensing.miscentering` -- the by-parts + law-of-cosines reduction
           of docs/miscentering_math.md section 5, one smooth GL integral.
  DES Y3   `y3_cluster_cpp/data/nfw_off_center/` -- a precomputed
           (ln x_mis, ln x) grid built from an analytic inner disc plus an
           arccos band quadrature, read by bilinear interpolation.

Both are pure NFW with r_s = 1 and Sigma_0 = 2 rho_s r_s = 1, so they are
directly comparable with no unit conversion.

Result (see docs/miscentering_math.md section 9.3): the STORED values agree
to 1e-8..1e-5, so the two derivations are consistent. The shipped table
nonetheless loses accuracy near the cusp at x = x_mis, because bilinear
interpolation on (ln x_mis, ln x) cuts across the ridge -- up to a wrong
SIGN at x = x_mis = 0.01. Tabulating against ln(x/x_mis) instead puts the
cusp on a grid line and fixes it with fewer nodes.

Set Y3_CLUSTER_CPP_DIR to point at the reference checkout; the script skips
cleanly if the tables are absent.
"""
import os
import sys
from pathlib import Path

import numpy as np

from clenspy.lensing.miscentering import miscentered_deltasigma

DEFAULT_Y3 = Path.home() / "Documents/Dev/github/y3_cluster_cpp"
Y3 = Path(os.environ.get("Y3_CLUSTER_CPP_DIR", DEFAULT_Y3))
TABLE_DIR = Y3 / "data" / "nfw_off_center"
STEM = "table_1000_1e-03_5e+03"


def nfw_sigma(x):
    """Dimensionless centred NFW Sigma, Sigma_0 = 1 (Wright & Brainerd 2000)."""
    x = np.maximum(np.abs(np.asarray(x, float)), 1e-12)
    out = np.empty_like(x)
    lo, hi = x < 1 - 1e-8, x > 1 + 1e-8
    eq = ~(lo | hi)
    xl, xh = x[lo], x[hi]
    out[lo] = 1 / (xl * xl - 1) * (
        1 - 2 / np.sqrt(1 - xl * xl) * np.arctanh(np.sqrt((1 - xl) / (1 + xl))))
    out[hi] = 1 / (xh * xh - 1) * (
        1 - 2 / np.sqrt(xh * xh - 1) * np.arctan(np.sqrt((xh - 1) / (1 + xh))))
    out[eq] = 1 / 3.0
    return out


def nfw_mean_sigma(x):
    """Dimensionless centred mean Sigmabar(<x), Sigma_0 = 1."""
    x = np.maximum(np.abs(np.asarray(x, float)), 1e-12)
    g = np.empty_like(x)
    lo, hi = x < 1 - 1e-8, x > 1 + 1e-8
    eq = ~(lo | hi)
    xl, xh = x[lo], x[hi]
    g[lo] = np.log(xl / 2) + np.arccosh(1 / xl) / np.sqrt(1 - xl * xl)
    g[hi] = np.log(xh / 2) + np.arccos(1 / xh) / np.sqrt(xh * xh - 1)
    g[eq] = 1 + np.log(0.5)
    return 2 / x**2 * g


def clenspy_value(x, x_mis, n_nodes=1024):
    """clenspy's DeltaSigma_mis at one (x, x_mis)."""
    return float(miscentered_deltasigma(
        nfw_sigma, nfw_mean_sigma, np.array([float(x)]), float(x_mis),
        n_nodes=n_nodes)[0])


def main():
    if not TABLE_DIR.is_dir():
        print(f"SKIP: reference tables not found at {TABLE_DIR}")
        print("      set Y3_CLUSTER_CPP_DIR to the y3_cluster_cpp checkout")
        return 0

    from scipy.interpolate import RegularGridInterpolator

    lnx = np.loadtxt(TABLE_DIR / f"{STEM}_single_logx.txt")
    lnxm = np.loadtxt(TABLE_DIR / f"{STEM}_single_logxmis.txt")
    tab = np.loadtxt(TABLE_DIR / f"{STEM}_deltasigma_signed_single.txt")
    table = RegularGridInterpolator((lnxm, lnx), tab, method="linear",
                                    bounds_error=False, fill_value=None)
    print(f"reference table {tab.shape} (x_mis, x), "
          f"{100 * (tab < 0).mean():.1f}% negative entries\n")

    # 1. Stored values at grid nodes -- do the two derivations agree?
    print("1. STORED VALUES AT NODES (no interpolation)")
    worst_node = 0.0
    for im in (0, 60, 125, 190, 249):
        x_mis = float(np.exp(lnxm[im]))
        i_diag = int(np.argmin(np.abs(lnx - lnxm[im])))
        for ix in (i_diag, min(len(lnx) - 1, i_diag + 1)):
            x = float(np.exp(lnx[ix]))
            ref = clenspy_value(x, x_mis, n_nodes=2048)
            rel = abs(tab[im, ix] - ref) / max(abs(ref), 1e-30)
            worst_node = max(worst_node, rel)
    print(f"   worst relative difference near the diagonal: {worst_node:.2e}")
    assert worst_node < 1e-3, f"derivations disagree: {worst_node:.2e}"
    print("   -> the two derivations agree; the stored grid is sound\n")

    # 2. Interpolated values on the cusp -- does bilinear survive it?
    print("2. INTERPOLATION ACROSS THE CUSP  x = x_mis")
    print(f"   {'x=x_mis':>10} {'clenspy':>14} {'table':>14} {'rel err':>10}")
    worst_cusp, sign_flips = 0.0, 0
    for x_mis in (0.01, 0.1, 0.37, 1.0, 10.0, 100.0):
        ref = clenspy_value(x_mis, x_mis, n_nodes=2048)
        got = float(table(np.array([[np.log(x_mis), np.log(x_mis)]]))[0])
        rel = abs(got - ref) / abs(ref)
        worst_cusp = max(worst_cusp, rel)
        flag = ""
        if np.sign(got) != np.sign(ref):
            sign_flips += 1
            flag = "  <-- SIGN"
        print(f"   {x_mis:10.3g} {ref:+14.6e} {got:+14.6e} {rel:10.2e}{flag}")
    print(f"\n   worst on-cusp error: {worst_cusp:.2e} "
          f"({sign_flips} sign flip(s))")
    print("   -> interpolation, not the physics; see "
          "docs/miscentering_math.md section 9.3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
