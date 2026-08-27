#!/usr/bin/env python
r"""Build the packaged miscentered-NFW lookup table, using cluster_toolkit.

Writes ``src/clenspy/data/nfw_miscentering.npz``: the dimensionless
single-offset profiles

    sigma_hat_mis(x, x_mis)  = Sigma_mis / Sigma_0
    ds_hat_mis(x, x_mis)     = DeltaSigma_mis / Sigma_0,   SIGNED

with Sigma_0 = 2 r_s rho_s, x = R / r_s and x_mis = R_mis / r_s. Mass,
concentration and cosmology enter only that prefactor, so one grid serves
every halo (docs/miscentering_math.md section 9.1).

ENVIRONMENT
-----------
Needs `cluster_toolkit` (the marcpaterno fork), which lives in the
``y3cl_je_macos`` conda env, not in the project venv. See
``y3_cluster_cpp/docs/source/building_macos.md``. Run with:

    PYTHONPATH=src \
      /opt/homebrew/Caskroom/miniforge/base/envs/y3cl_je_macos/bin/python \
      tools/make_miscentering_table.py

    ... --tune     grid-convergence scan instead of a build

DOMAIN
------
Matches the DES Y3 table (``y3_cluster_cpp/data/nfw_off_center``):

    x     in [1e-3, 5e3]
    x_mis in [1e-2, 5e2]

The x_mis floor of 1e-2 is not cosmetic. DeltaSigma_mis is the difference
Sigmabar_mis - Sigma_mis, and that difference collapses as x_mis shrinks:
at x_mis = 5e-3 it is 1.6e-5 of Sigma itself. cluster_toolkit forms it by
subtracting two nearly-equal numbers, so below ~1e-2 the result is
cancellation-limited -- 21% error at x_mis = 1e-2 and the WRONG SIGN by
5e-3 -- and refining the Rsigma grid does not help (measured: 30k -> 120k
points moved x_mis = 5e-3 not at all). Outside the tabulated range the
reader clamps; see `clenspy.halo.miscentering_table`.

AXES
----
(ln x_mis, ln q) with q = x / x_mis, and ln q = 0 an EXACT node.

DeltaSigma_mis has a cusp along x = x_mis where it crosses zero. On axes
(ln x_mis, ln x) that ridge runs at 45 degrees to the grid and bilinear
interpolation cuts across it. Measured at equal node budget (200 x 300)
against the reference quadrature:

                        on the cusp                global
                  median      max    sign flips    median
    (ln x_mis, ln x)   6.1e-01  2.3e+03    25/60     7.9e-04
    (ln x_mis, ln q)   1.9e-04  1.4e-03     0/60     2.4e-03

The ln q = 0 node is what does the work: with 0 on a cell boundary no cell
ever straddles the cusp. An even node count that misses 0 throws the
benefit away, so it is asserted below. The ln q axis is also the binding
one -- pinning x_mis to a node leaves 1.9e-3, pinning ln q leaves 1.4e-4 --
so it gets three tiers, densest on the cusp.

CROSS-CHECK
-----------
`clenspy.halo.miscentering_kernel` is the independent implementation (the
by-parts reduction of docs/miscentering_math.md section 5). It agrees with
ct's Sigma_mis to 1e-11..1e-14 and is used by ``--tune`` and by
``validation/validate_miscentering_table.py`` to bound the generator error.
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

OUT = (Path(__file__).resolve().parents[1]
       / "src" / "clenspy" / "data" / "nfw_miscentering.npz")

# Domain, matching the DES Y3 table.
X_RANGE = (1e-3, 5e3)
XM_RANGE = (1e-2, 5e2)
N_XM = 250                        # as y3

# Ratio axis, three tiers. Every worst-case point sits at |ln q| < 0.2.
INNER, N_INNER = 0.6, 241         # odd, so ln q = 0 is exact
MID, N_MID = 3.0, 100             # per side
N_WING = 200                      # per side

# cluster_toolkit needs the centred Sigma tabulated on a grid that brackets
# every query. Chosen by --tune; see the module docstring.
RSIG_N, RSIG_LO, RSIG_HI = 300_000, -8.0, 6.0

# ct's signature carries (M, conc, Omega_m); with r_s = 1 they only serve to
# fix r_200 = conc, so the profile ct sees is exactly the dimensionless one.
CONC, OMEGA_M = 4.0, 0.3
RHO_CRIT = 2.77533742639e11       # Msun h^2 / Mpc^3
MASS = 4 * np.pi / 3 * 200 * (OMEGA_M * RHO_CRIT) * CONC**3


def ratio_axis():
    """ln q grid, clustered on the cusp, with 0 an exact node."""
    lo = np.log(X_RANGE[0]) - np.log(XM_RANGE[1])
    hi = np.log(X_RANGE[1]) - np.log(XM_RANGE[0])
    inner = np.linspace(-INNER, INNER, N_INNER)
    assert np.any(inner == 0.0), "ln q = 0 must be an exact node"
    mid_lo = np.linspace(-MID, -INNER, N_MID, endpoint=False)
    mid_hi = np.linspace(INNER, MID, N_MID + 1)[1:]
    wing_lo = np.linspace(lo, -MID, N_WING, endpoint=False)
    wing_hi = np.linspace(MID, hi, N_WING + 1)[1:]
    return np.concatenate([wing_lo, mid_lo, inner, mid_hi, wing_hi])


def _centred_sigma_hat(x):
    """Dimensionless centred NFW Sigma / Sigma_0, from clenspy."""
    from clenspy.halo.miscentering_kernel import nfw_sigma_hat
    return nfw_sigma_hat(x)


def _row(args):
    """One x_mis row: ct's Sigma_mis and DeltaSigma_mis at that row's x."""
    from cluster_toolkit import miscentering as ctm

    ln_xm, ln_q = args
    x_mis = float(np.exp(ln_xm))
    x = np.clip(x_mis * np.exp(ln_q), *X_RANGE)

    r_sig = np.logspace(RSIG_LO, RSIG_HI, RSIG_N)
    sig_c = _centred_sigma_hat(r_sig)
    # Sigma_mis on the full grid first -- DeltaSigma_mis_at_R integrates
    # Sigmabar off it, so it needs the profile, not just the query points.
    smis_grid = ctm.Sigma_mis_single_at_R(
        r_sig, r_sig, sig_c, MASS, CONC, OMEGA_M, x_mis)
    s = ctm.Sigma_mis_single_at_R(x, r_sig, sig_c, MASS, CONC, OMEGA_M, x_mis)
    d = ctm.DeltaSigma_mis_at_R(x, r_sig, smis_grid)
    return np.asarray(s, float), np.asarray(d, float)


def build(n_proc):
    ln_xm = np.linspace(np.log(XM_RANGE[0]), np.log(XM_RANGE[1]), N_XM)
    ln_q = ratio_axis()
    print(f"x_mis : {N_XM} nodes in [{XM_RANGE[0]:.0e}, {XM_RANGE[1]:.0e}]")
    print(f"ln q  : {ln_q.size} nodes in [{ln_q[0]:.2f}, {ln_q[-1]:.2f}], "
          f"finest step {np.min(np.diff(ln_q)):.5f}")
    print(f"Rsigma: {RSIG_N} pts, 1e{RSIG_LO:+.0f}..1e{RSIG_HI:+.0f}")
    print(f"generator: cluster_toolkit, {n_proc} processes")

    t0 = time.time()
    with Pool(n_proc) as pool:
        rows = pool.map(_row, [(lm, ln_q) for lm in ln_xm], chunksize=1)
    sig = np.array([r[0] for r in rows])
    dsg = np.array([r[1] for r in rows])
    print(f"built in {time.time() - t0:.1f}s")

    assert np.all(np.isfinite(sig)), "non-finite sigma_hat_mis"
    assert np.all(np.isfinite(dsg)), "non-finite ds_hat_mis"
    assert np.all(sig > 0), "sigma_hat_mis must be positive"
    print(f"ds_hat_mis: {(dsg < 0).mean() * 100:.1f}% negative (the lobe)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # float32 for the values: ~1e-7 relative, well below the interpolation
    # floor, and it halves the wheel payload.
    np.savez_compressed(
        OUT, ln_x_mis=ln_xm, ln_q=ln_q,
        sigma_hat_mis=sig.astype(np.float32),
        ds_hat_mis=dsg.astype(np.float32),
        generator=np.array("cluster_toolkit"),
        rsig=np.array([RSIG_N, RSIG_LO, RSIG_HI]),
    )
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)")


def _tune_one(cfg):
    """Accuracy of one Rsigma grid against the independent quadrature."""
    from cluster_toolkit import miscentering as ctm

    from clenspy.halo.miscentering_kernel import (
        miscentered_deltasigma, nfw_mean_sigma_hat, nfw_sigma_hat,
    )
    npts, lo, hi = cfg
    r_sig = np.logspace(lo, hi, npts)
    sig_c = nfw_sigma_hat(r_sig)
    err, edge = [], None
    for x_mis in (1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0):
        xs = np.array([x_mis, x_mis * 1.1, x_mis * 0.9, 1.0, 10.0])
        xs = xs[(xs > r_sig[0] * 10) & (xs < r_sig[-1] / 10)]
        smis = ctm.Sigma_mis_single_at_R(
            r_sig, r_sig, sig_c, MASS, CONC, OMEGA_M, x_mis)
        got = ctm.DeltaSigma_mis_at_R(xs, r_sig, smis)
        ref = miscentered_deltasigma(
            nfw_sigma_hat, nfw_mean_sigma_hat, xs, x_mis, n_nodes=4096)
        for a, b in zip(got, ref):
            if abs(b) > 1e-13:
                err.append(abs(a - b) / abs(b))
        if x_mis == 1e-2:
            edge = (abs(got[0] - ref[0]) / abs(ref[0]),
                    np.sign(got[0]) == np.sign(ref[0]))
    err = np.array(err)
    return cfg, np.median(err), np.percentile(err, 90), err.max(), edge


def tune(n_proc):
    cfgs = [(30_000, -6.0, 4.0), (120_000, -8.0, 4.0), (300_000, -8.0, 6.0),
            (600_000, -9.0, 6.0), (1_000_000, -10.0, 6.0)]
    with Pool(min(n_proc, len(cfgs))) as pool:
        out = pool.map(_tune_one, cfgs)
    print(f"{'Rsigma grid':>28} {'median':>9} {'p90':>9} {'max':>9} "
          f"{'@x_mis=1e-2':>12} {'sign':>6}")
    for cfg, med, p90, mx, edge in out:
        tag = f"{cfg[0]} pts 1e{cfg[1]:+.0f}..1e{cfg[2]:+.0f}"
        print(f"{tag:>28} {med:9.2e} {p90:9.2e} {mx:9.2e} "
              f"{edge[0]:12.2e} {str(edge[1]):>6}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tune", action="store_true",
                    help="scan Rsigma grids instead of building")
    ap.add_argument("-j", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="worker processes")
    a = ap.parse_args()
    tune(a.j) if a.tune else build(a.j)
