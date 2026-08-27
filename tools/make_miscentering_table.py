#!/usr/bin/env python
r"""Build the packaged miscentered-NFW lookup table.

Writes ``src/clenspy/data/nfw_miscentering.npz``: the dimensionless
single-offset profiles

    sigma_hat_mis(x, x_mis)  = Sigma_mis / Sigma_0
    ds_hat_mis(x, x_mis)     = DeltaSigma_mis / Sigma_0,   SIGNED

with Sigma_0 = 2 r_s rho_s, x = R / r_s and x_mis = R_mis / r_s. Mass,
concentration and cosmology enter only the prefactor, so one grid serves
every halo (docs/miscentering_math.md section 9.1).

AXES: (ln x_mis, ln q) with q = x / x_mis, and ln q = 0 an EXACT node.

DeltaSigma_mis has a cusp along x = x_mis where it crosses zero. On the
obvious axes (ln x_mis, ln x) that ridge runs at 45 degrees to the grid and
bilinear interpolation cuts across it. Measured against the quadrature, at
equal node budget (200 x 300):

                        on the cusp                global
                  median      max    sign flips    median
    (ln x_mis, ln x)   6.1e-01  2.3e+03    25/60     7.9e-04
    (ln x_mis, ln q)   1.9e-04  1.4e-03     0/60     2.4e-03

Ratio axes win by 3-6 orders where it matters and cost almost nothing
elsewhere. The ln q = 0 node is what does the work -- with 0 on a cell
boundary no cell ever straddles the cusp. An even node count that misses 0
throws the whole benefit away, so it is asserted below.

GENERATOR: clenspy.halo.miscentering_kernel, the by-parts + law-of-cosines
reduction of docs/miscentering_math.md section 5. Not cluster_toolkit:
ct.miscentering is excellent off the cusp (agrees to 1e-11 .. 1e-14) but
builds DeltaSigma by integrating Sigmabar numerically off a Sigma_mis grid,
which costs it 1-2% at x = x_mis -- precisely the error the ratio axes
exist to remove. ct is used instead as the independent cross-check in
validation/validate_miscentering_table.py.

Run:  python tools/make_miscentering_table.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from clenspy.halo.miscentering_kernel import (
    miscentered_deltasigma,
    miscentered_sigma,
    nfw_mean_sigma_hat,
    nfw_sigma_hat,
)

OUT = (Path(__file__).resolve().parents[1]
       / "src" / "clenspy" / "data" / "nfw_miscentering.npz")

# Domain. x covers 0.001-1000 r_s (well past any fitted radial range) and
# x_mis covers 0.001-100 r_s (tau_mis R_lambda / r_s is ~0.4 in DES Y3).
LNXM = (np.log(1e-3), np.log(1e2))
N_XM = 256

# Ratio axis, three tiers. The ln q axis is what limits accuracy: with
# x_mis pinned to a node the error is still 1.9e-3, while pinning ln q
# instead drops it to 1.4e-4. So nodes go here, concentrated on the cusp,
# where the curvature is -- every worst-case point sits at |ln q| < 0.2.
INNER, N_INNER = 0.6, 241         # |ln q| <= 0.6, step ~0.005 (odd: 0 exact)
MID, N_MID = 3.0, 80              # per side
N_WING = 200                      # per side, out to the reachable ends
N_NODES = 1024                    # GL nodes in the generator quadrature


def build_ratio_axis():
    """ln q grid, clustered around 0, with 0 an exact node."""
    # reachable ratio span: x in [1e-3, 1e3] over x_mis in LNXM
    lo, hi = np.log(1e-3) - LNXM[1], np.log(1e3) - LNXM[0]
    inner = np.linspace(-INNER, INNER, N_INNER)
    assert np.any(inner == 0.0), "ln q = 0 must be an exact node"
    mid_lo = np.linspace(-MID, -INNER, N_MID, endpoint=False)
    mid_hi = np.linspace(INNER, MID, N_MID + 1)[1:]
    wing_lo = np.linspace(lo, -MID, N_WING, endpoint=False)
    wing_hi = np.linspace(MID, hi, N_WING + 1)[1:]
    return np.concatenate([wing_lo, mid_lo, inner, mid_hi, wing_hi])


def main():
    lnxm = np.linspace(*LNXM, N_XM)
    lnq = build_ratio_axis()
    print(f"grid: x_mis {N_XM} nodes in "
          f"[{np.exp(lnxm[0]):.1e}, {np.exp(lnxm[-1]):.1e}]")
    print(f"      ln q  {lnq.size} nodes in [{lnq[0]:.2f}, {lnq[-1]:.2f}], "
          f"finest step {np.min(np.diff(lnq)):.5f}")

    sig = np.empty((N_XM, lnq.size))
    dsg = np.empty((N_XM, lnq.size))
    t0 = time.time()
    for i, lm in enumerate(lnxm):
        x_mis = float(np.exp(lm))
        x = x_mis * np.exp(lnq)
        sig[i] = miscentered_sigma(nfw_sigma_hat, x, x_mis, n_nodes=N_NODES)
        dsg[i] = miscentered_deltasigma(
            nfw_sigma_hat, nfw_mean_sigma_hat, x, x_mis, n_nodes=N_NODES)
        if i % 64 == 0:
            print(f"  row {i:4d}/{N_XM}  x_mis={x_mis:.3e}  "
                  f"ds in [{dsg[i].min():+.2e}, {dsg[i].max():+.2e}]")
    print(f"built in {time.time()-t0:.1f}s")

    assert np.all(np.isfinite(sig)), "non-finite sigma_hat_mis"
    assert np.all(np.isfinite(dsg)), "non-finite ds_hat_mis"
    assert np.all(sig > 0), "sigma_hat_mis must be positive"
    neg = (dsg < 0).mean() * 100
    print(f"ds_hat_mis: {neg:.1f}% negative entries (the physical lobe)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # float32 for the values: ~1e-7 relative, two orders below the
    # interpolation floor, and it halves the wheel payload.
    np.savez_compressed(
        OUT, ln_x_mis=lnxm, ln_q=lnq,
        sigma_hat_mis=sig.astype(np.float32),
        ds_hat_mis=dsg.astype(np.float32),
        n_nodes=np.array(N_NODES),
    )
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
