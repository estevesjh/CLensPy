r"""Validate SelBiasEngine/SigmaPrj against Costanzi et al. (2026) Fig. 6
(their own model curve, digitized -- see ``data/costanzi2026_fig6.csv``).

Unlike ``validate_sigma_prj_mock.py`` (model vs. our own re-measurement of
the mock, with real shot noise), this compares model vs. a *published
theory curve*: any residual is attributable to a model/input mismatch, not
measurement noise on either side (digitization error aside).

    SELECTION_BIAS_DIR=../SelectionBias python validation/validate_fig6_digitized.py --plot
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_sigma_prj_mock as V  # noqa: E402
from clenspy.lensing import SigmaPrj, SigmaPrjConfig  # noqa: E402
from clenspy.selection import SelBiasEngine  # noqa: E402
from clenspy.selection.scaling_relation import HodMor  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "costanzi2026_fig6.csv"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    digi = np.loadtxt(DATA, delimiter=",", skiprows=13)
    lam_lo, lam_hi, z_lo, z_hi, R, ratio = digi.T

    xi_nl, hmf, bias, _ = V.build_halo_model()
    pk_prj, hmf_prj, two_halo_prj, bias_prj = V.build_projection_products()
    engine = SelBiasEngine(
        sigma_prj=SigmaPrj(cosmology=V.COSMO, hmf=hmf, bias=bias,
                           xi_nl=xi_nl).build(),
        mor=HodMor.buzzard(),
    )
    prj = SigmaPrj(
        cosmology=V.COSMO, pk=pk_prj, hmf=hmf_prj, two_halo=two_halo_prj,
        bias=bias_prj,
        config=SigmaPrjConfig(
            n_theta=128, theta_perp_range=(1e-3, 2.0 * V.APERTURE_HINV / V.H),
            los_depth=V.LOS_HALF_DEPTH_HINV / V.H, exclusion="counter",
            r_trunc=V.APERTURE_HINV / V.H,
        ),
    )

    b_eff_ij, n_ij, lam_ij, zrep_ij = V.b_eff_table()

    panels = sorted(set(zip(lam_lo, lam_hi, z_lo, z_hi)))
    rows = []
    for (ll, lh, zl, zh) in panels:
        i = int(np.where(np.isclose(V.LAMBDA_EDGES[:-1], ll))[0][0])
        j = int(np.where(np.isclose(V.Z_EDGES[:-1], zl))[0][0])
        lob_rep = float(lam_ij[i, j])
        zob_rep = float(zrep_ij[i, j])
        beff = float(b_eff_ij[i, j])

        m = ((lam_lo == ll) & (lam_hi == lh) & (z_lo == zl) & (z_hi == zh))
        R_panel = R[m]
        ratio_digi = ratio[m]

        bsel = engine.marginalised_bias(lob_rep, zob_rep, b_eff=beff)
        sel = prj.sigma_prj(R_panel / V.H, lob_rep, zob_rep, bsel, channel="sum")
        rnd = prj.sigma_prj(R_panel / V.H, lob_rep, zob_rep,
                            lambda th: beff, channel="sum")
        ratio_model = sel / rnd

        resid = ratio_model / ratio_digi - 1.0
        print(f"lam[{ll:.0f},{lh:.0f}) z[{zl:.2f},{zh:.2f}) "
              f"lob_rep={lob_rep:6.1f} b_eff={beff:5.2f}  "
              f"n={m.sum():3d}  frac resid: "
              f"min={resid.min():+.3f} max={resid.max():+.3f} "
              f"med={np.median(resid):+.3f}")
        for k in range(m.sum()):
            rows.append((ll, lh, zl, zh, R_panel[k], ratio_digi[k],
                        ratio_model[k], resid[k]))

    rows = np.asarray(rows)
    print(f"\noverall: {len(rows)} points, "
          f"median |frac resid| = {np.median(np.abs(rows[:, 7])):.3f}, "
          f"max |frac resid| = {np.max(np.abs(rows[:, 7])):.3f}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, len(panels) // 3, figsize=(15, 4),
                                sharex=True, sharey=True)
        colors = {0.20: "tab:blue", 0.35: "tab:orange", 0.50: "tab:green"}
        for (ll, lh, zl, zh) in panels:
            i = int(np.where(np.isclose(V.LAMBDA_EDGES[:-1], ll))[0][0])
            ax = axs[i] if len(panels) // 3 > 1 else axs
            m = (rows[:, 0] == ll) & (rows[:, 2] == zl)
            c = colors[zl]
            ax.plot(rows[m, 4], rows[m, 5], "o", color=c, ms=3, alpha=0.6)
            ax.plot(rows[m, 4], rows[m, 6], "-", color=c, lw=1.6)
            ax.set_xscale("log")
            ax.set_title(f"lam[{ll:.0f},{lh:.0f})")
        fig.tight_layout()
        out = Path("validation/data/fig6_digitized_check.png")
        fig.savefig(out, dpi=150)
        print(f"\nwrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
