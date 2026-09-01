r"""Validate SelBiasEngine/SigmaPrj against Costanzi et al. (2026) Fig. 6
(their own published model curve, digitized -- a theory curve with no
mock shot noise, so any residual is a model/input mismatch).

Writes data/processed/fig6_points.csv (per-point) and
data/processed/fig6_panels.csv (per-panel summary).

    SELECTION_BIAS_DIR=../../SelectionBias python scripts/make_fig6.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "validation"))

import validate_sigma_prj_mock as V  # noqa: E402
from clenspy.lensing import SigmaPrj, SigmaPrjConfig  # noqa: E402
from clenspy.selection import SelBiasEngine  # noqa: E402
from clenspy.selection.scaling_relation import HodMor  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "processed"
DIGI = REPO / "validation" / "data" / "costanzi2026_fig6.csv"


def main() -> int:
    digi = np.loadtxt(DIGI, delimiter=",", skiprows=13)
    lam_lo, lam_hi, z_lo, z_hi, R, ratio_digi_all = digi.T

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
    point_rows, panel_rows = [], []
    for (ll, lh, zl, zh) in panels:
        i = int(np.where(np.isclose(V.LAMBDA_EDGES[:-1], ll))[0][0])
        j = int(np.where(np.isclose(V.Z_EDGES[:-1], zl))[0][0])
        lob_rep = float(lam_ij[i, j])
        zob_rep = float(zrep_ij[i, j])
        beff = float(b_eff_ij[i, j])

        m = ((lam_lo == ll) & (lam_hi == lh) & (z_lo == zl) & (z_hi == zh))
        R_panel = R[m]
        ratio_digi = ratio_digi_all[m]

        bsel = engine.marginalised_bias(lob_rep, zob_rep, b_eff=beff)
        sel = prj.sigma_prj(R_panel / V.H, lob_rep, zob_rep, bsel, channel="sum")
        rnd = prj.sigma_prj(R_panel / V.H, lob_rep, zob_rep,
                            lambda th: beff, channel="sum")
        ratio_model = sel / rnd
        resid = ratio_model / ratio_digi - 1.0

        for k in range(m.sum()):
            point_rows.append((ll, lh, zl, zh, R_panel[k], ratio_digi[k],
                               ratio_model[k], resid[k]))
        panel_rows.append((ll, lh, zl, zh, lob_rep, zob_rep, beff,
                           int(m.sum()), float(np.median(resid)),
                           float(np.max(resid)), float(np.min(resid)),
                           float(np.median(np.abs(resid))),
                           float(np.max(np.abs(resid)))))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "fig6_points.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lam_lo", "lam_hi", "z_lo", "z_hi", "R_hinv_cMpc",
                    "ratio_digi", "ratio_model", "frac_resid"])
        w.writerows(point_rows)
    with open(OUT / "fig6_panels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lam_lo", "lam_hi", "z_lo", "z_hi", "lob_rep", "zob_rep",
                    "b_eff", "n", "median_resid", "max_resid", "min_resid",
                    "median_abs_resid", "max_abs_resid"])
        w.writerows(panel_rows)
    print(f"wrote {OUT/'fig6_points.csv'} ({len(point_rows)} rows)")
    print(f"wrote {OUT/'fig6_panels.csv'} ({len(panel_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
