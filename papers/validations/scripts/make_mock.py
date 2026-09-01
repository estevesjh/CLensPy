r"""Validate SelBiasEngine/SigmaPrj against the Costanzi mock catalogue
(legs A/C/D/E of validation/validate_sigma_prj_mock.py, reused here and
written out as CSVs for the report build instead of only printed).

Writes data/processed/mock_legD.csv (12 lambda/z bins), mock_legC.csv
(absolute-normalisation check), mock_legE.csv (two-halo diagnostic).

    SELECTION_BIAS_DIR=../../SelectionBias python scripts/make_mock.py
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
from clenspy.cosmology.pkgrid import PkGrid  # noqa: E402
from clenspy.halo.twohalo import TwoHaloTerm  # noqa: E402
from clenspy.lensing import SigmaPrj, SigmaPrjConfig  # noqa: E402
from clenspy.selection import SelBiasEngine  # noqa: E402
from clenspy.selection.scaling_relation import HodMor  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "processed"
R_2H_HINV = V.R_2H_HINV
TOL_RATIO_ABS = V.TOL_RATIO_ABS
TOL_ABSOLUTE_FRAC = V.TOL_ABSOLUTE_FRAC


def main() -> int:
    mock_dir = Path(os.environ.get("SELECTION_BIAS_DIR", "../../SelectionBias")).expanduser()
    fits_path = mock_dir / "mock_lob_sigma_catalog.fits"
    from astropy.io import fits as afits

    print(f"reading {fits_path} ...")
    with afits.open(fits_path, memmap=True) as hdul:
        data = hdul[1].data
        m200 = np.asarray(data["M200"], dtype=np.float64)
        z_true = np.asarray(data["Z"], dtype=np.float64)
        lam_ob = np.asarray(data["LAMBDA_OB_LOB"], dtype=np.float64)
        lam_tr = np.asarray(data["LAMBDA_TR_LOB"], dtype=np.float64)
        sigma_prj = np.asarray(data["SIGMA_PRJ_of_R"], dtype=np.float64)
    cond0 = ~np.all(sigma_prj == 0.0, axis=1)
    print(f"  {cond0.sum():,} profiled halos of {cond0.size:,}")

    lnM_all = np.log(m200[cond0])
    z_all = z_true[cond0]
    prof_all = sigma_prj[cond0]
    lam_all = lam_ob[cond0]
    ltr_all = lam_tr[cond0]
    dlam_all = (lam_ob - lam_tr)[cond0]

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

    score_r = V.R_MID_HINV > R_2H_HINV
    score_r &= np.arange(V.R_MID_HINV.size) >= V.I_RBIN_MIN
    inner_r = ~score_r & (np.arange(V.R_MID_HINV.size) >= V.I_RBIN_MIN)

    legd_rows = []
    for j in range(V.Z_EDGES.size - 1):
        for i in range(V.LAMBDA_EDGES.size - 1):
            in_bin = ((lam_all >= V.LAMBDA_EDGES[i]) & (lam_all < V.LAMBDA_EDGES[i + 1])
                      & (z_all >= V.Z_EDGES[j]) & (z_all < V.Z_EDGES[j + 1]))
            n_sel = int(in_bin.sum())
            if n_sel < 50:
                continue
            lob_rep = float(lam_ij[i, j])
            zob_rep = float(zrep_ij[i, j])
            beff = float(b_eff_ij[i, j])

            sel_mean = prof_all[in_bin].mean(axis=0)
            sel_err = prof_all[in_bin].std(axis=0) / np.sqrt(n_sel)
            rnd_stack, rnd_err = V.stacked_profile_weighted_by_mass_redshift(
                lnM_all[in_bin], z_all[in_bin], lnM_all, z_all, prof_all)
            ratio_mock = sel_mean / rnd_stack
            sig_ratio = np.sqrt((sel_err / rnd_stack) ** 2
                                + (sel_mean * rnd_err / rnd_stack**2) ** 2)

            P1, I1, I2 = engine.operators(lob_rep, zob_rep)
            delta_rnd_model = P1 + beff * I2
            delta_mock = float(dlam_all[in_bin].mean())
            ltr_mock_mean = float(ltr_all[in_bin].mean())

            delta = engine.excess_delta(lob_rep, zob_rep, beff)
            ltr_implied = lob_rep - delta_rnd_model * (1.0 + delta)

            bsel = engine.marginalised_bias(lob_rep, zob_rep, b_eff=beff)
            model_sel, _ = V.model_annulus_average(prj, lob_rep, zob_rep, bsel)
            model_rnd, _ = V.model_annulus_average(
                prj, lob_rep, zob_rep, lambda th: beff)
            ratio_model = model_sel / model_rnd

            resid = np.abs(ratio_model - ratio_mock)[score_r]
            resid_in = np.abs(ratio_model - ratio_mock)[inner_r]
            ok = np.all(resid < np.maximum(2.0 * sig_ratio[score_r], TOL_RATIO_ABS))

            legd_rows.append((i, j, V.LAMBDA_EDGES[i], V.LAMBDA_EDGES[i + 1],
                              V.Z_EDGES[j], V.Z_EDGES[j + 1], n_sel, lob_rep,
                              zob_rep, beff, bsel.b_small, bsel.b_large,
                              delta_mock, delta_rnd_model, ltr_mock_mean,
                              ltr_implied, float(resid.max()),
                              float(resid_in.max()), int(ok)))

    # -------------------------------------------------------- leg C
    in_c = ((lam_all >= 19.5) & (lam_all < 20.5)
            & (z_all >= 0.475) & (z_all < 0.525))
    legc_rows = []
    if in_c.sum() > 50:
        lob_c = float(lam_all[in_c].mean())
        zob_c = float(z_all[in_c].mean())
        mock_c = prof_all[in_c].mean(axis=0) * V.H
        bsel_c = engine.marginalised_bias(lob_c, zob_c, b_eff=float(b_eff_ij[0, 2]))
        model_c, _ = V.model_annulus_average(prj, lob_c, zob_c, bsel_c)
        band = (V.R_MID_HINV >= R_2H_HINV) & (V.R_MID_HINV <= 25.0)
        frac_c = np.abs(model_c[band] / mock_c[band] - 1.0)
        ok_c = frac_c.max() < TOL_ABSOLUTE_FRAC
        legc_rows.append((lob_c, zob_c, int(in_c.sum()), float(frac_c.max()),
                          int(ok_c)))

    # -------------------------------------------------------- leg E
    pk_lin = PkGrid(cosmo=V.COSMO, nonlinear=True)
    zob_e = 0.425
    twoh = TwoHaloTerm(pk_lin.k, pk_lin(pk_lin.k, zob_e))
    rho_m = prj.rho_m
    R_big = np.array([10.0, 20.0, 30.0])
    prj.sigma_prj(R_big, 25.0, zob_e, lambda th: 1.0)
    cl_unit_b = prj.cl.copy()
    sig2h = np.asarray(twoh.sigma(R_big, zob_e)) * rho_m
    lege_rows = [(float(rb), float(cl_unit_b[k]), float(sig2h[k]),
                 float(cl_unit_b[k] / sig2h[k])) for k, rb in enumerate(R_big)]

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "mock_legD.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "j", "lam_lo", "lam_hi", "z_lo", "z_hi", "n", "lob_rep",
                    "zob_rep", "b_eff", "b_small", "b_large", "delta_mock",
                    "delta_rnd_model", "ltr_mock_mean", "ltr_implied",
                    "resid_2h_max", "resid_inner_max", "ok"])
        w.writerows(legd_rows)
    with open(OUT / "mock_legC.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lob_c", "zob_c", "n", "max_frac_dev", "ok"])
        w.writerows(legc_rows)
    with open(OUT / "mock_legE.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["R_cMpc", "cl_unit_b", "rho_m_sigma_2h", "ratio"])
        w.writerows(lege_rows)

    print(f"wrote {OUT/'mock_legD.csv'} ({len(legd_rows)} rows, "
          f"{sum(r[-1] for r in legd_rows)}/{len(legd_rows)} pass)")
    print(f"wrote {OUT/'mock_legC.csv'} ({len(legc_rows)} rows)")
    print(f"wrote {OUT/'mock_legE.csv'} ({len(lege_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
