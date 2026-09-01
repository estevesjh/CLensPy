"""Produce every number in the report: bin summaries, ratio profiles, the
absolute comparison, and a channel decomposition, as CSVs under
data/processed/ with unit-and-configuration headers.

Reuses the validation machinery verbatim (one implementation of the
stacking estimator and of the model configuration):
validation/validate_sigma_prj_mock.py.

    SELECTION_BIAS_DIR=../SelectionBias python scripts/make_results.py
"""

from __future__ import annotations

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

CONFIG = dict(
    los_half_depth_hinv=V.LOS_HALF_DEPTH_HINV,
    exclusion="counter", r_trunc_hinv=V.APERTURE_HINV, n_theta=128,
    theta_perp_max_hinv=2 * V.APERTURE_HINV, xi_clip=False,
    hod="buzzard",
    cosmology=(f"Buzzard v1.1 (Om={V.OMEGA_M}, h={V.H}, "
               f"s8={V.COSMO.sigma8})"),
)


def _header(cols: str) -> str:
    cfg = ", ".join(f"{k}={v}" for k, v in CONFIG.items())
    return f"# config: {cfg}\n# {cols}\n"


def main() -> int:
    mock_dir = Path(os.environ.get("SELECTION_BIAS_DIR",
                                   str(REPO.parent / "SelectionBias")))
    from astropy.io import fits as afits

    with afits.open(mock_dir / "mock_lob_sigma_catalog.fits",
                    memmap=True) as hdul:
        d = hdul[1].data
        m200 = np.asarray(d["M200"], float)
        z = np.asarray(d["Z"], float)
        lam_ob = np.asarray(d["LAMBDA_OB_LOB"], float)
        lam_tr = np.asarray(d["LAMBDA_TR_LOB"], float)
        prof = np.asarray(d["SIGMA_PRJ_of_R"], float)
    cond0 = ~np.all(prof == 0.0, axis=1)
    n_total, n_profiled = cond0.size, int(cond0.sum())
    lnM_all, z_all = np.log(m200[cond0]), z[cond0]
    prof_all, lam_all = prof[cond0], lam_ob[cond0]
    dlam_all = (lam_ob - lam_tr)[cond0]

    xi_nl, hmf, bias, _ = V.build_halo_model()
    pk_prj, hmf_prj, two_halo_prj, bias_prj = V.build_projection_products()
    engine = SelBiasEngine(
        sigma_prj=SigmaPrj(cosmology=V.COSMO, hmf=hmf, bias=bias,
                           xi_nl=xi_nl).build(),
        mor=HodMor.buzzard(),
    )
    prj = SigmaPrj(cosmology=V.COSMO,
                   pk=pk_prj,
                   hmf=hmf_prj,
                   two_halo=two_halo_prj,
                   bias=bias_prj,
                   config=SigmaPrjConfig(
                       n_theta=CONFIG["n_theta"],
                       theta_perp_range=(1e-3,
                                         CONFIG["theta_perp_max_hinv"] / V.H),
                       los_depth=CONFIG["los_half_depth_hinv"] / V.H,
                       exclusion="counter",
                       r_trunc=CONFIG["r_trunc_hinv"] / V.H,
                   ))
    b_eff_ij, n_ij, lam_ij, zrep_ij = V.b_eff_table()

    bins_rows, prof_rows = [], []
    for j in range(V.Z_EDGES.size - 1):
        for i in range(V.LAMBDA_EDGES.size - 1):
            in_bin = ((lam_all >= V.LAMBDA_EDGES[i])
                      & (lam_all < V.LAMBDA_EDGES[i + 1])
                      & (z_all >= V.Z_EDGES[j])
                      & (z_all < V.Z_EDGES[j + 1]))
            n_sel = int(in_bin.sum())
            # forward-model bin representatives: N[lambda]/N[1], N[z]/N[1]
            lob_rep = float(lam_ij[i, j])
            zob_rep = float(zrep_ij[i, j])
            sel_mean = prof_all[in_bin].mean(axis=0)
            sel_err = prof_all[in_bin].std(axis=0) / np.sqrt(n_sel)
            rnd_stack, rnd_err = \
                V.stacked_profile_weighted_by_mass_redshift(
                    lnM_all[in_bin], z_all[in_bin], lnM_all, z_all,
                    prof_all)
            ratio_mock = sel_mean / rnd_stack
            sig_ratio = np.sqrt((sel_err / rnd_stack) ** 2
                                + (sel_mean * rnd_err
                                   / rnd_stack**2) ** 2)

            P1, I1, I2 = engine.operators(lob_rep, zob_rep)
            beff = float(b_eff_ij[i, j])
            delta_rnd = P1 + beff * I2
            delta_mock = float(dlam_all[in_bin].mean())
            bsel = engine.marginalised_bias(lob_rep, zob_rep, b_eff=beff)
            model_sel, _ = V.model_annulus_average(
                prj, lob_rep, zob_rep, bsel)
            model_rnd, _ = V.model_annulus_average(
                prj, lob_rep, zob_rep, lambda th: beff)
            # correlated channel alone; the background follows by
            # linearity of the annulus average, bkg = tot - cl
            model_cl, _ = V.model_annulus_average(
                prj, lob_rep, zob_rep, bsel, channel="cl")
            ratio_model = model_sel / model_rnd

            # b_lambda-ob(R)/b_eff -- paper Fig. 6's dashed line. R is a
            # comoving transverse separation at zob, so theta = R/chi(zob)
            # (the (1+z) factors in r_lambda's own physical->comoving
            # conversion cancel exactly against angular-diameter distance).
            theta_of_R = (V.R_MID_HINV / V.H) / engine.chi(zob_rep)
            bsel_over_beff = bsel(theta_of_R) / beff

            keep = np.arange(V.R_MID_HINV.size) >= V.I_RBIN_MIN
            score = keep & (V.R_MID_HINV > V.R_2H_HINV)
            resid = np.abs(ratio_model - ratio_mock)
            bins_rows.append([
                i, j, V.LAMBDA_EDGES[i], V.LAMBDA_EDGES[i + 1],
                V.Z_EDGES[j], V.Z_EDGES[j + 1], n_sel, lob_rep, zob_rep,
                beff, bsel.b_small, bsel.b_large, delta_mock, delta_rnd,
                resid[score].max(), np.median(resid[score]),
                resid[keep & ~score].max(),
            ])
            for k in np.where(keep)[0]:
                prof_rows.append([
                    i, j, V.R_MID_HINV[k], ratio_mock[k], sig_ratio[k],
                    ratio_model[k], sel_mean[k] * V.H, rnd_stack[k] * V.H,
                    model_sel[k], model_rnd[k], sel_err[k] * V.H,
                    model_cl[k], bsel_over_beff[k],
                ])
            print(f"bin ({i},{j}) done: n={n_sel}, "
                  f"2h max resid={resid[score].max():.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "bins_summary.csv", "w") as f:
        f.write(_header(
            "i, j, lam_lo, lam_hi, z_lo, z_hi, n_sel, lob_rep, zob_rep, "
            "b_eff, b_small, b_large, delta_prj_mock, delta_prj_rnd_model, "
            "resid2h_max, resid2h_med, resid_inner_max "
            "[richness/redshift dimensionless; residuals of the "
            "sel/RND ratio]"))
        np.savetxt(f, np.asarray(bins_rows), delimiter=",", fmt="%.6g")
    with open(OUT / "ratio_profiles.csv", "w") as f:
        f.write(_header(
            "i, j, R [cMpc/h], ratio_mock, sigma_ratio, ratio_model, "
            "sigma_sel_mock [Msun/cMpc^2 h-free], sigma_rnd_mock [Msun/cMpc^2 h-free], "
            "sigma_sel_model [Msun/cMpc^2 h-free], "
            "sigma_rnd_model [Msun/cMpc^2 h-free], "
            "sigma_sel_mock_err [Msun/cMpc^2 h-free], "
            "sigma_cl_model [Msun/cMpc^2 h-free, correlated channel only], "
            "bsel_over_beff [b_lambda-ob(R)/b_eff, dimensionless]"))
        np.savetxt(f, np.asarray(prof_rows), delimiter=",", fmt="%.6g")

    # absolute comparison at the Costanzi notebook's cell-19 selection
    in_c = ((lam_all >= 19.5) & (lam_all < 20.5)
            & (z_all >= 0.475) & (z_all < 0.525))
    lob_c = float(lam_all[in_c].mean())
    zob_c = float(z_all[in_c].mean())
    mock_c = prof_all[in_c].mean(axis=0) * V.H
    mock_c_err = prof_all[in_c].std(axis=0) / np.sqrt(in_c.sum()) * V.H
    bsel_c = engine.marginalised_bias(lob_c, zob_c,
                                      b_eff=float(b_eff_ij[0, 2]))
    model_c, _ = V.model_annulus_average(prj, lob_c, zob_c, bsel_c)
    with open(OUT / "absolute.csv", "w") as f:
        f.write(_header(
            f"R [cMpc/h], sigma_mock, sigma_mock_err, sigma_model "
            f"[all Msun/cMpc^2 h-free]; lob={lob_c:.2f} zob={zob_c:.4f} "
            f"n={int(in_c.sum())}"))
        np.savetxt(f, np.column_stack([V.R_MID_HINV, mock_c, mock_c_err,
                                       model_c]),
                   delimiter=",", fmt="%.6g")

    # channel decomposition at the representative bin (0, 1)
    R_dec = np.geomspace(0.1, 40.0, 32)  # h-free comoving Mpc
    bsel_d = engine.marginalised_bias(23.9, 0.425,
                                      b_eff=float(b_eff_ij[0, 1]))
    prj.sigma_prj(R_dec, 23.9, 0.425, bsel_d)
    with open(OUT / "decomposition.csv", "w") as f:
        f.write(_header("R [cMpc h-free], rnd, cl, sum "
                        "[Msun/cMpc^2 h-free]; lob=23.9 zob=0.425"))
        np.savetxt(f, np.column_stack([R_dec, prj.rnd, prj.cl,
                                       prj.rnd + prj.cl]),
                   delimiter=",", fmt="%.6g")

    with open(OUT / "catalog_meta.csv", "w") as f:
        f.write(_header("n_total, n_profiled"))
        np.savetxt(f, [[n_total, n_profiled]], delimiter=",", fmt="%d")
    print("wrote CSVs to", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
