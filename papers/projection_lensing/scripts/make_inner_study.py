"""Inner-region study: the b_small marginalisation bracket.

Matteo's closure is algebraically identical to `SelBiasEngine._closure`
(his ``bar_delta_prj_bkg`` = P1 + boost*b_eff*I1, his ``numerator2`` =
I2 - I1), but the lambda_tr marginalisation his notebook feeds to the
theta-grid interpolator (``b_sel_lob_theta_grid_inter``) is not defined
in the notebook. This script brackets the defensible choices at the
reference bin (lambda_ob in [20,30), z in [0.35,0.50)) and fits the
mock-implied inner plateau:

  A. "y3"    -- Y3 EMG kernel marginalisation (the package default)
  B. "self"  -- self-consistent exponential kernel x n(lambda_tr) prior
  C. "point" -- single point lambda_tr = lambda_ob - Delta_RND
                (the closure evaluated at its own mean boost)
  D. "fit"   -- b_small fitted to the mock inner ratio (R < 1 cMpc/h)

Writes data/processed/inner_study.csv (profiles) and
inner_study_summary.csv (the b_small values).

    SELECTION_BIAS_DIR=../SelectionBias python scripts/make_inner_study.py
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
from clenspy.lensing import SigmaPrj  # noqa: E402
from clenspy.selection import PhysicalMassMor, SelBiasEngine, SigmoidBias  # noqa: E402
from clenspy.selection.scaling_relation import HodMor  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "processed"

LAM_LO, LAM_HI, Z_LO, Z_HI = 20.0, 30.0, 0.35, 0.50
B_EFF = 3.02  # leg-A value for this bin (bins_summary.csv)


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
    lnM_all, z_all = np.log(m200[cond0]), z[cond0]
    prof_all, lam_all = prof[cond0], lam_ob[cond0]
    dlam_all = (lam_ob - lam_tr)[cond0]

    in_bin = ((lam_all >= LAM_LO) & (lam_all < LAM_HI)
              & (z_all >= Z_LO) & (z_all < Z_HI))
    lob = float(lam_all[in_bin].mean())
    zob = float(z_all[in_bin].mean())
    delta_mock = float(dlam_all[in_bin].mean())

    sel_mean = prof_all[in_bin].mean(axis=0)
    sel_err = prof_all[in_bin].std(axis=0) / np.sqrt(in_bin.sum())
    rnd_stack, rnd_err = V.stacked_profile_weighted_by_mass_redshift(
        lnM_all[in_bin], z_all[in_bin], lnM_all, z_all, prof_all)
    ratio_mock = sel_mean / rnd_stack
    sig_ratio = np.sqrt((sel_err / rnd_stack) ** 2
                        + (sel_mean * rnd_err / rnd_stack**2) ** 2)

    xi_nl, hmf, bias, _ = V.build_halo_model()
    engine = SelBiasEngine(cosmology=V.COSMO, xi_nl=xi_nl, hmf=hmf,
                           bias=bias,
                           mor=PhysicalMassMor(HodMor.buzzard(), V.H))
    prj = SigmaPrj(cosmology=V.COSMO, xi_nl=xi_nl, hmf=hmf, bias=bias,
                   n_theta=128, theta_perp_range=(1e-3, 60.0 / V.H),
                   los_window="hard", los_depth=50.0 / V.H,
                   exclusion="counter", r_trunc=30.0 / V.H)

    P1, I1, I2 = engine.operators(lob, zob)
    mean_d, var_d = engine.delta_stats(lob, zob, B_EFF)
    theta_lam = engine._theta_lob(lob, zob)

    def closure_point(delta):
        """b_small at a single Delta (Matteo's closure, scalar input)."""
        d_arr = np.array([lob - delta])   # ltr with gap = delta
        _, bs, bl = engine._closure(lob, P1, I1, I2, B_EFF, d_arr)
        return float(bs[0]), float(bl[0])

    variants = {}
    bs, bl = engine.plateaus(lob, zob, b_eff=B_EFF, plob_mode="y3")
    variants["y3"] = (bs, bl)
    bs, bl = engine.plateaus(lob, zob, b_eff=B_EFF, plob_mode="self")
    variants["self"] = (bs, bl)
    variants["point"] = closure_point(mean_d)

    model_rnd, _ = V.model_annulus_average(prj, lob, zob,
                                           lambda th: B_EFF)

    def ratio_for(b_small, b_large):
        bsel = SigmoidBias(lob=lob, zob=zob, theta_lambda=theta_lam,
                           b_small=b_small, b_large=b_large)
        m_sel, _ = V.model_annulus_average(prj, lob, zob, bsel)
        return m_sel / model_rnd

    # D: fit b_small to the mock inner ratio (R < 1 cMpc/h, skip first 4)
    inner = (np.arange(V.R_MID_HINV.size) >= V.I_RBIN_MIN) \
        & (V.R_MID_HINV < 1.0)
    b_large_y3 = variants["y3"][1]

    def cost(bs):
        r = ratio_for(bs, b_large_y3)
        return np.sum(((r - ratio_mock)[inner] / sig_ratio[inner]) ** 2)

    from scipy.optimize import minimize_scalar
    fit = minimize_scalar(cost, bounds=(1.0, 40.0), method="bounded")
    variants["fit"] = (float(fit.x), b_large_y3)

    rows, summary = [], []
    for name, (bs, bl) in variants.items():
        r = ratio_for(bs, bl)
        for k in np.where(np.arange(V.R_MID_HINV.size) >= V.I_RBIN_MIN)[0]:
            rows.append([{"y3": 0, "self": 1, "point": 2, "fit": 3}[name],
                         V.R_MID_HINV[k], ratio_mock[k], sig_ratio[k], r[k]])
        summary.append([{"y3": 0, "self": 1, "point": 2, "fit": 3}[name],
                        bs, bl])
        print(f"{name:6s}: b_small={bs:7.2f} b_large={bl:5.2f}")
    print(f"Delta: mock={delta_mock:.2f}, model RND={mean_d:.2f}, "
          f"var={var_d:.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "inner_study.csv", "w") as f:
        f.write("# variant (0=y3 EMG, 1=self kernel, 2=point Delta_RND, "
                "3=fit), R [cMpc/h], ratio_mock, sigma_ratio, ratio_model\n"
                f"# bin lam [{LAM_LO},{LAM_HI}) z [{Z_LO},{Z_HI}); "
                f"lob={lob:.2f} zob={zob:.4f} b_eff={B_EFF}; "
                f"delta_mock={delta_mock:.3f} delta_rnd={mean_d:.3f} "
                f"var={var_d:.3f}\n")
        np.savetxt(f, np.asarray(rows), delimiter=",", fmt="%.6g")
    with open(OUT / "inner_study_summary.csv", "w") as f:
        f.write("# variant, b_small, b_large\n")
        np.savetxt(f, np.asarray(summary), delimiter=",", fmt="%.6g")
    print("wrote", OUT / "inner_study.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
