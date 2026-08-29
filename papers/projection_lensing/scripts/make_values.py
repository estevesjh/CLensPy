"""CSVs -> build/values.tex (macros) + build/table_bins.tex.

Every number quoted in the prose comes from here; nothing is hard-coded
in the .tex sources.
"""

from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed"
BUILD = BASE / "build"


def main():
    bins = np.loadtxt(DATA / "bins_summary.csv", delimiter=",")
    meta = np.loadtxt(DATA / "catalog_meta.csv", delimiter=",")
    R, mock, err, model = np.loadtxt(DATA / "absolute.csv",
                                     delimiter=",", unpack=True)

    resid2h_max = bins[:, 14]
    resid2h_med = bins[:, 15]
    band = (R > 3.0) & (R <= 25.05)
    legc = np.abs(model[band] / mock[band] - 1.0)
    dd = np.abs(bins[:, 13] / bins[:, 12] - 1.0)  # |Delta_RND/Delta_mock - 1|

    BUILD.mkdir(exist_ok=True)
    with open(BUILD / "values.tex", "w") as f:
        w = lambda name, val: f.write(f"\\newcommand{{\\{name}}}{{{val}}}\n")
        w("NHalosTotal", f"{int(meta[0]):,}".replace(",", "\\,"))
        w("NHalosProfiled", f"{int(meta[1]):,}".replace(",", "\\,"))
        w("NBins", f"{bins.shape[0]:d}")
        w("MaxTwoHaloResid", f"{resid2h_max.max():.3f}")
        w("MinTwoHaloResid", f"{resid2h_max.min():.3f}")
        w("MedTwoHaloResid", f"{np.median(resid2h_med):.3f}")
        w("MaxInnerResid", f"{bins[:, 16].max():.2f}")
        w("LegCMaxDev", f"{legc.max() * 100:.1f}\\%")
        w("DeltaPrjAgreeMin", f"{dd.min() * 100:.1f}\\%")
        w("DeltaPrjAgreeMax", f"{dd.max() * 100:.0f}\\%")
        w("BeffMin", f"{bins[:, 9].min():.2f}")
        w("BeffMax", f"{bins[:, 9].max():.2f}")

    inner = np.loadtxt(DATA / "inner_study_summary.csv", delimiter=",")
    with open(BUILD / "values.tex", "a") as f:
        w = lambda name, val: f.write(f"\\newcommand{{\\{name}}}{{{val}}}\n")
        w("BsmallYthree", f"{inner[0, 1]:.1f}")
        w("BsmallSelf", f"{inner[1, 1]:.0f}")
        w("BsmallPoint", f"{inner[2, 1]:.2f}")
        w("BsmallFit", f"{inner[3, 1]:.2f}")
        prof = np.loadtxt(DATA / "inner_study.csv", delimiter=",")
        mfit = prof[:, 0] == 3
        pull = np.abs(prof[mfit, 4] - prof[mfit, 2]) / prof[mfit, 3]
        w("FitMaxPull", f"{pull.max():.2f}")
        w("FitMeanPull", f"{pull.mean():.2f}")

    # per-bin table
    with open(BUILD / "table_bins.tex", "w") as f:
        f.write("\\begin{tabular}{cc r rrr rr r}\n\\toprule\n")
        f.write("$\\lambda^{\\rm ob}$ & $z$ & $N_{\\rm cl}$ & "
                "$b_{\\rm eff}$ & $b_{\\rm small}$ & $b_{\\rm large}$ & "
                "$\\Delta^{\\rm prj}_{\\rm mock}$ & "
                "$\\Delta^{\\rm prj}_{\\rm RND}$ & "
                "$\\max|\\delta r|_{2h}$\\\\\n\\midrule\n")
        for row in bins:
            (i, j, lam_lo, lam_hi, z_lo, z_hi, n_sel, _lob, _zob, beff,
             bs, bl, dmock, drnd, r2h, _r2m, _rin) = row
            f.write(f"$[{lam_lo:.0f},{lam_hi:.0f})$ & "
                    f"$[{z_lo:.2f},{z_hi:.2f})$ & {int(n_sel):d} & "
                    f"{beff:.2f} & {bs:.1f} & {bl:.2f} & "
                    f"{dmock:.2f} & {drnd:.2f} & {r2h:.3f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("wrote", BUILD / "values.tex", "and", BUILD / "table_bins.tex")


if __name__ == "__main__":
    main()
