r"""Reduce data/processed/*.csv into build/values.tex (macros) and
build/table_*.tex (booktabs tables) for the report -- no number in the
report prose is hand-typed; everything traces back to one of these CSVs.

    python scripts/make_values.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
BUILD = ROOT / "build"


def read_csv(name):
    with open(DATA / name) as f:
        r = csv.DictReader(f)
        rows = list(r)
    return rows


def col(rows, name, dtype=float):
    return np.array([dtype(r[name]) for r in rows])


def elasticity(x, y):
    """d ln y / d ln x, log-log least squares over the grid."""
    lx, ly = np.log(x), np.log(y)
    A = np.column_stack([lx, np.ones_like(lx)])
    slope, _ = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope)


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def main() -> int:
    BUILD.mkdir(parents=True, exist_ok=True)
    out = []

    # ---------------------------------------------------------- (a) Fig. 6
    panels = read_csv("fig6_panels.csv")
    points = read_csv("fig6_points.csv")
    resid_all = np.abs(col(points, "frac_resid"))
    out.append(macro("FigSixNPoints", len(points)))
    out.append(macro("FigSixNPanels", len(panels)))
    out.append(macro("FigSixMedianAbsResid", f"{100*np.median(resid_all):.1f}\\%"))
    out.append(macro("FigSixMaxAbsResid", f"{100*np.max(resid_all):.1f}\\%"))

    lines = []
    for p in panels:
        lines.append(
            rf"$\lambda\in[{float(p['lam_lo']):.0f},{float(p['lam_hi']):.0f})$ & "
            rf"$[{float(p['z_lo']):.2f},{float(p['z_hi']):.2f})$ & "
            rf"{float(p['lob_rep']):.1f} & {float(p['b_eff']):.2f} & "
            rf"{100*float(p['median_resid']):+.1f}\% & "
            rf"{100*float(p['max_resid']):+.1f}\% \\"
        )
    with open(BUILD / "table_fig6.tex", "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---------------------------------------------------------- (b) mock
    legd = read_csv("mock_legD.csv")
    legc = read_csv("mock_legC.csv")
    n_pass = sum(int(r["ok"]) for r in legd)
    out.append(macro("MockNBins", len(legd)))
    out.append(macro("MockNPass", n_pass))
    resid2h = col(legd, "resid_2h_max")
    out.append(macro("MockTwoHaloMinResid", f"{np.min(resid2h):.3f}"))
    out.append(macro("MockTwoHaloMaxResid", f"{np.max(resid2h):.3f}"))
    delta_mock = col(legd, "delta_mock")
    delta_rnd = col(legd, "delta_rnd_model")
    drnd_frac = np.abs(delta_rnd / delta_mock - 1.0)
    out.append(macro("MockDeltaRndMinResid", f"{100*np.min(drnd_frac):.0f}\\%"))
    out.append(macro("MockDeltaRndMaxResid", f"{100*np.max(drnd_frac):.0f}\\%"))
    ltr_mock = col(legd, "ltr_mock_mean")
    ltr_model = col(legd, "ltr_implied")
    ltr_frac = np.abs(ltr_model / ltr_mock - 1.0)
    out.append(macro("MockLtrMedianResid", f"{100*np.median(ltr_frac):.1f}\\%"))
    out.append(macro("MockLtrMaxResid", f"{100*np.max(ltr_frac):.1f}\\%"))
    if legc:
        out.append(macro("MockLegCMaxDev", f"{100*float(legc[0]['max_frac_dev']):.1f}\\%"))
        out.append(macro("MockLegCLob", f"{float(legc[0]['lob_c']):.1f}"))
        out.append(macro("MockLegCZob", f"{float(legc[0]['zob_c']):.3f}"))

    lege = read_csv("mock_legE.csv")
    lines = [rf"{float(r['R_cMpc']):.1f} & {float(r['cl_unit_b']):.3e} & "
            rf"{float(r['rho_m_sigma_2h']):.3e} & {float(r['ratio']):.3f} \\"
            for r in lege]
    with open(BUILD / "table_legE.tex", "w") as f:
        f.write("\n".join(lines) + "\n")

    lines = []
    for r in legd:
        lines.append(
            rf"$\lambda\in[{float(r['lam_lo']):.0f},{float(r['lam_hi']):.0f})$ & "
            rf"$[{float(r['z_lo']):.2f},{float(r['z_hi']):.2f})$ & "
            rf"{int(float(r['n']))} & {float(r['lob_rep']):.1f} & "
            rf"{float(r['b_eff']):.2f} & {float(r['b_small']):.2f} & "
            rf"{float(r['b_large']):.2f} & {float(r['delta_mock']):.2f} & "
            rf"{float(r['delta_rnd_model']):.2f} & "
            rf"{float(r['resid_2h_max']):.3f} & "
            rf"{'ok' if int(r['ok']) else 'FAIL'} \\"
        )
    with open(BUILD / "table_mock.tex", "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---------------------------------------------------- (d) alpha sensitivity
    alpha = read_csv("alpha_sensitivity.csv")
    lam_lo_vals = sorted(set(float(r["lam_lo"]) for r in alpha))
    for lo in lam_lo_vals:
        rows_i = [r for r in alpha if float(r["lam_lo"]) == lo]
        e_small = [elasticity(col([r for r in rows_i if int(r["j"]) == j], "alpha"),
                              col([r for r in rows_i if int(r["j"]) == j], "b_small"))
                   for j in sorted(set(int(r["j"]) for r in rows_i))]
        e_large = [elasticity(col([r for r in rows_i if int(r["j"]) == j], "alpha"),
                              col([r for r in rows_i if int(r["j"]) == j], "b_large"))
                   for j in sorted(set(int(r["j"]) for r in rows_i))]
        tag = "Low" if lo < 30 else "High"
        out.append(macro(f"Alpha{tag}ElastSmallMin", f"{min(e_small):.2f}"))
        out.append(macro(f"Alpha{tag}ElastSmallMax", f"{max(e_small):.2f}"))
        out.append(macro(f"Alpha{tag}ElastLargeMin", f"{min(e_large):.2f}"))
        out.append(macro(f"Alpha{tag}ElastLargeMax", f"{max(e_large):.2f}"))

    # ---------------------------------------------------- (e) cosmo sensitivity
    cosmo = read_csv("cosmo_sensitivity.csv")
    max_abs_dratio = 0.0
    for i in sorted(set(int(r["i"]) for r in cosmo)):
        for param, col_name in (("Om", "Omega_m"), ("s8", "sigma8")):
            rows_p = [r for r in cosmo if int(r["i"]) == i and r["param"] == param]
            fid = [r for r in rows_p if abs(float(r["frac"]) - 1.0) < 1e-9]
            for rv in sorted(set(float(r["R_hinv_cMpc"]) for r in rows_p)):
                rr = sorted([r for r in rows_p if float(r["R_hinv_cMpc"]) == rv],
                           key=lambda r: float(r["frac"]))
                x = col(rr, col_name)
                y = col(rr, "ratio")
                fid_ratio = y[len(y) // 2]
                dratio = np.max(np.abs(y / fid_ratio - 1.0))
                max_abs_dratio = max(max_abs_dratio, dratio)
    out.append(macro("CosmoMaxAbsRatioShift", f"{100*max_abs_dratio:.2f}\\%"))

    with open(BUILD / "values.tex", "w") as f:
        f.writelines(out)
    print(f"wrote {BUILD/'values.tex'} ({len(out)} macros)")
    print(f"wrote {BUILD/'table_fig6.tex'}, {BUILD/'table_mock.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
