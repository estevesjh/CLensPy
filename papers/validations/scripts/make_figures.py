r"""Figures for the report, generated from data/processed/*.csv only.

    python scripts/make_figures.py
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DATA = ROOT / "data" / "processed"
BUILD = ROOT / "build"


def read_csv(name):
    with open(DATA / name) as f:
        return list(csv.DictReader(f))


def col(rows, name, dtype=float):
    return np.array([dtype(r[name]) for r in rows])


def fig_fig6():
    rows = read_csv("fig6_points.csv")
    lam_bins = sorted(set((float(r["lam_lo"]), float(r["lam_hi"])) for r in rows))
    z_colors = {0.20: "tab:blue", 0.35: "tab:orange", 0.50: "tab:green"}

    fig, axs = plt.subplots(1, len(lam_bins), figsize=(6.5 * len(lam_bins), 4.2),
                            sharey=True)
    if len(lam_bins) == 1:
        axs = [axs]
    for ax, (ll, lh) in zip(axs, lam_bins):
        sub = [r for r in rows if float(r["lam_lo"]) == ll and float(r["lam_hi"]) == lh]
        for zl in sorted(set(float(r["z_lo"]) for r in sub)):
            s = sorted([r for r in sub if float(r["z_lo"]) == zl],
                      key=lambda r: float(r["R_hinv_cMpc"]))
            R = col(s, "R_hinv_cMpc")
            ax.plot(R, col(s, "ratio_digi"), "o", ms=3, alpha=0.6,
                   color=z_colors.get(zl, "grey"))
            ax.plot(R, col(s, "ratio_model"), "-", lw=1.8,
                   color=z_colors.get(zl, "grey"),
                   label=rf"$z\in[{zl:.2f},{zl+0.15:.2f})$")
        ax.set_xscale("log")
        ax.set_xlabel(r"$R\;[h^{-1}{\rm cMpc}]$")
        ax.set_title(rf"$\lambda\in[{ll:.0f},{lh:.0f})$")
        ax.legend(fontsize=8, frameon=False)
    axs[0].set_ylabel(r"$\langle\Sigma^{\rm prj}\rangle_\lambda/"
                      r"\langle\Sigma^{\rm prj}\rangle_{\rm RND}$")
    fig.tight_layout()
    fig.savefig(BUILD / "fig_fig6.pdf")
    plt.close(fig)


def fig_alpha_sensitivity():
    rows = read_csv("alpha_sensitivity.csv")
    lam_bins = sorted(set((float(r["lam_lo"]), float(r["lam_hi"])) for r in rows))
    z_colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for (ll, lh), ls in zip(lam_bins, ["-", "--"]):
        sub = [r for r in rows if float(r["lam_lo"]) == ll and float(r["lam_hi"]) == lh]
        for j in sorted(set(int(r["j"]) for r in sub)):
            s = sorted([r for r in sub if int(r["j"]) == j], key=lambda r: float(r["alpha"]))
            alpha = col(s, "alpha")
            axs[0].plot(alpha, col(s, "b_small"), ls, color=z_colors[j], lw=1.8,
                       label=rf"$\lambda[{ll:.0f},{lh:.0f})$, z-bin {j}")
            axs[1].plot(alpha, col(s, "b_large"), ls, color=z_colors[j], lw=1.8)
    axs[0].set(xlabel=r"$\alpha$", ylabel=r"$B_{\rm small}$")
    axs[1].set(xlabel=r"$\alpha$", ylabel=r"$B_{\rm large}$")
    axs[0].legend(fontsize=7, frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(BUILD / "fig_alpha_sensitivity.pdf")
    plt.close(fig)


def fig_cosmo_sensitivity():
    rows = read_csv("cosmo_sensitivity.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    colors = {1.0: "k", 3.0: "tab:red"}
    for ax, param, xcol, xlabel in zip(
        axs, ("Om", "s8"), ("Omega_m", "sigma8"),
        (r"$\Omega_m$", r"$\sigma_8$"),
    ):
        for i in sorted(set(int(r["i"]) for r in rows)):
            sub = [r for r in rows if r["param"] == param and int(r["i"]) == i]
            for rv in sorted(set(float(r["R_hinv_cMpc"]) for r in sub)):
                s = sorted([r for r in sub if float(r["R_hinv_cMpc"]) == rv],
                          key=lambda r: float(r[xcol]))
                x = col(s, xcol)
                y = col(s, "ratio")
                y_fid = y[len(y) // 2]
                ax.plot(x, y / y_fid - 1.0, "-o", ms=3,
                       label=rf"bin {i}, $R={rv:.0f}$" if i == sorted(set(int(r["i"]) for r in rows))[0] else None)
        ax.axhline(0.0, color="grey", lw=0.8, ls=":")
        ax.set_xlabel(xlabel)
    axs[0].set_ylabel(r"ratio$/$ratio$_{\rm fid}-1$")
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(BUILD / "fig_cosmo_sensitivity.pdf")
    plt.close(fig)


def fig_mock_ratio():
    """Reused as-is from validate_sigma_prj_mock.py --plot (regenerate
    that script's figure to refresh this one; documented in the Makefile
    dependency, not regenerated from this report's own scripts)."""
    src = REPO / "docs" / "_static" / "validation" / "sigma_prj_ratio_grid.png"
    dst = BUILD / "fig_mock_ratio.png"
    shutil.copyfile(src, dst)


if __name__ == "__main__":
    BUILD.mkdir(parents=True, exist_ok=True)
    fig_fig6()
    fig_alpha_sensitivity()
    fig_cosmo_sensitivity()
    fig_mock_ratio()
    print(f"wrote figures to {BUILD}")

