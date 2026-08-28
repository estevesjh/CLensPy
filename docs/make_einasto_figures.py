"""Generate the Einasto validation figures embedded in einasto_math.md.

Each figure compares the analytic pipeline against the retained numerical
cross-check method, with an inset showing the fractional difference at
+-0.05% limits (points outside the band clip -- that is the message: the
numerical methods, not the analytic pipeline, are the limiting factor).

    uv run python docs/make_einasto_figures.py
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import sanzo_wada as sw
import seaborn as sns

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from clenspy.halo.einasto import EinastoProfile  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "_static" / "img"

# Sanzo Wada combination vol1-114 for the analytic-vs-numerical pair;
# C_REF is a plain neutral reference line (the inset's zero line), not a
# data category, so it stays grey.
C_ANA, C_NUM = [c.hex for c in sw.get_combination("vol1-114").colors]
C_REF = "grey"

sns.set_theme(context="talk", style="white")


def _residual_inset(ax, x, frac, xlabel, loc=(0.13, 0.12)):
    """Inset axes inside `ax`: fractional difference in percent, +-0.05%."""
    ins = ax.inset_axes([loc[0], loc[1], 0.5, 0.34])
    ins.axhline(0.0, color=C_REF, lw=1.2, zorder=1)
    ins.semilogx(x, 100.0 * frac, color=C_NUM, lw=1.6, zorder=2)
    ins.set_ylim(-0.05, 0.05)
    ins.set_ylabel("diff. [%]", fontsize=11)
    ins.set_xlabel(xlabel, fontsize=11)
    ins.tick_params(labelsize=9)
    return ins


def fig_deltasigma(n=4.0):
    e = EinastoProfile(alpha=1.0 / n, rho_0=1.0, r_s=1.0)
    R = np.logspace(-2.2, 2.2, 250) * e.r_s    # physical range, r_s units
    ds_ana = e.deltasigma(R)
    ds_num = e._deltasigma_numerical(R)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.loglog(R / e.r_s, ds_ana, color=C_ANA, lw=2.5,
              label="analytic series backend")
    ax.loglog(R / e.r_s, ds_num, color=C_NUM, lw=2.0, ls="--",
              label="Abel + cumtrapz (numerical)")
    ax.set_xlabel(r"$R/r_s$")
    ax.set_ylabel(r"$\Delta\Sigma \; / \; \rho_0 r_s$")
    ax.set_title(rf"Einasto $\Delta\Sigma$,  $n = {n:g}$", fontsize=16)
    ax.legend(loc="upper right", fontsize=12, frameon=False)
    _residual_inset(ax, R / e.r_s, ds_num / ds_ana - 1.0, r"$R/r_s$",
                    loc=(0.46, 0.10))
    fig.tight_layout()
    fig.savefig(OUT / "einasto_deltasigma_validation.png", dpi=150)
    plt.close(fig)


def fig_pk(n=4.0):
    e = EinastoProfile(alpha=1.0 / n, rho_0=1.0, r_s=1.0)
    krs = np.logspace(-2.2, 2.2, 250)          # k in 1/r_s units
    k = krs / e.r_s
    pk_ana = e.power_spectrum(k)
    pk_num = e._power_spectrum_numerical(k)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.loglog(krs, pk_ana, color=C_ANA, lw=2.5, label="analytic dispatch")
    ax.loglog(krs, pk_num, color=C_NUM, lw=2.0, ls="--",
              label="FFTLog (numerical)")
    ax.set_xlabel(r"$k\, r_s$")
    ax.set_ylabel(r"$P(k) \; / \; \rho_0 r_s^3$")
    ax.set_title(rf"Einasto $P(k)$,  $n = {n:g}$", fontsize=16)
    ax.legend(loc="lower left", fontsize=12, frameon=False,
              bbox_to_anchor=(0.02, 0.5))
    _residual_inset(ax, krs, pk_num / pk_ana - 1.0, r"$k\, r_s$")
    fig.tight_layout()
    fig.savefig(OUT / "einasto_pk_validation.png", dpi=150)
    plt.close(fig)


def fig_three_panel(n=4.0):
    """rho(r), Sigma(R), DeltaSigma(R) side by side, analytic vs numerical."""
    e = EinastoProfile(alpha=1.0 / n, rho_0=1.0, r_s=1.0)
    r = np.logspace(-2.2, 2.2, 250) * e.r_s

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].loglog(r, e.density(r), color=C_ANA, lw=2.5)
    axs[0].set(xlabel=r"$r/r_s$", ylabel=r"$\rho / \rho_0$",
               title=r"$\rho(r)$")
    for ax, ana, num, name in (
            (axs[1], e.sigma(r), e._sigma_numerical(r), r"$\Sigma(R)$"),
            (axs[2], e.deltasigma(r), e._deltasigma_numerical(r),
             r"$\Delta\Sigma(R)$")):
        ax.loglog(r, ana, color=C_ANA, lw=2.5, label="analytic")
        ax.loglog(r, num, color=C_NUM, lw=1.8, ls="--", label="numerical")
        ax.set(xlabel=r"$R/r_s$", title=name)
        ax.legend(fontsize=12, frameon=False)
    axs[1].set_ylabel(r"$\Sigma / \rho_0 r_s$")
    axs[2].set_ylabel(r"$\Delta\Sigma / \rho_0 r_s$")
    fig.suptitle(rf"Einasto profile, $n = {n:g}$", fontsize=17)
    fig.tight_layout()
    fig.savefig(OUT / "einasto_profiles_overview.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_three_panel()
    fig_deltasigma()
    fig_pk()
    print(f"wrote figures to {OUT}")
