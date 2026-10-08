"""Generate the figures embedded in boost_factor.md and later selection/
survey Theory pages.

    uv run python docs/make_selection_figures.py
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import sanzo_wada as sw
import seaborn as sns

from clenspy.cosmology import fiducial_cosmology
from clenspy.kernels import LensingKernel, sigma_critical
from clenspy.lensing import SigmaPrj
from clenspy.selection import (
    EmgParams,
    HodMor,
    LogNormalMor,
    SelBiasEngine,
    SelectionFunction,
    boost_factor_nfw,
)
from clenspy.survey import Survey, available_configs, deg2, omega_des_y1, omega_sdss

#: Valid z-ranges for the two footprint fits plotted below
#: (clenspy.survey.survey.DES_Y1_Z_RANGE / SDSS_Z_RANGE) -- not exported,
#: so repeated here rather than imported from a private module path.
DES_Y1_Z_RANGE = (0.20, 0.65)
SDSS_Z_RANGE = (0.10, 0.33)

OUT = pathlib.Path(__file__).resolve().parent / "_static" / "img"

# Sanzo Wada combinations: vol1-114 for 2-curve comparisons, vol2-100 (a
# 4-color combination) for 4 distinct categories (e.g. 4 richness bins),
# reordered orange/teal/tan for 3-curve comparisons, matching
# docs/make_cosmology_figures.py's palette.
C2 = [c.hex for c in sw.get_combination("vol1-114").colors]
C4 = [c.hex for c in sw.get_combination("vol2-100").colors]
C3 = [C4[3], C4[2], C4[1]]

sns.set_theme(style="white", context="talk", font_scale=0.8)


def fig_boost_factor():
    """B(R) for a few amplitudes B0, at a fixed NFW scale radius."""
    rs = 0.35  # Mpc, an NFW scale radius for M ~ 1e14
    R = np.logspace(-1.5, 1.2, 200)  # Mpc

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for B0, c in zip((0.05, 0.10, 0.20), C3):
        ax.semilogx(R, boost_factor_nfw(R, B0, rs), color=c, lw=2.5,
                    label=rf"$B_0={B0:.2f}$")
    ax.axhline(1.0, color="grey", lw=1.0, ls=":")
    ax.set(xlabel=r"$R \; [{\rm Mpc}]$", ylabel=r"$B(R)$")
    ax.set_title(rf"NFW boost factor, $r_s={rs:g}$ Mpc", fontsize=15)
    ax.legend(fontsize=12, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "boost_factor.png", dpi=150)
    plt.close(fig)


def fig_selection_function():
    """S_i(M, z=0.3): probability of landing in each richness bin, vs M."""
    lam_edges = np.array([20.0, 30.0, 45.0, 60.0, 200.0])  # DES Y1
    z_edges = np.array([0.20, 0.35, 0.50, 0.65])
    params = EmgParams(delta_mu=-1.5, sigma=3.0, f_prj=0.3, tau=0.12)
    sel = SelectionFunction(lam_edges, z_edges, LogNormalMor(), params,
                            sigma_z=0.01)

    M = np.logspace(13.0, 15.3, 150)  # h^-1 Msun
    S = sel.S_i(np.log(M), 0.3)  # (nM, n_lambda_bins)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for i, c in enumerate(C4):
        lo, hi = lam_edges[i], lam_edges[i + 1]
        ax.semilogx(M, S[:, i], color=c, lw=2.5,
                    label=rf"$\lambda\in[{lo:g},{hi:g})$")
    ax.semilogx(M, S.sum(axis=1), color="grey", lw=1.5, ls="--",
                label="sum (any bin)")
    ax.set(xlabel=r"$M \; [h^{-1}M_\odot]$", ylabel=r"$S_i(M, z=0.3)$")
    ax.set_title("Selection function by richness bin", fontsize=15)
    ax.legend(fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "selection_function.png", dpi=150)
    plt.close(fig)


def fig_selection_bias():
    """b_sel(theta): the sigmoid between b_small and b_large.

    Shares its halo model with SigmaPrj (Tinker(2008) mass function,
    Tinker(2010) bias, CAMB halofit xi_NL), as in SelBiasEngine's own
    __main__ demo -- PkGrid disk-caches the CAMB call.
    """
    cosmo = fiducial_cosmology()
    engine = SelBiasEngine(
        sigma_prj=SigmaPrj(cosmology=cosmo).build(), mor=HodMor.des_y1(),
        n_z=32, n_M=16, n_theta=8, n_ltr=40, ltr_grid_size=10,
    )
    lob, zob = 40.0, 0.4
    profile = engine.marginalised_bias(lob, zob)

    fracs = np.linspace(0.0, 6.0, 200)
    b_sel = np.array([profile(f * profile.theta_lambda) for f in fracs])

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.plot(fracs, b_sel, color=C3[0], lw=2.5, label=r"$b_{\rm sel}(\theta)$")
    ax.axhline(profile.b_small, color=C3[1], lw=1.5, ls="--",
               label=rf"$b_{{\rm small}}={profile.b_small:.1f}$")
    ax.axhline(profile.b_large, color=C3[2], lw=1.5, ls=":",
               label=rf"$b_{{\rm large}}={profile.b_large:.1f}$")
    ax.axvline(1.0, color="grey", lw=1.0, ls=":")
    ax.set(xlabel=r"$\theta / \theta_\lambda$", ylabel=r"$b_{\rm sel}(\theta)$")
    ax.set_title(rf"$\lambda^{{\rm ob}}={lob:g}$, $z^{{\rm ob}}={zob:g}$ "
                 "(toy hmf/bias/xi_NL)", fontsize=14)
    ax.legend(fontsize=12, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "selection_bias.png", dpi=150)
    plt.close(fig)


def fig_survey():
    """p(z_s) for every shipped config, and the DES Y1 / SDSS footprints."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))

    z_s = np.linspace(0.0, 2.0, 300)
    for name, c in zip(available_configs(), C3):
        su = Survey.from_config(name)
        axs[0].plot(z_s, su.pz_src(z_s), color=c, lw=2.5, label=su.name)
    axs[0].set(xlabel=r"$z_s$", ylabel=r"$p(z_s)$")
    axs[0].set_title("Source redshift distributions", fontsize=15)
    axs[0].legend(fontsize=12, frameon=False)

    z_y1 = np.linspace(*DES_Y1_Z_RANGE, 200)
    z_sdss = np.linspace(*SDSS_Z_RANGE, 200)
    axs[1].plot(z_y1, deg2(omega_des_y1(z_y1)), color=C2[0], lw=2.5,
               label="DES Y1")
    axs[1].plot(z_sdss, deg2(omega_sdss(z_sdss)), color=C2[1], lw=2.5,
               label="SDSS")
    axs[1].set(xlabel=r"$z$", ylabel=r"$\Omega(z) \; [{\rm deg}^2]$")
    axs[1].set_title("Effective footprint", fontsize=15)
    axs[1].legend(fontsize=12, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "survey.png", dpi=150)
    plt.close(fig)


def fig_lensing_kernel():
    """Sigma_crit(z_l fixed, z_s) and <Sigma_crit^-1>(z_l) for DES Y1."""
    cosmo = fiducial_cosmology()
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))

    z_l = 0.35
    z_s = np.linspace(z_l + 0.02, 2.5, 200)
    sc = np.array([sigma_critical(z_l, zsi, cosmo) for zsi in z_s])
    axs[0].semilogy(z_s, sc, color=C2[0], lw=2.5)
    axs[0].axvline(z_l, color="grey", lw=1.0, ls=":")
    axs[0].annotate(rf"$z_l={z_l:g}$", (z_l, axs[0].get_ylim()[0]),
                    fontsize=11, ha="left", va="bottom")
    axs[0].set(xlabel=r"$z_s$",
              ylabel=r"$\Sigma_{\rm crit}(z_l,z_s) \; [M_\odot\,{\rm Mpc}^{-2}]$")
    axs[0].set_title("Critical surface density", fontsize=15)

    lk = LensingKernel(survey=Survey.from_config("des_y1"), cosmology=cosmo)
    z_l_grid = np.linspace(0.1, 0.65, 100)
    axs[1].plot(z_l_grid, lk.mean_inverse_sigma_crit(z_l_grid), color=C2[0],
               lw=2.5, label=r"$\langle\Sigma_{\rm crit}^{-1}\rangle$")
    axs[1].plot(z_l_grid, 1.0 / lk.mean_sigma_crit(z_l_grid), color=C2[1],
               lw=2.5, ls="--", label=r"$1/\langle\Sigma_{\rm crit}\rangle$")
    axs[1].set(xlabel=r"$z_l$", ylabel=r"${\rm Mpc}^2\,M_\odot^{-1}$")
    axs[1].set_title("DES Y1 source-averaged weights", fontsize=15)
    axs[1].legend(fontsize=12, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "lensing_kernel.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_boost_factor()
    fig_selection_function()
    fig_selection_bias()
    fig_survey()
    fig_lensing_kernel()
    print(f"wrote figures to {OUT}")
