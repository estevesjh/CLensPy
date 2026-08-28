"""Generate the figures embedded in density_profiles.md,
projected_profiles.md, two_halo_term.md, and lensing_profile.md.

    uv run python docs/make_halo_figures.py
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import sanzo_wada as sw
import seaborn as sns

from clenspy.cosmology import PkGrid, fiducial_cosmology, mean_matter_density
from clenspy.halo import EinastoProfile, NfwProfile, TwoHaloTerm
from clenspy.lensing import LensingProfile, MiscenteringProfile

OUT = pathlib.Path(__file__).resolve().parent / "_static" / "img"

# Sanzo Wada combinations: vol1-114 for 2-curve comparisons, vol2-100 (a
# 4-color combination) for 4-curve comparisons, reordered orange/teal/tan
# for 3-curve comparisons.
C2 = [c.hex for c in sw.get_combination("vol1-114").colors]
C4 = [c.hex for c in sw.get_combination("vol2-100").colors]
C3 = [C4[3], C4[2], C4[1]]
C_NFW, C_EIN = C2

sns.set_theme(style="white", context="talk", font_scale=0.8)

M200, C200, ALPHA = 1e14, 5.0, 0.25


def _matched_halos():
    """NFW and a mass-matched Einasto: same r_s = r200/c200, same enclosed
    mass at r200 -- a shape-only comparison, not a rho_0/r_s coincidence."""
    nfw = NfwProfile(m200=M200, c200=C200)
    rho0_unit = EinastoProfile(alpha=ALPHA, rho_0=1.0, r_s=nfw.rs, tol=1e-4)
    rho0 = M200 / rho0_unit.enclosed_mass(nfw.r200)
    einasto = EinastoProfile(alpha=ALPHA, rho_0=rho0, r_s=nfw.rs, tol=1e-4)
    return nfw, einasto


def fig_density_profiles():
    """rho(r) and the mass-normalized u(k), NFW vs a mass-matched Einasto,
    plus the untruncated NFW to show *why* NFW needs the truncation that
    Einasto doesn't."""
    m200, alpha = M200, ALPHA
    nfw, einasto = _matched_halos()

    r = np.logspace(-2.0, np.log10(3.0 * nfw.r200), 200)  # Mpc
    k = np.logspace(np.log10(0.007), 1.5, 200)  # 1/Mpc -- low enough to see
    # the untruncated NFW curve visibly still rising (it never flattens --
    # see below); the two truncated/finite-mass curves are already flat here.

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))

    axs[0].loglog(r, nfw.density(r), color=C_NFW, lw=2.5, label="NFW")
    axs[0].loglog(r, einasto.density(r), color=C_EIN, lw=2.5,
                  label=rf"Einasto ($\alpha={alpha:g}$, untruncated)")
    axs[0].axvline(nfw.r200, color=C_NFW, lw=1.0, ls=":")
    axs[0].annotate(r"$r_{200}$", (nfw.r200, axs[0].get_ylim()[0]),
                     fontsize=11, ha="left", va="bottom")
    axs[0].set(xlabel=r"$r \; [{\rm Mpc}]$",
               ylabel=r"$\rho(r) \; [M_\odot\,{\rm Mpc}^{-3}]$")
    axs[0].legend(fontsize=12, frameon=False)

    # All three divided by the *shared* m200, not each profile's own total
    # mass: Einasto's total_mass runs to r -> infinity but stays finite
    # (mass beyond r200 that the NFW truncation discards), so normalizing
    # each by its own total mass would compare different physical masses.
    # The untruncated NFW has no finite total mass at all -- see below --
    # so it is plotted only to show that divergence, not as a third
    # "shape-only" comparison curve.
    axs[1].loglog(k, nfw.fourier(k) / m200, color=C_NFW, lw=2.5,
                  label="NFW (truncated at $r_{200}$)")
    axs[1].loglog(k, nfw.fourier(k, truncated=False) / m200,
                  color=C_NFW, lw=1.5, ls=":",
                  label="NFW (untruncated -- diverges)")
    axs[1].loglog(k, einasto.fourier(k) / m200, color=C_EIN,
                  lw=2.5, label=rf"Einasto ($\alpha={alpha:g}$, untruncated)")
    axs[1].axhline(1.0, color="grey", lw=1.0, ls=":")
    k200 = 1.0 / nfw.r200  # the k-space equivalent of r200: k*r200 = 1
    axs[1].axvline(k200, color=C_NFW, lw=1.0, ls=":")
    axs[1].annotate(r"$k_{200}$", (k200, axs[1].get_ylim()[1]),
                     fontsize=11, ha="left", va="top")
    axs[1].set(xlabel=r"$k \; [{\rm Mpc}^{-1}]$",
               ylabel=r"$\tilde\rho(k) / M_{200}$")
    axs[1].legend(fontsize=11, frameon=False)

    fig.suptitle(rf"$M_{{200}}={m200:.0e}\,M_\odot$, $c_{{200}}={C200:g}$",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "density_profiles.png", dpi=150)
    plt.close(fig)


def fig_projected_profiles():
    """Sigma(R) and DeltaSigma(R), NFW vs the same mass-matched Einasto."""
    nfw, einasto = _matched_halos()

    R = np.logspace(-2.0, np.log10(3.0 * nfw.r200), 200)  # Mpc

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))

    axs[0].loglog(R, nfw.sigma(R), color=C_NFW, lw=2.5, label="NFW")
    axs[0].loglog(R, einasto.sigma(R), color=C_EIN, lw=2.5, ls="--",
                  label=rf"Einasto ($\alpha={ALPHA:g}$)")
    axs[0].axvline(nfw.r200, color=C_NFW, lw=1.0, ls=":")
    axs[0].set(xlabel=r"$R \; [{\rm Mpc}]$",
               ylabel=r"$\Sigma(R) \; [M_\odot\,{\rm Mpc}^{-2}]$")
    axs[0].legend(fontsize=12, frameon=False)

    axs[1].loglog(R, nfw.deltasigma(R), color=C_NFW, lw=2.5, label="NFW")
    axs[1].loglog(R, einasto.deltasigma(R), color=C_EIN, lw=2.5, ls="--",
                  label=rf"Einasto ($\alpha={ALPHA:g}$)")
    axs[1].axvline(nfw.r200, color=C_NFW, lw=1.0, ls=":")
    axs[1].set(xlabel=r"$R \; [{\rm Mpc}]$",
               ylabel=r"$\Delta\Sigma(R) \; [M_\odot\,{\rm Mpc}^{-2}]$")
    axs[1].legend(fontsize=12, frameon=False)

    fig.suptitle(rf"$M_{{200}}={M200:.0e}\,M_\odot$, $c_{{200}}={C200:g}$",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "projected_profiles.png", dpi=150)
    plt.close(fig)


def fig_two_halo():
    """xi(r,z) and DeltaSigma_2h(R,z) from a real CAMB linear P(k,z)."""
    cosmo = fiducial_cosmology()
    pk_grid = PkGrid(cosmo=cosmo, nonlinear=False)
    k = pk_grid.k
    rho_m = mean_matter_density(cosmo)
    zvec = (0.0, 0.5, 1.0)
    R = np.logspace(-1.0, 2.0, 60)  # Mpc

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    for z, c in zip(zvec, C3):
        two_halo = TwoHaloTerm(k, pk_grid(k, z=z), zvec=z)
        axs[0].loglog(R, two_halo.xi(R, z), color=c, lw=2.5, label=f"$z={z:g}$")
        axs[1].loglog(R, two_halo.deltasigma(R, z) * rho_m, color=c, lw=2.5,
                      label=f"$z={z:g}$")

    axs[0].set(xlabel=r"$r \; [{\rm Mpc}]$", ylabel=r"$\xi(r,z)$")
    axs[0].legend(fontsize=12, frameon=False)
    axs[1].set(xlabel=r"$R \; [{\rm Mpc}]$",
               ylabel=r"$\Delta\Sigma_{\rm 2h}(R,z) \; [M_\odot\,{\rm Mpc}^{-2}]$")
    axs[1].legend(fontsize=12, frameon=False)

    fig.suptitle("The two-halo term, from a linear CAMB P(k,z)", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "two_halo_term.png", dpi=150)
    plt.close(fig)


def fig_lensing_profile():
    """DeltaSigma(R): 1-halo term, 2-halo term, and their sum."""
    z_cluster, m200, c200 = 0.3, 1e14, 4.0
    lp = LensingProfile(z_cluster=z_cluster, m200=m200, concentration=c200)
    lp_1h = LensingProfile(z_cluster=z_cluster, m200=m200, concentration=c200,
                            include_2halo=False)

    R = np.logspace(-1.5, 1.7, 60)  # Mpc
    ds_tot = lp.deltasigma(R)
    ds_1h = lp_1h.deltasigma(R)
    ds_2h = ds_tot - ds_1h

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.loglog(R, ds_1h, color=C3[0], lw=2.0, ls="--", label="1-halo")
    ax.loglog(R, ds_2h, color=C3[2], lw=2.0, ls=":", label="2-halo")
    ax.loglog(R, ds_tot, color=C3[1], lw=2.5, label="1-halo + 2-halo")
    ax.set(xlabel=r"$R \; [{\rm Mpc}]$",
           ylabel=r"$\Delta\Sigma(R) \; [M_\odot\,{\rm Mpc}^{-2}]$")
    ax.set_title(rf"$M_{{200}}={m200:.0e}\,M_\odot$, $c_{{200}}={c200:g}$, "
                 rf"$z={z_cluster:g}$", fontsize=15)
    ax.legend(fontsize=12, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "lensing_profile.png", dpi=150)
    plt.close(fig)


#: DES Y1/Y3 miscentering scale (Rykoff et al. 2014 R_lambda; tau_mis is
#: the redMaPPer-calibrated offset scale, e.g. docs/miscentering_math.md
#: Sec. 9.3's own worked example, lambda ~ 25).
TAU_MIS = 0.17
RICHNESS = 25.0


def _gamma_averaged_deltasigma_mis(z_cluster, m200, R, theta_mpc, n=80,
                                    rmax_factor=8.0):
    """Population-averaged DeltaSigma_mis over the Gamma offset law DES
    Y1/Y3 fit to redMaPPer (Hoshino et al. 2015; McClintock et al. 2019):

        p(r_mis | theta) = (r_mis/theta^2) exp(-r_mis/theta)

    a Gamma(shape=2, scale=theta) distribution, theta = tau_mis * R_lambda.
    `MiscenteringProfile` only evaluates one fixed r_mis at a time (see the
    class Notes); the average over the distribution is genuinely the
    caller's job, done here with a plain trapezoid quadrature over a grid
    of single-offset profiles.
    """
    r_mis_grid = np.linspace(1e-3, rmax_factor * theta_mpc, n)
    pdf = (r_mis_grid / theta_mpc**2) * np.exp(-r_mis_grid / theta_mpc)
    pdf /= np.trapezoid(pdf, r_mis_grid)
    grid = np.array([
        MiscenteringProfile(z_cluster=z_cluster, m200=m200, r_mis=rm,
                             include_2halo=False).deltasigma_mis(R)
        for rm in r_mis_grid
    ])
    return np.trapezoid(pdf[:, None] * grid, r_mis_grid, axis=0)


def fig_miscentering():
    """DeltaSigma(R): centered, 1-halo alone, single-offset, and
    Gamma-averaged (DES Y1/Y3) miscentered -- cluster-toolkit's own
    DeltaSigma figure style, log-log throughout.
    """
    z_cluster, m200 = 0.25, 2e14
    R = np.logspace(-2.0, np.log10(50.0), 100)  # Mpc

    # R_lambda = (lambda/100)^0.2 h^-1 Mpc (Rykoff et al. 2014); clenspy's
    # lensing classes are h-free, so divide by h once, visibly, here.
    h = fiducial_cosmology().h
    R_lambda_hinv = (RICHNESS / 100.0) ** 0.2
    theta_mpc = TAU_MIS * R_lambda_hinv / h

    lp = LensingProfile(z_cluster=z_cluster, m200=m200, include_2halo=True)
    lp_1h = LensingProfile(z_cluster=z_cluster, m200=m200, include_2halo=False)
    ds_tot = lp.deltasigma(R)
    ds_1h = lp_1h.deltasigma(R)
    ds_single = MiscenteringProfile(
        z_cluster=z_cluster, m200=m200, r_mis=theta_mpc, include_2halo=False,
    ).deltasigma_mis(R)
    ds_gamma = _gamma_averaged_deltasigma_mis(z_cluster, m200, R, theta_mpc)

    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    ax.loglog(R, ds_tot, color=C4[0], lw=2.5, label=r"$\Delta\Sigma$")
    ax.loglog(R, ds_1h, color=C4[1], lw=2.0, ls="--",
              label=r"$\Delta\Sigma_{\rm 1h}$ (NFW alone)")
    ax.loglog(R, ds_gamma, color=C4[2], lw=2.0, ls=":",
              label=rf"$\Delta\Sigma_{{\rm mis}}$, Gamma law "
                    rf"($\lambda={RICHNESS:g}$, $\tau_{{\rm mis}}={TAU_MIS:g}$)")
    ax.loglog(R, ds_single, color=C4[3], lw=2.0, ls=":",
              label=rf"$\Delta\Sigma_{{\rm mis}}$, single offset "
                    rf"$R_{{\rm mis}}={theta_mpc:.2f}$ Mpc")
    ax.set(xlabel=r"$R \; [{\rm Mpc}]$",
           ylabel=r"$\Delta\Sigma(R) \; [M_\odot\,{\rm Mpc}^{-2}]$")
    ax.set_title(rf"$M_{{200}}={m200:.0e}\,M_\odot$, $z={z_cluster:g}$",
                 fontsize=15)
    ax.legend(fontsize=11, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "miscentering.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_density_profiles()
    fig_projected_profiles()
    fig_two_halo()
    fig_lensing_profile()
    fig_miscentering()
    print(f"wrote figures to {OUT}")
