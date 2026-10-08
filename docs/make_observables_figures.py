"""Generate the figures embedded in stacked_shear.md, covariance.md, and
covariance_halo_to_halo.md.

    uv run python docs/make_observables_figures.py
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import sanzo_wada as sw
import seaborn as sns
from matplotlib.colors import LogNorm

from clenspy.cosmology import BiasModel, PkGrid, fiducial_cosmology, mean_matter_density
from clenspy.covariance import (
    ALL_TERMS,
    DeltaSigmaGaussianCovariance,
    DeltaSigmaHaloToHaloCovariance,
)
from clenspy.halo import NfwProfile, TwoHaloTerm
from clenspy.observables import ClusterCounts, StackedDeltaSigma
from clenspy.selection import EmgParams, LogNormalMor, SelectionFunction
from clenspy.survey import omega_des_y1

OUT = pathlib.Path(__file__).resolve().parent / "_static" / "img"

# Sanzo Wada combinations: vol2-100 (a 4-color combination) for 4 distinct
# categories (e.g. 4 richness bins); vol2-233 (6 colors) sliced to 5 for
# ALL_TERMS' 5 covariance terms; vol1-114's gold/teal pair, as a
# teal-white-gold diverging map, for covariance/correlation heatmaps.
C4 = [c.hex for c in sw.get_combination("vol2-100").colors]
C5 = [c.hex for c in sw.get_combination("vol2-233").colors][:5]
_GOLD, _TEAL = [c.hex for c in sw.get_combination("vol1-114").colors]
CMAP_COV = sw.diverging_cmap(_TEAL, "white", _GOLD)

sns.set_theme(style="white", context="talk", font_scale=0.8)

LAM_EDGES = np.array([20.0, 30.0, 45.0, 60.0, 200.0])
Z_EDGES = np.array([0.20, 0.35, 0.50, 0.65])


def _toy_abundance():
    """ClusterCounts on an analytic dn/dlnM stand-in -- no CAMB needed."""
    cosmo = fiducial_cosmology()

    def mass_function(ln_mass, z):
        lnm, zz = np.broadcast_arrays(np.asarray(ln_mass, float), np.asarray(z, float))
        m = np.exp(lnm)
        return 1e-5 * (m / 1e14) ** -1.0 * np.exp(-m / 5e14) / (1.0 + zz)

    sel = SelectionFunction(LAM_EDGES, Z_EDGES, LogNormalMor(),
                            EmgParams(-1.5, 3.0, 0.3, 0.12), sigma_z=0.01)
    ln_mass = np.log(np.logspace(13.5, 15.3, 24))
    z = np.linspace(0.16, 0.70, 32)
    return ClusterCounts(ln_mass, z, mass_function, sel, cosmo, omega_des_y1), cosmo


def fig_observables():
    """DeltaSigma_ij(R): the second contraction of the counts weight."""
    abundance, cosmo = _toy_abundance()
    radii = np.logspace(-1.2, 1.2, 40)  # Mpc

    def nfw_deltasigma(r, mass, z_cluster):
        rho_m = cosmo.critical_density0.to_value("Msun/Mpc^3") * cosmo.Om0
        return NfwProfile(m200=mass, c200=4.0, rho_ref=rho_m).deltasigma(r)

    stack = StackedDeltaSigma.from_profile(abundance, nfw_deltasigma, radii)
    ds = stack.profile()  # (n_lambda, n_z, n_r)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for i, c in enumerate(C4):
        lo, hi = LAM_EDGES[i], LAM_EDGES[i + 1]
        ax.loglog(radii, ds[i, 0], color=c, lw=2.5,
                  label=rf"$\lambda\in[{lo:g},{hi:g})$")
    ax.set(xlabel=r"$R \; [{\rm Mpc}]$",
           ylabel=r"$\Delta\Sigma_{ij}(R) \; [M_\odot\,{\rm Mpc}^{-2}]$")
    ax.set_title(rf"Stacked profile, $z\in[{Z_EDGES[0]:g},{Z_EDGES[1]:g})$",
                 fontsize=15)
    ax.legend(fontsize=12, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "observables.png", dpi=150)
    plt.close(fig)


def fig_covariance_terms():
    """Fractional contribution of the 5 Gaussian-covariance terms vs rp.

    Power-law spectra stand-ins (as in the class's own __main__), so the
    demo needs no Limber run -- the point is the *decomposition*, not the
    absolute amplitude.
    """
    rp_edges = np.logspace(np.log10(0.2), np.log10(30.0), 12)  # Mpc
    chi_h = 1100.0  # Mpc, roughly z = 0.4
    f_sky = 1500.0 * (np.pi / 180.0) ** 2 / (4.0 * np.pi)

    def c_hh(ell):
        return 1e-5 * (np.asarray(ell, float) / 100.0) ** -1.0

    def c_ss(ell):
        return 4e26 * (np.asarray(ell, float) / 100.0) ** -1.2

    def c_hs(ell):
        return np.sqrt(c_hh(ell) * c_ss(ell))

    n_h, shape_noise = 3.0e5, 1.0e26
    cov = DeltaSigmaGaussianCovariance(rp_edges, chi_h, f_sky, c_hh, c_ss,
                                       c_hs, n_h, shape_noise)
    parts = cov.components()
    total = np.diag(cov.cov())
    rp_mid = np.sqrt(rp_edges[:-1] * rp_edges[1:])

    term_colors = dict(zip(ALL_TERMS, C5))
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for name in ALL_TERMS:
        # cross = c_hS^2 = c_hh*c_SS exactly for this toy's linear-bias
        # c_hS = sqrt(c_hh c_SS), so it is identical to lss_lss by
        # construction -- dashed so the degenerate curves stay visible.
        ls = "--" if name == "cross" else "-"
        ax.semilogx(rp_mid, np.diag(parts[name]) / total, color=term_colors[name],
                   lw=2.5, ls=ls, label=name.replace("_", " "))
    ax.set(xlabel=r"$r_p \; [{\rm Mpc}]$",
           ylabel="fraction of diagonal covariance")
    ax.set_title(r"$\Delta\Sigma$ Gaussian covariance decomposition",
                fontsize=15)
    ax.legend(fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "covariance_terms.png", dpi=150)
    plt.close(fig)


def fig_covariance_matrix():
    """The full Gaussian-field covariance matrix and its correlation, for
    one halo-redshift slice -- same toy spectra as `fig_covariance_terms`.
    """
    rp_edges = np.logspace(np.log10(0.2), np.log10(30.0), 12)  # Mpc
    chi_h = 1100.0  # Mpc, roughly z = 0.4
    f_sky = 1500.0 * (np.pi / 180.0) ** 2 / (4.0 * np.pi)

    def c_hh(ell):
        return 1e-5 * (np.asarray(ell, float) / 100.0) ** -1.0

    def c_ss(ell):
        return 4e26 * (np.asarray(ell, float) / 100.0) ** -1.2

    def c_hs(ell):
        return np.sqrt(c_hh(ell) * c_ss(ell))

    n_h, shape_noise = 3.0e5, 1.0e26
    cov = DeltaSigmaGaussianCovariance(rp_edges, chi_h, f_sky, c_hh, c_ss,
                                       c_hs, n_h, shape_noise)
    c = cov.cov()
    d = np.sqrt(np.diag(c))
    corr = c / np.outer(d, d)

    log_rp = np.log10(np.sqrt(rp_edges[:-1] * rp_edges[1:]))
    extent = [log_rp[0], log_rp[-1], log_rp[0], log_rp[-1]]

    fig, axs = plt.subplots(1, 2, figsize=(12.0, 5.5))
    im0 = axs[0].imshow(c, origin="lower", extent=extent, norm=LogNorm(),
                        cmap=CMAP_COV)
    axs[0].set_title(r"${\rm Cov}^{\rm Gauss}$  $[(M_\odot\,{\rm Mpc}^{-2})^2]$",
                     fontsize=14)
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(corr, origin="lower", extent=extent, vmin=0.0,
                        vmax=1.0, cmap=CMAP_COV)
    axs[1].set_title("correlation", fontsize=14)
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set(xlabel=r"$\log_{10} r_p \; [{\rm Mpc}]$",
              ylabel=r"$\log_{10} r_p \; [{\rm Mpc}]$")

    fig.suptitle(r"Gaussian-field $\Delta\Sigma$ covariance, one halo-redshift"
                 " slice", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "covariance_matrix.png", dpi=150)
    plt.close(fig)


def fig_halo_to_halo():
    """Fractional sigma_intr(R) per richness bin -- rises with richness,
    since a broader mass population means more per-cluster scatter."""
    abundance, cosmo = _toy_abundance()
    pk_grid = PkGrid(cosmo=cosmo, nonlinear=False)
    k_camb = pk_grid.k
    z_eff = 0.28
    Pk_eff = pk_grid(k_camb, z=z_eff)
    rho_m0 = mean_matter_density(cosmo)
    twohalo = TwoHaloTerm(k_camb, Pk_eff, zvec=z_eff)
    bias_model = BiasModel(k_camb, Pk_eff, cosmo=cosmo)
    intrinsic = DeltaSigmaHaloToHaloCovariance(abundance, twohalo, bias_model,
                                               rho_m0, z_eff=z_eff)

    R = np.logspace(-1.2, 1.2, 40)  # Mpc
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for i, c in enumerate(C4):
        lo, hi = LAM_EDGES[i], LAM_EDGES[i + 1]
        s = np.sqrt(np.diag(intrinsic.cov(R, i, 0)))
        m = intrinsic.mean_profile(R, i, 0)
        ax.semilogx(R, s / m, color=c, lw=2.5,
                   label=rf"$\lambda\in[{lo:g},{hi:g})$")
    ax.set(xlabel=r"$R \; [{\rm Mpc}]$",
           ylabel=r"$\sigma_{\rm intr}(R) / \langle\Delta\Sigma(R)\rangle$")
    ax.set_title("Halo-to-halo (intrinsic) fractional scatter", fontsize=15)
    ax.legend(fontsize=12, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "halo_to_halo.png", dpi=150)
    plt.close(fig)


def fig_halo_to_halo_matrix(i=0, j=0):
    """The intrinsic covariance for one richness/redshift bin: actual
    values (log scale) and the correlation matrix, side by side --
    McClintock et al. (2019) Fig. 6's own presentation of their SAC matrix.
    """
    abundance, cosmo = _toy_abundance()
    pk_grid = PkGrid(cosmo=cosmo, nonlinear=False)
    k_camb = pk_grid.k
    z_eff = 0.28
    Pk_eff = pk_grid(k_camb, z=z_eff)
    rho_m0 = mean_matter_density(cosmo)
    twohalo = TwoHaloTerm(k_camb, Pk_eff, zvec=z_eff)
    bias_model = BiasModel(k_camb, Pk_eff, cosmo=cosmo)
    intrinsic = DeltaSigmaHaloToHaloCovariance(abundance, twohalo, bias_model,
                                               rho_m0, z_eff=z_eff)

    R = np.logspace(-1.2, 1.2, 20)  # Mpc
    c = intrinsic.cov(R, i, j)
    d = np.sqrt(np.diag(c))
    corr = c / np.outer(d, d)

    log_r = np.log10(R)
    extent = [log_r[0], log_r[-1], log_r[0], log_r[-1]]

    fig, axs = plt.subplots(1, 2, figsize=(12.0, 5.5))
    im0 = axs[0].imshow(c, origin="lower", extent=extent, norm=LogNorm(),
                        cmap=CMAP_COV)
    axs[0].set_title(r"$C^{\rm intr}$  $[(M_\odot\,{\rm Mpc}^{-2})^2]$",
                     fontsize=14)
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(corr, origin="lower", extent=extent, vmin=0.0,
                        vmax=1.0, cmap=CMAP_COV)
    axs[1].set_title("correlation", fontsize=14)
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set(xlabel=r"$\log_{10} R \; [{\rm Mpc}]$",
              ylabel=r"$\log_{10} R \; [{\rm Mpc}]$")

    lo, hi = LAM_EDGES[i], LAM_EDGES[i + 1]
    zlo, zhi = Z_EDGES[j], Z_EDGES[j + 1]
    fig.suptitle(rf"Halo-to-halo covariance, $\lambda\in[{lo:g},{hi:g})$, "
                 rf"$z\in[{zlo:g},{zhi:g})$", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "halo_to_halo_matrix.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_observables()
    fig_covariance_terms()
    fig_covariance_matrix()
    fig_halo_to_halo()
    fig_halo_to_halo_matrix()
    print(f"wrote figures to {OUT}")
