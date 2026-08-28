"""Generate the figures embedded in cosmology.md, power_spectrum.md,
halo_bias.md, mass_function.md, and concentration.md.

Needs sanzo-wada (not a project dependency -- Python >=3.11 only):

    uv pip install "git+https://github.com/estevesjh/sanzo-wada-colors"
    uv run python docs/make_cosmology_figures.py
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import sanzo_wada as sw
import seaborn as sns

from clenspy.cosmology import (
    BiasModel,
    LinearPk,
    PkGrid,
    SigmaGrid,
    TinkerMassFunction,
    child18,
    child18_powerlaw,
    duffy08,
    fiducial_cosmology,
    growth_factor,
    m_star_hinv,
)

OUT = pathlib.Path(__file__).resolve().parent / "_static" / "img"

# Sanzo Wada combinations: vol1-114 for 2-curve comparisons, vol2-100 (a
# 4-color combination) reordered orange/teal/tan for 3-curve comparisons.
C2 = [c.hex for c in sw.get_combination("vol1-114").colors]
_C4 = [c.hex for c in sw.get_combination("vol2-100").colors]
C3 = [_C4[3], _C4[2], _C4[1]]

#: Common mass ceiling for every M-axis plot -- the point beyond which the
#: production analyses have essentially no clusters.
MAX_MASS = 2e15  # Msun, or h^-1 Msun where that is the plotted convention

sns.set_theme(style="white", context="talk", font_scale=0.8)


def fig_cosmology():
    """chi(z), D_A(z) and the growth factor D(z), side by side."""
    cosmo = fiducial_cosmology(H0=70.0, Om0=0.3)
    z = np.linspace(0.0, 3.0, 200)
    chi = cosmo.comoving_distance(z).value
    d_a = cosmo.angular_diameter_distance(z).value
    d_z = growth_factor(z, cosmo)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(z, chi, color=C2[0], lw=2.5, label=r"$\chi(z)$ (comoving)")
    axs[0].plot(z, d_a, color=C2[1], lw=2.5, ls="--",
                label=r"$D_A(z)$ (angular diameter)")
    axs[0].set(xlabel="$z$", ylabel="distance [Mpc]")
    axs[0].legend(fontsize=12, frameon=False)

    axs[1].plot(z, d_z, color=C2[0], lw=2.5)
    axs[1].set(xlabel="$z$", ylabel="$D(z)$", ylim=(0, 1.05))
    axs[1].axhline(1.0, color=C2[1], lw=1.0, ls=":")

    fig.suptitle(f"Flat $\\Lambda$CDM: $H_0={cosmo.H0.value:g}$, "
                 f"$\\Omega_m={cosmo.Om0:g}$", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT / "cosmology_distances_growth.png", dpi=150)
    plt.close(fig)


def fig_power_spectrum():
    """CAMB linear P(k) at two redshifts."""
    cosmo = fiducial_cosmology()
    pk_grid = PkGrid(cosmo=cosmo, nonlinear=False)
    k = pk_grid.k

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for zi, c in zip((0.0, 1.0), C2):
        ax.loglog(k, pk_grid(k, z=zi), color=c, lw=2.5, label=f"$z={zi:g}$")
    ax.set(xlabel=r"$k$ [Mpc$^{-1}$]", ylabel=r"$P(k)$ [Mpc$^3$]")
    ax.set_title("Linear matter power spectrum (CAMB)", fontsize=16)
    ax.legend(fontsize=12, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "power_spectrum.png", dpi=150)
    plt.close(fig)

    return pk_grid, cosmo


def fig_sigma_r(pk_grid, cosmo):
    """sigma(R) at z=0 and z=1, in the h-scaled convention SigmaGrid uses."""
    h = cosmo.h
    k_h, pk_h3 = pk_grid.k / h, pk_grid(pk_grid.k, z=0.0) * h**3
    sigma_grid = SigmaGrid(LinearPk(k_h, pk_h3))
    r_hinv = np.logspace(-1.0, 2.0, 80)  # Mpc/h
    sigma_z0 = np.array([sigma_grid.sigma(r) for r in r_hinv])

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for zi, c in zip((0.0, 1.0), C2):
        d_z = growth_factor(zi, cosmo)
        ax.loglog(r_hinv, sigma_z0 * d_z, color=c, lw=2.5, label=f"$z={zi:g}$")
    sigma8 = sigma_grid.sigma(8.0)
    ax.scatter([8.0], [sigma8], color=C2[0], zorder=5)
    ax.annotate(rf"$\sigma_8={sigma8:.2f}$", (8.0, sigma8),
               textcoords="offset points", xytext=(10, 8), fontsize=12)
    ax.set(xlabel=r"$R \; [{\rm Mpc}/h]$", ylabel=r"$\sigma(R)$")
    ax.set_title(r"Top-hat variance $\sigma(R)$", fontsize=16)
    ax.legend(fontsize=12, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "sigma_r.png", dpi=150)
    plt.close(fig)


def fig_halo_bias(cosmo):
    """b(M) at three redshifts, h-free mass axis (Msun, not h^-1 Msun).

    One instance: BiasModel(cosmo=cosmo) builds its PkGrid/SigmaGrid
    lazily; sigma(M,z) = D(z) sigma(M,0) is applied per call, inside
    bias(M, z=zi) -- no separate instance needed per redshift.
    """
    zvec = np.array([0.0, 0.5, 1.0])
    model = BiasModel(cosmo=cosmo)
    M = np.logspace(12.5, np.log10(MAX_MASS), 60)  # Msun

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for zi, c in zip(zvec, C3):
        ax.loglog(M, model.bias(M, z=zi), color=c, lw=2.5,
                  label=f"$z={zi:g}$")
    ax.set(xlabel=r"$M \; [M_\odot]$", ylabel=r"$b(M,z)$")
    ax.set_title("Tinker (2010) linear halo bias", fontsize=16)
    ax.legend(fontsize=12, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "halo_bias.png", dpi=150)
    plt.close(fig)


def fig_mass_function(cosmo):
    """dn/dlnM at three redshifts, mass axis converted to h^-1 Msun.

    One instance: cosmo -> PkGrid -> SigmaGrid -> dndlnm_grid runs once,
    lazily, and the D(z)^2 growth scaling happens inside dndlnm_grid for
    every z in zvec -- not once per redshift by hand.
    """
    zvec = np.array([0.0, 0.5, 1.0])
    hmf = TinkerMassFunction(cosmo=cosmo, zvec=zvec)
    min_mass_hinv = 10.0**12.5  # h^-1 Msun, matches fig_halo_bias's M-axis floor
    r_min = hmf.radius_of_mass(min_mass_hinv / cosmo.Om0)
    r_max = hmf.radius_of_mass(MAX_MASS / cosmo.Om0)  # MAX_MASS h^-1 Msun
    r_hinv = np.logspace(np.log10(r_min), np.log10(r_max), 60)  # Mpc/h
    m_h = hmf.mass_of_radius(r_hinv)
    m_hinv = m_h * cosmo.Om0  # Omega_m h^-1 Msun -> h^-1 Msun

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for zi, c in zip(zvec, C3):
        ax.loglog(m_hinv, hmf.dndlnm(m_h, z=zi), color=c, lw=2.5,
                  label=f"$z={zi:g}$")
    ax.set(xlabel=r"$M \; [h^{-1}M_\odot]$",
           ylabel=r"$dn/d\ln M \; [h^3\,{\rm Mpc}^{-3}]$")
    ax.set_title("Tinker (2008) halo mass function", fontsize=16)
    ax.legend(fontsize=12, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "mass_function.png", dpi=150)
    plt.close(fig)


def fig_concentration():
    """c(M) at z=0.3: child18, child18_powerlaw, and duffy08 side by side.

    All three take M_200c in h^-1 Msun; capped at MAX_MASS (h^-1 Msun here,
    since concentration relations are the one place clenspy is h-scaled).
    """
    z = 0.3
    ms = m_star_hinv(z)
    m200c_hinv = np.logspace(13.0, np.log10(MAX_MASS), 60)  # h^-1 Msun

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.semilogx(m200c_hinv, child18(m200c_hinv, z, ms), color=C3[0], lw=2.5,
                label="child18 (Eq. 18)")
    ax.semilogx(m200c_hinv, child18_powerlaw(m200c_hinv, z), color=C3[1],
                lw=2.5, ls="--", label="child18_powerlaw (Eq. 19)")
    ax.semilogx(m200c_hinv, duffy08(m200c_hinv, z, mass_def="200c"),
                color=C3[2], lw=2.5, label="duffy08")
    ax.set(xlabel=r"$M_{200c} \; [h^{-1}M_\odot]$", ylabel=r"$c_{200c}(M)$")
    ax.set_title(f"Concentration-mass relations at $z={z:g}$", fontsize=16)
    ax.legend(fontsize=12, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "concentration.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_cosmology()
    pk_grid, cosmo = fig_power_spectrum()
    fig_sigma_r(pk_grid, cosmo)
    fig_halo_bias(cosmo)
    fig_mass_function(cosmo)
    fig_concentration()
    print(f"wrote figures to {OUT}")
