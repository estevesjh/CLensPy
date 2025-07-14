# tests/test_nfw.py

import numpy as np
import pytest

from clenspy.halo.nfw import NFWProfile

try:
    import pyccl as ccl
except ImportError:
    ccl = None

is_plot = False  # Set to True to enable plotting in tests

@pytest.mark.skipif(ccl is None, reason="pyccl not installed")
@pytest.mark.parametrize("truncated", [False, True])
def test_nfw_fourier_matches_pyccl(truncated):
    """Test NFWProfile.fourier matches pyccl's analytic NFW FT (fractional RMS < 1e-3)."""
    # --- User's implementation ---
    m200 = 1e14  # Msun
    c200 = 4.0
    k = np.logspace(-3, 2, 200)  # 1/Mpc

    import pyccl as ccl

    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)

    # 2.  Mass definition + concentration-mass relation
    mdef = ccl.halos.massdef.MassDef200m  # 200×ρ̄_m
    conc = ccl.halos.concentration.constant.ConcentrationConstant(c200, mass_def=mdef)
    # c_of_m = ccl.halos.concentration.ConcentrationDuffy08(mass_def=mdef)

    # 3.  Analytic NFW profile in k-space
    p_nfw = ccl.halos.profiles.HaloProfileNFW(
        mass_def=mdef, concentration=conc, fourier_analytic=True, truncated=truncated
    )  # analytic FT :contentReference[oaicite:0]{index=0}

    # Fourier transform of NFW profile in k-space
    uk_ccl = p_nfw.fourier(
        cosmo, k, m200, 1
    )
    # 1.  clenspy NFW Fourier transform
    nfw = NFWProfile(m200, c200)
    uk_clenspy = nfw.fourier(k, truncated=truncated)

    if is_plot:
        import matplotlib.pyplot as plt
        # --- Plot and compare ---
        plt.figure()
        plt.loglog(k, np.abs(uk_clenspy), label="clenspy NFW FT")
        plt.loglog(k, np.abs(uk_ccl), ls="--", label="pyccl NFW FT")
        plt.xlabel(r"$k$ [Mpc$^{-1}$]")
        plt.ylabel(r"$|u_{\mathrm{NFW}}(k)|$")
        plt.legend()
        plt.title("NFW Fourier Transform: clenspy vs pyccl")
        plt.tight_layout()
        plt.show()

    # Avoid nan at k=0 for truncated case (pyccl returns nan)
    valid = np.isfinite(uk_ccl)
    frac_diff = (uk_clenspy[valid] - uk_ccl[valid]) / uk_ccl[valid]

    # Assert: max fractional diff and RMS are both small (<1e-3, tweakable)
    assert np.nanmax(np.abs(frac_diff)) < 3e-3  # allow some tolerance
    assert np.sqrt(np.nanmean(frac_diff**2)) < 1e-3


@pytest.mark.skipif(ccl is None, reason="pyccl not installed")
def test_nfw_surface_density_matches_pyccl():
    """Test NFWProfile.surface_density matches pyccl analytic NFW projected profile."""

    m200 = 1e14  # Msun
    c200 = 4.0
    R = np.logspace(-3, 1.3, 100)  # R in Mpc, up to ~20 Mpc for outskirts

    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)

    mdef = ccl.halos.massdef.MassDef200m
    conc = ccl.halos.concentration.constant.ConcentrationConstant(c200, mass_def=mdef)

    # Get halo profile
    p_nfw = ccl.halos.profiles.HaloProfileNFW(
        mass_def=mdef, concentration=conc, projected_analytic=True,
        truncated=False  # Use full profile for comparison
    )
    # CCL: projected surface density Sigma(R)
    sig_ccl = p_nfw.projected(cosmo, R, m200, 1)  # M_sun / Mpc^2

    # clenspy: NFW surface density Sigma(R)
    nfw = NFWProfile(m200, c200)
    sig_clenspy = nfw.sigmaR(R)

    # For strict normalization, match scale radius to CCL:
    # rs_ccl = mdef.get_radius(cosmo, m200, 1) / conc(cosmo, m200, 1)
    # nfw.rs = rs_ccl  # Sync scale radius
    # nfw.rhom = cosmo.rho_x(1,'matter')# cosmo.critical_density(0).to_value("Msun/Mpc3")
    # nfw.rho_s = nfw._calc_rhos(m200, c200)

    valid = np.isfinite(sig_ccl) & (sig_ccl > 0)
    frac_diff = (sig_clenspy[valid] - sig_ccl[valid]) / sig_ccl[valid]
    
    if is_plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(R, sig_clenspy, label="clenspy NFW FT")
        plt.loglog(R, sig_ccl, ls="--", label="pyccl NFW FT")
        plt.xlabel(r"$R$ [Mpc]")
        plt.ylabel(r"$\Sigma_{\mathrm{NFW}}(R)$ [M$_\odot$ / Mpc$^2$]")
        plt.legend()
        plt.title("NFW Surface Density: clenspy vs pyccl")
        plt.tight_layout()
        plt.show()
    # Assert: Require close match
    assert np.nanmax(np.abs(frac_diff)) < 5e-3
    assert np.sqrt(np.nanmean(frac_diff**2)) < 2e-3

@pytest.mark.skipif(ccl is None, reason="pyccl not installed")
def test_nfw_deltasigma_matches_pyccl():
    """Test NFWProfile.deltasigma matches pyccl analytic NFW DeltaSigma (RMS < 0.2%)."""
    m200 = 1e14  # Msun
    c200 = 4.0
    R = np.logspace(-3, 1.3, 100)  # [Mpc], up to ~20 Mpc

    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)

    mdef = ccl.halos.massdef.MassDef200m
    conc = ccl.halos.concentration.constant.ConcentrationConstant(c200, mass_def=mdef)

    p_nfw = ccl.halos.profiles.HaloProfileNFW(
        mass_def=mdef, concentration=conc,
        projected_analytic=True, cumul2d_analytic=True, truncated=False
    )

    # Projected surface density Sigma(R) for reference
    # sig_ccl = p_nfw.projected(cosmo, R, m200, z)

    # pyccl: excess surface density DeltaSigma(R)
    sigma_mean = p_nfw.cumul2d(cosmo, R, m200, 1) 
    delta_ccl = sigma_mean - p_nfw.projected(cosmo, R, m200, 1)

    # clenspy: NFW excess surface density DeltaSigma(R)
    nfw = NFWProfile(m200, c200)
    # Ensure you use the same mass definition and normalization!
    delta_clenspy = nfw.deltasigmaR(R)

    if is_plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(R, delta_clenspy, label="clenspy DeltaSigma")
        plt.loglog(R, delta_ccl, ls="--", label="pyccl DeltaSigma")
        plt.xlabel(r"$R$ [Mpc]")
        plt.ylabel(r"$\Delta\Sigma_{\mathrm{NFW}}(R)$ [M$_\odot$ / Mpc$^2$]")
        plt.legend()
        plt.title("NFW Excess Surface Density: clenspy vs pyccl")
        plt.tight_layout()
        plt.show()

    valid = np.isfinite(delta_ccl) & (delta_ccl > 0)
    frac_diff = (delta_clenspy[valid] - delta_ccl[valid]) / delta_ccl[valid]

    print("Max fractional difference:", np.nanmax(np.abs(frac_diff)))
    print("RMS fractional difference:", np.sqrt(np.nanmean(frac_diff**2)))

    # Assert: close match
    assert np.nanmax(np.abs(frac_diff)) < 5e-3
    assert np.sqrt(np.nanmean(frac_diff**2)) < 2e-3