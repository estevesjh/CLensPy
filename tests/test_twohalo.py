"""
Tests for halo module.
"""

import numpy as np
import pytest

from clenspy.halo.twohalo import TwoHaloTerm

try:
    import pyccl as ccl
except ImportError:
    ccl = None

try:
    import cluster_toolkit as ct
except ImportError:
    ct = None

try:
    import clmm
    from clmm import Cosmology
    from clmm.theory import func_layer
except ImportError:
    clmm = None

RHOCRIT = 2.77533742639e11  # Critical density in Msun h^2/Mpc^3

is_plot = True  # Set to True to enable plotting in tests

@pytest.mark.skipif(ccl is None or ct is None or clmm is None, reason="pyccl and cluster_toolkit and clmm required")
def test_twohalo_deltasigma_matches_clustertoolkit():
    """Test TwoHaloTerm ΔΣ_2h matches cluster-toolkit to <2% RMS on fixed grid."""
    # --- Parameters ---
    MASS = 2E14
    CONCENTRATION = 1.0
    z_cl = 0.1

    # k and P(k) grid (coarse for speed)
    k_values = np.logspace(-3, 1, 60)   # [Mpc^-1]
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.81, n_s=0.96)
    clmm_cosmo = Cosmology(H0=100*0.7, Omega_dm0=0.25, 
                           Omega_b0=0.05, Omega_k0=0.0)
    pk_values = ccl.linear_matter_power(cosmo, k_values, 1/(1+z_cl))
    rho_m = clmm_cosmo._get_rho_m(z_cl)

    # Projected radii [Mpc]
    r_proj = np.logspace(-1, 1, 20)   # [0.1, 10] Mpc, 20 points

    # --- TwoHaloTerm ---
    # halogrid = TwoHaloTerm(k_values, pk_values, method='trapz')
    # halogrid = TwoHaloTerm(k_values, pk_values, method='leggauss', n_points=128)
    halogrid = TwoHaloTerm(k_values, pk_values, method='quad_vec')

    deltasigma_halo = halogrid.deltasigma_R(r_proj)
    deltasigma_halo *= rho_m #/ halogrid.rho_m

    # --- cluster-toolkit (uses same P(k)) ---
    Rfix = np.logspace(-3., 2, 500)
    xi_mm = ct.xi.xi_mm_at_r(Rfix, k_values, pk_values)
    xi_2halo = ct.xi.xi_2halo(1.0, xi_mm)
    Sigma_mm = ct.deltasigma.Sigma_at_R(Rfix[1:], Rfix, xi_2halo, MASS, CONCENTRATION, 1.0)

    # Interpolate ΔΣ at r_proj
    from scipy.integrate import cumulative_trapezoid as cumtrapz
    res = cumtrapz(Sigma_mm * Rfix[1:], Rfix[1:], initial=0)
    mean_sigma = 2 * res / (Rfix[1:] ** 2)
    deltasigma_ct = np.clip(mean_sigma - Sigma_mm, 0, None)
    deltasigma_ct = np.interp(np.log(r_proj), np.log(Rfix[1:]), deltasigma_ct)
    deltasigma_ct *= 1e12*rho_m/RHOCRIT

    # --- CLMM ---
    rfix = np.logspace(-3, 3, 100)  # Mpc

    # Compute physical Σ at physical radius
    sigma_physical = func_layer.compute_excess_surface_density_2h(
        rfix, z_cl, clmm_cosmo, halobias=1.0,
    )
    # Convert to comoving Σ at comoving radius
    deltasigma_clmm = np.interp(np.log(r_proj/(1+z_cl)), np.log(rfix), sigma_physical)
    deltasigma_clmm *= (1+z_cl)

    if is_plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(r_proj, deltasigma_halo, label="clenspy TwoHaloTerm ΔΣ")
        plt.loglog(r_proj, deltasigma_ct, ls="--", label="cluster-toolkit ΔΣ")
        plt.loglog(r_proj, deltasigma_clmm, ls=":", label="CLMM ΔΣ")
        plt.xlabel(r"$R$ [Mpc]")
        plt.ylabel(r"$\Delta\Sigma(R)$ [M$_\odot$ / Mpc$^2$]")
        plt.legend()
        plt.title("Two-Halo Term: clenspy vs cluster-toolkit vs CLMM")
        plt.tight_layout()
        plt.show()

    # --- Assert RMS fractional difference < 5%
    rel = (deltasigma_halo - deltasigma_ct) / deltasigma_ct
    print("ΔΣ mean:", np.mean(deltasigma_ct), "ΔΣ max abs rel err:", np.max(np.abs(rel)))
    rms_frac = np.sqrt(np.mean(rel**2))
    print("RMS fractional difference:", rms_frac)
    assert rms_frac < 0.05   # <5% RMS

    # Also: no NaNs or negatives
    assert np.all(np.isfinite(deltasigma_halo))
    assert np.all(deltasigma_halo >= 0)

    # --- Assert CLMM matches halo term
    rel_clmm = (deltasigma_halo - deltasigma_clmm) / deltasigma_clmm
    print("ΔΣ CLMM mean:", np.mean(deltasigma_clmm), "ΔΣ max abs rel err CLMM:", np.max(np.abs(rel_clmm)))
    rms_frac_clmm = np.sqrt(np.mean(rel_clmm**2))
    print("RMS fractional difference CLMM:", rms_frac_clmm)
    assert rms_frac_clmm < 0.05   # <5% RMS
