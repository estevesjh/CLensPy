"""
Basic usage example for CLensPy.

This script demonstrates how to:
1. Build an NFW halo from mass and concentration.
2. Compute its 3D density, projected surface density Sigma(R), and excess
   surface density (weak-lensing shear proxy) DeltaSigma(R).
3. Compare it against an Einasto profile with a matched scale radius.
4. Estimate the linear halo bias for the same halo mass.

Run it with:
    python examples/demo_basic_usage.py
"""

import matplotlib.pyplot as plt
import numpy as np

from clenspy.halo import BiasModel, EinastoProfile, NfwProfile

# ---------------------------------------------------------------------------
# 1. NFW profile for a massive cluster-scale halo
# ---------------------------------------------------------------------------
M200 = 1e14  # Halo mass [Msun]
C200 = 5.0  # Concentration

nfw = NfwProfile(m200=M200, c200=C200)
print(f"NFW halo: M200={M200:.1e} Msun, c200={C200}")
print(
    f"  r200 = {nfw.r200:.3f} Mpc, rs = {nfw.rs:.3f} Mpc, "
    f"rho_s = {nfw.rho_s:.3e} Msun/Mpc^3"
)

R = np.logspace(-2, 1, 60)  # Projected radius [Mpc]
r = np.logspace(-2, 1, 60)  # 3D radius [Mpc]

rho_nfw = nfw.density(r)
sigma_nfw = nfw.sigma(R)
deltasigma_nfw = nfw.deltasigma(R)

# ---------------------------------------------------------------------------
# 2. Einasto profile with a matched scale radius, for comparison
# ---------------------------------------------------------------------------
einasto = EinastoProfile(alpha=0.2, rho_0=nfw.rho_s, r_s=nfw.rs, tol=1e-4)
print(f"Einasto halo: n={einasto.n_index:.1f}, r_s = {einasto.r_s:.3f} Mpc")

rho_einasto = einasto.density(r)
sigma_einasto = einasto.sigma(R)
deltasigma_einasto = einasto.deltasigma(R)

# ---------------------------------------------------------------------------
# 3. Linear halo bias for the same mass, from a toy power-law P(k)
# ---------------------------------------------------------------------------
k = np.logspace(-3, 1, 200)  # Wavenumber [h/Mpc]
Pk = 2e4 * (k / 0.05) ** (-1.5)  # Illustrative power spectrum, not a real P(k)

bias_model = BiasModel(k, Pk)
bias = bias_model.bias(M200)
print(f"Linear bias b(M200) = {bias:.3f} (toy power spectrum, illustrative only)")

# ---------------------------------------------------------------------------
# 4. Plot density, Sigma(R), and DeltaSigma(R) for both profiles
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].loglog(r, rho_nfw, label="NFW")
axes[0].loglog(r, rho_einasto, ls="--", label="Einasto")
axes[0].set_xlabel(r"$r$ [Mpc]")
axes[0].set_ylabel(r"$\rho(r)$ [M$_\odot$/Mpc$^3$]")
axes[0].legend()

axes[1].loglog(R, sigma_nfw, label="NFW")
axes[1].loglog(R, sigma_einasto, ls="--", label="Einasto")
axes[1].set_xlabel(r"$R$ [Mpc]")
axes[1].set_ylabel(r"$\Sigma(R)$ [M$_\odot$/Mpc$^2$]")
axes[1].legend()

axes[2].loglog(R, deltasigma_nfw, label="NFW")
axes[2].loglog(R, deltasigma_einasto, ls="--", label="Einasto")
axes[2].set_xlabel(r"$R$ [Mpc]")
axes[2].set_ylabel(r"$\Delta\Sigma(R)$ [M$_\odot$/Mpc$^2$]")
axes[2].legend()

fig.suptitle(f"CLensPy: NFW vs Einasto, M200={M200:.0e} Msun, c200={C200}")
fig.tight_layout()
fig.savefig("clenspy_basic_usage.png", dpi=150)
print("Saved plot to clenspy_basic_usage.png")
plt.show()
