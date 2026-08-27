"""Physical constants, in the package's h-free absolute unit system.

NOTE: CLensPy works throughout in h-free absolute units -- mass in Msun,
lengths in Mpc, densities in Msun/Mpc^3, wavenumbers in 1/Mpc. Every
constant below is expressed in that system, and its units are stated in a
trailing comment. A constant carrying a different convention (an h^2 in a
density, say) does not belong in this module: convert at the boundary where
it enters, in one visible multiplication.
"""

C_LIGHT = 299792.458      # km/s
G_NEWTON = 4.302e-9       # Mpc Msun^-1 (km/s)^2

__all__ = ["C_LIGHT", "G_NEWTON"]


if __name__ == "__main__":
    print("physical constants, h-free absolute units")
    print(f"  C_LIGHT  = {C_LIGHT:.6f} km/s")
    print(f"  G_NEWTON = {G_NEWTON:.6e} Mpc Msun^-1 (km/s)^2")
    # the combination every Sigma_crit carries
    import numpy as np

    print(f"  c^2/(4 pi G) = {C_LIGHT**2 / (4 * np.pi * G_NEWTON):.6e} Msun/Mpc")
    print("  NOTE: c is the exact 299792.458, not the rounded 3e5 some")
    print("        references use -- that rounding is a 0.14% offset.")
