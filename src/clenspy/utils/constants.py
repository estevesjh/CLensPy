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
