"""
NFW (Navarro-Frenk-White) density profile implementation.

The NFW profile is the most commonly used model for dark matter halo
density profiles in weak lensing analysis.
"""

import numpy as np
from astropy import cosmology
from scipy.special import sici

from ..config import DEFAULT_COSMOLOGY
from ..utils.decorators import scalar_array_output

# from ..config import RHOCRIT
RHOCRIT = 2.77536627e11  # Critical density in Msun/Mpc^3/h^2


class NFWProfile:
    """
    Analytical NFW lensing profile for a single halo or a vector of halos.

    Parameters
    ----------
    m200 : float, array-like
        Halo mass M_200 [Msun]. Can be scalar or array.
    c200 : float, array-like
        Concentration c_200 (dimensionless). Can be scalar or array.
    cosmo : astropy.cosmology instance
        Cosmology instance to use for calculations.

    Notes
    -----
    All methods are vectorized for (n_halo, ...) broadcasting.
    """

    def __init__(
        self,
        m200: np.ndarray | float,
        c200: np.ndarray | float = 4.0,
        cosmo: cosmology = DEFAULT_COSMOLOGY,
    ) -> None:
        # Broadcast shapes for mass and concentration
        m200, c200 = np.broadcast_arrays(m200, c200)
        self.m200 = m200
        self.c200 = c200

        # Critical density in Msun/Mpc^3
        rhoc = cosmo.critical_density(0).to_value("Msun/Mpc^3")
        self.rhom = rhoc * cosmo.Om0  # Msun/Mpc^3

        # Calculate r200 and rs
        self.r200 = self._calculateAtR200(self.m200)  # (n_halo,)
        self.rs = self.r200 / self.c200  # (n_halo,)
        self.rho_s = self._calculateRhos(self.m200, self.c200)  # (n_halo,)

    def _calculateAtR200(self, m200: np.ndarray | float) -> np.ndarray | float:
        """Calculate r200 [Mpc] for given m200 [Msun]."""
        m200 = np.asarray(m200)
        return (3 * m200 / (4 * np.pi * 200 * self.rhom)) ** (1.0 / 3.0)

    def _calculateRhos(
        self, m200: np.ndarray | float, c200: np.ndarray | float
    ) -> np.ndarray | float:
        """Calculate NFW scale density rho_s [Msun/Mpc^3]."""
        c200 = np.asarray(c200)
        rho_s = m200 / (4 * np.pi * self.rs**3 * (np.log(1 + c200) - c200 / (1 + c200)))
        return rho_s

    @scalar_array_output
    def density(self, r: np.ndarray | float) -> np.ndarray | float:
        """
        Calculate 3D density profile for NFW.
        Parameters
        ----------
        r : float or np.ndarray
            Radius [Mpc]. Can be scalar or array.
        Returns
        -------
        rho : np.ndarray
            Density [Msun/Mpc^3], shape = broadcast(n_halo, n_r)
        """
        r = np.atleast_1d(r)
        rs = self.rs[..., None]
        rho_s = self.rho_s[..., None]
        x = r / rs
        return rho_s / (x * (1 + x) ** 2)

    @scalar_array_output
    def fourier(
        self, k: np.ndarray | float, truncated: bool = True
    ) -> np.ndarray | float:
        """
        Analytical Fourier transform of the NFW density profile (normalized by M_200).
        This is the standard u(k|M) ≡ (1/M) ∫ d^3r ρ_NFW(r) exp(i k⋅r),
        evaluated using the closed formula:
        see e.g. pyccl, eq. 34 in Cooray & Sheth 2002.

        Parameters
        ----------
        k : float or np.ndarray
            Wavenumber array [1/Mpc].
        truncated : bool, optional
            If True, use truncated Fourier transform (default: True).

        Returns
        -------
        uk : np.ndarray
            Dimensionless Fourier transform. Shape: (n_halo, n_k)
        """
        m200, c200, rs = self.m200, self.c200, self.rs
        k = np.atleast_1d(k)
        m200, c200, rs = np.broadcast_arrays(m200, c200, rs)
        x = rs[..., None] * k
        norm = np.log(1 + c200)[..., None] - (c200[..., None] / (1 + c200[..., None]))
        P1 = m200[..., None] / norm
        Si2, Ci2 = sici(x)
        if truncated:
            Si1, Ci1 = sici((1 + c200)[..., None] * x)
            P2 = np.sin(x) * (Si1 - Si2) + np.cos(x) * (Ci1 - Ci2)
            P3 = np.sin(c200[..., None] * x) / ((1 + c200[..., None]) * x)
            prof = P1[..., None] * (P2 - P3)
        else:
            P2 = np.sin(x) * (0.5 * np.pi - Si2) - np.cos(x) * Ci2
            prof = P1[..., None] * P2

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(m200) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof

    @scalar_array_output
    def sigma(self, R: np.ndarray | float) -> np.ndarray | float:
        """
        Projected surface density Σ(R) for NFW, in [Msun/Mpc^2].

        Parameters
        ----------
        R : float or np.ndarray
            Projected radius [Mpc].

        Returns
        -------
        sigma : np.ndarray
            Projected surface density, shape = broadcast(n_halo, n_R)
        """
        R = np.atleast_1d(R)
        rs = self.rs[..., None]
        rho_s = self.rho_s[..., None]
        Rs = R / rs
        sigma = 2 * rs * rho_s * self._fNfw(Rs)
        return sigma

    @scalar_array_output
    def deltaSigma(self, R: np.ndarray | float) -> np.ndarray | float:
        """
        Excess surface density ΔΣ(R) for NFW, in [Msun/Mpc^2].

        Parameters
        ----------
        R : float or np.ndarray
            Projected radius [Mpc].

        Returns
        -------
        deltasigma : np.ndarray
            Excess surface density, shape = broadcast(n_halo, n_R)
        """
        R = np.atleast_1d(R)
        rs = self.rs[..., None]
        rho_s = self.rho_s[..., None]
        x = R / rs
        deltasigma = rs * rho_s * self._gNfw(x)
        return deltasigma

    @staticmethod
    def _fNfw(x):
        """Projected NFW profile kernel f(x)."""
        x = np.array(x, dtype=float)
        result = np.zeros_like(x)
        mask1 = x < 1
        mask2 = x == 1
        mask3 = x > 1
        x1 = x[mask1]
        x3 = x[mask3]
        # For x < 1
        result[mask1] = (
            1.0
            / (x1**2 - 1.0)
            * (1 - 2 / np.sqrt(1 - x1**2) * np.arctanh(np.sqrt((1 - x1) / (1 + x1))))
        )
        # For x == 1
        result[mask2] = 1.0 / 3.0
        # For x > 1
        result[mask3] = (
            1.0
            / (x3**2 - 1.0)
            * (1 - 2 / np.sqrt(x3**2 - 1) * np.arctan(np.sqrt((x3 - 1) / (x3 + 1))))
        )
        return result

    @staticmethod
    def _gNfw(x, eps=1e-9):
        """Mean enclosed projected NFW kernel g(x)."""
        x = np.array(x, dtype=float)
        res = np.zeros_like(x)
        # x == 1 (central value, analytic)
        mask_c = np.abs(x - 1) <= eps
        res[mask_c] = 10.0 / 3.0 + 4 * np.log(1 / 2.0)

        # x < 1
        mask_l = x < 1 - eps
        sqrt1mx2 = np.sqrt(1.0 - x[mask_l] ** 2)
        atanh = np.arctanh(sqrt1mx2 / (1.0 + x[mask_l]))
        term1 = 8.0 * atanh / (x[mask_l] ** 2 * sqrt1mx2)
        term2 = 4.0 / x[mask_l] ** 2 * np.log(x[mask_l] / 2.0)
        term3 = -2.0 / (x[mask_l] ** 2 - 1.0)
        term4 = 4.0 * atanh / ((x[mask_l] ** 2 - 1.0) * sqrt1mx2)
        res[mask_l] = term1 + term2 + term3 + term4

        # x > 1
        mask_g = x > 1 + eps
        sqrtx2m1 = np.sqrt(x[mask_g] ** 2 - 1.0)
        atan = np.arctan(sqrtx2m1 / (1.0 + x[mask_g]))
        term1 = 8.0 * atan / (x[mask_g] ** 2 * sqrtx2m1)
        term2 = 4.0 / x[mask_g] ** 2 * np.log(x[mask_g] / 2.0)
        term3 = -2.0 / (x[mask_g] ** 2 - 1.0)
        term4 = 4.0 * atan / ((x[mask_g] ** 2 - 1.0) ** 1.5)
        res[mask_g] = term1 + term2 + term3 + term4

        return res


__all__ = ["NFWProfile"]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    from clenspy.halo.nfw import NFWProfile

    is_truncated = True  # Set to False for full profile

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
        mass_def=mdef, concentration=conc, fourier_analytic=True, truncated=is_truncated
    )  # analytic FT :contentReference[oaicite:0]{index=0}

    uk_ccl = p_nfw.fourier(
        cosmo, k, m200, 1
    )  # Fourier transform of NFW profile in k-space
    rs_ccl = mdef.get_radius(cosmo, m200, 1) / conc(cosmo, m200, 1)

    # 1.  clenspy NFW Fourier transform
    nfw = NFWProfile(m200, c200)
    uk_clenspy = nfw.fourier(k, truncated=is_truncated)
    # nfw.rs = rs_ccl  # Use CCL's rs for consistency

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

    plt.figure()
    frac_diff = (uk_clenspy - uk_ccl) / uk_ccl
    plt.semilogx(k, frac_diff)
    plt.xlabel(r"$k$ [Mpc$^{-1}$]")
    plt.ylabel("Fractional diff (clenspy - pyccl)/pyccl")
    plt.title("Fractional difference")
    plt.axhline(0, color="k", lw=1)
    plt.tight_layout()
    plt.show()

    print("Max fractional diff:", np.nanmax(np.abs(frac_diff)))
    print("RMS fractional diff:", np.sqrt(np.nanmean(frac_diff**2)))
