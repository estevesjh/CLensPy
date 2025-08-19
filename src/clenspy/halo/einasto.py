"""
Module to compute the Einasto profile for dark matter halos.
"""

import numpy as np
from scipy.special import expn, gamma, gammainc, gammaincc


def expn_fast(nu, z):
    """
    Compute the generalized exponential integral E_nu(z).
    If nu is a positive integer, use scipy.special.expn.
    Otherwise, use the relation with the upper incomplete gamma function.
    """
    nu_int = np.floor(nu).astype(int)
    if np.all((nu == nu_int) & (nu_int > 0)):
        return expn(nu_int, z)
    else:
        return z**(nu-1) * gammaincc(1 - nu, z) * gamma(1 - nu)

    # TODO: Implement asymptotic expansion for large nu
    # TODO: Implement einasto power-spectrum (fourier)

class Einasto:
    """Class to represent an Einasto profile."""
    
    def __init__(self, alpha, rho_s, r_s, order=10):
        """
        Initialize the Einasto profile with shape parameter alpha,
        scale density rho_s, and scale radius r_s.
        """
        self.alpha = alpha
        self.rho_s = rho_s
        self.r_s = r_s
        self.order = order

        # Compute the index n from alpha
        self.n_index = 1 / alpha
        self.h = self.r_s / ( 2 * self.n_index )**(self.n_index)

    def density(self, r):
        """
        Compute the density at radius r.

        Parameters:
        r : float or array-like
            Radius at which to compute the density.

        Returns:
        float or array-like
            Density at radius r.
        """
        x = r/self.h
        exponent = - x**(1/self.n_index)
        return self.rho_s * np.exp(exponent)

    def enclosed_mass(self, r):
        """
        Compute the mass enclosed within radius r.

        Parameters:
        r : float
            Radius within which to compute the enclosed mass.

        Returns:
        float
            Mass enclosed within radius r.
        """
        x = r / self.r_s
        prefactor = 4 * np.pi * self.rho_s * self.r_s**3 * (self.alpha / 2)**(3/self.alpha)
        return prefactor * gammainc(3/self.alpha, (2/self.alpha) * x**self.alpha) * gamma(3/self.alpha)
    
    @property(cache=True)
    def _k_vector(self):
        """Return array [1, 2, ..., order]."""
        return np.arange(1, self.order + 1)

    @property(cache=True)
    def _Ck_array(self):
        """
        Compute array of C_k coefficients for k = 1 to order using:
        
        ```
        \math{C_k = \frac{(2k - 3)!!}{2^k\,k!} \quad (k \geq 1)}
        ```
        """
        k = self._k_vector
        Ck = (1.0 / 4**k) * np.array([np.math.comb(2*kk, kk) for kk in k])
        return Ck

    def _nu_vector(self):
        """
        Compute the vector of ν_k values for k = 1 to order:
        
        ```
        \math{\nu_k = 1 + 2nk - n}
        ```
        """
        return 1 + self.n_index * (2 * self._k_vector - 1)
        
    def sigmaR(self, R):
        """
        Compute the projected surface density Σ(R) 
        at projected radius R using the analytic series.

        Parameters:
        R : float or array-like
            Projected radius at which to compute the projected surface density.

        Returns:
        float or array-like
            Projected surface density at radius R.

        Formula:
        ```
        \math{\Sigma(R) = 4 \rho_0\, n R \sum_{k=1}^{\infty} k\, C_k\, E_{\nu_k}\left(X^{1/n}\right)}
        ```
        where
        ```
        \math{C_k = \frac{1}{4^k} \binom{2k}{k}, \quad \nu_k = 1 + n(2k - 1), \quad X = \frac{R}{h}, \quad n = \frac{1}{\alpha}}
        ```
        """
        # Prepare input as numpy array for broadcasting
        R = np.asarray(R)
        n = self.n_index
        h = self.h
        rho_0 = self.rho_s
        X = R / h
        Xpow = X**(1/n)
        k = self._k_vector  # shape (order,)
        Ck = self._Ck_array  # shape (order,)
        nu_k = self._nu_vector()  # shape (order,)

        # Broadcast Xpow and nu_k to shape (R.size, order)
        Xpow_broad = np.expand_dims(Xpow, axis=-1)  # shape (R.size,1)
        nu_k_broad = np.expand_dims(nu_k, axis=0)  # shape (1, order)

        E_nu = expn_fast(nu_k_broad, Xpow_broad)  # shape (R.size, order)
        sumval = np.sum(k * Ck * E_nu, axis=-1)  # shape (R.size,)

        result = 4 * rho_0 * n * R * sumval
        return result


    def deltasigmaR(self, R):
        """
        Excess surface density ΔΣ(R) = Σ̄(<R) - Σ(R) for the Einasto profile,
        using the analytic expression

        ```
        \math{
        \Delta \Sigma(R) = 4 \rho_0\, n R \left( X^{-3} \gamma(3n, X^{1/n}) - \sum_{k=1}^{\infty} (k-1)\, C_k\, E_{\nu_k}\left(X^{1/n}\right) \right)
        }
        ```
        where
        ```
        \math{
        C_k = \frac{1}{4^k} \binom{2k}{k}, \quad
        \nu_k = 1 + n(2k - 1), \quad
        n = \frac{1}{\alpha}, \quad
        X = \frac{R}{h}
        }
        ```
        and \(E_\nu\) is the generalized exponential integral.
        """
        R = np.asarray(R)
        n = self.n_index
        h = self.h
        rho_0 = self.rho_s

        # Compute X and its power
        X = R / h
        Xpow = X**(1/n)

        # call the precomputed arrays
        k = self._k_vector
        Ck = self._Ck_array
        nu_k = self._nu_vector

        Xpow_broad = np.expand_dims(Xpow, axis=-1)
        nu_k_broad = np.expand_dims(nu_k, axis=0)

        E_nu = expn_fast(nu_k_broad, Xpow_broad)
        sumval = np.sum((k - 1) * Ck * E_nu, axis=-1)

        gamma_term = gammainc(3*n, Xpow) * gamma(3*n)
        result = 4 * rho_0 * n * R * (X**(-3) * gamma_term - sumval)
        return result

    def enclosed_mass_2D(self, R):
        """
        Compute the 2D enclosed mass:
        
        ```
        \math{
        M_{2D}(R) = 4 \pi \rho_0\, n R^3 \left( X^{-3} \gamma(3n, X^{1/n}) + \sum_{k=1}^{\infty} C_k\, E_{\nu_k}\left(X^{1/n}\right) \right)
        }
        ```
        where
        ```
        \math{
        X = \frac{R}{h}, \quad
        C_k = \frac{(2k - 3)!!}{2^k k!}, \quad k \geq 1, \quad
        \nu_k = 2 n k - n
        }
        ```
        """
        R = np.asarray(R)
        n = self.n_index
        h = self.h
        rho_0 = self.rho_s
        X = R / h
        Xpow = X**(1/n)

        # Compute C_k = (2k - 3)!! / (2^k k!) for k=1..order
        Ck = self._Ck_array  # shape (order,)
        nu_k = self._nu_vector()  # shape (order,)

        Xpow_broad = np.expand_dims(Xpow, axis=-1)
        nu_k_broad = np.expand_dims(nu_k, axis=0)

        E_nu = expn_fast(nu_k_broad, Xpow_broad)
        sumval = np.sum(Ck * E_nu, axis=-1)

        gamma_term = gammainc(3*n, Xpow) * gamma(3*n)
        result = 4 * np.pi * rho_0 * n * R**3 * (X**(-3) * gamma_term + sumval)
        return result

    def convergence(self, R):
        """
        Compute the convergence 
        ```
        \math{
        \kappa(R) = \frac{\Sigma(R)}{\Sigma_{\mathrm{crit}}}
        }
        ```
        for the Einasto profile.

        Parameters:
        R : float or array-like
            Projected radius at which to compute the convergence.
        order : int
            Order of the series expansion.

        Returns:
        float or array-like
            Convergence at radius R.
        """
        # Placeholder for Σ_crit; in practice, this should be provided based on lensing geometry
        Sigma_crit = 1.0  
        Sigma_R = self.sigmaR(R)
        return Sigma_R / Sigma_crit
    
    def shear(self, R):
        """
        Compute the shear:
        ```
        \math{
        \gamma(R) = \frac{\Delta \Sigma(R)}{\Sigma_{\mathrm{crit}}}
        }
        ```
        for the Einasto profile.

        Parameters:
        R : float or array-like
            Projected radius at which to compute the shear.
        order : int
            Order of the series expansion.

        Returns:
        float or array-like
            Shear at radius R.
        """
        # Placeholder for Σ_crit; in practice, this should be provided based on lensing geometry
        Sigma_crit = 1.0  
        Delta_Sigma_R = self.deltasigmaR(R)
        return Delta_Sigma_R / Sigma_crit