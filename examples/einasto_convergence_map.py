"""
Convergence map for the Einasto projected-series truncation order.

The projected quantities Sigma, DeltaSigma, M_2D are convergent Catalan
series (docs/einasto_proj_density.tex). Their term decay is algebraic,
O(k^{-3/2}), so the relative truncation error falls only as ~K^{-1/2} and
depends strongly on the shape index n. This script maps

    max_R | series(order=K) / ground_truth - 1 |

over a grid of (n, K), using direct numerical quadrature as ground truth.
The output replaces the arbitrary default ``order`` with a data-driven
choice: read off the smallest K meeting a target tolerance for the n of
interest (n = 4-5 for typical spiral haloes).

Run:  PYTHONPATH=../src python einasto_convergence_map.py
"""

import importlib.util
import os

import numpy as np
from scipy.integrate import quad

# Load the einasto module directly (avoids importing clenspy.halo.__init__,
# which pulls optional deps like mcfit).
_HERE = os.path.dirname(__file__)
_SRC = os.path.join(_HERE, "..", "src", "clenspy", "halo", "einasto.py")
_spec = importlib.util.spec_from_file_location("einasto", _SRC)
einasto = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(einasto)
EinastoProfile = einasto.EinastoProfile


def sigma_quad(n, h, rho_0, R):
    """Ground-truth Sigma(R) via the Abel integral (any n)."""
    def integrand(r):
        return rho_0 * np.exp(-(r / h) ** (1.0 / n)) * r / np.sqrt(r * r - R * R)
    val, _ = quad(integrand, R, np.inf, limit=200)
    return 2.0 * val


def m2d_quad(n, h, rho_0, R):
    """Ground-truth M_2D(R) by direct cylindrical integration."""
    def integrand(r):
        rho = rho_0 * np.exp(-(r / h) ** (1.0 / n))
        if r <= R:
            return 4.0 * np.pi * r * r * rho
        # spherical-cap cross section beyond R
        return 4.0 * np.pi * r * r * rho * (1.0 - np.sqrt(1.0 - (R / r) ** 2))
    val, _ = quad(integrand, 0.0, np.inf, limit=400)
    return val


def build_map(quantity="sigma", n_list=(0.5, 1.0, 2.0, 4.0, 5.0, 6.0),
              orders=(2, 5, 10, 20, 40, 80), Rh=(0.3, 0.5, 1.0, 2.0, 3.0),
              rho_0=1.0, r_s=1.0):
    """Print max|rel err| vs order for the chosen projected quantity."""
    Rh = np.asarray(Rh, float)
    truth_fn = {"sigma": sigma_quad, "m2d": m2d_quad}[quantity]
    series_attr = {"sigma": "sigma", "m2d": "enclosed_mass_2D"}[quantity]

    print(f"\n=== {quantity}: max|rel err| over R/h in {list(Rh)} ===")
    header = f"{'n':>5} | " + " ".join(f"K={K:<4}" for K in orders)
    print(header)
    print("-" * len(header))
    for n in n_list:
        alpha = 1.0 / n
        h = EinastoProfile(alpha=alpha, rho_0=rho_0, r_s=r_s, order=2).h
        ref = np.array([truth_fn(n, h, rho_0, rh * h) for rh in Rh])
        row = []
        for K in orders:
            prof = EinastoProfile(alpha=alpha, rho_0=rho_0, r_s=r_s, order=K)
            got = np.asarray(getattr(prof, series_attr)(Rh * h), float)
            row.append(np.max(np.abs(got / ref - 1.0)))
        print(f"{n:>5} | " + " ".join(f"{v:.1e}" for v in row))


def suggest_order(n, tol=1e-4, Rh=(0.3, 0.5, 1.0, 2.0, 3.0),
                  rho_0=1.0, r_s=1.0, kmax=400):
    """Smallest order K with max_R |Sigma_K/Sigma_quad - 1| < tol."""
    Rh = np.asarray(Rh, float)
    alpha = 1.0 / n
    h = EinastoProfile(alpha=alpha, rho_0=rho_0, r_s=r_s, order=2).h
    ref = np.array([sigma_quad(n, h, rho_0, rh * h) for rh in Rh])
    for K in range(1, kmax + 1):
        prof = EinastoProfile(alpha=alpha, rho_0=rho_0, r_s=r_s, order=K)
        got = np.asarray(prof.sigma(Rh * h), float)
        if np.max(np.abs(got / ref - 1.0)) < tol:
            return K
    return None


if __name__ == "__main__":
    build_map("sigma")
    build_map("m2d")

    print("\n=== suggested Sigma order vs tolerance ===")
    print(f"{'n':>5} | " + " ".join(f"tol={t:<7.0e}" for t in (1e-2, 1e-3, 1e-4)))
    for n in (0.5, 1.0, 2.0, 4.0, 5.0, 6.0):
        ks = [suggest_order(n, tol=t) for t in (1e-2, 1e-3, 1e-4)]
        print(f"{n:>5} | " + " ".join(f"{str(k):<11}" for k in ks))
