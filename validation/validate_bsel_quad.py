r"""Independent quadrature cross-check of `SelBiasEngine.operators()`
(:math:`P_1, I_1, I_2`) -- same method as
``RichnessSelection/validations/quad_validate.py``: adaptive
``scipy.integrate.quad`` over the line-of-sight redshift, dense fixed
grids (no Gauss-Legendre, no z-ring/outer split) for the coupled
:math:`(\theta, M, \lambda^{\rm tr})` integral at each z. If this agrees
with the production engine (``bsel.py``'s GL-node, ring-split machinery)
to a few parts in 1e-3, the issue #5 discrepancy is not in
`SelBiasEngine`'s numerics -- see ``validate_mor_notebook.py`` for the
part that is.

Physics reproduced verbatim from `SelBiasEngine.operators()` (the
docstring's :math:`\mathcal P[X]` with :math:`X \in \{1, b\xi, b\xi\sigma\}`);
only the *integration method* differs, deliberately.

Usage::

    python validation/validate_bsel_quad.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_sigma_prj_mock as V  # noqa: E402

from clenspy.cosmology.distances import (  # noqa: E402
    ComovingDistance,
    comoving_volume_element,
)
from clenspy.kernels.photoz import (  # noqa: E402
    photoz_projection,
    photoz_projection_support,
    y3_photoz_window,
)
from clenspy.selection import PhysicalMassMor, SelBiasEngine  # noqa: E402
from clenspy.selection.geometry import (  # noqa: E402
    area_overlap,
    r_lambda,
    sigmoid_theta,
    theta_lambda,
)
from clenspy.selection.scaling_relation import HodMor  # noqa: E402

N_TH, N_M, N_LTR = 200, 150, 150  # dense fixed grids, no GL


def build_quad_operators(engine: SelBiasEngine):
    """Return a callable ``operators_quad(lob, zob) -> (P1, I1, I2)``
    that reimplements `SelBiasEngine.operators` with scipy.quad over z
    and dense trapz over (theta, M, lambda) -- independent of `_z_grid`,
    `gl_nodes`, and the engine's cache."""
    cosmo, h = engine.cosmo, engine.h
    dist = ComovingDistance(cosmo)
    window = y3_photoz_window()

    def _inner_at_z(z, zob, theta_lob, theta_max, R_excl, chi_o, weight_type):
        chi_z = float(dist.chi(z))
        cos_e = np.clip(
            (chi_z**2 + chi_o**2 - R_excl**2) / (2.0 * chi_z * chi_o + 1e-30),
            -1.0, 1.0,
        )
        theta_excl = 1e-6 if cos_e >= 1.0 - 1e-12 else np.arccos(cos_e)
        th_lo = max(theta_excl, 1e-6)
        if th_lo >= theta_max:
            return 0.0

        thetas = np.linspace(th_lo, theta_max, N_TH)
        dchi = np.sqrt(np.maximum(
            chi_z**2 + chi_o**2 - 2.0 * chi_z * chi_o * np.cos(thetas), 0.0
        ))
        xi = np.maximum(engine.xi_nl(dchi, zob), 0.0)
        sig = sigmoid_theta(thetas, theta_lob, engine.damping, engine.theta0_frac)

        m_grid = np.geomspace(engine.min_mass, 10.0**engine.log10_M_max, N_M)
        ln_m = np.log(m_grid)
        lam_grid = np.linspace(1e-6, lob_cur[0], N_LTR)

        theta_lam = r_lambda(lam_grid, h) * (1.0 + z) / chi_z      # (Nltr,)
        fA = area_overlap(thetas, theta_lob, theta_lam)             # (Nth,Nltr)

        if weight_type == "P1":
            ang = np.trapezoid(np.sin(thetas)[:, None] * fA, thetas, axis=0)
        elif weight_type == "I2":
            ang = np.trapezoid(
                np.sin(thetas)[:, None] * fA * xi[:, None], thetas, axis=0
            )
        elif weight_type == "I1":
            ang = np.trapezoid(
                np.sin(thetas)[:, None] * fA * xi[:, None] * sig[:, None],
                thetas, axis=0,
            )
        else:
            raise ValueError(weight_type)
        ang *= 2.0 * np.pi                                          # (Nltr,)

        p_lm = engine.mor.pdf(lam_grid[:, None], m_grid[None, :], z)  # (Nltr,NM)
        lam_int = np.trapezoid(
            p_lm * (lam_grid * ang)[:, None], lam_grid, axis=0
        )                                                            # (NM,)

        n_m = engine.hmf(m_grid, z)
        if weight_type == "P1":
            M_integrand = m_grid * n_m * lam_int
        else:
            b_m = engine.bias(m_grid, z)
            M_integrand = m_grid * n_m * b_m * lam_int
        M_int = np.trapezoid(M_integrand, ln_m)

        wz_val = float(photoz_projection(np.array([z]), zob, window,
                                         n_sigma=1.0)[0])
        dV = comoving_volume_element(z, cosmo)
        return dV * wz_val * M_int

    lob_cur = [None]  # closure cell, set per call (avoids re-plumbing lob)

    def operators_quad(lob, zob):
        lob_cur[0] = float(lob)
        theta_lob = float(theta_lambda(lob, zob, dist.chi, h))
        theta_max = 2.0 * theta_lob
        chi_o = float(dist.chi(zob))
        R_excl = float(r_lambda(lob, h) * (1.0 + zob))
        z_lo, z_hi = photoz_projection_support(zob, window, n_sigma=1.0)

        out = {}
        for weight in ("P1", "I2", "I1"):
            args = (zob, theta_lob, theta_max, R_excl, chi_o, weight)
            v_fg, _ = quad(_inner_at_z, z_lo, zob, args=args,
                          epsrel=1e-4, limit=200)
            v_bg, _ = quad(_inner_at_z, zob, z_hi, args=args,
                          epsrel=1e-4, limit=200)
            out[weight] = v_fg + v_bg
        return out["P1"], out["I1"], out["I2"]

    return operators_quad


def main():
    print("[setup] building Buzzard-config halo model (shared with "
          "validate_sigma_prj_mock.py) ...")
    t0 = time.time()
    xi_nl, hmf, bias, _ = V.build_halo_model()
    print(f"[setup] done in {time.time() - t0:.1f}s\n")

    lob, zob = 20.0, 0.5

    for label, mor in (
        ("HodMor.from_lognormal() [current production MOR]",
         PhysicalMassMor(HodMor.from_lognormal(), V.H)),
        ("HodMor.buzzard()", PhysicalMassMor(HodMor.buzzard(), V.H)),
    ):
        print(f"=== MOR = {label} ===")
        engine = SelBiasEngine(cosmology=V.COSMO, xi_nl=xi_nl, hmf=hmf,
                               bias=bias, mor=mor)
        t0 = time.time()
        p1_prod, i1_prod, i2_prod = engine.operators(lob, zob)
        t_prod = time.time() - t0

        operators_quad = build_quad_operators(engine)
        t0 = time.time()
        p1_q, i1_q, i2_q = operators_quad(lob, zob)
        t_quad = time.time() - t0

        print(f"{'':>4s} {'production (GL+ring)':>22s} {'quad (scipy+dense)':>20s} "
              f"{'rel diff':>10s}")
        for name, prod, q in (("P1", p1_prod, p1_q), ("I1", i1_prod, i1_q),
                              ("I2", i2_prod, i2_q)):
            rel = abs(prod - q) / abs(q) if q != 0 else float("nan")
            print(f"{name:>4s} {prod:22.6e} {q:20.6e} {rel:10.2%}")
        print(f"  [production {t_prod:.2f}s, quad {t_quad:.2f}s]\n")


if __name__ == "__main__":
    main()
