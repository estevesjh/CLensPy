r"""Sensitivity of the SelBiasEngine closure (b_small, b_large, b_eff) to
the HOD mass-observable-relation slope alpha in HodMor.buzzard(), at the
6 richness/z bins used throughout the Fig.6/mock validation
(lambda in [20,30) and [60,500), z in the 3 mock redshift bins).

alpha is scaled relative to buzzard()'s own value, holding log10_Mmin,
log10_M1, epsilon, sigma_intr, z_pivot fixed; b_eff (and the bin
representatives lob_rep/zob_rep) are recomputed fresh per alpha variant
since they depend on the MOR too.

Writes data/processed/alpha_sensitivity.csv.

    SELECTION_BIAS_DIR=../../SelectionBias python scripts/make_alpha_sensitivity.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "validation"))

import validate_sigma_prj_mock as V  # noqa: E402
from clenspy.cosmology import TinkerMassFunction  # noqa: E402
from clenspy.cosmology.bias import BiasModel  # noqa: E402
from clenspy.cosmology.growth import growth_factor  # noqa: E402
from clenspy.lensing import SigmaPrj  # noqa: E402
from clenspy.observables import ClusterCounts  # noqa: E402
from clenspy.selection import EmgParams, SelBiasEngine, SelectionFunction  # noqa: E402
from clenspy.selection.scaling_relation import HodMor  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "processed"
ALPHA_SCALES = [0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
BINS = [(0, 0), (0, 1), (0, 2), (3, 0), (3, 1), (3, 2)]


def b_eff_table_for(mor):
    """b_eff_table()'s recipe, parametrised on the MOR (default hard-codes
    HodMor.buzzard()); everything else identical."""
    sel = SelectionFunction(V.LAMBDA_EDGES, V.Z_EDGES, mor,
                            EmgParams.from_y3_table(), sigma_z=5e-3)
    ln_mass = np.log(np.logspace(13.0, 15.7, 64))
    z_grid = np.linspace(0.15, 0.70, 48)
    tmf_counts = TinkerMassFunction(cosmo=V.COSMO, zvec=np.linspace(0.0, 1.0, 21))
    counts_grid = tmf_counts.dndlnm(np.exp(ln_mass) / V.H, z_grid)
    counts_interp = RegularGridInterpolator(
        (ln_mass, z_grid), counts_grid, bounds_error=False, fill_value=None)

    def mass_function(lnm, z):
        m_h, zz = np.broadcast_arrays(np.exp(np.asarray(lnm, float)), np.asarray(z, float))
        points = np.column_stack((np.log(m_h.ravel()), zz.ravel()))
        return np.asarray(counts_interp(points)).reshape(m_h.shape)

    counts = ClusterCounts(ln_mass, z_grid, mass_function, sel, V.COSMO,
                           omega=lambda z: np.full_like(np.asarray(z, float), np.pi))
    bm = BiasModel(cosmo=V.COSMO)
    m_phys = np.exp(ln_mass) / V.H
    sigma0 = np.asarray(bm.sigma_tophat(m_phys, z=0.0), float)
    growth = np.asarray(growth_factor(z_grid, V.COSMO), float)
    nu = 1.686 / (sigma0[:, None] * growth[None, :])
    bias_grid = np.asarray(bm.bias_at_nu(nu), float)
    return (counts.average(bias_grid), counts.counts(),
            counts.mean_richness(), counts.mean_redshift())


def main() -> int:
    xi_nl, hmf, bias, _ = V.build_halo_model()
    base = HodMor.buzzard()

    rows = []
    for a_scale in ALPHA_SCALES:
        mor = HodMor(log10_Mmin=base.log10_Mmin, log10_M1=base.log10_M1,
                    alpha=base.alpha * a_scale, epsilon=base.epsilon,
                    sigma_intr=base.sigma_intr, z_pivot=base.z_pivot)
        b_eff_ij, n_ij, lam_ij, zrep_ij = b_eff_table_for(mor)
        engine = SelBiasEngine(
            sigma_prj=SigmaPrj(cosmology=V.COSMO, hmf=hmf, bias=bias,
                               xi_nl=xi_nl).build(),
            mor=mor,
        )
        for (i, j) in BINS:
            lob = float(lam_ij[i, j])
            zob = float(zrep_ij[i, j])
            beff = float(b_eff_ij[i, j])
            b_small, b_large = engine.b_small_large(lob, zob, b_eff=beff)
            delta = engine.excess_delta(lob, zob, beff)
            gamma = engine.gamma_lambda(lob, zob)
            rows.append((i, j, V.LAMBDA_EDGES[i], V.LAMBDA_EDGES[i + 1],
                        V.Z_EDGES[j], V.Z_EDGES[j + 1], a_scale, base.alpha * a_scale,
                        lob, zob, beff, b_small, b_large, delta, gamma))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "alpha_sensitivity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "j", "lam_lo", "lam_hi", "z_lo", "z_hi", "alpha_scale",
                    "alpha", "lob", "zob", "b_eff", "b_small", "b_large",
                    "delta", "gamma"])
        w.writerows(rows)
    print(f"wrote {OUT/'alpha_sensitivity.csv'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
