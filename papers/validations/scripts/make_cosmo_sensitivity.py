r"""Sensitivity of the selected/RND SigmaPrj ratio to the assumed
cosmology (Omega_m, sigma8) around the Buzzard v1.1 fiducial, at fixed
b_eff (letting b_eff respond self-consistently to cosmology is a
second-order effect not tested here -- see the report caveat).

Reuses validate_sigma_prj_mock.build_halo_model/build_projection_products
by temporarily overriding the module's COSMO global (those functions
read it at call time, so this reproduces the exact production chain for
each cosmology variant without duplicating it).

Writes data/processed/cosmo_sensitivity.csv.

    SELECTION_BIAS_DIR=../../SelectionBias python scripts/make_cosmo_sensitivity.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "validation"))

import validate_sigma_prj_mock as V  # noqa: E402
from clenspy.lensing import SigmaPrj, SigmaPrjConfig  # noqa: E402
from clenspy.selection import SelBiasEngine  # noqa: E402
from clenspy.selection.scaling_relation import HodMor  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "processed"
R_VALS = np.array([1.0, 3.0, 10.0, 20.0])   # cMpc/h
FRACS = [0.90, 0.95, 1.00, 1.05, 1.10]
BINS = [(0, 2), (3, 2)]   # lambda[20,30) and [60,500), z in [0.50,0.65)

FIDUCIAL_OM = V.OMEGA_M
FIDUCIAL_S8 = V.BuzzardCosmology.sigma8


def ratio_at(cosmo, lob, zob, beff):
    orig = V.COSMO
    V.COSMO = cosmo
    try:
        xi_nl, hmf, bias, _ = V.build_halo_model()
        pk_prj, hmf_prj, two_halo_prj, bias_prj = V.build_projection_products()
    finally:
        V.COSMO = orig
    engine = SelBiasEngine(
        sigma_prj=SigmaPrj(cosmology=cosmo, hmf=hmf, bias=bias, xi_nl=xi_nl).build(),
        mor=HodMor.buzzard(),
    )
    prj = SigmaPrj(
        cosmology=cosmo, pk=pk_prj, hmf=hmf_prj, two_halo=two_halo_prj, bias=bias_prj,
        config=SigmaPrjConfig(
            n_theta=128, theta_perp_range=(1e-3, 2.0 * V.APERTURE_HINV / V.H),
            los_depth=V.LOS_HALF_DEPTH_HINV / V.H, exclusion="counter",
            r_trunc=V.APERTURE_HINV / V.H,
        ),
    )
    bsel = engine.marginalised_bias(lob, zob, b_eff=beff)
    sel = prj.sigma_prj(R_VALS / V.H, lob, zob, bsel, channel="sum")
    rnd = prj.sigma_prj(R_VALS / V.H, lob, zob, lambda th: beff, channel="sum")
    return sel / rnd


def main() -> int:
    b_eff_ij, n_ij, lam_ij, zrep_ij = V.b_eff_table()

    rows = []
    for (i, j) in BINS:
        lob = float(lam_ij[i, j])
        zob = float(zrep_ij[i, j])
        beff = float(b_eff_ij[i, j])

        for param in ("Om", "s8"):
            for frac in FRACS:
                if param == "Om":
                    om, s8 = FIDUCIAL_OM * frac, FIDUCIAL_S8
                else:
                    om, s8 = FIDUCIAL_OM, FIDUCIAL_S8 * frac
                cosmo = V.BuzzardCosmology(H0=100.0 * V.H, Om0=om, Ob0=0.046)
                cosmo.sigma8 = s8
                ratio = ratio_at(cosmo, lob, zob, beff)
                for k, rv in enumerate(R_VALS):
                    rows.append((i, j, V.LAMBDA_EDGES[i], V.LAMBDA_EDGES[i + 1],
                                param, frac, om, s8, float(rv), float(ratio[k])))
        print(f"bin ({i},{j}) done")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "cosmo_sensitivity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "j", "lam_lo", "lam_hi", "param", "frac", "Omega_m",
                    "sigma8", "R_hinv_cMpc", "ratio"])
        w.writerows(rows)
    print(f"wrote {OUT/'cosmo_sensitivity.csv'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
