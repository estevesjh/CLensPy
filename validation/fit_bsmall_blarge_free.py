r"""Free (b_small, b_large) fit per (lam, z) bin -- NOT a valid physical
fit (the closure ties them through one delta, they are not independent
degrees of freedom) -- but isolates whether SigmoidBias's fixed shape
(k=damping/theta_lob, theta0=theta0_frac*theta_lob, from the paper's own
eqn) can match Matteo Costanzi's Fig 6 curve SHAPE at all, independent of
amplitude/closure.

    python validation/fit_bsmall_blarge_free.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "validation"))

import validate_sigma_prj_mock as V  # noqa: E402
from clenspy.lensing import SigmaPrj  # noqa: E402
from clenspy.selection import SelBiasEngine, SigmoidBias  # noqa: E402
from clenspy.selection.scaling_relation import HodMor  # noqa: E402
from fit_boost_slope_global import custom_b_eff, load_all  # noqa: E402


def main() -> int:
    digi = load_all()
    lam_lo, lam_hi, z_lo, z_hi, R, bsel_digi = digi.T

    xi_nl, hmf, bias, _ = V.build_halo_model()
    engine = SelBiasEngine(
        sigma_prj=SigmaPrj(cosmology=V.COSMO, hmf=hmf, bias=bias, xi_nl=xi_nl).build(),
        mor=HodMor.buzzard(),
    )
    b_eff_ij, _, lam_ij, zrep_ij = V.b_eff_table()

    bins = sorted(set(zip(lam_lo, lam_hi, z_lo, z_hi)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lam_groups = sorted(set((ll, lh) for (ll, lh, _, _) in bins))
    fig, axes = plt.subplots(len(lam_groups), 3, figsize=(13, 4.0 * len(lam_groups)),
                             sharex=True)

    print(f"{'bin':<28}{'beff':>7}{'bs_fit':>9}{'bl_fit':>9}{'bs/beff':>9}{'bl/beff':>9}{'med|resid|':>12}")
    for row, (ll, lh) in enumerate(lam_groups):
        row_bins = sorted([b for b in bins if b[0] == ll and b[1] == lh],
                          key=lambda b: b[2])
        for ax, (ll_, lh_, zl, zh) in zip(axes[row], row_bins):
            m = (lam_lo == ll_) & (lam_hi == lh_) & (z_lo == zl) & (z_hi == zh)
            R_b, bsel_b = R[m], bsel_digi[m]

            aligned = np.isclose(V.LAMBDA_EDGES[:-1], ll_).any()
            if aligned:
                i = int(np.where(np.isclose(V.LAMBDA_EDGES[:-1], ll_))[0][0])
                j = int(np.where(np.isclose(V.Z_EDGES[:-1], zl))[0][0])
                lob_rep = float(lam_ij[i, j]); zob_rep = float(zrep_ij[i, j])
                beff = float(b_eff_ij[i, j])
            else:
                lob_rep, zob_rep, beff = custom_b_eff(ll_, lh_, zl, zh)

            chi_o = float(engine.chi(zob_rep))
            D_A = chi_o / (1.0 + zob_rep)
            theta = (R_b / V.H) / D_A
            theta_lob = engine._theta_lob(lob_rep, zob_rep)

            def resid(params):
                bs, bl = params
                s = SigmoidBias(lob=lob_rep, zob=zob_rep, theta_lambda=theta_lob,
                                b_small=bs, b_large=bl, damping=engine.damping,
                                theta0_frac=engine.theta0_frac)
                return s(theta) / beff - bsel_b

            res = least_squares(resid, x0=[beff * 1.3, beff],
                                bounds=([0.1, 0.1], [200.0, 200.0]))
            bs_fit, bl_fit = res.x
            med = np.median(np.abs(resid(res.x)))
            print(f"lam[{ll_:.0f},{lh_:.0f}) z[{zl:.2f},{zh:.2f}) "
                  f"{beff:7.2f}{bs_fit:9.2f}{bl_fit:9.2f}{bs_fit/beff:9.3f}"
                  f"{bl_fit/beff:9.3f}{med:12.4f}")

            theta_fine = np.geomspace(theta.min(), theta.max(), 200)
            curve = SigmoidBias(lob=lob_rep, zob=zob_rep, theta_lambda=theta_lob,
                                b_small=bs_fit, b_large=bl_fit, damping=engine.damping,
                                theta0_frac=engine.theta0_frac)
            ax.plot(theta, bsel_b, "o", ms=4, color="k", label="Matteo (digitized)")
            ax.plot(theta_fine, curve(theta_fine) / beff, "-", color="C2",
                    label="free (b_small,b_large) fit")
            ax.set_xscale("log")
            ax.set_title(f"$\\lambda\\in[{ll_:.0f},{lh_:.0f})$ "
                         f"$z\\in[{zl:.2f},{zh:.2f})$", fontsize=9)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\theta$ [rad]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$b_{\rm sel}/b_{\rm eff}$")
    fig.tight_layout()
    out = REPO / "docs" / "_static" / "validation" / "bsel_free_bsmall_blarge_fit.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
