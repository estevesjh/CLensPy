r"""RECEIPTS: CLensPy's shipped SelBiasEngine.b_small_large() (default
excess_delta, boost_slope=0.13) vs Matteo Costanzi's Fig 6 digitized data,
vs the free (b_small,b_large) fit -- shows the closure's b_small is wildly
over-amplified (excess_delta is wrong in both magnitude and z-trend), while
SigmoidBias's shape formula itself (b_small+(b_large-b_small)*sigmoid) is
NOT the bug -- confirmed by matching geometry.sigmoid_theta line-by-line
against costanzi_2026.tex eq. (b_sel_theta).

    python validation/validate_bsel_closure_bug.py
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
            theta_fine = np.geomspace(theta.min(), theta.max(), 200)
            R_fine = theta_fine * D_A * V.H  # back to h^-1 comoving Mpc, matching CSV

            # (1) CLensPy as shipped: default excess_delta, boost_slope=0.13
            bsel_curve = engine.marginalised_bias(lob_rep, zob_rep, b_eff=beff)
            ratio_default = bsel_curve(theta_fine) / beff

            # (2) free (b_small, b_large) fit -- best this shape can do
            def resid(params):
                bs, bl = params
                s = SigmoidBias(lob=lob_rep, zob=zob_rep, theta_lambda=theta_lob,
                                b_small=bs, b_large=bl, damping=engine.damping,
                                theta0_frac=engine.theta0_frac)
                return s(theta) / beff - bsel_b
            res = least_squares(resid, x0=[beff * 1.3, beff],
                                bounds=([0.1, 0.1], [200.0, 200.0]))
            bs_fit, bl_fit = res.x
            curve_fit = SigmoidBias(lob=lob_rep, zob=zob_rep, theta_lambda=theta_lob,
                                    b_small=bs_fit, b_large=bl_fit,
                                    damping=engine.damping, theta0_frac=engine.theta0_frac)
            ratio_fit = curve_fit(theta_fine) / beff

            ax.plot(R_b, bsel_b, "o", ms=4, color="k", zorder=5,
                    label="Costanzi+26 Fig 6 (digitized)")
            ax.plot(R_fine, ratio_default, "-", color="C3", lw=2,
                    label="CLensPy as shipped (excess_delta, s=0.13)")
            ax.plot(R_fine, ratio_fit, "--", color="C2",
                    label="best (b_small,b_large) this shape can do")
            ax.set_xscale("log")
            ax.set_title(f"$\\lambda^{{\\rm ob}}\\in[{ll_:.0f},{lh_:.0f})$ "
                         f"$z^{{\\rm ob}}\\in[{zl:.2f},{zh:.2f})$", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel(r"$R\ [h^{-1}{\rm cMpc}]$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$b_{\rm sel}/b_{\rm eff}$")
    fig.suptitle("CLensPy b_sel closure vs Costanzi et al. 2026 Fig 6", y=1.005)
    fig.tight_layout()
    out = REPO / "docs" / "_static" / "validation" / "bsel_closure_bug_receipts.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")
    return str(out)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
