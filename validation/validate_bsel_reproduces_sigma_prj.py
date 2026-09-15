r"""Does Matteo Costanzi's own digitized b_sel(theta)/b_eff curve, fed
through OUR SigmaPrj integral, reproduce his own digitized Sigma_prj
ratio curve (Fig 6)? Isolates the Sigma_prj/n_los_integral machinery
from the b_sel(theta) closure entirely: b_sel here is read directly off
the digitized data (log-log interpolated, flat-extrapolated), not
computed by SelBiasEngine at all.

theta<->R convention: matches SigmaPrj's OWN internal usage (R_comoving
= theta * chi(zob), i.e. theta = R_comoving/chi(zob)) -- NOT the
R/D_A(zob) convention used in the bsel-only diagnostic scripts earlier
this session (those differ by a factor of (1+zob); flagged, unresolved,
and irrelevant to the power-law-in-R finding, but it matters here since
we must hand SigmaPrj a b_sel(theta) it will query at ITS OWN theta
nodes).

    SELECTION_BIAS_DIR=../SelectionBias python validation/validate_bsel_reproduces_sigma_prj.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "validation"))

import validate_sigma_prj_mock as V  # noqa: E402
from clenspy.lensing import SigmaPrj, SigmaPrjConfig  # noqa: E402
from fit_boost_slope_global import load_all as load_bsel  # noqa: E402

SIGMA_PRJ_CSV = REPO / "validation" / "data" / "costanzi2026_fig6.csv"


def bsel_interpolator(R_pts_hinv, ratio_pts, beff, chi_o):
    """theta -> b_sel, by looking up the digitized bsel/beff(R) curve at
    R_comoving_hinv = theta * chi_o * H (SigmaPrj's own theta<->R
    convention), flat-extrapolated outside the digitized R range."""
    order = np.argsort(R_pts_hinv)
    logR = np.log(R_pts_hinv[order])
    logRatio = np.log(ratio_pts[order])
    lo, hi = logR[0], logR[-1]

    def bsel(theta):
        theta = np.asarray(theta, dtype=float)
        R_hinv = theta * chi_o * V.H
        logR_q = np.clip(np.log(np.maximum(R_hinv, 1e-12)), lo, hi)
        logRatio_q = np.interp(logR_q, logR, logRatio)
        return beff * np.exp(logRatio_q)

    return bsel


def main() -> int:
    prj_digi = np.loadtxt(SIGMA_PRJ_CSV, delimiter=",", skiprows=13)
    p_lam_lo, p_lam_hi, p_z_lo, p_z_hi, p_R, p_ratio = prj_digi.T

    bsel_digi = load_bsel()
    b_lam_lo, b_lam_hi, b_z_lo, b_z_hi, b_R, b_ratio = bsel_digi.T

    xi_nl, hmf, bias, _ = V.build_halo_model()
    pk_prj, hmf_prj, two_halo_prj, bias_prj = V.build_projection_products()
    prj = SigmaPrj(
        cosmology=V.COSMO, pk=pk_prj, hmf=hmf_prj, two_halo=two_halo_prj,
        bias=bias_prj,
        config=SigmaPrjConfig(los_depth=V.LOS_HALF_DEPTH_HINV / V.H,
                              exclusion="counter"),
    )
    b_eff_ij, _, lam_ij, zrep_ij = V.b_eff_table()

    # only richness bins present in BOTH digitized datasets (top bin
    # differs: bsel has [60,200), sigma_prj has [60,500) -- skipped)
    p_bins = set(zip(p_lam_lo, p_lam_hi))
    b_bins = set(zip(b_lam_lo, b_lam_hi))
    shared = sorted(p_bins & b_bins)
    print(f"shared richness bins: {shared}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(shared), 3, figsize=(13, 4.0 * len(shared)),
                             sharex=True)
    if len(shared) == 1:
        axes = axes[None, :]

    all_resid = []
    for row, (ll, lh) in enumerate(shared):
        i = int(np.where(np.isclose(V.LAMBDA_EDGES[:-1], ll))[0][0])
        zpanels = sorted(set(zip(p_z_lo[(p_lam_lo == ll)],
                                 p_z_hi[(p_lam_lo == ll)])))
        for ax, (zl, zh) in zip(axes[row], zpanels):
            j = int(np.where(np.isclose(V.Z_EDGES[:-1], zl))[0][0])
            lob_rep = float(lam_ij[i, j]); zob_rep = float(zrep_ij[i, j])
            beff = float(b_eff_ij[i, j])
            chi_o = float(prj.distance.chi(zob_rep))

            mp = ((p_lam_lo == ll) & (p_lam_hi == lh)
                  & (p_z_lo == zl) & (p_z_hi == zh))
            R_panel, ratio_digi = p_R[mp], p_ratio[mp]

            mb = ((b_lam_lo == ll) & (b_lam_hi == lh)
                  & (b_z_lo == zl) & (b_z_hi == zh))
            bsel = bsel_interpolator(b_R[mb], b_ratio[mb], beff, chi_o)

            sel = prj.sigma_prj(R_panel / V.H, lob_rep, zob_rep, bsel, channel="sum")
            rnd = prj.sigma_prj(R_panel / V.H, lob_rep, zob_rep,
                                lambda th: beff, channel="sum")
            ratio_model = sel / rnd

            resid = ratio_model / ratio_digi - 1.0
            all_resid.append(resid)
            print(f"lam[{ll:.0f},{lh:.0f}) z[{zl:.2f},{zh:.2f}) "
                  f"lob_rep={lob_rep:6.1f} beff={beff:5.2f}  "
                  f"frac resid: min={resid.min():+.3f} max={resid.max():+.3f} "
                  f"med={np.median(resid):+.3f}")

            ax.plot(R_panel, ratio_digi, "o", ms=4, color="k",
                    label="Costanzi+26 Fig 6 (Sigma_prj ratio, digitized)")
            ax.plot(R_panel, ratio_model, "-", color="C0",
                    label="our SigmaPrj, fed digitized b_sel(theta)")
            ax.set_xscale("log")
            ax.set_title(f"$\\lambda\\in[{ll:.0f},{lh:.0f})\\ z\\in[{zl:.2f},{zh:.2f})$",
                         fontsize=9)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel(r"$R\ [h^{-1}{\rm cMpc}]$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\langle\Sigma^{\rm prj}\rangle_{\rm sel}/"
                      r"\langle\Sigma^{\rm prj}\rangle_{\rm RND}$")
    fig.suptitle("Digitized b_sel(theta) fed into our SigmaPrj, vs digitized "
                 "Sigma_prj ratio (both Costanzi+26 Fig 6)", y=1.005)
    fig.tight_layout()
    out = REPO / "docs" / "_static" / "validation" / "bsel_into_sigma_prj_check.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nwrote {out}")

    all_resid = np.concatenate(all_resid)
    print(f"\noverall ({len(shared)} richness bins x 3 z, "
          f"{all_resid.size} points): median |frac resid| = "
          f"{np.median(np.abs(all_resid)):.3f}, "
          f"max |frac resid| = {np.max(np.abs(all_resid)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
