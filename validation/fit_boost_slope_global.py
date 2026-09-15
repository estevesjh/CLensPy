r"""Joint fit of ONE shared boost_slope (s) across independent richness
bins, testing "the s_fit should be the same across the binnings" against
Matteo Costanzi's own b_sel(theta)/b_eff reference curve (Fig 6 webdigitized,
~3% precision, some outliers).

Per (lam, z) bin: free delta_i, shared s. A_s(s) recomputed inside the
residual (A_s and s are NOT independent -- see plan-bsel-stable-closure.md
sec 2). b_small/b_large are NOT fit independently -- they are both driven
by the single delta_i through the real closure algebra.

    python validation/fit_boost_slope_global.py
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
from clenspy.observables import ClusterCounts  # noqa: E402
from clenspy.selection import EmgParams, SelBiasEngine, SelectionFunction  # noqa: E402
from clenspy.selection.scaling_relation import HodMor  # noqa: E402
from clenspy.cosmology import TinkerMassFunction  # noqa: E402
from clenspy.cosmology.bias import BiasModel  # noqa: E402
from clenspy.cosmology.growth import growth_factor  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

DATA_FILES = [
    (REPO / "validation" / "data" / "costanzi_bsel_lam30_45.csv", 5),
    (REPO / "validation" / "data" / "costanzi_bsel_more_bins.csv", 6),
]


def load_all():
    rows = []
    for f, skip in DATA_FILES:
        d = np.loadtxt(f, delimiter=",", skiprows=skip)
        rows.append(d)
    return np.concatenate(rows, axis=0)


def custom_b_eff(lam_lo, lam_hi, z_lo, z_hi):
    """Same recipe as V.b_eff_table(), single bespoke (lam, z) bin."""
    sel = SelectionFunction(
        np.array([lam_lo, lam_hi]), np.array([z_lo, z_hi]),
        HodMor.buzzard(), EmgParams.from_y3_table(), sigma_z=5e-3,
    )
    ln_mass = np.log(np.logspace(13.0, 15.7, 64))
    z_grid = np.linspace(max(0.0, z_lo - 0.05), z_hi + 0.05, 48)
    tmf_counts = TinkerMassFunction(cosmo=V.COSMO, zvec=np.linspace(0.0, 1.0, 21))
    counts_grid = tmf_counts.dndlnm(np.exp(ln_mass) / V.H, z_grid)
    counts_interp = RegularGridInterpolator(
        (ln_mass, z_grid), counts_grid, bounds_error=False, fill_value=None
    )

    def mass_function(lnm, z):
        m_h, zz = np.broadcast_arrays(np.exp(np.asarray(lnm, float)), np.asarray(z, float))
        pts = np.column_stack((np.log(m_h.ravel()), zz.ravel()))
        return np.asarray(counts_interp(pts)).reshape(m_h.shape)

    counts = ClusterCounts(ln_mass, z_grid, mass_function, sel, V.COSMO,
                           omega=lambda z: np.full_like(np.asarray(z, float), np.pi))
    bm = BiasModel(cosmo=V.COSMO)
    m_phys = np.exp(ln_mass) / V.H
    sigma0 = np.asarray(bm.sigma_tophat(m_phys, z=0.0), float)
    growth = np.asarray(growth_factor(z_grid, V.COSMO), float)
    nu = 1.686 / (sigma0[:, None] * growth[None, :])
    bias_grid = np.asarray(bm.bias_at_nu(nu), float)
    beff = float(np.asarray(counts.average(bias_grid)).squeeze())
    lob_rep = float(np.asarray(counts.mean_richness()).squeeze())
    zob_rep = float(np.asarray(counts.mean_redshift()).squeeze())
    return lob_rep, zob_rep, beff


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
    print(f"{len(bins)} bins loaded")

    per_bin = []  # (lob_rep, zob_rep, beff, theta_lob, theta, bsel_digi, D, P1, I1, I2)
    for (ll, lh, zl, zh) in bins:
        m = (lam_lo == ll) & (lam_hi == lh) & (z_lo == zl) & (z_hi == zh)
        R_b, bsel_b = R[m], bsel_digi[m]

        aligned = np.isclose(V.LAMBDA_EDGES[:-1], ll).any()
        if aligned:
            i = int(np.where(np.isclose(V.LAMBDA_EDGES[:-1], ll))[0][0])
            j = int(np.where(np.isclose(V.Z_EDGES[:-1], zl))[0][0])
            lob_rep = float(lam_ij[i, j]); zob_rep = float(zrep_ij[i, j])
            beff = float(b_eff_ij[i, j])
        else:
            lob_rep, zob_rep, beff = custom_b_eff(ll, lh, zl, zh)

        chi_o = float(engine.chi(zob_rep))
        D_A = chi_o / (1.0 + zob_rep)
        theta = (R_b / V.H) / D_A
        theta_lob = engine._theta_lob(lob_rep, zob_rep)

        P1, I1, I2 = engine.operators(lob_rep, zob_rep)
        D = I2 - I1
        per_bin.append(dict(ll=ll, lh=lh, zl=zl, zh=zh, lob_rep=lob_rep,
                            zob_rep=zob_rep, beff=beff, theta_lob=theta_lob,
                            theta=theta, bsel_digi=bsel_b, P1=P1, I1=I1, I2=I2, D=D))
        print(f"  lam[{ll:.0f},{lh:.0f}) z[{zl:.2f},{zh:.2f}) "
              f"lob_rep={lob_rep:.1f} zob_rep={zob_rep:.3f} beff={beff:.3f}  "
              f"({m.sum()} pts)")

    from clenspy.selection import SigmoidBias

    def sigma_from(bin_, s, delta):
        beff = bin_["beff"]
        A_s = (bin_["P1"] + beff * bin_["I2"] - s * beff * bin_["I1"]) / bin_["D"]
        b_small = beff + delta * A_s
        b_large = beff * (1.0 + s * delta)
        curve = SigmoidBias(lob=bin_["lob_rep"], zob=bin_["zob_rep"],
                            theta_lambda=bin_["theta_lob"], b_small=b_small,
                            b_large=b_large, damping=engine.damping,
                            theta0_frac=engine.theta0_frac)
        return curve(bin_["theta"]) / beff

    n_bins = len(per_bin)

    def resid(params):
        s = params[0]
        deltas = params[1:]
        out = []
        for bin_, d in zip(per_bin, deltas):
            out.append(sigma_from(bin_, s, d) - bin_["bsel_digi"])
        return np.concatenate(out)

    x0 = [0.13] + [0.5] * n_bins
    res = least_squares(resid, x0=x0)
    s_fit = res.x[0]
    deltas_fit = res.x[1:]

    print(f"\nGLOBAL joint fit: shared s_fit = {s_fit:.4f}  "
          f"(pipeline default boost_slope = {engine.boost_slope})")
    total_resid = np.median(np.abs(resid(res.x)))
    print(f"median |residual| (all bins, all points) = {total_resid:.4f}\n")
    for bin_, d in zip(per_bin, deltas_fit):
        r = sigma_from(bin_, s_fit, d) - bin_["bsel_digi"]
        med = np.median(np.abs(r))
        print(f"  lam[{bin_['ll']:.0f},{bin_['lh']:.0f}) z[{bin_['zl']:.2f},{bin_['zh']:.2f}) "
              f"delta_fit={d:.3f}  med|resid|={med:.4f}")

    # per-bin independent s fit, for comparison -- is a single shared s
    # actually a good model, or does the "best s" itself drift bin to bin?
    print("\nPer-bin INDEPENDENT (s_i, delta_i) fits (2 dof each, for comparison):")
    for bin_ in per_bin:
        def resid_i(params):
            return sigma_from(bin_, params[0], params[1]) - bin_["bsel_digi"]
        r = least_squares(resid_i, x0=[0.13, 0.5])
        med = np.median(np.abs(resid_i(r.x)))
        print(f"  lam[{bin_['ll']:.0f},{bin_['lh']:.0f}) z[{bin_['zl']:.2f},{bin_['zh']:.2f}) "
              f"s_i={r.x[0]:.3f} delta_i={r.x[1]:.3f}  med|resid|={med:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(13, 15), sharex=True)
    for row, (ll, lh) in enumerate(sorted(set((b["ll"], b["lh"]) for b in per_bin))):
        cols = [b for b in per_bin if b["ll"] == ll and b["lh"] == lh]
        cols.sort(key=lambda b: b["zl"])
        for ax, bin_, d in zip(axes[row], cols,
                               [deltas_fit[per_bin.index(b)] for b in cols]):
            theta_fine = np.geomspace(bin_["theta"].min(), bin_["theta"].max(), 200)
            bin_fine = dict(bin_, theta=theta_fine)
            ax.plot(bin_["theta"], bin_["bsel_digi"], "o", ms=4, color="k",
                    label="Matteo (digitized, ~3%)")
            ax.plot(theta_fine, sigma_from(bin_fine, s_fit, d), "-", color="C0",
                    label=f"global fit (s={s_fit:.2f})")
            ax.plot(theta_fine, sigma_from(bin_fine, engine.boost_slope, d), "--",
                    color="C3", alpha=0.6, label=f"old default (s={engine.boost_slope})")
            ax.set_xscale("log")
            ax.set_title(f"$\\lambda\\in[{ll:.0f},{lh:.0f})$ "
                         f"$z\\in[{bin_['zl']:.2f},{bin_['zh']:.2f})$", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\theta$ [rad]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$b_{\rm sel}/b_{\rm eff}$")
    fig.tight_layout()
    out = REPO / "docs" / "_static" / "validation" / "bsel_boost_slope_global_fit.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
