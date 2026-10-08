r"""V1: lensing kernels against ``cluster-lensing-cov`` at the pin.

Rung V1 of the covariance ladder. Compares `clenspy.kernels.LensingKernel`
with her ``LensingKernel`` **after** commit cddbb2a, whose conventions are:

* :math:`\langle\Sigma_{\rm crit}\rangle(z_h)` is the **conditional**
  average over sources behind the lens, normalised by
  :math:`\int p\,dz_s` over the same range;
* both :math:`\langle\Sigma_{\rm crit}\rangle` and :math:`f_{\rm src}`
  start the source integral at :math:`z_h + 0.1` (was 0.01);
* :math:`\langle\Sigma_{\rm crit}^{-1}\rangle` and :math:`q_\Sigma` keep
  :math:`z_l + 0.01`.

Where the conventions differ, the quantity the covariance actually uses is
checked (``mean_sigma_crit(z, 0.1) / f_src_behind(z, 0.1)``, as
`LimberProjector.shape_noise_Sigma` forms it since the cddbb2a fix, with
``MIN_LENS_SOURCE_SEPARATION_NOISE = 0.1``) and must pass tightly; the
method *defaults* (0.01 cut, unnormalised) are a different quantity by
definition and are reported as INFO.

Tolerance budget (all measured, none fitted)
--------------------------------------------
Both codes evaluate these integrals on **identical** nodes
(``linspace(max(z + cut, zs_min), zs_max, 100)``, trapezoid), so the only
legitimate differences are:

* :math:`c`: hers 3e5 km/s, ours 299792.458 -- an exact factor
  :math:`(c_{\rm ours}/c_{\rm hers})^{\pm2}` removed analytically;
* the :math:`p(z_s)` normalisation: hers trapezoid on
  ``arange(zs_min, zs_max, 0.01)`` (drops the last 0.01), ours 601 nodes --
  an exact constant ratio, computed here from both definitions and
  removed analytically.

After those two exact corrections the residual is floating-point only, so
the matched tolerance is 1e-6 -- three orders above the measured residual
and three below any convention effect (the smallest, zs_max 3 -> 5, is
1e-3 on :math:`f_{\rm src}`).

Run::

    python validation/validate_cov_kernels.py [--plot]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _clc_reference as ref  # noqa: E402

TOL_MATCHED = 1e-6


def her_pz_norm(zs_min, zs_max):
    """Her Survey.norm, by her rule (a deliberate second copy)."""
    s = ref.REF_SOURCES
    z = np.arange(zs_min, zs_max, 0.01)
    shape = z ** s["m"] * np.exp(-((z / s["z_star"]) ** s["beta"]))
    return np.trapezoid(shape, x=z)


def main(plot=False):
    from clenspy.kernels import LensingKernel
    from clenspy.kernels.lensing_kernel import MIN_LENS_SOURCE_SEPARATION_NOISE

    snap = ref.load_snapshot()
    v = ref.Verdicts("V1 kernels")
    cosmo = ref.ours_cosmology()
    survey = ref.ours_survey()
    lk = LensingKernel(survey, cosmo)

    c2 = (ref.C_OURS / ref.C_HERS) ** 2          # Sigma_crit(ours/hers)
    p_ratio = her_pz_norm(0.0, 3.0) / survey.norm  # p_ours / p_hers
    v.info("p(z) normalisation ratio ours/hers - 1", p_ratio - 1.0,
           "her arange drops [2.99, 3]; removed exactly")
    v.info("c^2 ratio ours/hers - 1", c2 - 1.0, "removed exactly")

    # -- distances: same astropy model on both sides
    chi = cosmo.comoving_distance(snap["chi_z"]).value
    v.check("chi(z) max rel diff", np.max(np.abs(chi[1:] / snap["chi_val"][1:]
                                                 - 1.0)), 1e-10)

    # -- <Sigma_crit^-1>(z_l): 0.01 cut on both sides
    zl = snap["kz_zl"]
    mine = lk.mean_inverse_sigma_crit(zl)
    dev = np.max(np.abs(mine * c2 / p_ratio / snap["kz_val"] - 1.0))
    v.check("<Sc^-1>(z_l) on her grid", dev, TOL_MATCHED)

    # -- q_Sigma(z_l; z_h): dimensionless, c cancels
    for zh, row in zip(snap["ks_zh"], snap["ks_val"]):
        q = lk.q_sigma(snap["ks_zl"], zh)
        good = np.abs(row) > 1e-3 * np.max(np.abs(row))
        dev = np.max(np.abs(q[good] / p_ratio / row[good] - 1.0))
        v.check(f"q_Sigma(z_l; z_h={zh:.3f})", dev, TOL_MATCHED)

    # -- <Sigma_crit>(z_h) and f_src(z_h): the conventions that changed
    zh = snap["zh_grid"]
    zh = zh[zh + 0.1 < survey.zs_max]
    sel = np.isin(snap["zh_grid"], zh)
    msc_her, fsrc_her = snap["mean_sigma_crit"][sel], snap["fsrc_behind"][sel]

    # the method defaults (0.01 cut, unnormalised) are a different quantity
    # by definition; the shape noise does not use them
    shipped_msc = lk.mean_sigma_crit(zh)                  # 0.01, unnormalised
    shipped_f = lk.f_src_behind(zh)                       # 0.01
    # what LimberProjector.shape_noise_Sigma uses (MIN_..._NOISE = 0.1)
    cut = MIN_LENS_SOURCE_SEPARATION_NOISE
    f01 = lk.f_src_behind(zh, min_separation=cut)
    matched_msc = lk.mean_sigma_crit(zh, min_separation=cut) / f01

    r = shipped_msc / c2 / msc_her
    v.info("<Sc>(z_h) method default (0.01, unnormalised)/hers, min",
           r.min(), f"spans [{r.min():.3f}, {r.max():.3f}]; not her quantity")
    v.check("<Sc>(z_h) as used by shape noise (0.1, / f_src)",
            np.max(np.abs(matched_msc / c2 / msc_her - 1.0)), TOL_MATCHED)
    r = shipped_f / p_ratio / fsrc_her
    v.info("f_src(z_h) method default (cut 0.01)/hers, max", r.max(),
           f"spans [{r.min():.3f}, {r.max():.3f}]; kernel cut, not noise cut")
    v.check("f_src(z_h) as used by shape noise (cut 0.1)",
            np.max(np.abs(f01 / p_ratio / fsrc_her - 1.0)), TOL_MATCHED)

    # -- decompose the as-shipped <Sc> gap into its two causes
    cut_only = lk.mean_sigma_crit(zh, min_separation=0.1) / c2 / msc_her
    v.info("<Sc>: normalisation alone (= f_src(0.1)), min", cut_only.min(),
           "unnormalised ours is f_src x conditional mean")
    gap = shipped_msc / lk.mean_sigma_crit(zh, min_separation=0.1)
    v.info("<Sc>: cut 0.01 vs 0.1 alone, max ratio", gap.max(),
           "the near-lens spike the 0.1 cut removes")

    # -- zs_max 3 -> 5 (her new default): measured, not absorbed
    lk5 = LensingKernel(ref.ours_survey(zs_max=5.0), cosmo)
    zh5 = snap["zh_grid"]
    p5 = her_pz_norm(0.0, 5.0) / lk5.survey.norm
    f5 = lk5.f_src_behind(zh5, min_separation=cut)
    m5 = lk5.mean_sigma_crit(zh5, min_separation=0.1) / f5
    v.check("zs_max=5: <Sc> matched", np.max(np.abs(
        m5 / c2 / snap["mean_sigma_crit_zs5"] - 1.0)), TOL_MATCHED)
    v.check("zs_max=5: f_src matched", np.max(np.abs(
        f5 / p5 / snap["fsrc_behind_zs5"] - 1.0)), TOL_MATCHED)
    eff_m = snap["mean_sigma_crit_zs5"][sel] / msc_her
    eff_f = snap["fsrc_behind_zs5"][sel] / fsrc_her
    v.info("her zs_max 3->5: <Sc> change, max|.-1|",
           np.max(np.abs(eff_m - 1.0)), "z_h <= 1.2")
    v.info("her zs_max 3->5: f_src change, max|.-1|",
           np.max(np.abs(eff_f - 1.0)))

    status = v.report()
    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="white", context="talk", font_scale=0.8)
        ref.FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].plot(zh, msc_her * 1e-12, color="black", lw=3, label="hers (cddbb2a)")
        ax[0].plot(zh, matched_msc / c2 * 1e-12, color="firebrick", ls="--", lw=2,
                   label="clenspy, shape-noise convention")
        ax[0].plot(zh, shipped_msc / c2 * 1e-12, color="grey", ls=":", lw=2,
                   label="clenspy, method default")
        ax[0].set(xlabel=r"$z_h$",
                  ylabel=r"$\langle\Sigma_{\rm crit}\rangle$ [$M_\odot$/pc$^2$]")
        ax[1].plot(zh, fsrc_her, color="black", lw=3, label="hers (cut 0.1)")
        ax[1].plot(zh, shipped_f, color="grey", ls=":", lw=2, label="clenspy default (cut 0.01)")
        ax[1].set(xlabel=r"$z_h$", ylabel=r"$f_{\rm src}(z_h)$")
        for a in ax:
            a.legend(frameon=False, fontsize=12)
        sns.despine(fig)
        fig.tight_layout()
        out = ref.FIG_DIR / "cov_kernels_vs_clc.png"
        fig.savefig(out, dpi=140)
        print(f"wrote {out}")
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    sys.exit(main(**vars(parser.parse_args())))
