r"""Diagnostic (not a rung): the foreground pole in :math:`q_\Sigma`, and three candidate definitions.

For :math:`z_{\rm lss} < z_h` the source integral of :math:`q_\Sigma` (keyed on
the LSS slab, as in Wu et al. 2019 eq. F_Sigma and her ``calc_kernel_Sigma``)
crosses :math:`z_s = z_h`, where :math:`\Sigma_{\rm crit}(z_s, z_h)` has a simple
pole and changes sign. Candidates evaluated:

``A``  signed, sources from :math:`z_{\rm lss} + 0.01` (hers and ours, shipped);
``B``  sources behind the halo only, from :math:`\max(z_{\rm lss}, z_h) + 0.01`;
``C``  the shape-noise source sample: from :math:`\max(z_{\rm lss} + 0.01, z_h + 0.1)`,
       divided by :math:`f_{\rm src}(z_h; 0.1)`.

Prints the integrand across the pole, the node-count dependence of ``A``, the
slab-width convergence of :math:`C^{\Sigma\Sigma}` and :math:`C^{h\Sigma}`, and the
covariance diagonal at the largest radial bin. Findings are in
``covariance_review_REPORT.md`` (task 3). Takes ~30 s. Run::

    python validation/diagnose_qsigma_pole.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _clc_reference as ref  # noqa: E402
import validate_cov_noise as v3  # noqa: E402
from clenspy.kernels.lensing_kernel import sigma_crit_comoving  # noqa: E402
from clenspy.covariance.deltasigma import DeltaSigmaGaussianCovariance  # noqa
import clenspy.kernels  # noqa
LIM = sys.modules["clenspy.kernels.limber"]

snap = ref.load_snapshot()
survey, lk, proj = v3.build(snap, n_ell=2000)
cosmo = lk.cosmo


def q_variant(zl_arr, zh, mode, n_nodes=100):
    """A: signed (hers); B: sources behind z_h+0.01; C: behind z_h+0.1, / f_src(0.1)."""
    zl_arr = np.atleast_1d(np.asarray(zl_arr, float))
    out = np.zeros(zl_arr.shape)
    f01 = lk.f_src_behind(zh, min_separation=0.1, n_nodes=n_nodes).item()
    for i, zl in enumerate(zl_arr):
        if mode == "A":
            lo = zl + 0.01
        elif mode == "B":
            lo = max(zl + 0.01, zh + 0.01)
        else:
            lo = max(zl + 0.01, zh + 0.1)
        lo = max(lo, survey.zs_min)
        if lo >= survey.zs_max:
            continue
        zs = np.linspace(lo, survey.zs_max, n_nodes)
        sc_h = sigma_crit_comoving(zh, zs, cosmo, signed=True)
        sc_l = sigma_crit_comoving(float(zl), zs, cosmo)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = sc_h / sc_l
        r = np.where(np.isfinite(r), r, 0.0)
        out[i] = np.trapezoid(survey.pz_src(zs) * r, x=zs)
        if mode == "C":
            out[i] /= f01
    return out


print("(i) the pole: integrand p(zs) Sc(zs,zh)/Sc(zs,zl) near zs = zh, zh=0.425, zl=0.1528")
zh, zl = 0.425, 0.1528
for zs in (0.40, 0.42, 0.424, 0.4249, 0.4251, 0.426, 0.43, 0.45):
    v = survey.pz_src(np.array([zs])) * sigma_crit_comoving(zh, zs, cosmo, signed=True) \
        / sigma_crit_comoving(zl, zs, cosmo)
    print(f"   zs={zs:7.4f}  integrand={v.item(): .4e}")
print("   q_A(zl=0.1528; zh=0.425) vs node count (signed definition):")
for n in (100, 101, 200, 400, 800, 1600, 3200):
    print(f"     n={n:5d}  q={q_variant(zl, zh, 'A', n).item(): .4f}")
print("   same for B (behind halo): ", [round(q_variant(zl, zh, 'B', n).item(), 4)
                                     for n in (100, 400, 1600)])

print("\n    q_Sigma(z_lss) on a fine z grid, signed A, n=100 vs n=400:")
for zh in (0.275, 0.425, 0.575):
    zg = np.linspace(0.1, zh - 0.02, 8)
    a1, a4 = q_variant(zg, zh, "A", 100), q_variant(zg, zh, "A", 400)
    print(f"   zh={zh}: zl={np.round(zg, 3)}")
    print(f"      n=100 {np.round(a1, 3)}")
    print(f"      n=400 {np.round(a4, 3)}")
    print(f"      B     {np.round(q_variant(zg, zh, 'B'), 4)}")

print("\n(iii) slab-width convergence of C_SS and C_hS, rel. change vs DZ=0.01, band 10..1e5")
band = (proj.ell >= 10) & (proj.ell <= 1e5)
rp = ref.RP_EDGES
results = {}
for mode in ("A", "B", "C"):
    proj.q_sigma = (lambda m: (lambda z, h: q_variant(z, h, m)))(mode)
    for case in ("thin0", "thin1", "thin2"):
        p = f"{case}_"
        zlo, zhi = (float(x) for x in snap[p + "zbin"])
        zh = 0.5 * (zlo + zhi)
        bias = float(snap[p + "bias"])
        top = min(2.0, ref.REF_SOURCES["zs_max"] - 0.1)
        res = {}
        for dz in (0.1, 0.05, 0.02, 0.01):
            LIM.DZ_SLAB = dz
            res[dz] = (proj.C_ell_SS(0.1, top, zh), proj.C_ell_hS(zlo, zhi, bias, zh),
                       proj.C_ell_hh(zlo, zhi, bias))
        LIM.DZ_SLAB = 0.1
        ref_ss, ref_hs = res[0.01][0], res[0.01][1]
        line = []
        for dz in (0.1, 0.05, 0.02):
            dss = np.max(np.abs(res[dz][0][band] / ref_ss[band] - 1))
            dhs = np.max(np.abs(res[dz][1][band] / ref_hs[band] - 1))
            line.append(f"dz={dz}: SS {dss:7.2%} hS {dhs:7.2%}")
        print(f"   {mode} {case}: " + " | ".join(line))
        results[(mode, case)] = res

print("\n    covariance diagonal, largest radial bin (11), DZ=0.1 and 0.01; ")
print("    using our spectra, her shot noise, our (shipped) shape noise; converged ell")
for case in ("thin0", "thin1", "thin2"):
    p = f"{case}_"
    zlo, zhi = (float(x) for x in snap[p + "zbin"])
    zh = 0.5 * (zlo + zhi)
    chi = float(snap[p + "chi_h"])
    n_h = 1.0 / float(snap[p + "shot_noise"])
    sn = proj.shape_noise_Sigma(zh)
    ell = proj.ell
    base = None
    for mode in ("A", "B", "C"):
        for dz in (0.1, 0.01):
            css, chs, chh = results[(mode, case)][dz]
            ii = lambda c: (lambda x: np.interp(np.log(x), np.log(ell), c))  # noqa
            cov = DeltaSigmaGaussianCovariance(rp, chi, ref.F_SKY, ii(chh), ii(css), ii(chs),
                                               n_h, sn)
            comp = cov.components()
            diag = {t: comp[t][11, 11] for t in comp}
            tot = sum(diag.values())
            lss = diag["lss_lss"] + diag["shot_lss"] + diag["cross"]
            if base is None:
                base = tot
            print(f"   {case} {mode} dz={dz:4.2f}: total/total(A,0.1)={tot / base:7.4f}"
                  f"  lss share={lss / tot:6.1%}  cross share={diag['cross'] / tot:6.2%}")
