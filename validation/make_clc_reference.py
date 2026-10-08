r"""V0: run ``cluster-lensing-cov`` at the pin and snapshot its numbers.

Rung V0 of the covariance ladder (see ``covariance_review_REPORT.md``).
Every other ``validate_cov_*`` script compares clenspy against the file this
writes, ``validation/data/clc_cddbb2a_reference.npz``, so they run without
her repository and against a reference that cannot drift.

What is snapshotted (her code, commit `_clc_reference.PIN`, matched
configuration `_clc_reference.REF_COSMO` / `REF_SOURCES`):

* her linear :math:`P(k, z)` as :math:`P(k,0)\,D^2(z)`, with the
  scale-independence of :math:`D` asserted and the interpolation error
  measured, so clenspy can be driven by the *same* spectrum;
* the lensing kernels on her own grids: :math:`\langle\Sigma_{\rm
  crit}^{-1}\rangle(z_l)`, :math:`q_\Sigma(z_l; z_h)`, the new normalised
  :math:`\langle\Sigma_{\rm crit}\rangle(z_h)` and
  :math:`f_{\rm src}(z_h)` (0.1 cut), at ``zs_max`` 3 and 5;
* for three thin halo slices and the DES Y1 wide bin: her spectra
  :math:`C^{\Sigma\Sigma}, C^{hh}, C^{h\Sigma}` (every 4th point of her
  8000-point grid), both noise levels, and the Gaussian covariance split
  into all **five** terms (three runs with her ``cosmic_shear_no_shot`` /
  ``halo_shot_noise_only`` switches), plus two runs with her
  :math:`\ell` integration widened/refined, for the tolerance budget;
* her :math:`\gamma_t` covariance (``cov_gammat``) for one slice, with
  :math:`C^{\kappa\kappa}` and the kappa shape noise;
* her counts / sample variance for one bin, for the units of
  :math:`\sigma_W`;
* her Tinker mass function and two Abacus covariance files, for V6.

NOTE: her ``CovDeltaSigma.__init__`` with ``halo_shot_noise_only=True``
writes ``self.cosmic_shear_no_shot == False`` -- a comparison, not an
assignment -- on an attribute that does not exist yet, so the constructor
itself raises ``AttributeError``. The two switches are set after
construction here; that reproduces her intent and changes no arithmetic.

Run::

    python validation/make_clc_reference.py

Needs ``cluster-lensing-cov`` (``$CLUSTER_LENSING_COV_DIR``); takes a few
minutes, dominated by her per-pair :math:`C^{\kappa\kappa}` recomputation in
``cov_gammat``. Exits nonzero if any internal consistency check fails.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _clc_reference as ref  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

#: Subsampling stride for her 8000-point ell grid when stored.
ELL_STRIDE = 4

ABACUS_CASES = (
    "z0.3/1e+14_2e+14_R0.1_100_nrp15",
    "z0.3/2e+14_4e+14_R0.1_100_nrp15",
)
#: Abacus "planck" cosmology (AbacusCosmos Planck 2015 box): her grafting
#: script uses FlatLambdaCDM(H0=67.3, Om0=0.314).
ABACUS_COSMO = dict(h=0.6726, OmegaM=0.314, OmegaDE=0.686, sigma8=0.83,
                    OmegaB=0.0491, ns=0.9652, tau=0.088, w0=-1.0, wa=0.0)
ABACUS_BOX_HINV = 720.0


def quiet(fn, *a, **kw):
    """Call her code with its print() chatter swallowed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def her_objects():
    from clens.util.parameters import CosmoParameters
    from clens.util.survey import Survey

    co = CosmoParameters(**ref.REF_COSMO)
    s = ref.REF_SOURCES

    def survey(zs_max):
        return quiet(Survey, z_star_src=s["z_star"], m_src=s["m"],
                     beta_src=s["beta"], sigma_gamma=s["sigma_gamma"],
                     n_src_arcmin=s["n_src_arcmin"], zs_min=s["zs_min"],
                     zs_max=zs_max)

    return co, survey(s["zs_max"]), survey(5.0)


#: See the note in `snapshot_pk`: the V2 C_hh tolerance.
PK_GROWTH_SCALE_DEP_TOL = 1e-8


def snapshot_pk(co, out, problems):
    from clens.ying.lineartheory import LinearTheory
    from clens.ying.param_w0wa import CosmoParams

    cy = CosmoParams(omega_M_0=co.OmegaM, omega_b_0=co.OmegaB,
                     omega_lambda_0=co.OmegaDE, h=co.h, sigma_8=co.sigma8,
                     n=co.ns, tau=co.tau)
    k = np.geomspace(1e-5, 1e5, 4001)
    p0 = LinearTheory(cosmo=cy, z=0.0).power_spectrum(k)
    zs = np.linspace(0.0, 3.0, 301)
    d2 = np.empty_like(zs)
    worst = 0.0
    probe = np.array([1e-4, 1e-2, 1.0, 1e2, 1e4])
    p0_probe = LinearTheory(cosmo=cy, z=0.0).power_spectrum(probe)
    for i, z in enumerate(zs):
        lin = LinearTheory(cosmo=cy, z=z)
        ratio = lin.power_spectrum(probe) / p0_probe
        d2[i] = ratio[2]
        worst = max(worst, float(np.max(np.abs(ratio / ratio[2] - 1.0))))
    # Threshold = the tightest downstream tolerance on anything built from
    # P(k, z): V2's hard C_hh check, 1e-8. A scale dependence eps of the
    # growth factor moves a Limber spectrum by at most eps, so below 1e-8
    # the factorised snapshot P(k,0) D^2(z) cannot fail that check. The
    # measured value (5.1e-9 at cddbb2a) is fp noise in her per-z growth
    # normalisation -- Eisenstein-Hu without neutrinos is scale independent
    # analytically -- and sits below the snapshot's own interpolation error
    # (1.2e-7, recorded as pk_interp_err). The former 1e-10 was tighter
    # than her arithmetic and made this script exit 1 on a sound snapshot.
    if worst > PK_GROWTH_SCALE_DEP_TOL:
        problems.append(f"P(k,z)/P(k,0) is scale dependent: {worst:.2e}")
    out.update(pk_k=k, pk_p0=p0, pk_z=zs, pk_d2=d2, pk_growth_scale_dep=worst)

    # interpolation error of the stored representation, measured off-grid
    pk = ref.pk_from_snapshot(out)
    k_off = np.sqrt(k[1:] * k[:-1])[::7]
    err = 0.0
    for z in (0.0, 0.137, 0.4251, 1.333, 2.71):
        direct = LinearTheory(cosmo=cy, z=z).power_spectrum(k_off)
        err = max(err, float(np.max(np.abs(pk(k_off, z) / direct - 1.0))))
    out["pk_interp_err"] = err
    print(f"P(k): growth scale-dependence {worst:.1e}, "
          f"snapshot interpolation error {err:.1e}")
    return cy


def snapshot_kernels(co, su, su5, out):
    from clens.lensing.lensing_kernel import LensingKernel

    lk = quiet(LensingKernel, co=co, su=su)
    lk5 = quiet(LensingKernel, co=co, su=su5)
    out["kz_zl"] = np.asarray(lk.kernel_z_interp.x)
    out["kz_val"] = np.asarray(lk.kernel_z_interp.y)

    zh = np.round(np.arange(0.10, 1.2001, 0.025), 4)
    out["zh_grid"] = zh
    out["mean_sigma_crit"] = np.array([lk.mean_Sigma_crit(z) for z in zh])
    out["fsrc_behind"] = np.array([lk.fsrc_behind_zh(z) for z in zh])
    out["mean_sigma_crit_zs5"] = np.array([lk5.mean_Sigma_crit(z) for z in zh])
    out["fsrc_behind_zs5"] = np.array([lk5.fsrc_behind_zh(z) for z in zh])

    zh_sigma = np.array([0.5 * sum(b) for b in ref.THIN_SLICES]
                        + [0.5 * sum(ref.WIDE_BIN)])
    ks = []
    for z in zh_sigma:
        quiet(lk.calc_kernel_Sigma, z)
        ks.append(np.asarray(lk.kernel_Sigma_z_interp.y))
    out["ks_zl"] = np.asarray(lk.kernel_Sigma_z_interp.x)
    out["ks_zh"] = zh_sigma
    out["ks_val"] = np.array(ks)
    chi_z = np.linspace(0.0, 3.0, 61)
    out["chi_z"] = chi_z
    out["chi_val"] = lk.chi(chi_z).value
    print("kernels: done")


def slice_counts(zmin, zmax):
    """Thin-slice counts at the wide bin's comoving number density."""
    cosmo = ref.ours_cosmology()

    def vol(a, b):
        return (cosmo.comoving_distance(b).value ** 3
                - cosmo.comoving_distance(a).value ** 3)

    return ref.WIDE_COUNTS * vol(zmin, zmax) / vol(*ref.WIDE_BIN)


def run_cov(co, su, zmin, zmax, counts, bias, *, mode="normal", bf=None):
    from clens.lensing.cov_DeltaSigma import CovDeltaSigma
    from clens.util.scaling_relation import PrecalculatedCountsBias

    sr = PrecalculatedCountsBias(lens_counts=counts, lens_bias=bias,
                                 fsky=ref.F_SKY)
    cds = CovDeltaSigma(co=co, su=su, sr=sr, fsky=ref.F_SKY,
                        cosmic_shear_no_shot=(mode == "no_shot"))
    if mode == "shot_only":
        # her __init__ crashes for halo_shot_noise_only=True (see NOTE), so
        # set the two switches after construction, as she intended
        cds.halo_shot_noise_only = True
        cds.cosmic_shear_no_shot = False
    if bf:
        for key, val in bf.items():
            setattr(cds.bf, key, val)
    quiet(cds.calc_cov, rp_min=ref.RP_EDGES[0], rp_max=ref.RP_EDGES[-1],
          n_rp=ref.RP_EDGES.size - 1, zh_min=zmin, zh_max=zmax,
          diag_only=False)
    return cds


def snapshot_cov_cases(co, su, out, problems):
    cases = [("thin%d" % i, lo, hi, slice_counts(lo, hi), ref.SLICE_BIAS)
             for i, (lo, hi) in enumerate(ref.THIN_SLICES)]
    cases.append(("wide", *ref.WIDE_BIN, ref.WIDE_COUNTS, ref.WIDE_BIAS))
    out["case_names"] = np.array([c[0] for c in cases])
    for name, lo, hi, counts, bias in cases:
        t0 = time.time()
        n = run_cov(co, su, lo, hi, counts, bias)
        a = run_cov(co, su, lo, hi, counts, bias, mode="no_shot")
        s = run_cov(co, su, lo, hi, counts, bias, mode="shot_only")
        edges = np.concatenate([n.rp_min_list, n.rp_max_list[-1:]])
        if not np.allclose(edges, ref.RP_EDGES, rtol=1e-12):
            problems.append(f"{name}: her rp edges differ from RP_EDGES")
        # the split must reassemble her own grouping exactly
        for label, whole, parts in (
            ("cosmic", n.cov_cosmic_shear,
             a.cov_cosmic_shear + s.cov_cosmic_shear),
            ("shape", n.cov_shape_noise,
             a.cov_shape_noise + s.cov_shape_noise),
        ):
            dev = np.max(np.abs(parts / whole - 1.0))
            if dev > 1e-10:
                problems.append(f"{name}: five-term split of {label} off "
                                f"by {dev:.2e}")
        # her matrices are in (Msun/pc^2)^2; store (Msun/Mpc^2)^2
        to_mpc = 1e24
        p = f"{name}_"
        out[p + "zbin"] = np.array([lo, hi])
        out[p + "counts"] = counts
        out[p + "bias"] = bias
        out[p + "chi_h"] = n.chi(0.5 * (lo + hi)).value
        out[p + "lss_lss"] = a.cov_cosmic_shear * to_mpc
        out[p + "lss_shape"] = a.cov_shape_noise * to_mpc
        out[p + "shot_lss"] = s.cov_cosmic_shear * to_mpc
        out[p + "shot_shape"] = s.cov_shape_noise * to_mpc
        out[p + "cross"] = n.cov_cross * to_mpc
        aps = n.aps
        out["ell"] = aps.ell[::ELL_STRIDE]
        out[p + "C_SS"] = aps.C_ell_Sigma[::ELL_STRIDE]
        out[p + "C_hh"] = aps.C_ell_h[::ELL_STRIDE]
        out[p + "C_hS"] = aps.C_ell_h_Sigma[::ELL_STRIDE]
        out[p + "shot_noise"] = aps.shot_noise
        out[p + "shape_noise_Sigma"] = aps.shape_noise_for_Sigma
        # interpolation error of the stride, measured on the dropped points
        lne = np.log(aps.ell)
        err = 0.0
        for c in (aps.C_ell_Sigma, aps.C_ell_h, aps.C_ell_h_Sigma):
            back = np.exp(np.interp(lne, lne[::ELL_STRIDE],
                                    np.log(c[::ELL_STRIDE])))
            err = max(err, float(np.max(np.abs(back / c - 1.0))))
        out[p + "stride_interp_err"] = err
        # her ell-integration settings, varied for the tolerance budget
        for tag, bf in (("fine", dict(dlnell=5e-4)),
                        ("wide", dict(scaling_for_ell_min=0.3,
                                      scaling_for_ell_max=300.0))):
            v = run_cov(co, su, lo, hi, counts, bias, bf=bf)
            out[p + f"var_{tag}_cosmic"] = v.cov_cosmic_shear * to_mpc
            out[p + f"var_{tag}_shape"] = v.cov_shape_noise * to_mpc
            out[p + f"var_{tag}_cross"] = v.cov_cross * to_mpc
        print(f"cov case {name} z=[{lo},{hi}] N={counts:.1f}: "
              f"{time.time() - t0:.1f}s")
    out["bf_defaults"] = np.array([1.0, 100.0, 1e-3])  # ell_min/max scale, dlnell


def snapshot_gammat(co, su, out):
    from clens.lensing.cov_gammat import Covgammat
    from clens.util.scaling_relation import PrecalculatedCountsBias

    lo, hi = ref.THIN_SLICES[1]
    counts = slice_counts(lo, hi)
    sr = PrecalculatedCountsBias(lens_counts=counts, lens_bias=ref.SLICE_BIAS,
                                 fsky=ref.F_SKY)
    cg = Covgammat(co=co, su=su, sr=sr, fsky=ref.F_SKY)
    arcmin = np.pi / (180.0 * 60.0)
    th_lo, th_hi, nth = 2.0 * arcmin, 60.0 * arcmin, 8
    t0 = time.time()
    quiet(cg.calc_cov_gammat_integration, thmin=th_lo, thmax=th_hi, nth=nth,
          zh_min=lo, zh_max=hi, lambda_min=None, lambda_max=None,
          diag_only=False)
    out["gt_zbin"] = np.array([lo, hi])
    out["gt_counts"] = counts
    out["gt_theta_edges"] = np.geomspace(th_lo, th_hi, nth + 1)
    out["gt_cov_cosmic"] = cg.cov_cosmic_shear
    out["gt_cov_shape"] = cg.cov_shape_noise
    out["gt_C_kk"] = cg.aps.C_ell_kappa[::ELL_STRIDE]
    out["gt_C_hh"] = cg.aps.C_ell_h[::ELL_STRIDE]
    out["gt_shape_noise"] = cg.aps.shape_noise
    out["gt_shot_noise"] = cg.aps.shot_noise
    print(f"gamma_t: {time.time() - t0:.1f}s")


def snapshot_counts(co, out):
    from clens.util.cluster_counts import ClusterCounts
    from clens.util.scaling_relation import FiducialScalingRelation

    cc = ClusterCounts(cosmo_parameters=co,
                       scaling_relation=FiducialScalingRelation())
    counts, sv, bias, _, _ = quiet(cc.calc_counts, zmin=ref.WIDE_BIN[0],
                                   zmax=ref.WIDE_BIN[1], lambda_min=20,
                                   lambda_max=30,
                                   survey_area_sq_deg=ref.SURVEY_AREA_DEG2)
    cosmo = ref.ours_cosmology()
    vol = ref.F_SKY * 4.0 * np.pi / 3.0 * (
        cosmo.comoving_distance(ref.WIDE_BIN[1]).value ** 3
        - cosmo.comoving_distance(ref.WIDE_BIN[0]).value ** 3)
    out.update(cnt_counts=counts, cnt_sv=sv, cnt_bias=bias, cnt_vol=vol,
               cnt_r_eff_mpc=(3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0),
               cnt_sigma_w=np.sqrt(sv) / (bias * counts))
    print(f"counts: N={counts:.1f} b={bias:.3f} "
          f"sigma_W={out['cnt_sigma_w']:.5f}")


def snapshot_abacus(out, root):
    from clens.util.parameters import CosmoParameters
    from clens.ying.density_w0wa import Density
    from clens.ying.halostat import HaloStat
    from clens.ying.lineartheory import LinearTheory
    from clens.ying.param_w0wa import CosmoParams

    co = CosmoParameters(**ABACUS_COSMO)
    cy = CosmoParams(omega_M_0=co.OmegaM, omega_b_0=co.OmegaB,
                     omega_lambda_0=co.OmegaDE, h=co.h, sigma_8=co.sigma8,
                     n=co.ns, tau=co.tau)
    z = 0.3
    dlnm = 0.01
    lnm = np.arange(np.log(5e13), np.log(1e15), dlnm)  # Msun, h-free
    den = Density(cosmo=cy)
    hs = quiet(HaloStat, cosmo=cy, z=z, DELTA_HALO=200.0,
               rho_mean_0=den.rho_mean_z(0.0), mass=np.exp(lnm), dlogm=dlnm)
    out["ab_lnm_hfree"] = lnm
    out["ab_dndlnm"] = hs.mass_function * np.exp(lnm)  # 1/Mpc^3
    out["ab_bias_tinker10"] = hs.bias_function
    k = np.geomspace(1e-4, 1e3, 1400)
    out["ab_pk_k"] = k
    out["ab_pk_z0"] = LinearTheory(cosmo=cy, z=0.0).power_spectrum(k)
    out["ab_pk_z"] = LinearTheory(cosmo=cy, z=z).power_spectrum(k)
    out["ab_cosmo"] = np.array([ABACUS_COSMO[k] for k in
                                ("h", "OmegaM", "sigma8", "OmegaB", "ns")])
    out["ab_z"] = z
    out["ab_box_hinv"] = ABACUS_BOX_HINV
    base = root / "data" / "abacus_scatter0" / "720_planck_10percent"
    for i, case in enumerate(ABACUS_CASES):
        r, m = np.loadtxt(base / case / "mean_DeltaSigma.dat", unpack=True)
        out[f"ab{i}_case"] = np.array(case)
        out[f"ab{i}_rp_hinv"] = r
        out[f"ab{i}_mean_h"] = m  # h Msun/pc^2
        out[f"ab{i}_cov_h"] = np.loadtxt(base / case / "cov_DeltaSigma.dat")
    print("abacus: done")


def main():
    root = ref.import_clens()
    head = ref.git("rev-parse", "HEAD", repo=root)
    print(f"cluster-lensing-cov at {root}\nHEAD {head}")
    if head != ref.PIN:
        print(f"HEAD is not the pin {ref.PIN}")
        return 1
    problems: list[str] = []
    out: dict = {"pin": np.array(ref.PIN)}

    co, su, su5 = her_objects()
    snapshot_pk(co, out, problems)
    snapshot_kernels(co, su, su5, out)
    snapshot_cov_cases(co, su, out, problems)
    snapshot_gammat(co, su, out)
    snapshot_counts(co, out)
    snapshot_abacus(out, root)

    ref.DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ref.SNAPSHOT, **out)
    log = ref.git("log", "-1", "--format=%H %ad %an %s", "--date=short",
                  repo=root)
    ref.PIN_FILE.write_text(
        "hywu/cluster-lensing-cov reference pin for the covariance ladder\n"
        f"{log}\n"
        "https://github.com/hywu/cluster-lensing-cov/commit/" + ref.PIN + "\n"
        "Regenerate: python validation/make_clc_reference.py\n"
        "Shim: np.trapz = np.trapezoid (NumPy 2.4 removed the alias).\n"
        f"numpy {np.__version__}\n")
    size = ref.SNAPSHOT.stat().st_size / 1024
    print(f"wrote {ref.SNAPSHOT} ({size:.0f} kB) and {ref.PIN_FILE}")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
