r"""V3: the two noise terms of the Gaussian covariance, against ``cluster-lensing-cov``.

* halo shot noise :math:`1/n_h` -- pure algebra, one footprint number
  driving :math:`f_{\rm sky}` and the area;
* shape noise on :math:`\Sigma`,
  :math:`\langle\Sigma_{\rm crit}\rangle^2\sigma_\gamma^2 / (n_s f_{\rm src})`.

The second is where her Sep 2026 commit ``cddbb2a`` changed the
convention. Three conventions are evaluated on our side so the table shows
which one her number is:

``old``
    what `LimberProjector.shape_noise_Sigma` did before the cddbb2a fix:
    unnormalised :math:`\langle\Sigma_{\rm crit}\rangle` (so it already
    contains the behind-the-lens fraction) divided by :math:`f_{\rm src}`,
    lens-source cut 0.01;
``cut 0.1``
    the same, with her 0.1 cut on both :math:`\langle\Sigma_{\rm crit}\rangle`
    and :math:`f_{\rm src}`;
``conditional, cut 0.1``
    :math:`\langle\Sigma_{\rm crit}\rangle/f_{\rm src}`, the average over
    sources behind the lens only, then divided by :math:`f_{\rm src}`
    again -- her new convention.

``current`` is `LimberProjector.shape_noise_Sigma` as shipped; since the
fix it *is* the conditional convention, so its column must equal the last
one and is checked against hers with the same tolerance.

Because ``old`` carries one factor :math:`f_{\rm src}` in the numerator
and one in the denominator, it differs from her convention by
:math:`f_{\rm src}^2` (plus the cut). That is the number to look at.

Run::

    python validation/validate_cov_noise.py

Exits nonzero if the shot noise, the matching convention or the shipped
`shape_noise_Sigma` fails.

Tolerance for the matching convention: same formula on identical nodes
(both sides use the 100-point linspace trapezoid from :math:`z_h + 0.1`),
so the raw ratio is set by two exact constants: :math:`c` (hers 3e5 km/s,
ours 299792.458), which enters as :math:`\Sigma_{\rm crit}^2 \propto c^4`
and gives the constant -2.77e-3 seen in the raw columns, and the
:math:`p(z_s)` normalisation (her ``arange`` drops the last 0.01, 2.8e-7).
The raw rows use a loose tolerance, 3x the 100-vs-800-node quadrature
spread (a scale for "same convention"); the **exact** rows remove both
constants analytically, as V1 does, and must agree to 1e-6.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _clc_reference as ref  # noqa: E402
from clenspy.kernels.lensing_kernel import LensingKernel  # noqa: E402
from clenspy.kernels.limber import ARCMIN_TO_RAD, LimberProjector  # noqa: E402

CUT_HERS = 0.1
TOL_EXACT = 1e-6


def her_pz_norm(zs_min, zs_max):
    """Her Survey.norm, by her rule (as in validate_cov_kernels.py)."""
    s = ref.REF_SOURCES
    z = np.arange(zs_min, zs_max, 0.01)
    return np.trapezoid(z ** s["m"] * np.exp(-((z / s["z_star"]) ** s["beta"])), x=z)


def build(snap, n_ell=64):
    cosmo = ref.ours_cosmology()
    survey = ref.ours_survey()
    lk = LensingKernel(survey, cosmo)
    proj = LimberProjector(
        chi=lambda z: cosmo.comoving_distance(z).value,
        pk_lin=ref.pk_from_snapshot(snap),
        rho_mean0=ref.her_rho_mean(),
        q_sigma=lk.q_sigma,
        mean_sigma_crit=lk.mean_sigma_crit,
        f_src_behind=lk.f_src_behind,
        sigma_gamma=survey.sigma_gamma,
        n_src_arcmin2=survey.n_src_arcmin,
        n_ell=n_ell,   # the noise terms do not use the ell grid; V2 passes 8000
    )
    return survey, lk, proj


def shape_noise(survey, lk, z_h, *, cut, conditional, n_nodes=None):
    n_src_sr = survey.n_src_arcmin / ARCMIN_TO_RAD**2
    f = float(lk.f_src_behind(z_h, min_separation=cut, n_nodes=n_nodes)[0])
    mean = float(np.ravel(lk.mean_sigma_crit(z_h, cut, n_nodes))[0])
    if conditional:
        mean /= f
    return survey.sigma_gamma**2 / (n_src_sr * f) * mean**2, f, mean


def main():
    snap = ref.load_snapshot()
    survey, lk, proj = build(snap)
    failed = []

    print("shot noise 1/n_h  [sr]   ours = 4 pi f_sky / counts")
    for case in snap["case_names"]:
        p = f"{case}_"
        counts = float(snap[p + "counts"])
        ours = proj.shot_noise_h(counts, 4.0 * np.pi * ref.F_SKY)
        err = ours / float(snap[p + "shot_noise"]) - 1.0
        good = abs(err) < 1e-12
        if not good:
            failed.append(("shot", case, err))
        print(f"  {case:6s} ours/hers - 1 = {err:9.2e}  {'PASS' if good else 'FAIL'}")

    print("\nshape noise on Sigma  [(Msun/Mpc^2)^2 sr]   ours/hers - 1")
    print(f"  {'case':6s} {'z_h':>6s} {'f_src(0.1)':>10s}"
          f" {'old':>10s} {'cut 0.1':>10s} {'cond., 0.1':>11s} {'f_src^2':>9s} {'current':>10s}")
    spread = 0.0
    rows = []
    for case in snap["case_names"]:
        p = f"{case}_"
        z_h = 0.5 * float(np.sum(snap[p + "zbin"]))
        hers = float(snap[p + "shape_noise_Sigma"])
        cur = shape_noise(survey, lk, z_h, cut=0.01, conditional=False)[0]
        cut = shape_noise(survey, lk, z_h, cut=CUT_HERS, conditional=False)[0]
        con, f, _ = shape_noise(survey, lk, z_h, cut=CUT_HERS, conditional=True)
        # quadrature spread of the matching convention: 100 vs 800 nodes
        a = shape_noise(survey, lk, z_h, cut=CUT_HERS, conditional=True, n_nodes=100)[0]
        b = shape_noise(survey, lk, z_h, cut=CUT_HERS, conditional=True, n_nodes=800)[0]
        spread = max(spread, abs(a / b - 1.0))
        shipped = proj.shape_noise_Sigma(z_h)
        rows.append((case, con / hers - 1.0, shipped / hers - 1.0))
        print(f"  {case:6s} {z_h:6.3f} {f:10.4f} {cur / hers - 1:10.2%} {cut / hers - 1:10.2%}"
              f" {con / hers - 1:11.2e} {f**2 - 1:9.2%} {shipped / hers - 1:10.2e}")
    tol = max(3.0 * spread, 1e-3)
    print(f"\n  quadrature spread of the matching convention (100 vs 800 nodes): {spread:.1e};"
          f" tolerance {tol:.1e}")
    for case, err, err_shipped in rows:
        good = abs(err) < tol
        if not good:
            failed.append(("shape", case, err))
        print(f"  {case:6s} conditional, cut 0.1 vs hers: {err:9.2e}  {'PASS' if good else 'FAIL'}")
        good = abs(err_shipped) < tol
        if not good:
            failed.append(("shipped", case, err_shipped))
        print(f"  {case:6s} shipped shape_noise_Sigma vs hers: {err_shipped:9.2e}  "
              f"{'PASS' if good else 'FAIL'}")

    # exact constants removed: c^4 and the p(z) normalisation (1/f_src)
    c4 = (ref.C_OURS / ref.C_HERS) ** 4
    p_ratio = her_pz_norm(ref.REF_SOURCES["zs_min"], ref.REF_SOURCES["zs_max"]) / survey.norm
    print(f"\n  exact: c^4 ratio - 1 = {c4 - 1:.3e}, p(z) norm ratio - 1 = {p_ratio - 1:.2e}"
          f" removed; tolerance {TOL_EXACT:.0e}")
    for case, _, err_shipped in rows:
        err = (1.0 + err_shipped) * p_ratio / c4 - 1.0
        good = abs(err) < TOL_EXACT
        if not good:
            failed.append(("exact", case, err))
        print(f"  {case:6s} shipped shape_noise_Sigma vs hers, exact: {err:9.2e}  "
              f"{'PASS' if good else 'FAIL'}")

    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("\nV3 PASS (shipped shape_noise_Sigma = her cddbb2a convention; `old` is"
          " the pre-fix convention, tabulated above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
