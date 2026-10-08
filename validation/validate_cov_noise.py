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

Tolerance for the matching convention: our formula integrated with her
rule (the 100-point linspace trapezoid from :math:`z_h + 0.1`, see
``_clc_reference.trapz_her_nodes``), so the raw ratio is set by two exact constants: :math:`c` (hers 3e5 km/s,
ours 299792.458), which enters as :math:`\Sigma_{\rm crit}^2 \propto c^4`
and gives the constant -2.77e-3 seen in the raw columns, and the
:math:`p(z_s)` normalisation (her ``arange`` drops the last 0.01, 2.8e-7).
The **exact** rows remove both constants analytically, as V1 does, and must
agree to 1e-6. The shipped `shape_noise_Sigma` integrates with Gauss-Legendre
(converged), so it differs from hers by *her* trapezoid error, which is
reported and held to a measured budget.
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

#: Budget for the shipped (converged) value vs hers: her 100-node trapezoid error
#: on <Sigma_crit>^2/f_src. Measured 1.6e-3..4.2e-3 across the four cases; the
#: per-integral trapezoid errors are f: 2e-4, <Sigma_crit>(0.1): 5e-4..2e-3
#: (kernels/lensing_kernel.py N_ZS_GL note). 1e-2 is 2.4x the worst measured.
BUDGET_HER_QUADRATURE = 1e-2
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


def shape_noise(survey, lk, z_h, *, cut, conditional):
    """Our formula with HER quadrature (100-node trapezoid on her nodes).

    The shipped `LimberProjector.shape_noise_Sigma` uses Gauss-Legendre and
    differs from this by her quadrature error; see the rows below.
    """
    n_src_sr = survey.n_src_arcmin / ARCMIN_TO_RAD**2
    f = float(ref.trapz_her_nodes(lk, z_h, cut, "f")[0])
    mean = float(ref.trapz_her_nodes(lk, z_h, cut, "sc")[0])
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
    rows = []
    for case in snap["case_names"]:
        p = f"{case}_"
        z_h = 0.5 * float(np.sum(snap[p + "zbin"]))
        hers = float(snap[p + "shape_noise_Sigma"])
        cur = shape_noise(survey, lk, z_h, cut=0.01, conditional=False)[0]
        cut = shape_noise(survey, lk, z_h, cut=CUT_HERS, conditional=False)[0]
        con, f, _ = shape_noise(survey, lk, z_h, cut=CUT_HERS, conditional=True)
        shipped = proj.shape_noise_Sigma(z_h)
        rows.append((case, con / hers - 1.0, shipped / hers - 1.0))
        print(f"  {case:6s} {z_h:6.3f} {f:10.4f} {cur / hers - 1:10.2%} {cut / hers - 1:10.2%}"
              f" {con / hers - 1:11.2e} {f**2 - 1:9.2%} {shipped / hers - 1:10.2e}")
    print("\n  raw ratios below still contain the two exact constants (c^4, p(z) norm);"
          " the exact rows remove them")

    # exact constants removed: c^4 and the p(z) normalisation (1/f_src)
    c4 = (ref.C_OURS / ref.C_HERS) ** 4
    p_ratio = her_pz_norm(ref.REF_SOURCES["zs_min"], ref.REF_SOURCES["zs_max"]) / survey.norm
    print(f"\n  exact: c^4 ratio - 1 = {c4 - 1:.3e}, p(z) norm ratio - 1 = {p_ratio - 1:.2e}"
          f" removed; tolerance {TOL_EXACT:.0e}")
    print(f"\n  matching convention (our formula, her 100-node trapezoid) vs hers, exact: "
          f"tolerance {TOL_EXACT:.0e}")
    for case, err_match, err_shipped in rows:
        err = (1.0 + err_match) * p_ratio / c4 - 1.0
        good = abs(err) < TOL_EXACT
        if not good:
            failed.append(("exact", case, err))
        print(f"  {case:6s} conditional, cut 0.1, her rule vs hers, exact: {err:9.2e}  "
              f"{'PASS' if good else 'FAIL'}")
    print(f"\n  shipped shape_noise_Sigma (Gauss-Legendre, converged) vs hers, exact"
          f"  = her quadrature error; budget {BUDGET_HER_QUADRATURE:.0e}")
    for case, _, err_shipped in rows:
        err = (1.0 + err_shipped) * p_ratio / c4 - 1.0
        good = abs(err) < BUDGET_HER_QUADRATURE
        if not good:
            failed.append(("shipped", case, err))
        print(f"  {case:6s} shipped vs hers, exact: {err:9.2e}  "
              f"{'PASS' if good else 'FAIL'}")

    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("\nV3 PASS (shipped shape_noise_Sigma = her cddbb2a convention; `old` is"
          " the pre-fix convention, tabulated above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
