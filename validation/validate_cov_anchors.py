r"""V8: physical anchors for the Gaussian :math:`\Delta\Sigma` covariance.

None of this uses ``cluster-lensing-cov``. It asks whether the numbers are
right, not whether they agree with someone else's.

A. **Shot noise x shape noise, in other units.** The code works per
   steradian with :math:`\Omega_{\rm ann} = A_{\rm ann}/\chi_h^2`. Here the
   same quantity is rebuilt from the *physical sample*: ``n_s`` in
   arcmin^-2, the annulus in arcmin^2 (:math:`\Omega\,(180\cdot60/\pi)^2`)
   and the halo count :math:`N_{\rm halo} = n_h\,4\pi f_{\rm sky}`, giving
   :math:`{\rm Var}[\Delta\Sigma_i] = \langle\Sigma_{\rm crit}\rangle^2
   \sigma_\gamma^2 / (n_s^{\rm arcmin}\,A^{\rm arcmin}_i\,N_{\rm halo})`.
   A missing :math:`\chi_h^2`, or a mixed-up unit, shows up here at any
   :math:`\chi_h`.

B. **Monte-Carlo stack.** Seeded, deterministic. Each realisation draws
   Poisson source counts per annulus for the whole stack and the mean
   tangential noise, and the estimator variance over realisations is
   compared with the code, within its statistical error.

C. **Where the line-of-sight integral lives.** The converged integral
   includes :math:`\ell` below Wu et al.'s cut :math:`1/\theta_{\max}`.
   This reports the fraction of each bin's ``lss_lss`` that comes from
   :math:`\ell < 1/\theta_{\max}` and from :math:`\ell < 10`, where the
   flat-sky Limber approximation is not trusted. If the missing part were
   all at :math:`\ell < 10` the cut would be harmless; it is reported, not
   assumed.

Run::

    python validation/validate_cov_anchors.py

Needs the V0 snapshot only for sample numbers (counts, chi_h, noise).
Exits nonzero if A or B fail.

Tolerances: A is algebra, 1e-9. B has a statistical error of
:math:`\sqrt{2/R}` on a variance from R realisations (R = 4000: 2.2%), and
the Poisson-weighting bias :math:`1/N_i`; the tolerance is 4 sigma plus
that bias, ~10%.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _clc_reference as ref  # noqa: E402
from clenspy.covariance.deltasigma import DeltaSigmaGaussianCovariance  # noqa: E402

ARCMIN_PER_RAD = 180.0 * 60.0 / np.pi
N_REAL = 4000
SEED = 20261008


def zero(ell):
    return np.zeros_like(np.asarray(ell, float))


def sample(d, case="thin0"):
    """Sample numbers from the snapshot: chi_h, counts, sigma_crit, noises."""
    p = f"{case}_"
    chi = float(d[p + "chi_h"])
    counts = float(d[p + "counts"])
    n_s = ref.REF_SOURCES["n_src_arcmin"]                       # arcmin^-2
    sigma_gamma = ref.REF_SOURCES["sigma_gamma"]
    n_s_sr = n_s * ARCMIN_PER_RAD**2
    shape_noise = float(d[p + "shape_noise_Sigma"])             # Sigma_crit^2 sigma^2 / n_s_sr
    sigma_crit = np.sqrt(shape_noise * n_s_sr) / sigma_gamma    # recovered, Msun/Mpc^2
    n_halo = counts
    n_h = n_halo / (4.0 * np.pi * ref.F_SKY)                    # per sr
    return dict(chi=chi, n_s=n_s, sigma_gamma=sigma_gamma, sigma_crit=sigma_crit,
                shape_noise=shape_noise, n_halo=n_halo, n_h=n_h)


def anchor_a(s, rp):
    print("A. shot_shape vs the physical-sample formula (arcmin units)")
    ok = True
    for chi in (1.0, 300.0, s["chi"], 2500.0):
        cov = DeltaSigmaGaussianCovariance(
            rp, chi, ref.F_SKY, zero, zero, zero, s["n_h"], s["shape_noise"])
        for exact in (True, False):
            if not exact and chi != s["chi"]:
                continue
            c = cov if exact else DeltaSigmaGaussianCovariance(
                rp, chi, ref.F_SKY, zero, zero, zero, s["n_h"], s["shape_noise"],
                exact_shot_shape=False)
            got = np.diag(c.components()["shot_shape"])
            area_mpc2 = np.pi * (rp[1:] ** 2 - rp[:-1] ** 2)
            area_arcmin2 = area_mpc2 / chi**2 * ARCMIN_PER_RAD**2
            want = (s["sigma_crit"] ** 2 * s["sigma_gamma"] ** 2
                    / (s["n_s"] * area_arcmin2 * s["n_halo"]))
            err = np.max(np.abs(got / want - 1.0))
            tol = 1e-9 if exact else 5e-3
            good = err < tol
            ok &= good
            print(f"   chi_h = {chi:7.1f} Mpc  {'closed form' if exact else 'quadrature ':11s}"
                  f"  max |code/physical - 1| = {err:9.2e}   (tol {tol:.0e})  "
                  f"{'PASS' if good else 'FAIL'}")
    return ok


def anchor_b(s, rp):
    print(f"\nB. Monte-Carlo stack, {N_REAL} realisations, seed {SEED}")
    rng = np.random.default_rng(SEED)
    area_arcmin2 = np.pi * (rp[1:] ** 2 - rp[:-1] ** 2) / s["chi"] ** 2 * ARCMIN_PER_RAD**2
    mean_n = s["n_s"] * area_arcmin2 * s["n_halo"]               # sources per annulus, whole stack
    n_draw = rng.poisson(mean_n, size=(N_REAL, mean_n.size))
    # mean of N iid Normal(0, sigma^2) is Normal(0, sigma^2 / N): exact, no source loop needed
    est = s["sigma_crit"] * rng.normal(0.0, s["sigma_gamma"] / np.sqrt(np.maximum(n_draw, 1)))
    mc_var = est.var(axis=0, ddof=1)
    cov = DeltaSigmaGaussianCovariance(
        rp, s["chi"], ref.F_SKY, zero, zero, zero, s["n_h"], s["shape_noise"])
    code = np.diag(cov.components()["shot_shape"])
    ratio = mc_var / code
    stat = np.sqrt(2.0 / (N_REAL - 1))
    bias = 1.0 / mean_n                                          # E[1/N] ~ (1 + 1/N)/N for Poisson
    dev = np.abs(ratio - (1.0 + bias)) / stat
    good = bool(np.all(dev < 4.0))
    print(f"   sources per annulus: {mean_n.min():.0f} .. {mean_n.max():.0f}")
    print(f"   MC variance / code: min {ratio.min():.3f} max {ratio.max():.3f}"
          f"   (statistical error {stat:.3f}); max deviation {dev.max():.1f} sigma  "
          f"{'PASS' if good else 'FAIL'}")
    return good


def anchor_c(d, case="thin0"):
    print(f"\nC. Where the lss_lss integral lives ({case}); bins 0 / 3 / 6 / 9 / 11")
    p = f"{case}_"
    rp = ref.RP_EDGES
    ell = d["ell"]
    chi = float(d[p + "chi_h"])
    lx = {k: np.log(np.abs(d[p + k])) for k in ("C_hh", "C_SS", "C_hS")}
    fs = [(lambda v: (lambda x: np.exp(np.interp(np.log(np.asarray(x, float)), np.log(ell), v))))(lx[k])
          for k in ("C_hh", "C_SS", "C_hS")]
    n_h = 1.0 / float(d[p + "shot_noise"])
    sn = float(d[p + "shape_noise_Sigma"])

    def diag(ell_lo):
        return np.diag(DeltaSigmaGaussianCovariance(
            rp, chi, ref.F_SKY, *fs, n_h, sn, k_range=(ell_lo / chi, ell[-1] / chi),
            n_k=16384).components()["lss_lss"])

    full = diag(ell[0])
    below10 = 1.0 - diag(10.0) / full
    theta_max = rp[1:] / chi
    # fraction below each bin's own 1/theta_max, via the same integral
    below_cut = np.array([1.0 - diag(1.0 / theta_max[i])[i] / full[i] for i in range(rp.size - 1)])
    idx = (0, 3, 6, 9, 11)
    print("   bin                       " + "".join(f"{i:9d}" for i in idx))
    print("   ell < 10  (flat-sky doubtful)" + "".join(f"{below10[i]:9.1%}" for i in idx))
    print("   ell < 1/theta_max (her cut) " + "".join(f"{below_cut[i]:9.1%}" for i in idx))
    print("   (reported, not asserted: where the part her range drops lives)")
    return True


def main():
    d = np.load(ref.SNAPSHOT)
    s = sample(d)
    rp = ref.RP_EDGES
    ok = anchor_a(s, rp)
    ok &= anchor_b(s, rp)
    anchor_c(d)
    print("\nV8", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
