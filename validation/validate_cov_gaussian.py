r"""V4: Gaussian :math:`\Delta\Sigma` covariance, term by term, against ``cluster-lensing-cov``.

Feeds `DeltaSigmaGaussianCovariance` **her** spectra and noises (the V0
snapshot, ``make_clc_reference.py``) and compares each of the five terms
with her matrices. Spectra are shared, so this isolates the covariance
integral from the Limber step (V2) and from the kernels (V1).

The comparison uses **her ell range**, entry by entry, through the public
``DeltaSigmaGaussianCovariance(..., ell_range="wu2019")`` option. Her
``_calc_C_ell_integration`` integrates each pair of radial bins over

.. math::
    \ell \in \left[\frac{1}{\theta_{\max}},\;\frac{100}{\theta_{\min}}\right],
    \qquad \theta_{\max} = \max(\theta_{\max,i},\theta_{\max,j}),\;
    \theta_{\min} = \min(\theta_{\min,i},\theta_{\min,j}),

with step :math:`d\ln\ell = 10^{-3}`. That range is a *choice*, not the
integral: it drops the :math:`\ell < 1/\theta_{\max}` tail, so for the
line-of-sight terms at small :math:`r_p` her matrices are not converged
(her own widened-range run, stored in the snapshot, moves ``cross`` by
tens of percent). It is reproduced here so the two integrators can be
compared on the same integral; ``--converged`` also reports how far her
default range sits from the converged value.

Run::

    python validation/validate_cov_gaussian.py [--converged]

Exits nonzero if any term misses its tolerance.

TOLERANCE BUDGET (per term, error normalised by :math:`\sqrt{C_{ii}C_{jj}}`
so near-zero off-diagonal entries do not dominate):

* same integral on the same ``np.arange`` grid (ours read off a cumulative
  sum, hers ``np.trapz`` per pair); the residual is the
  re-interpolation of her stored spectra (every 4th point, log-log) and
  her ``interp1d`` (linear) of the full table: measured ~1e-5. The
  ``shot_shape`` term has no spectrum and agrees to 1e-12, which pins the
  grid itself (start, step, exclusive end) to hers;
* ``1e-3`` is the tolerance: ten times the largest measured difference,
  and one decade below the smallest physically meaningful effect checked in
  V8 (the 0.9% truncation bias on ``shot_shape``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _clc_reference as ref  # noqa: E402
from clenspy.covariance.deltasigma import DeltaSigmaGaussianCovariance  # noqa: E402

TERMS = ("lss_lss", "lss_shape", "shot_lss", "shot_shape", "cross")
TOL = 1e-3


def loglog(ell, c):
    lx, ly = np.log(ell), np.log(np.abs(c))
    sign = np.sign(c[0])
    return lambda x: sign * np.exp(np.interp(np.log(np.asarray(x, float)), lx, ly))


def scaled_error(ours, hers):
    """max |ours - hers| / sqrt(C_ii C_jj), plus the diagonal-only relative error."""
    di = np.diag(hers)
    full = np.max(np.abs(ours - hers) / np.sqrt(np.outer(di, di)))
    diag = np.max(np.abs(np.diag(ours) / di - 1.0))
    return full, diag


def ours_with_her_range(d, case, rp):
    """Her per-pair range, through the public ``ell_range="wu2019"`` option.

    ``shot_shape`` is then the quadrature over her range (the option's
    default), since her truncation of that term is part of the comparison.
    """
    p = f"{case}_"
    ell = d["ell"]
    chi = float(d[p + "chi_h"])
    n_h = 1.0 / float(d[p + "shot_noise"])
    shape_noise = float(d[p + "shape_noise_Sigma"])
    spectra = [loglog(ell, d[p + k]) for k in ("C_hh", "C_SS", "C_hS")]
    return DeltaSigmaGaussianCovariance(
        rp, chi, ref.F_SKY, *spectra, n_h, shape_noise, ell_range="wu2019",
    ).components()


def converged(d, case, rp):
    """Ours over a wide fixed range, for the her-vs-converged comparison."""
    p = f"{case}_"
    ell = d["ell"]
    spectra = [loglog(ell, d[p + k]) for k in ("C_hh", "C_SS", "C_hS")]
    chi = float(d[p + "chi_h"])
    return DeltaSigmaGaussianCovariance(
        rp, chi, ref.F_SKY, *spectra, 1.0 / float(d[p + "shot_noise"]),
        float(d[p + "shape_noise_Sigma"]),
        k_range=(1e-4, ell[-1] / chi), n_k=16384,
    ).components()


def main():
    show_converged = "--converged" in sys.argv
    d = np.load(ref.SNAPSHOT)
    rp = ref.RP_EDGES
    failed = []
    print(f"V4  pin {str(d['pin'])[:7]}   tolerance {TOL:.0e} (error / sqrt(C_ii C_jj))\n")
    print(f"{'case':6s} {'term':10s} {'full':>10s} {'diagonal':>10s}  verdict")
    for case in d["case_names"]:
        ours = ours_with_her_range(d, case, rp)
        for t in TERMS:
            full, diag = scaled_error(ours[t], d[f"{case}_{t}"])
            ok = full < TOL
            if not ok:
                failed.append((case, t, full))
            print(f"{case:6s} {t:10s} {full:10.2e} {diag:10.2e}  {'PASS' if ok else 'FAIL'}")
    if show_converged:
        print("\nher default range vs the converged integral (ours, wide range),"
              " diagonal ours/hers - 1, bins 0 / 5 / 11:")
        for case in d["case_names"]:
            conv = converged(d, case, rp)
            for t in TERMS:
                r = np.diag(conv[t]) / np.diag(d[f"{case}_{t}"]) - 1.0
                print(f"{case:6s} {t:10s} {r[0]:9.1%} {r[5]:9.1%} {r[11]:9.1%}")
    if failed:
        print(f"\nFAILED: {len(failed)} term(s) outside tolerance")
        return 1
    print("\nall terms within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
