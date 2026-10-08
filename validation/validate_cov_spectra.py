r"""V2: the Limber spectra :math:`C^{hh}, C^{\Sigma\Sigma}, C^{h\Sigma}` against her code.

Same inputs on both sides: her linear :math:`P(k,z)` (the V0 snapshot), her
comoving mean density, the same bias and the same :math:`\ell` grid (her
8000-point log grid; the snapshot keeps every 4th point). What can differ
is the Limber implementation: slab decomposition, windows, and the lensing
kernel behind :math:`F_\Sigma`.

Run::

    python validation/validate_cov_spectra.py

Reported as the maximum relative difference over the :math:`\ell` range that
carries covariance weight (:math:`10^1 \le \ell \le 10^5`, the range the
radial bins of the matched configuration probe), and over the full grid.

Verdicts. ``C^{hh}`` is a hard check (1e-8; measured ~1e-10): same slabs,
same :math:`P(k,z)`, same window. ``C^{\Sigma\Sigma}`` and ``C^{h\Sigma}``
are reported as INFO, because they are not reproducible between two sound
implementations, for a reason measured rather than assumed:

* V1 shows :math:`q_\Sigma(z_{\rm lss}; z_h)` agrees with her kernel to 1e-12
  *on her nodes*. She stores it on a 100-point grid (dz = 0.029) and
  interpolates linearly; ours evaluates it exactly at each slab.
* But :math:`q_\Sigma` is ill-conditioned for foreground slabs
  (:math:`z_{\rm lss} < z_h`): the integrand has a pole where the source
  redshift crosses :math:`z_h`, so its value depends on node placement.
  Re-running ours with the slab width :math:`\Delta z` = 0.1, 0.05, 0.02,
  0.01 changes :math:`C^{\Sigma\Sigma}` by 40-98% (non-monotonically), while
  :math:`C^{h\Sigma}`, which only samples the lens bin, moves by ~1%.

So agreement of :math:`C^{\Sigma\Sigma}` with her is not a meaningful target
until :math:`F_\Sigma` is defined for foreground slabs without the pole.

The second table measures that directly: the maximum relative change of
:math:`C^{\Sigma\Sigma}` and :math:`C^{h\Sigma}` over the band when the slab
width ``DZ_SLAB`` goes 0.05 -> 0.01 and 0.02 -> 0.01, with the shipped
(signed) :math:`q_\Sigma`. Reported as INFO: a converged definition would
move by < 1%, the signed one does not (see
``validation/covariance_review_REPORT.md``, task 3).

NOTE: ``clenspy.kernels.limber`` is shadowed by the exported ``limber``
function in the package ``__init__``, so the module is reached through
``sys.modules``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _clc_reference as ref  # noqa: E402
import validate_cov_noise as v3  # noqa: E402

BAND = (1e1, 1e5)


def rel(ours, hers, ell):
    band = (ell >= BAND[0]) & (ell <= BAND[1])
    r = np.abs(ours / hers - 1.0)
    return r[band].max(), np.median(r[band]), r.max()


def main():
    snap = ref.load_snapshot()
    survey, lk, proj = v3.build(snap, n_ell=8000)
    ell_her = snap["ell"]
    n_stride = round(proj.ell.size / ell_her.size)
    if proj.ell.size % ell_her.size or not np.allclose(proj.ell[::n_stride], ell_her):
        print("ell grids differ; V2 needs the same grid")
        return 2
    ell = ell_her
    print(f"ell grid: {proj.ell.size} points, snapshot every {n_stride}th;"
          f" band {BAND[0]:.0e}..{BAND[1]:.0e}\n")
    failed = False
    print(f"{'case':6s} {'spectrum':8s} {'max (band)':>11s} {'median':>9s} {'max (all)':>10s}")
    for case in snap["case_names"]:
        p = f"{case}_"
        zlo, zhi = (float(x) for x in snap[p + "zbin"])
        z_h = 0.5 * (zlo + zhi)
        bias = float(snap[p + "bias"])
        top = min(2.0, ref.REF_SOURCES["zs_max"] - 0.1)
        ours = {
            "C_hh": proj.C_ell_hh(zlo, zhi, bias)[::n_stride],
            "C_SS": proj.C_ell_SS(0.1, top, z_h)[::n_stride],
            "C_hS": proj.C_ell_hS(zlo, zhi, bias, z_h)[::n_stride],
        }
        for name, c in ours.items():
            mx, md, mall = rel(c, snap[p + name], ell)
            hard = name == "C_hh"
            verdict = ("PASS" if mx < 1e-8 else "FAIL") if hard else "INFO"
            failed |= hard and mx >= 1e-8
            print(f"{case:6s} {name:8s} {mx:11.2e} {md:9.2e} {mall:10.2e}  {verdict}")

    print("\nslab-width convergence (shipped signed q_Sigma), max |C(dz)/C(0.01) - 1| in band")
    limber_mod = sys.modules["clenspy.kernels.limber"]
    dz_default = limber_mod.DZ_SLAB
    band = (proj.ell >= BAND[0]) & (proj.ell <= BAND[1])
    print(f"{'case':6s} {'spectrum':8s} {'dz=0.05':>10s} {'dz=0.02':>10s}")
    try:
        for case in [c for c in snap["case_names"] if c.startswith("thin")]:
            p = f"{case}_"
            zlo, zhi = (float(x) for x in snap[p + "zbin"])
            z_h = 0.5 * (zlo + zhi)
            bias = float(snap[p + "bias"])
            top = min(2.0, ref.REF_SOURCES["zs_max"] - 0.1)
            res = {}
            for dz in (0.05, 0.02, 0.01):
                limber_mod.DZ_SLAB = dz
                res[dz] = (proj.C_ell_SS(0.1, top, z_h)[band],
                           proj.C_ell_hS(zlo, zhi, bias, z_h)[band])
            for k, name in enumerate(("C_SS", "C_hS")):
                d = [np.max(np.abs(res[dz][k] / res[0.01][k] - 1.0)) for dz in (0.05, 0.02)]
                print(f"{case:6s} {name:8s} {d[0]:10.2e} {d[1]:10.2e}  INFO")
    finally:
        limber_mod.DZ_SLAB = dz_default
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
