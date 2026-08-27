r"""`LensingKernel` against the frozen covariance reference.

``cluster-lensing-cov`` froze its Stage-A kernel inputs to
``validation/frozen_inputs/kernels.npz`` precisely so that a refactor could
be shown to be equivalent. That file holds ``q_plain``
(:math:`\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)`), ``q_sigma``,
``mean_sigma_crit`` and ``f_src``, computed by the exemplar's own
`LensingKernel` for a DES Y1 source population.

NOTE: two of the four quantities are **logarithmically divergent** and
exist only relative to a convention -- the minimum lens-source separation,
and (through the endpoint's trapezoid weight) the node count. This script
therefore pins conventions, not just numbers: it runs at the exemplar's
0.01 and 100 nodes, which is what makes the comparison meaningful at all.

NOTE: a residual of exactly 0.1383% is **expected** on the two quantities
carrying :math:`c^2`. The reference uses :math:`c = 3\times10^5` km/s;
`clenspy` uses the exact 299792.458. The ratio is

.. math::
    (299792.458 / 3\times10^5)^2 = 0.99861687,

and `clenspy` is the one that is right. The script reports the residual
both raw and after removing that factor, and requires the corrected
residual to be small -- so it would catch a real disagreement while not
failing on a rounded constant.

Run
---
::

    python validation/validate_lensing_kernel.py
    python validation/validate_lensing_kernel.py --plot

Needs ``cluster-lensing-cov`` checked out; set ``CLUSTER_LENSING_COV_DIR``
if it is not at the default path. Exits nonzero on failure.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from clenspy.kernels import LensingKernel
from clenspy.survey import Survey

DEFAULT_REPO = Path.home() / "Documents/Dev/github/cluster-lensing-cov"
REPO = Path(os.environ.get("CLUSTER_LENSING_COV_DIR", DEFAULT_REPO))
FROZEN = REPO / "validation" / "frozen_inputs" / "kernels.npz"

FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "_static" / "validation"

#: The reference's cosmology: ``configs/des_y1.json`` -> h = 0.7,
#: OmegaM = 0.3, OmegaDE = 0.7. The exemplar builds a ``w0waCDM`` with no
#: radiation, so ``Tcmb0 = 0`` here matches it.
REF_COSMO = dict(H0=70.0, Om0=0.3, Tcmb0=0.0, Ob0=0.05)

#: The reference's source population, same config file.
REF_SOURCES = dict(z_star=0.74, m=1.68, beta=2.33, sigma_gamma=0.3,
                   n_src_arcmin=6.28, zs_min=0.0, zs_max=3.0)

#: :math:`(c_{\rm exact} / c_{\rm rounded})^2`, the expected offset on any
#: quantity carrying :math:`c^2 / 4\pi G`.
C_RATIO_SQUARED = (299792.458 / 3.0e5) ** 2

#: Tolerance on the residual *after* removing `C_RATIO_SQUARED`. Set to
#: catch a real disagreement while allowing the quadrature-layout
#: differences that remain between two independent implementations.
TOL = 3e-3


def report(name, mine, ref, c_power=0):
    """Print the residual, before and after the constants correction.

    ``c_power`` is the power of :math:`c^2` the quantity carries: +1 for a
    :math:`\Sigma_{\rm crit}`, -1 for its inverse, 0 for a probability or
    a ratio. The expected offset is `C_RATIO_SQUARED` to that power.
    """
    mine, ref = np.ravel(np.asarray(mine)), np.ravel(np.asarray(ref))
    good = np.abs(ref) > 0
    raw = mine[good] / ref[good] - 1.0
    factor = C_RATIO_SQUARED ** c_power
    corrected = mine[good] / (ref[good] * factor) - 1.0
    worst = float(np.max(np.abs(corrected)))
    ok = worst < TOL
    tag = f" (c^2 power {c_power:+d})" if c_power else ""
    print(f"  {name:<24s} raw max |resid| {np.max(np.abs(raw)):.3e}   "
          f"corrected {worst:.3e}{tag}   {'PASS' if ok else 'FAIL'}")
    return ok, raw, corrected


def main(plot=False):
    if not FROZEN.is_file():
        print(f"frozen reference not found at {FROZEN}")
        print("set CLUSTER_LENSING_COV_DIR to the checkout; skipping")
        return 0

    d = np.load(FROZEN, allow_pickle=True)
    zl, zh = np.asarray(d["zl"]), np.asarray(d["z_h"])
    print(f"frozen reference: {FROZEN}")
    print(f"  zl: {zl.size} nodes over [{zl[0]:.2f}, {zl[-1]:.2f}], "
          f"z_h = {zh}, zs_max = {float(d['zs_max'])}")
    print(f"  expected c^2 offset: {C_RATIO_SQUARED:.8f}\n")

    lk = LensingKernel(Survey.smail(**REF_SOURCES),
                       FlatLambdaCDM(**REF_COSMO))

    results = [
        # <Sigma_crit^-1> and <Sigma_crit> both carry c^2/4piG
        # <Sigma_crit^-1> ~ 4 pi G / c^2, so it carries the INVERSE offset
        report("q_plain <Sc^-1>(z_l)", lk.mean_inverse_sigma_crit(zl),
               d["q_plain"], c_power=-1),
        report("mean_sigma_crit(z_h)", lk.mean_sigma_crit(zh),
               d["mean_sigma_crit"], c_power=+1),
        # f_src is a pure probability -- no constants
        report("f_src(z_h)", lk.f_src_behind(zh), d["f_src"]),
        # q_sigma is a ratio of two Sigma_crit, so c^2 cancels
        report("q_sigma(z_l; z_h)",
               np.array([lk.q_sigma(zl, float(z)) for z in zh]),
               d["q_sigma"]),
    ]

    # the sign structure of q_sigma is part of the definition, not noise
    ref_q = np.asarray(d["q_sigma"])
    print(f"\n  q_sigma in the reference runs [{ref_q.min():.3f}, "
          f"{ref_q.max():.3f}] -- it is SIGNED. Do not clamp it.")

    if plot:
        import matplotlib.pyplot as plt

        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].semilogy(zl, np.asarray(d["q_plain"]), "k-", lw=3,
                         label="frozen reference")
        axes[0].semilogy(zl, lk.mean_inverse_sigma_crit(zl), "--",
                         color="firebrick", lw=2, label="clenspy")
        axes[0].set_xlabel(r"$z_l$")
        axes[0].set_ylabel(r"$\langle\Sigma_{\rm crit}^{-1}\rangle$"
                           r"  [Mpc$^2$/M$_\odot$]")
        axes[0].set_title("Averaged inverse critical density")
        axes[0].legend()

        for i, z in enumerate(zh):
            axes[1].plot(zl, ref_q[i], "-", color=f"C{i}", lw=3,
                         label=f"ref $z_h$={z:.3f}")
            axes[1].plot(zl, lk.q_sigma(zl, float(z)), "--", color="k", lw=1)
        axes[1].axhline(0, color="0.5", lw=0.8)
        axes[1].set_xlabel(r"$z_l$")
        axes[1].set_ylabel(r"$q_\Sigma(z_l; z_h)$")
        axes[1].set_title("Signed $\\Sigma_{\\rm crit}$-weighted kernel")
        axes[1].legend(fontsize=9)
        fig.tight_layout()
        out = FIG_DIR / "lensing_kernel_vs_frozen.png"
        fig.savefig(out, dpi=140)
        print(f"\n  wrote {out}")

    ok = all(r[0] for r in results)
    print("\nall comparisons pass" if ok else "\nSOME COMPARISONS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true",
                        help="write the comparison figure into docs/_static")
    sys.exit(main(**vars(parser.parse_args())))
