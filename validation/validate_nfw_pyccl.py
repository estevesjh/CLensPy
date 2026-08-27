r"""`NfwProfile` against `pyccl`'s analytic NFW.

Compares all three closed forms `clenspy` carries -- the Fourier transform
:math:`u(k)`, the projected :math:`\Sigma(R)`, and the excess
:math:`\Delta\Sigma(R)` -- against `pyccl.halos.HaloProfileNFW` with
``fourier_analytic``, ``projected_analytic`` and ``cumul2d_analytic``. Both
sides are then closed-form NFW, so any disagreement is an algebra or
normalisation error rather than an integration tolerance.

NOTE: the mass definition must match on both sides or the comparison is
meaningless. `pyccl`'s ``MassDef200m`` is 200x the mean matter density,
which is what `NfwProfile` means by ``m200`` when ``rho_ref`` defaults to
:math:`\Omega_{m,0}\rho_{c,0}`. Concentration is pinned with
``ConcentrationConstant`` so no c(M) relation enters.

NOTE: units are h-free absolute -- Msun, Mpc, Msun/Mpc^2, 1/Mpc.

Run
---
::

    python validation/validate_nfw_pyccl.py            # numbers only
    python validation/validate_nfw_pyccl.py --plot     # + figures

Exits nonzero if any comparison misses its tolerance.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyccl as ccl

from clenspy.halo import NfwProfile

#: Figures land in the docs so the validation page can show them.
FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "_static" / "validation"

M200 = 1e14  # Msun, M_200m
C200 = 4.0

#: pyccl needs a cosmology to define the mass-radius relation. Only Omega_m
#: and h matter here: sigma8 and n_s enter no closed form being compared.
CCL_COSMO = dict(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)

#: NOTE: both sides evaluate closed forms of the *same* function, so the
#: only error is floating point -- measured agreement is 1e-10 on all four
#: comparisons. These thresholds sit two decades above that, which leaves
#: room for a compiler or library difference while still catching a real
#: algebra change. The tolerances these checks carried as unit tests were
#: 5e-3, seven orders of magnitude looser than the agreement, so they could
#: not have detected anything short of a wholesale error.
TOL_MAX = 1e-8
TOL_RMS = 1e-8


def ccl_nfw(**profile_kwargs):
    """A `pyccl` NFW at fixed concentration, matched to `NfwProfile`."""
    mass_def = ccl.halos.massdef.MassDef200m
    conc = ccl.halos.concentration.constant.ConcentrationConstant(
        C200, mass_def=mass_def
    )
    return ccl.halos.profiles.HaloProfileNFW(
        mass_def=mass_def, concentration=conc, **profile_kwargs
    )


def report(name, ours, theirs, x, tol_max, tol_rms):
    """Print the two error norms and say whether they pass."""
    valid = np.isfinite(theirs) & (theirs != 0)
    frac = (ours[valid] - theirs[valid]) / theirs[valid]
    err_max = float(np.nanmax(np.abs(frac)))
    err_rms = float(np.sqrt(np.nanmean(frac**2)))
    ok = err_max < tol_max and err_rms < tol_rms
    print(f"  {name:<14s} max {err_max:.2e} (< {tol_max:.0e})   "
          f"rms {err_rms:.2e} (< {tol_rms:.0e})   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok, x[valid], ours[valid], theirs[valid]


def validate_fourier(truncated):
    r""":math:`u(k) = \tilde\rho(k)/M`, truncated at r200 or not."""
    k = np.logspace(-3, 2, 200)  # 1/Mpc; brackets 1/r200 ~ 1 by two decades
    cosmo = ccl.Cosmology(**CCL_COSMO)
    theirs = ccl_nfw(fourier_analytic=True, truncated=truncated).fourier(
        cosmo, k, M200, 1
    )
    ours = NfwProfile(M200, C200).fourier(k, truncated=truncated)
    label = "u(k) trunc" if truncated else "u(k)"
    return report(label, ours, theirs, k, TOL_MAX, TOL_RMS)


def validate_sigma():
    r""":math:`\Sigma(R)`, the projected surface density."""
    R = np.logspace(-3, 1.3, 100)  # Mpc, in to 1e-3 to exercise the small-x branch
    cosmo = ccl.Cosmology(**CCL_COSMO)
    theirs = ccl_nfw(projected_analytic=True, truncated=False).projected(
        cosmo, R, M200, 1
    )
    ours = NfwProfile(M200, C200).sigma(R)
    return report("Sigma(R)", ours, theirs, R, TOL_MAX, TOL_RMS)


def validate_deltasigma():
    r""":math:`\Delta\Sigma(R) = \bar\Sigma(<R) - \Sigma(R)`.

    Assembled on the pyccl side as ``cumul2d - projected``; on ours it is
    the single closed form :math:`\bar g(x) - f(x)`, which is the point --
    the small-:math:`R` branch of `NfwProfile._gbarNfw` exists because
    differencing the two cancels catastrophically there.
    """
    R = np.logspace(-3, 1.3, 100)  # Mpc
    cosmo = ccl.Cosmology(**CCL_COSMO)
    p = ccl_nfw(projected_analytic=True, cumul2d_analytic=True, truncated=False)
    theirs = p.cumul2d(cosmo, R, M200, 1) - p.projected(cosmo, R, M200, 1)
    ours = NfwProfile(M200, C200).deltasigma(R)
    return report("DeltaSigma(R)", ours, theirs, R, TOL_MAX, TOL_RMS)


def main(plot=False):
    print(f"NfwProfile vs pyccl {ccl.__version__}: "
          f"M_200m = {M200:.1e} Msun, c_200 = {C200}")
    results = [
        validate_fourier(truncated=False),
        validate_fourier(truncated=True),
        validate_sigma(),
        validate_deltasigma(),
    ]
    if plot:
        import matplotlib.pyplot as plt

        # One overlay panel per quantity, as the comparison was originally
        # plotted: the curves lie on top of each other, which is the result.
        panels = [
            (r"$k$ [Mpc$^{-1}$]", r"$|u_{\mathrm{NFW}}(k)|$",
             "NFW Fourier Transform"),
            (r"$k$ [Mpc$^{-1}$]", r"$|u_{\mathrm{NFW}}(k)|$",
             "NFW Fourier Transform (truncated)"),
            (r"$R$ [Mpc]", r"$\Sigma_{\mathrm{NFW}}(R)$ [M$_\odot$ / Mpc$^2$]",
             "NFW Surface Density"),
            (r"$R$ [Mpc]",
             r"$\Delta\Sigma_{\mathrm{NFW}}(R)$ [M$_\odot$ / Mpc$^2$]",
             "NFW Excess Surface Density"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, (xlabel, ylabel, title), (_, x, ours, theirs) in zip(
            axes.ravel(), panels, results
        ):
            ax.loglog(x, np.abs(ours), label="clenspy")
            ax.loglog(x, np.abs(theirs), ls="--", label="pyccl")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title}: clenspy vs pyccl")
            ax.legend()
        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        out = FIG_DIR / "nfw_vs_pyccl.png"
        fig.savefig(out, dpi=140)
        print(f"  wrote {out}")

    ok = all(r[0] for r in results)
    print("all comparisons pass" if ok else "SOME COMPARISONS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true",
                        help="write docs/_static/validation/nfw_vs_pyccl.png")
    sys.exit(main(**vars(parser.parse_args())))
