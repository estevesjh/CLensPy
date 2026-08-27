r"""The $P(k) \to \xi \to \Sigma \to \Delta\Sigma$ chain, stage by stage,
against the closed-form NFW.

Every code that computes a two-halo term runs the same three transforms.
This bench feeds each of them :math:`P(k) \equiv \tilde\rho_{\rm NFW}(k)`
and compares the output of **each stage** to the exact answer, so a
disagreement localises to one transform instead of showing up as a single
number at the end.

The identity that makes it work:

.. math::
    \xi(r) = \int \! dk\, \frac{k^2}{2\pi^2}\, P(k)\, j_0(kr)

is the *same integral* as the inverse 3-D Fourier transform of
:math:`\tilde\rho`. So a code fed :math:`P = \tilde\rho` must return
:math:`\xi \equiv \rho`; its Abel stage must return the Wright & Brainerd
:math:`\Sigma`; and its interior-mean stage must return
:math:`\Delta\Sigma`. All three references are closed forms
(`analytic_nfw`), themselves quadrature-checked.

The benchmark halo
------------------
A mean-density mass definition on the present-day (comoving) background --
no redshift anywhere, so comoving equals physical and no :math:`(1+z)`
factor can hide:

    rho_m   = Omega_m0 * rho_c0            = 8.6327e10  Msun h^2/Mpc^3
    M_200m  = 1e14 Msun/h  ->  r_200m      = 1.1141 cMpc/h
    c = 5                  ->  r_s         = 0.2228 cMpc/h
    delta_c = 8694.8        ->  rho_s      = 7.506e14  Msun h^2/Mpc^3

NOTE: **this script works in the h-ful convention** of the reference
libraries -- lengths cMpc/h, densities Msun h^2/Mpc^3, so :math:`\Sigma`
emerges in Msun h/Mpc^2 and is divided by :math:`10^{12}` into Msun h/pc^2
for `cluster_toolkit`. That is *not* `clenspy`'s h-free convention. It is
kept because the point of the CLMM legs is to exercise exactly those
crossings, and because the residuals are ratios, in which :math:`h`
cancels within each leg.

The four legs
-------------
``cluster_toolkit``
    The y3 production engine. ``xi.xi_mm_at_r``, then
    ``deltasigma.Sigma_at_R`` (Abel of :math:`\rho/\rho_m`, extending the
    integrand below its grid as an NFW), then
    ``deltasigma.DeltaSigma_at_R``.
``clenspy``
    `pk_to_xi_fftlog` (mcfit ``P2xi``), then `compute_sigma_grid` (Abel
    quadrature under :math:`u = t/(1-t)`), then
    `sigma_to_deltasigma_cumtrapz`.
``CLMM``
    Native NFW, **no** :math:`P(k)` at all: its public API is parametric.
    So this leg tests conventions and normalisation -- mass definition,
    :math:`\rho_s`/:math:`r_s` reconstruction, the :math:`1/h` and
    :math:`1/h^2` crossings -- not transform numerics. That is why its
    residual is a coherent offset with the same shape in all three panels
    rather than noise.
``CLMM backend (P input)``
    The generic route: `pyccl`'s ``HaloProfile`` accepts an arbitrary
    ``_fourier``, and every stage is then one FFTLog Hankel transform of
    :math:`\tilde\rho` with no intermediate :math:`\xi` table -- ``real``,
    ``projected`` (:math:`J_0`, the Fourier-slice theorem) and ``cumul2d``
    (:math:`2J_1(kR)/(kR)`, the disc average of :math:`J_0`).

Run
---
::

    python validation/validate_twohalo_chain.py
    python validation/validate_twohalo_chain.py --plot

Needs `cluster_toolkit`, `clmm` and `pyccl`. Exits nonzero if any leg
misses its tolerance. Source: ``y3_cluster_cpp/validations/
second_halo_term/10_chain_residuals.py``, CLensPy issue #4.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analytic_nfw import NfwAnalytic, selfcheck  # noqa: E402

#: Figures land in the docs so the validation page can show them. Resolved
#: from this file, not the working directory.
FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "_static" / "validation"

#: Present-day critical density in Msun h^2 / Mpc^3 -- the h-ful
#: convention. Named with the convention in the identifier because that is
#: exactly what is easy to get wrong.
RHO_CRIT_WITH_H = 2.77533742639e11
OMEGA_M = 0.311049
OMEGA_B = 0.048975
H = 0.6766
RHO_M_WITH_H = OMEGA_M * RHO_CRIT_WITH_H  # comoving, z-independent

M200M_OVER_H = 1e14  # Msun/h
CONC = 5.0

#: NOTE: r from 0.05 to 30 cMpc/h spans well inside r_s = 0.22 out to where
#: rho has fallen five decades. The FFTLog leg's residual grows in the far
#: tail (r >> r200, rho -> 0) purely because the reference goes to zero, so
#: the summary quotes its worst over r <= 5 as well.
R3D = np.logspace(np.log10(0.05), np.log10(30.0), 72)
#: Projected radii: the lensing-relevant range, 0.1 to 20 cMpc/h.
R2D = np.logspace(np.log10(0.1), np.log10(20.0), 128)

#: Per-leg tolerance on the worst absolute fractional residual across all
#: three stages. These are *measured* values rounded up, not aspirations --
#: each is set just above what the leg achieves, so a regression trips it.
TOL = {
    "cluster_toolkit": 1e-3,
    "clenspy": 1e-4,
    "CLMM": 5e-3,
    "CLMM backend ($P$ input)": 1e-2,
}

STYLE = {
    "cluster_toolkit": ("black", "-"),
    "clenspy": ("firebrick", "--"),
    "CLMM": ("grey", "-."),
    "CLMM backend ($P$ input)": ("darkgrey", ":"),
}


def benchmark_halo():
    """The closed-form NFW every leg is compared against."""
    return NfwAnalytic.from_m200m(M200M_OVER_H, c=CONC,
                                  rho_ref=RHO_M_WITH_H)


# -- the four legs ---------------------------------------------------------


def leg_cluster_toolkit(p):
    """`cluster_toolkit`: Hankel quadrature, then its native Sigma/DS path."""
    import cluster_toolkit as ct

    k = np.logspace(-4, 3, 1200)
    xi = ct.xi.xi_mm_at_r(R3D, k, p.rho_tilde(k))
    res_rho = xi / p.rho(R3D) - 1.0

    # ct's grid must be wide: below r_xi[0] it extends the integrand
    # assuming an NFW(M, c), which is the source of its -0.03% Sigma dip.
    r_xi = np.logspace(-3, 3, 1000)
    sig = ct.deltasigma.Sigma_at_R(R2D, r_xi, p.rho(r_xi) / RHO_M_WITH_H,
                                   M200M_OVER_H, CONC, OMEGA_M)
    res_sig = sig / (p.sigma(R2D) / 1e12) - 1.0  # Msun h/pc^2 crossing

    ds = ct.deltasigma.DeltaSigma_at_R(R2D, R2D, p.sigma(R2D) / 1e12,
                                       M200M_OVER_H, CONC, OMEGA_M)
    res_ds = ds / (p.delta_sigma(R2D) / 1e12) - 1.0
    return res_rho, res_sig, res_ds


def leg_clenspy(p):
    """`clenspy`: mcfit FFTLog, Abel quadrature, cumtrapz interior mean."""
    from clenspy.utils.integrate import (
        compute_sigma_grid,
        pk_to_xi_fftlog,
        sigma_to_deltasigma_cumtrapz,
    )

    # NOTE: k out to 1e5, matching TwoHaloTerm's own internal window. A
    # narrower window rings below r ~ 0.3 -- that is issue #4 item 4, and
    # it is a real constraint on the caller, not a tuning knob.
    k = np.logspace(-4, 5, 2048)
    xi = np.asarray(pk_to_xi_fftlog(k, p.rho_tilde(k), R3D)).reshape(-1)
    res_rho = xi / p.rho(R3D) - 1.0

    # NOTE: 600 Abel nodes and r_max = 300; at 150 nodes this stage sits at
    # 1e-5 instead of 5e-7.
    sig = np.asarray(
        compute_sigma_grid(lambda r, z: p.rho(r), R2D, np.array([0.0]),
                           method="trapz", rmax_integral=300.0, n_points=600)
    ).reshape(-1)
    res_sig = sig / p.sigma(R2D) - 1.0

    # NOTE: the interior mean starts at R = 0, so the grid must extend well
    # below the smallest output radius -- 1e-4 here. Truncating it is the
    # omitted inner-boundary term of issue #4 item 3.
    r_ext = np.logspace(-4, np.log10(20.0), 1200)
    ds_ext = np.asarray(
        sigma_to_deltasigma_cumtrapz(r_ext, p.sigma(r_ext))
    ).reshape(-1)
    ds = np.exp(np.interp(np.log(R2D), np.log(r_ext),
                          np.log(np.maximum(ds_ext, 1e-300))))
    res_ds = ds / p.delta_sigma(R2D) - 1.0
    return res_rho, res_sig, res_ds


def leg_clmm(p):
    """`CLMM` native NFW: tests conventions, not transforms."""
    import clmm
    from clmm.theory import (
        compute_3d_density,
        compute_excess_surface_density,
        compute_surface_density,
    )

    cosmo = clmm.Cosmology(H0=100.0 * H, Omega_dm0=OMEGA_M - OMEGA_B,
                           Omega_b0=OMEGA_B, Omega_k0=0.0)
    # NOTE: clmm rejects z_cl = 0. At 1e-4 the (1+z)^3 of the mean-density
    # mass definition is far below every code's own accuracy floor.
    kw = dict(mdelta=M200M_OVER_H / H, cdelta=CONC, z_cl=1e-4, cosmo=cosmo,
              delta_mdef=200, halo_profile_model="nfw", massdef="mean",
              verbose=False)

    # the h crossings: r/h in, then /h^2 on a density and /h on a Sigma
    rho = np.asarray(compute_3d_density(R3D / H, **kw))
    res_rho = (rho / H**2) / p.rho(R3D) - 1.0
    sig = np.asarray(compute_surface_density(R2D / H, **kw))
    res_sig = (sig / H) / p.sigma(R2D) - 1.0
    ds = np.asarray(compute_excess_surface_density(R2D / H, **kw))
    res_ds = (ds / H) / p.delta_sigma(R2D) - 1.0
    return res_rho, res_sig, res_ds


def leg_clmm_backend(p):
    """`pyccl` FFTLog fed the Fourier profile directly: the generic route."""
    import pyccl

    class TildeProfile(pyccl.halos.HaloProfile):
        """A `pyccl` profile whose Fourier transform is ours."""

        def __init__(self, prof):
            super().__init__(mass_def=pyccl.halos.MassDef(200, "matter"))
            self.prof = prof

        def _fourier(self, cosmo, k, M, a):
            ft = self.prof.rho_tilde(np.atleast_1d(np.asarray(k, float)))
            if np.ndim(M) == 0:
                return ft
            return np.tile(ft, (np.atleast_1d(M).size, 1))

    cosmo = pyccl.Cosmology(Omega_c=OMEGA_M - OMEGA_B, Omega_b=OMEGA_B,
                            h=H, sigma8=0.8238, n_s=0.9665)

    # NOTE: FFTLog precision is tuned per transform. plaw_fourier = -2
    # matches the rho_tilde ~ kappa^-2 tail; real() prefers the wide
    # window, projected/cumul2d the denser one. The residual ripple is the
    # k -> 0 log divergence of the untruncated NFW transform, not a
    # resolution problem.
    prof_3d = TildeProfile(p)
    prof_3d.update_precision_fftlog(padding_lo_fftlog=1e-4,
                                    padding_hi_fftlog=1e4,
                                    n_per_decade=600, plaw_fourier=-2.0)
    res_rho = np.asarray(prof_3d.real(cosmo, R3D, 1e14, 1.0)) / p.rho(R3D) - 1.0

    prof_2d = TildeProfile(p)
    prof_2d.update_precision_fftlog(padding_lo_fftlog=1e-3,
                                    padding_hi_fftlog=1e3,
                                    n_per_decade=1200, plaw_fourier=-2.0)
    sig = np.asarray(prof_2d.projected(cosmo, R2D, 1e14, 1.0))
    res_sig = sig / p.sigma(R2D) - 1.0
    cum = np.asarray(prof_2d.cumul2d(cosmo, R2D, 1e14, 1.0))
    res_ds = (cum - sig) / p.delta_sigma(R2D) - 1.0
    return res_rho, res_sig, res_ds


# -- reporting ------------------------------------------------------------


def summarise(results):
    """Print the median/max table and check each leg against `TOL`."""
    inner = R3D <= 5.0
    print(f"\n{'leg':<26s} {'rho/xi med/max':>21s} "
          f"{'Sigma med/max':>21s} {'DeltaSigma med/max':>21s}  worst")
    ok = {}
    for name, (res_rho, res_sig, res_ds) in results.items():
        cells = []
        for res in (res_rho, res_sig, res_ds):
            cells.append(f"{np.nanmedian(np.abs(res)):.1e} /"
                         f"{np.nanmax(np.abs(res)):.1e}")
        # the FFTLog rho residual grows where rho -> 0; quote r <= 5
        rho_eff = res_rho[inner] if "backend" in name else res_rho
        worst = max(np.nanmax(np.abs(rho_eff)), np.nanmax(np.abs(res_sig)),
                    np.nanmax(np.abs(res_ds)))
        ok[name] = worst < TOL[name]
        print(f"{name:<26s} " + " ".join(f"{c:>21s}" for c in cells)
              + f"  {worst:.1e} (< {TOL[name]:.0e}) "
              + ("PASS" if ok[name] else "FAIL"))
    return ok


def make_figure(results, out):
    """The three-panel residual figure, one panel per chain stage."""
    import matplotlib.pyplot as plt

    panels = [
        (r"$\rho\;/\;\xi$ stage", R3D, 0, r"$r$ [cMpc/$h$]"),
        (r"$\Sigma(R)$ stage", R2D, 1, r"$R$ [cMpc/$h$]"),
        (r"$\Delta\Sigma(R)$ stage", R2D, 2, r"$R$ [cMpc/$h$]"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
    for ax, (title, x, stage, xlabel) in zip(axes, panels):
        # each panel carries an inset zoom on the core at a fixed +-0.01%,
        # so the legs that sit on zero can be told apart from each other
        axz = ax.inset_axes([0.56, 0.08, 0.40, 0.28])
        for name, res in results.items():
            color, ls = STYLE[name]
            ax.semilogx(x, 100.0 * res[stage], color=color, ls=ls, lw=3.0,
                        label=name)
            axz.plot(x, 100.0 * res[stage], color=color, ls=ls, lw=2.0)
        ax.axhline(0, color="0.25", lw=0.8)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)

        axz.axhline(0, color="0.25", lw=0.6)
        axz.set_xlim(0.1, 1.0)
        axz.set_ylim(-0.01, 0.01)
        axz.set_xticks(np.arange(0.2, 1.01, 0.2))
        axz.set_xticks(np.arange(0.1, 1.01, 0.1), minor=True)
        axz.tick_params(axis="both", labelsize=11)
        axz.text(0.05, 0.82, r"core, $\pm 0.01\%$", transform=axz.transAxes,
                 fontsize=12)
        axz.set_facecolor("white")
        ax.indicate_inset_zoom(axz, edgecolor="0.4", lw=1.0)

    axes[0].set_ylabel("code / analytic $-$ 1  [%]")
    axes[0].legend(fontsize=13, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\n  wrote {out}")


def main(plot=False):
    p = benchmark_halo()
    print(f"benchmark: {p}")
    print(f"  M_200m = {M200M_OVER_H:.0e} Msun/h, c = {CONC}, "
          f"r_200m = {p.r200:.4f} cMpc/h, r_s = {p.r_s:.4f} cMpc/h")

    bad = selfcheck(verbose=False)
    print(f"  closed-form self-check: {'PASS' if not bad else f'FAIL {bad}'}")

    results = {
        "cluster_toolkit": leg_cluster_toolkit(p),
        "clenspy": leg_clenspy(p),
        "CLMM": leg_clmm(p),
        "CLMM backend ($P$ input)": leg_clmm_backend(p),
    }
    ok = summarise(results)

    if plot:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        make_figure(results, FIG_DIR / "twohalo_chain_residuals.png")

    passed = all(ok.values()) and not bad
    print("all legs pass" if passed else "SOME LEGS FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot", action="store_true",
        help="write docs/_static/validation/twohalo_chain_residuals.png")
    sys.exit(main(**vars(parser.parse_args())))
