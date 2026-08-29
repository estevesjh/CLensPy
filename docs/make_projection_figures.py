"""Generate the figure embedded in projection_lensing.md.

    uv run python docs/make_projection_figures.py

Real halo model (Tinker 2008/2010 + CAMB halofit xi_NL, disk-cached), the
Buzzard-mock configuration of validation/validate_sigma_prj_mock.py.
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import sanzo_wada as sw
import seaborn as sns
from astropy.cosmology import FlatLambdaCDM

from clenspy.cosmology import BiasModel, TinkerMassFunction
from clenspy.cosmology.pkgrid import PkGrid
from clenspy.lensing import SigmaPrj
from clenspy.selection import PhysicalMassMor, SelBiasEngine, XiNL
from clenspy.selection.scaling_relation import HodMor

OUT = pathlib.Path(__file__).resolve().parent / "_static" / "img"

C4 = [c.hex for c in sw.get_combination("vol2-100").colors]
C3 = [C4[3], C4[2], C4[1]]

sns.set_theme(style="white", context="talk", font_scale=0.8)

H, OMEGA_M = 0.7, 0.286


class BuzzardCosmology(FlatLambdaCDM):
    sigma8 = 0.82
    n_s = 0.96


COSMO = BuzzardCosmology(H0=100.0 * H, Om0=OMEGA_M, Ob0=0.046)


def halo_model():
    pk = PkGrid(cosmo=COSMO, nonlinear=True)
    xi_nl = XiNL(pk, clip=False)
    tmf = TinkerMassFunction(cosmo=COSMO, zvec=np.linspace(0.0, 1.0, 21))
    bm = BiasModel(cosmo=COSMO)

    def hmf(mass, z):
        m, zz = np.broadcast_arrays(np.asarray(mass, float),
                                    np.asarray(z, float))
        vals = tmf.dndlnm(m.ravel() * H / OMEGA_M, zz.ravel())
        return vals.reshape(m.shape) * H**3 / m

    def bias(mass, z):
        m, zz = np.broadcast_arrays(np.asarray(mass, float),
                                    np.asarray(z, float))
        return np.asarray(bm.bias(m.ravel(), zz.ravel())).reshape(m.shape)

    return xi_nl, hmf, bias


def fig_projection_lensing():
    """Left: the rnd/cl channel split of Sigma_prj. Right: the ratio
    observable (selected over b_eff-weighted random), b_sel on vs off."""
    xi_nl, hmf, bias = halo_model()
    lob, zob, b_eff = 23.9, 0.425, 3.02  # the mock's [20,30)x[0.35,0.5) bin
    engine = SelBiasEngine(cosmology=COSMO, xi_nl=xi_nl, hmf=hmf, bias=bias,
                           mor=PhysicalMassMor(HodMor.buzzard(), H))
    bsel = engine.marginalised_bias(lob, zob, b_eff=b_eff)
    prj = SigmaPrj(cosmology=COSMO, xi_nl=xi_nl, hmf=hmf, bias=bias,
                   los_window="hard", los_depth=50.0 / H,
                   exclusion="ball", theta_perp_range=(1e-3, 60.0 / H))

    R = np.geomspace(0.1, 40.0, 32)  # comoving Mpc
    tot = prj.sigma_prj(R, lob, zob, bsel, channel="sum")
    rnd, cl = prj.rnd.copy(), prj.cl.copy()
    prj.sigma_prj(R, lob, zob, lambda th: b_eff, channel="sum")
    tot_rnd_model = prj.rnd + prj.cl

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.6))
    ax1.loglog(R, tot, color="k", lw=2.5,
               label=r"$\Sigma_{\rm tot} = \Sigma_{\rm bkg} + \Sigma^{\rm prj}$")
    ax1.loglog(R, rnd, color=C3[0], lw=2.0, ls="--",
               label=r"$\Sigma_{\rm bkg}$ (background)")
    ax1.loglog(R, cl, color=C3[1], lw=2.0, ls=":",
               label=r"$\Sigma^{\rm prj}$ (correlated)")
    ax1.set_xlabel(r"$R$ [comoving Mpc]")
    ax1.set_ylabel(r"$\Sigma$ [$M_\odot\,$Mpc$^{-2}$]")
    ax1.legend(frameon=False)

    ax2.semilogx(R, tot / tot_rnd_model, color="k", lw=2.5,
                 label=r"$b_{\rm sel}(\theta)$")
    ax2.axhline(1.0, ls="--", color="gray", lw=1.0)
    ax2.set_xlabel(r"$R$ [comoving Mpc]")
    ax2.set_ylabel(r"$\Sigma_{\rm tot}^{\rm sel}/\Sigma_{\rm tot}"
                   r"^{b_{\rm eff}}$")
    ax2.legend(frameon=False)
    fig.suptitle(r"$\lambda^{\rm ob}=23.9$, $z^{\rm ob}=0.425$ "
                 "(Buzzard-mock configuration)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "projection_lensing.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_projection_lensing()
    print(f"wrote {OUT / 'projection_lensing.png'}")
