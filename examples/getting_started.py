# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CLensPy: Getting Started
#
# One notebook, one section per physical effect. Each section is the single
# source for the matching page under `docs/` — pulled in there via
# `{literalinclude}` against this file's jupytext-paired `.py` percent
# format, tag-delimited below. Run this notebook top to bottom to reproduce
# every snippet in the docs.

# %% [markdown]
# ## Cosmology

# %% tags=["cosmology"]
import numpy as np
from clenspy.cosmology import (
    comoving_to_theta,
    fiducial_cosmology,
    growth_factor,
    theta_to_comoving,
)

# every other layer takes a cosmology object as input; build one first.
# a fresh instance every call -- no shared, mutable module-level default.
cosmo = fiducial_cosmology(H0=70.0, Om0=0.3)  # flat LambdaCDM
print(cosmo)

z_lens = 0.35
D_c = np.array([0.1, 1.0, 10.0])  # comoving separations [Mpc]
theta = comoving_to_theta(D_c, z_lens, cosmo, unit="arcmin")
print("theta [arcmin] =", theta)
print("round trip     =", theta_to_comoving(theta, z_lens, cosmo, unit="arcmin"))

z = np.array([0.0, 0.35, 1.0, 2.0])
print("D(z) =", growth_factor(z, cosmo))  # normalised, D(0) = 1

# %% [markdown]
# ## Power spectrum

# %% tags=["power-spectrum"]
from clenspy.cosmology import PkGrid

# CAMB-backed linear P(k, z=0); cached to disk after the first call.
# h-free, like the rest of the package: k in 1/Mpc, P(k) in Mpc^3.
pk_grid = PkGrid(cosmo=cosmo, nonlinear=False)
k_camb = pk_grid.k
Pk_camb = pk_grid(k_camb, z=0.0)
print(f"P(k) from CAMB: k in [{k_camb[0]:.1e}, {k_camb[-1]:.1e}] 1/Mpc, "
      f"P in [{Pk_camb.min():.2e}, {Pk_camb.max():.2e}] Mpc^3")

# physical units end to end: k in 1/Mpc, P in Mpc^3, R in Mpc
from clenspy.cosmology import SigmaGrid

sigma_grid = SigmaGrid(k_camb, Pk_camb)
for r in (1.0, 8.0, 20.0):  # Mpc
    print(f"sigma(R={r:5.1f} Mpc) = {sigma_grid.sigma(r):.4f}")

# %% [markdown]
# ## Halo mass function

# %% tags=["mass-function"]
from clenspy.cosmology import TinkerMassFunction

# cosmo -> PkGrid -> SigmaGrid -> dndlnm_grid, all lazily, on first use:
# the shortcut for the (k, pk) chain built by hand above.
hmf = TinkerMassFunction(cosmo=cosmo)  # Delta = 200 (mean matter) by default

M = np.array([1e13, 1e14, 5e14, 1e15])  # Msun
print("M [Msun]           =", M)
print("dn/dlnM [Mpc^-3]   =", hmf.dndlnm(M, z=0.0))

# %% [markdown]
# ## Halo bias

# %% tags=["halo-bias"]
from clenspy.cosmology import BiasModel

# same physical chain and the same (M, z) grid idea as the mass function
bias_model = BiasModel(cosmo=cosmo)

print("nu(M)   =", bias_model.nu_at_mass(M))
print("b(M)    =", bias_model.bias(M, z=0.0))

# sigma(M,z) = D(z) sigma(M,0): b(M) rises with z at fixed mass, since a
# fixed mass is a rarer peak against a smaller, less-grown sigma.
for z in (0.0, 0.5, 1.0):
    print(f"b(M=1e14, z={z:.1f}) = {bias_model.bias(1e14, z=z):.4f}")

# %% [markdown]
# ## Concentration-mass relations

# %% tags=["concentration"]
from clenspy.cosmology import child18, child18_powerlaw, delta_c, duffy08, m_star_hinv, scatter

# these relations were calibrated in h^-1 Msun and on M_200c, not M_200m --
# see the module docstring's two NOTEs before mixing them with the rest of
# clenspy, which is h-free and M_200m.
m200c_hinv = 1e14  # h^-1 Msun
z = 0.3

ms = m_star_hinv(z)  # Child et al.'s own anchor line, this cosmology's M*
print(f"M_star(z={z}) = {ms:.3e} h^-1 Msun,  M/M_star = {m200c_hinv / ms:.1f}")

c18 = child18(m200c_hinv, z, ms)
c19 = child18_powerlaw(m200c_hinv, z)
cd8 = duffy08(m200c_hinv, z, mass_def="200c")
print(f"c_200c: child18 = {c18:.3f}, child18_powerlaw = {c19:.3f}, "
      f"duffy08 = {cd8:.3f}")

# Duffy08's WMAP-5 sigma_8 sits below Child et al.'s, so it under-predicts
# concentration at cluster scales (Child et al. Fig. 12).
print(f"child18 / duffy08 = {c18 / cd8:.3f}  (> 1, as expected)")

print(f"NFW delta_c(c={c18:.3f}) = {delta_c(c18):.1f}, "
      f"scatter sigma_c = {scatter(c18):.3f}")

# %% [markdown]
# ## Halo density profiles

# %% tags=["density-profiles"]
from clenspy.halo import EinastoProfile, NfwProfile

# h-free absolute units: mass in Msun, lengths in Mpc, densities in
# Msun/Mpc^3, wavenumbers in 1/Mpc -- no cosmology object needed, only the
# reference density rho_ref that mass_def is measured against (default:
# the comoving mean matter density, giving M_200m).
m200 = 1e14  # Msun
c200 = 5.0
nfw = NfwProfile(m200=m200, c200=c200)
print(f"r200 = {nfw.r200:.4f} Mpc, rs = {nfw.rs:.4f} Mpc, "
      f"rho_s = {nfw.rho_s:.3e} Msun/Mpc^3")

r = np.array([0.1, 0.5, 1.0, 2.0])  # Mpc
print("rho_NFW(r)     [Msun/Mpc^3] =", nfw.density(r))

# same r_s as the NFW halo (r_s = r200/c200), rho_0 solved so the enclosed
# mass at r200 matches m200 -- a fair shape-only comparison at fixed mass
# and scale radius. alpha=0.25 is a typical cluster shape (Retana-Montenegro
# et al. 2012 report alpha ~ 0.16-0.25 for clusters).
alpha = 0.25
rho0_unit = EinastoProfile(alpha=alpha, rho_0=1.0, r_s=nfw.rs, tol=1e-4)
rho0 = m200 / rho0_unit.enclosed_mass(nfw.r200)
einasto = EinastoProfile(alpha=alpha, rho_0=rho0, r_s=nfw.rs, tol=1e-4)
print("rho_Einasto(r) [Msun/Mpc^3] =", einasto.density(r))

# fourier() returns rho_tilde(k), the *unnormalized* FT -- units of mass,
# going to M as k -> 0, not the dimensionless mass-normalized u(k|M).
k = np.array([0.1, 1.0, 10.0])  # 1/Mpc
print("rho_tilde_NFW(k)     [Msun] =", nfw.fourier(k))
print("rho_tilde_Einasto(k) [Msun] =", einasto.fourier(k))

# %% [markdown]
# ## Projected density profiles

# %% tags=["projected-profiles"]
# same nfw/einasto halos as above; the line-of-sight projection of rho(r).
R = np.array([0.1, 0.5, 1.0, 2.0])  # Mpc, projected radius
print("Sigma_NFW(R)          [Msun/Mpc^2] =", nfw.sigma(R))
print("Sigma_Einasto(R)      [Msun/Mpc^2] =", einasto.sigma(R))
print("DeltaSigma_NFW(R)     [Msun/Mpc^2] =", nfw.deltasigma(R))
print("DeltaSigma_Einasto(R) [Msun/Mpc^2] =", einasto.deltasigma(R))

# both profiles satisfy Sigmabar(<R) = Sigma(R) + DeltaSigma(R) identically,
# even though mean_sigma is evaluated from its own closed form, not this sum
nfw_check = nfw.mean_sigma(R) / (nfw.sigma(R) + nfw.deltasigma(R)) - 1.0
print("NFW Sigmabar consistency, max|rel| =", np.max(np.abs(nfw_check)))

# %% [markdown]
# ## The two-halo term

# %% tags=["two-halo-term"]
from clenspy.cosmology import mean_matter_density
from clenspy.halo import TwoHaloTerm

# same (k_camb, Pk_camb) from the power-spectrum section above -- h-free,
# so no k_h/pk_h3 conversion needed here, unlike SigmaGrid/TinkerMassFunction.
z_halo = 0.3
Pk_z = pk_grid(k_camb, z=z_halo)
two_halo = TwoHaloTerm(k_camb, Pk_z, zvec=z_halo)

R = np.array([0.5, 1.0, 5.0, 10.0, 50.0])  # Mpc
xi = two_halo.xi(R, z_halo)
sigma_hat = two_halo.sigma(R, z_halo)
deltasigma_hat = two_halo.deltasigma(R, z_halo)
print("xi(R, z)              =", xi)

# sigma/deltasigma are UNNORMALISED -- units of Mpc, not Msun/Mpc^2 -- until
# multiplied by the comoving (not physical) mean matter density.
rho_m = mean_matter_density(cosmo)
print("Sigma_2h(R)      [Msun/Mpc^2] =", sigma_hat * rho_m)
print("DeltaSigma_2h(R) [Msun/Mpc^2] =", deltasigma_hat * rho_m)

# %% [markdown]
# ## The lensing profile (1-halo + 2-halo)

# %% tags=["lensing-profile"]
from clenspy.lensing import LensingProfile

# the constructor only stores; nothing is built (no Boltzmann solver call)
# until the first observable is evaluated -- see the class Notes.
lp = LensingProfile(z_cluster=0.3, m200=1e14, concentration=4.0)
print(lp)

R = np.array([0.1, 0.5, 1.0, 5.0])  # Mpc
ds = lp.deltasigma(R)
print("DeltaSigma [Msun/Mpc^2] =", ds)
print(f"b(M) = {lp.bias:.3f}   Sigma_crit = {lp.sigma_crit:.3e} Msun/Mpc^2")

# the 2-halo term is the correlated large-scale structure around the halo
# (TwoHaloTerm); it only matters at large R
ds_1h = LensingProfile(z_cluster=0.3, m200=1e14, include_2halo=False).deltasigma(R)
print("1-halo only             =", ds_1h)
print("2-halo fraction         =", 1.0 - ds_1h / ds)

print("shear(R)         =", lp.shear(R))
print("reduced_shear(R) =", lp.reduced_shear(R))

# %% [markdown]
# ## Miscentering

# %% tags=["miscentering"]
from clenspy.lensing import MiscenteringProfile

# miscentered observables are read from a packaged lookup table, never
# integrated at call time -- only NFW is tabulated today.
R = np.array([0.1, 0.3, 1.0, 3.0])  # Mpc
for r_mis in (0.0, 0.2, 1.0):  # Mpc, the assumed-to-true center offset
    p = MiscenteringProfile(z_cluster=0.25, m200=2e14, r_mis=r_mis,
                             include_2halo=False)
    ds = p.deltasigma_mis(R)
    print(f"r_mis={r_mis:.1f} Mpc  DeltaSigma_mis [Msun/Mpc^2] =", ds)

# %% [markdown]
# ## Boost factor

# %% tags=["boost-factor"]
from clenspy.selection import boost_factor_nfw

# B(R) is dimensionless and > 1: correlated cluster members diluting the
# source catalogue make the *effective* Sigma_crit larger, so the measured
# DeltaSigma must be multiplied up by B(R) to recover the true signal.
R = np.array([0.1, 0.3, 1.0, 3.0, 10.0])  # Mpc
rs = 0.35  # Mpc, an NFW scale radius for M ~ 1e14
for B0 in (0.05, 0.10, 0.20):
    print(f"B0={B0:.2f}  B(R) =", boost_factor_nfw(R, B0, rs))

# %% [markdown]
# ## The selection function

# %% tags=["selection-function"]
from clenspy.selection import EmgParams, LogNormalMor, SelectionFunction

# S_ij(M, z) = S_i(M, z) * S_j(z): probability a halo of mass M at z lands
# in richness bin i and redshift bin j. Factorises exactly (see the module
# docstring for why) into a richness piece (Gauss-Legendre + EMG kernel)
# and a redshift piece (Gaussian CDF difference).
lam_edges = np.array([20.0, 30.0, 45.0, 60.0, 200.0])  # DES Y1 richness bins
z_edges = np.array([0.20, 0.35, 0.50, 0.65])
params = EmgParams(delta_mu=-1.5, sigma=3.0, f_prj=0.3, tau=0.12)
sel = SelectionFunction(lam_edges, z_edges, LogNormalMor(), params, sigma_z=0.01)
print(sel)

print("\nS_i(M, z=0.3): probability of landing in each richness bin")
for m in (1e13, 5e13, 1e14, 3e14, 1e15):
    s = sel.S_i(np.log(m), 0.3)
    print(f"M={m:8.1e} h^-1 Msun  S_i={np.round(s, 4)}  sum={s.sum():.4f}  "
          f"bracket_miss={sel.residual(np.log(m), 0.3):.1e}")

# %% [markdown]
# ## The selection-affected bias b_sel

# %% tags=["selection-bias"]
from clenspy.lensing import SigmaPrj
from clenspy.selection import HodMor, SelBiasEngine

# a cluster selected at observed richness sits behind extra line-of-sight
# structure, so its effective two-halo bias is not b(M,z) but a
# theta-dependent b_sel interpolating between two plateaus (b_small inside
# the aperture, b_large well outside it).

# SelBiasEngine shares its halo model with SigmaPrj (Tinker(2008) mass
# function, Tinker(2010) bias, CAMB halofit xi_NL) -- one built chain, not
# two; PkGrid disk-caches the CAMB call.
engine = SelBiasEngine(
    sigma_prj=SigmaPrj(cosmology=cosmo).build(), mor=HodMor.des_y1(),
    n_z=32, n_M=16, n_theta=8, n_ltr=40, ltr_grid_size=10,
)
lob, zob = 40.0, 0.4  # observed richness, observed redshift
profile = engine.marginalised_bias(lob, zob)
print(f"theta_lambda = {profile.theta_lambda:.6f} rad, "
      f"b_small = {profile.b_small:.3f}, b_large = {profile.b_large:.3f}")

# b_sel(theta) interpolates smoothly between the two plateaus, 0.5 of the
# way there exactly at theta = theta_lambda by construction
for frac in (0.0, 0.5, 1.0, 2.0, 5.0):
    theta = frac * profile.theta_lambda
    print(f"theta/theta_lambda={frac:4.2f}  b_sel={profile(theta):.4f}")

# %% [markdown]
# ## Projection lensing Sigma_prj

# %% tags=["projection-lensing"]
from clenspy.cosmology import BiasModel as _BiasModel, TinkerMassFunction as _Tmf
from clenspy.cosmology.pkgrid import PkGrid as _PkGrid
from clenspy.lensing import SigmaPrj, SigmaPrjConfig
from clenspy.selection import XiNL

# the projected two-halo surface density around a richness-selected
# cluster (Costanzi 2026 eq. 13): an exact 2 pi sin(theta) d theta
# angular integral -- no Limber, no Bessel -- of the offset-NFW kernel
# against two channels, rnd (the uniform mean, no b_sel) and cl (the
# correlated excess, carrying b_sel(theta) from the engine above).
# Real halo model this time: PkGrid disk-caches CAMB, so it costs seconds
# once and nothing after.
_tmf = _Tmf(cosmo=cosmo, zvec=np.linspace(0.0, 1.0, 21))
_bm = _BiasModel(cosmo=cosmo, zvec=np.linspace(0.0, 1.0, 21))

xi_real = XiNL(_PkGrid(cosmo=cosmo, nonlinear=True), clip=False)  # signed BAO trough

# default exclusion="counter" (zeroes the neighbour count inside the
# ball): the one mode whose cl channel is the mode-invariant
# random-subtracted excess. Under "ball" the exclusion hole would be
# booked in rnd -- and the default channel="cl" of deltasigma_prj below
# would silently omit it.
prj = SigmaPrj(cosmology=cosmo, xi_nl=xi_real, hmf=_tmf, bias=_bm,
               config=SigmaPrjConfig(los_depth=71.4))  # Costanzi-mock window
R_prj = np.array([0.5, 2.0, 8.0, 25.0])  # comoving Mpc
# b_sel from the toy engine above: its SHAPE is right, its amplitude is
# not (see docs/selection_bias.md); the mutually calibrated pipeline is
# validation/validate_sigma_prj_mock.py
sigma_tot = prj.sigma_prj(R_prj, 20.0, 0.5, profile, channel="sum")
parts = prj.components()
print("Sigma_prj(R | lob=20, zob=0.5) [Msun/Mpc^2 comoving]:")
for k, r in enumerate(R_prj):
    print(f"  R={r:5.1f}  rnd={parts['rnd'][k]:.3e}  cl={parts['cl'][k]:.3e}"
          f"  sum={sigma_tot[k]:.3e}")

# DeltaSigma_prj is its OWN integral (the DeltaSigma_mis kernel inside the
# same operator) -- never a reconstruction from Sigma_prj. Its rnd channel
# cancels to a boundary term: the excess functional annihilates constants.
ds = prj.deltasigma_prj(R_prj, 20.0, 0.5, profile)
print("DeltaSigma_prj:", np.array2string(ds, precision=3),
      f"\n  rnd/cl at R=8: {prj.rnd[2] / prj.cl[2]:+.4f} (boundary term only)")

# %% [markdown]
# ## Survey

# %% tags=["survey"]
from clenspy.survey import Survey, deg2, omega_des_y1, survey_bins

# Survey is the source population only (p(z_s), sigma_gamma, n_src) -- no
# footprint, no bins. Omega(z) and the bin grid are separate, since the
# footprint cancels in the shear but not the counts (see the module NOTE).
survey = Survey.from_config("des_y1")
print(survey)

z_s = np.array([0.3, 0.8, 1.5])
print("p(z_s) =", survey.pz_src(z_s))

bins = survey_bins("des_y1")
print(f"{len(bins)} bins = {bins.n_lam} richness x {bins.n_z} redshift")

z_l = np.array([0.2, 0.35, 0.5, 0.65])
print("Omega(z) [deg^2] =", deg2(omega_des_y1(z_l)))

# %% [markdown]
# ## The lensing kernel

# %% tags=["lensing-kernel"]
from clenspy.kernels import LensingKernel, sigma_critical

# Sigma_crit for one lens-source pair: diverges as z_s -> z_l, flattens for
# distant sources.
z_l = 0.35
for z_s in (0.6, 1.0, 1.5, 2.0):
    print(f"z_s={z_s:.2f}  Sigma_crit={sigma_critical(z_l, z_s, cosmo):.3e} "
          "Msun/Mpc^2")

# a real survey's source population averages Sigma_crit^-1 over p(z_s) --
# average the inverse, never invert the average (they differ by the source
# weighting, seen in the ratio below).
lk = LensingKernel(survey=Survey.from_config("des_y1"), cosmology=cosmo)
z_l = np.array([0.2, 0.35, 0.5, 0.65])
inv = lk.mean_inverse_sigma_crit(z_l)
mean = lk.mean_sigma_crit(z_l)
print("\n<Sigma_crit^-1>   =", inv)
print("1/<Sigma_crit>    =", 1.0 / mean)
print("ratio (!= 1)      =", inv * mean)
print("f_src_behind(z_l) =", lk.f_src_behind(z_l))

# %% [markdown]
# ## Number counts

# %% tags=["number-counts"]
from clenspy.observables import ClusterCounts
from clenspy.selection import EmgParams, LogNormalMor, SelectionFunction
from clenspy.survey import omega_des_y1

# the counts are one contraction of the weight W_ij = Omega(z) dV/dz
# n(M,z) S_ij(M,z) -- a smooth analytic dn/dlnM stand-in avoids needing
# CAMB/sigma-grid just to demo the contraction.
def toy_mass_function(ln_mass, z):
    lnm, zz = np.broadcast_arrays(np.asarray(ln_mass, float), np.asarray(z, float))
    m = np.exp(lnm)
    return 1e-5 * (m / 1e14) ** -1.0 * np.exp(-m / 5e14) / (1.0 + zz)

sel = SelectionFunction(
    np.array([20.0, 30.0, 45.0, 60.0, 200.0]),
    np.array([0.20, 0.35, 0.50, 0.65]),
    LogNormalMor(), EmgParams(-1.5, 3.0, 0.3, 0.12), sigma_z=0.01,
)
ln_mass_grid = np.log(np.logspace(13.5, 15.3, 24))  # h^-1 Msun
z_grid = np.linspace(0.16, 0.70, 32)
abundance = ClusterCounts(ln_mass_grid, z_grid, toy_mass_function, sel, cosmo,
                          omega_des_y1)

print("<N_ij> (richness x redshift bins):")
print(abundance.counts())

# %% [markdown]
# ## Stacked shear

# %% tags=["stacked-shear"]
from clenspy.halo import NfwProfile
from clenspy.observables import StackedDeltaSigma

# the second contraction of the SAME weight `abundance` above, now against
# DeltaSigma(R|M,z) instead of 1 -- this is the halo's own (one-halo) profile;
# a real stacked measurement also carries the projected two-halo excess
# Sigma_prj computed in "Projection lensing" above, evaluated at the bin's
# representative (lambda_ob, z_ob) rather than contracted through W_ij.
radii = np.logspace(-1.0, 1.0, 6)  # Mpc

def nfw_deltasigma(r, mass, z_cluster):
    rho_m = cosmo.critical_density0.to_value("Msun/Mpc^3") * cosmo.Om0
    return NfwProfile(m200=mass, c200=4.0, rho_ref=rho_m).deltasigma(r)

stack = StackedDeltaSigma.from_profile(abundance, nfw_deltasigma, radii)
ds = stack.profile()
print("DeltaSigma_ij^1h(R) [Msun/Mpc^2], lowest redshift bin, rises with richness:")
print(ds[:, 0])

# the identity that proves the stack IS the counts' own weight
ones = np.ones_like(stack.profile_grid)
print("\nstacking DeltaSigma=1, max|result - 1| ="
      f" {np.max(np.abs(abundance.average(ones) - 1.0)):.2e}")

# %% [markdown]
# ## Shear projection

# %% tags=["shear-proj"]
# the total stacked shear a driver actually fits: the one-halo term (a
# representative M_200m for a lambda_ob=20 cluster, same illustrative
# mass/concentration as "The Lensing Profile" above) plus the projected
# two-halo excess from "Projection lensing" above -- two separate models,
# summed by hand, since no single class owns both pieces at the binned level.
ds_prj = prj.deltasigma_prj(R_prj, 20.0, 0.5, profile)
rho_m_z0 = cosmo.critical_density0.to_value("Msun/Mpc^3") * cosmo.Om0
ds_1h = NfwProfile(m200=1.0e14, c200=4.0, rho_ref=rho_m_z0).deltasigma(R_prj)
ds_tot = ds_1h + ds_prj
print("DeltaSigma(R) [Msun/Mpc^2] at (lambda_ob=20, z_ob=0.5):")
print(f"{'R [Mpc]':>9s} {'1h':>12s} {'prj':>12s} {'total':>12s} {'prj frac':>10s}")
for k, r in enumerate(R_prj):
    print(f"{r:9.2f} {ds_1h[k]:12.4e} {ds_prj[k]:12.4e} {ds_tot[k]:12.4e} "
          f"{ds_prj[k] / ds_tot[k]:10.4f}")

# %% [markdown]
# ## Covariance

# %% tags=["covariance"]
from clenspy.cosmology import growth_factor
from clenspy.covariance import CountsCovariance

# CountsCovariance: Poisson (shot noise) + sample variance (a coherent
# window mode shared by every cluster in a redshift slice). A DES-Y1-like
# toy counts/bias table; sigma_W(z) = sigma_R(R_eff) * D(z) (linear only).
# sigma_grid is the same real-CAMB SigmaGrid built in the power-spectrum
# section above -- never a toy/analytic P(k).
counts = np.array([[2500.0, 3100.0, 2700.0],
                   [900.0, 1150.0, 1000.0],
                   [300.0, 380.0, 330.0],
                   [110.0, 140.0, 120.0]])
bias = np.array([[2.1, 2.2, 2.3],
                 [2.6, 2.7, 2.8],
                 [3.2, 3.3, 3.5],
                 [4.3, 4.5, 4.8]])
z_mid = np.array([0.28, 0.43, 0.57])

sigma_w = sigma_grid.sigma(120.0, truncate=False) * growth_factor(z_mid)  # R_eff=120 Mpc/h

cc = CountsCovariance(counts, bias, sigma_w)
diag_p = np.sqrt(np.diag(cc.cov_poisson())) / counts.ravel()
diag_s = np.sqrt(np.diag(cc.cov_sample_variance())) / counts.ravel()
print("fractional error by component (Poisson falls with N; sample")
print("variance does not, since it is a coherent mode shared at fixed z):")
print("Poisson       =", np.round(diag_p, 4))
print("sample_var    =", np.round(diag_s, 4))

# %% [markdown]
# ## Halo-to-halo covariance

# %% tags=["covariance-halo-to-halo"]
from clenspy.cosmology import BiasModel
from clenspy.covariance import DeltaSigmaHaloToHaloCovariance
from clenspy.halo import TwoHaloTerm

# the Gaussian covariance treats halo+matter fields as Gaussian and gives
# the variance of the MEAN profile; each cluster in the stack also carries
# its OWN DeltaSigma (mass, concentration scatter), and the stack of N_cl
# of them inherits that population's covariance -- a sixth, independent
# term, scaling as 1/N_cl. Same abundance object as the observables
# section, and the same real CAMB P(k) at z_eff.
z_eff = 0.28
Pk_eff = pk_grid(k_camb, z=z_eff)
rho_m0 = mean_matter_density(cosmo)
twohalo = TwoHaloTerm(k_camb, Pk_eff, zvec=z_eff)
bias_model = BiasModel(k_camb, Pk_eff, cosmo=cosmo)

intrinsic = DeltaSigmaHaloToHaloCovariance(abundance, twohalo, bias_model,
                                           rho_m0, z_eff=z_eff)
print(intrinsic)

radii = np.logspace(-0.7, 1.0, 6)  # Mpc
mean_ds = intrinsic.mean_profile(radii, 0, 0)
sigma_intr = np.sqrt(np.diag(intrinsic.cov(radii, 0, 0)))
print("\nrichness bin 0, redshift bin 0:")
print("<DeltaSigma>  =", mean_ds)
print("sigma_intr    =", sigma_intr)
print("fractional    =", sigma_intr / mean_ds)

# it rises with richness, because the mass population is broader
print("\nmean fractional sigma_intr by richness bin (rises with richness):")
for i in range(sel.n_lambda_bins):
    s = np.sqrt(np.diag(intrinsic.cov(radii, i, 0)))
    m = intrinsic.mean_profile(radii, i, 0)
    print(f"  bin {i}: {np.mean(s / m):.4f}   N_cl = {abundance.counts()[i, 0]:.1f}")
