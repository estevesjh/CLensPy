# LLM/agent quickstart

This page is for consumers — human or LLM coding agent — calling CLensPy
from another project. It states the API contracts that are easy to get
wrong, then gives one verified recipe per common task. Every snippet on
this page runs as-is in an environment with CLensPy and CAMB installed.

## The three rules

Three mistakes account for most misuse of this package from the outside;
each has a one-line fix.

1. **Never `import camb`.** CLensPy wraps CAMB inside
   {class}`~clenspy.cosmology.pkgrid.PkGrid` — including the h-unit
   conversion (CLensPy is h-free: k in 1/Mpc, P in Mpc³, M in Msun, R in
   Mpc), the sigma8 renormalisation, and a disk cache. Configure the
   astropy cosmology object and let CLensPy call the Boltzmann solver.
2. **Never `np.trapz` the transforms.** P(k) → ξ(r) is an oscillatory
   Fourier–Bessel integral and Σ-from-ξ is a singular Abel projection;
   both go through the dedicated machinery in `clenspy.utils.integrate`
   (`pk_to_xi_fftlog` via FFTLog/mcfit, `compute_sigma_grid` on a
   cosh-substituted grid, `gl_nodes`/`mass_nodes` cached Gauss–Legendre
   rules) or through the classes that call it (`TwoHaloTerm`,
   `LensingProfile`). Trapezoids on linspace grids are the failure mode
   this package exists to prevent.
3. **Never loop over array inputs.** The interpolator-backed evaluators
   (`dndlnm`, `bias`, `xi`, `sigma`, `deltasigma`) are vectorized grid
   queries: vector `x` + vector `z` returns the outer `(nx, nz)` grid,
   vector + scalar returns `(nx,)`, scalar + scalar a float. Pass whole
   arrays. Two verified exceptions: `PkGrid.__call__` follows NumPy
   broadcasting instead — `pk(kvec, zvec)` with different lengths raises,
   use `pk(kvec[:, None], zvec)` for the `(nk, nz)` grid — and the
   low-level {class}`~clenspy.cosmology.sigma.SigmaGrid`, whose
   `sigma`/`sigma2` take scalar R (use its `sigma2_fftlog`, or the
   mass-function/bias wrappers below, for arrays).

## Setup: the cosmology object

Every layer takes one astropy cosmology; build it first and pass it
everywhere.

```python
import numpy as np
from clenspy.cosmology import fiducial_cosmology

cosmo = fiducial_cosmology(H0=70.0, Om0=0.286)  # Buzzard-like flat LCDM
```

`fiducial_cosmology` returns a fresh `FlatLambdaCDM` per call. sigma8 and
n_s are not carried by astropy cosmologies; `PkGrid` defaults them to 0.8
and 0.96 unless your cosmology object carries those attributes.

## ξ_NL(r, z) at a given cosmology

`PkGrid` computes halofit P(k, z) (CAMB inside, disk-cached), and
`TwoHaloTerm` FFTLogs it to the correlation function.

```python
from clenspy.cosmology import PkGrid
from clenspy.halo import TwoHaloTerm

pk_nl = PkGrid(cosmo=cosmo, nonlinear=True)      # halofit P(k, z)
z = 0.3
two_halo = TwoHaloTerm(pk_nl.k, pk_nl(pk_nl.k, z=z), zvec=z)
r = np.logspace(-0.5, 2.0, 40)                   # Mpc
xi_nl = two_halo.xi(r, z)                        # shape (40,)
```

For a multi-z interpolator, pass the full grid once:
`TwoHaloTerm(pk_nl.k, pk_nl.pk, zvec=pk_nl.z)`; then `xi(rvec, zvec)`
returns the outer `(nr, nz)` grid.

## σ(M, z), mass function, and bias

`BiasModel` and `TinkerMassFunction` build their own linear `PkGrid` from
the cosmology (lazily, disk-cached) — no manual P(k) plumbing needed. All
three queries below are array-in, array-out.

```python
from clenspy.cosmology import BiasModel, TinkerMassFunction

M = np.logspace(13.0, 15.0, 30)                  # Msun (M_200m)
bias_model = BiasModel(cosmo=cosmo)
sigma_M = bias_model.sigma_tophat(M, z=0.3)      # sigma(M, z), shape (30,)
b_M = bias_model.bias(M, z=0.3)                  # Tinker (2010), shape (30,)

hmf = TinkerMassFunction(cosmo=cosmo)            # Tinker (2008), Delta=200m
dndlnm = hmf.dndlnm(M, z=0.3)                    # Mpc^-3, shape (30,)
```

To integrate over the mass function, use the quadrature helpers rather
than a linspace trapezoid — e.g. the number density above 10¹⁴ Msun:

```python
from clenspy.utils.integrate import gl_nodes

lnM, w = gl_nodes(np.log(1e14), np.log(1e15), 32)
n_cl = np.sum(w * hmf.dndlnm(np.exp(lnM), z=0.3))   # Mpc^-3
```

(`mass_nodes` does the same in M with the dM Jacobian folded into the
weights.)

## Σ(R) and ΔΣ(R)

For a single halo without large-scale structure, `NfwProfile` is
closed-form and needs no cosmology object (mass definition rides on
`rho_ref`; the default is M_200m):

```python
from clenspy.halo import NfwProfile

nfw = NfwProfile(m200=1e14, c200=4.0)
R = np.logspace(-1, 1, 15)                       # Mpc, projected
sigma_1h = nfw.sigma(R)                          # Msun/Mpc^2
ds_1h = nfw.deltasigma(R)                        # Msun/Mpc^2
```

For the full observable (1-halo + linear-bias 2-halo term),
`LensingProfile` assembles the chain internally — the constructor only
stores; CAMB runs on the first evaluation:

```python
from clenspy.lensing import LensingProfile

lp = LensingProfile(z_cluster=0.3, m200=1e14, concentration=4.0,
                    cosmology=cosmo)
ds = lp.deltasigma(R)                            # Msun/Mpc^2, shape (15,)
```

Never reconstruct ΔΣ(R) from a tabulated Σ(R) by cumulative trapezoid —
ΔΣ is its own integral (see the house rule in the theory pages; the
`deltasigma` methods already do the right thing).

## Selection bias: `SelBiasEngine` basics

The engine computes the θ-dependent selection-affected bias
b_sel(θ) of a richness-selected cluster, sharing one halo-model chain
with {class}`~clenspy.lensing.projection.SigmaPrj`:

```python
from clenspy.lensing import SigmaPrj
from clenspy.selection import HodMor, SelBiasEngine

engine = SelBiasEngine(
    sigma_prj=SigmaPrj(cosmology=cosmo).build(), mor=HodMor.des_y1(),
    n_z=32, n_M=16, n_theta=8, n_ltr=40, ltr_grid_size=10,
)
profile = engine.marginalised_bias(40.0, 0.4)    # (lambda_ob, z_ob)
b_at = profile(profile.theta_lambda)             # b_sel at theta_lambda
```

`profile` interpolates between the plateaus `profile.b_small` (inside the
aperture) and `profile.b_large` (well outside). The low quadrature orders
above are demo-sized: the b_sel *shape* is right but its amplitude needs
the production orders (the defaults) and the calibrated pipeline — see
{doc}`selection_bias`.

## Executed shape contracts

Every line below was run in the repo's environment with
`cosmo = fiducial_cosmology(H0=70.0, Om0=0.286)`; the shapes and numbers
are the actual outputs, not descriptions. `kvec4 = np.logspace(-2, 0, 4)`,
`Mvec4 = np.logspace(13, 15, 4)`, `Rvec3 = np.array([1.0, 5.0, 20.0])`,
`zvec3 = np.array([0.0, 0.3, 0.6])`.

```python
# PkGrid (nonlinear=True): NumPy BROADCASTING, not an outer grid
pk(0.1, 0.3)                 # float: 7550.217
pk(kvec4, 0.3)               # (4,):  [59730.03, 21808.83, 2488.96, ...]
pk(0.1, zvec3)               # (3,):  [10285.90, 7550.22, 5577.48]
pk(kvec4, zvec3)             # RAISES ValueError (4,) vs (3,)
pk(kvec4[:, None], zvec3)    # (4, 3) grid: [[80910.18, 59730.03, ...], ...]

# SigmaGrid (linear P(k) of the same cosmology): scalar-R by design
sg.sigma(8.0)                          # float: 1.005278
sg.sigma(Rvec3)                        # RAISES TypeError
sg.sigma2_fftlog(np.log(Rvec3))[0]     # (3,) ln sigma^2: [1.978, 0.011, -1.259]

# TinkerMassFunction / BiasModel: outer (nM, nz) grid for vector+vector
hmf.dndlnm(1e14, 0.3)        # float: 1.6e-05
hmf.dndlnm(Mvec4, 0.3)       # (4,):  [2.12e-04, 4.10e-05, 5.0e-06, ...]
hmf.dndlnm(Mvec4, zvec3)     # (4, 3) outer grid
bm.bias(1e14, 0.3)           # float: 2.4658
bm.bias(Mvec4, zvec3)        # (4, 3) outer grid
bm.sigma_tophat(Mvec4, z=0.3)  # (4,): [1.279, 0.970, 0.706, ...]
bm.sigma_tophat(Mvec4, z=zvec3)  # RAISES ValueError -- z must be scalar

# TwoHaloTerm (built on the full (k, z) grid): outer grid for vector+vector
th.xi(5.0, 0.3)              # float: 1.0862
th.xi(Rvec3, 0.3)            # (3,):  [30.367, 1.086, 0.134]
th.xi(Rvec3, zvec3)          # (3, 3) outer grid
th.sigma(Rvec3, 0.3)         # (3,) unnormalised [Mpc]: [91.85, 23.20, 7.20]

# NfwProfile(m200=1e14, c200=4.0), scalar halo
nfw.sigma(0.5)               # float: 3.351e+13 Msun/Mpc^2
nfw.sigma(Rvec3)             # (3,):  [1.214e+13, 7.056e+11, 4.781e+10]
# NfwProfile with a VECTOR of halos, m200=[1e14, 5e14]
nfw2.sigma(Rvec3)            # (2, 3): one row per halo
nfw2.sigma(0.5)              # (2, 1) -- note the kept R axis
nfw2.fourier(0.5)            # (2,)   -- fourier squeezes the k axis
```

## Where to go next

- {doc}`index` — the Theory toctree: one page per physical effect, with
  prose, the governing equation, and a runnable snippet.
- {doc}`api/index` — the full API reference. Docstrings state the array
  contracts (shapes, vectorization, units) per method.
- `examples/getting_started.ipynb` in the repository — every snippet on
  the docs Theory pages, runnable top to bottom.
