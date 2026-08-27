# Notation

The symbol → code dictionary for CLensPy. One row per quantity: what it is
called, how it is written in the papers, what it means, and the units the code
carries it in.

**Sources.** The papers are the specification; the code transliterates them.

| Tag | Document |
|---|---|
| **E26** | `des-y1-cluster-optical-selection/main.tex` — Esteves, *Optical cluster cosmology with projection effects* |
| **RSF** | `RichnessSelection/docs/richness_selection_function.tex` — closed-form richness selection function |
| **RSZ** | `RichnessSelection/docs/richness_selection_frozen.tex` — frozen-physics variant |
| **EIN** | `CLensPy/docs/einasto_proj_density_v3.tex`, `einasto_math.md` |
| **MIS** | `CLensPy/docs/miscentering_math.md` |
| **C19a/b, C21, C26** | Costanzi et al. 2019a (arXiv:1807.11719), 2019b, 2021, 2026 |
| **McC19** | McClintock et al. 2019 (arXiv:1805.00039) |

**Status.** ✅ implemented · 🔶 partial · ⬜ not yet written.

---

## Unit convention

CLensPy is **h-free absolute** throughout: mass in $M_\odot$, length in Mpc,
density in $M_\odot\,{\rm Mpc}^{-3}$, wavenumber in ${\rm Mpc}^{-1}$. No
quantity in the package carries an $h$. Papers and external pipelines that use
$h^{-1}M_\odot$ / $h^{-1}$Mpc are converted **at the boundary**, in one visible
multiplication.

Comoving unless a name says otherwise. In particular $\rho_m$ is the *comoving*
mean matter density — evaluated at $z=0$ with no redshift dependence. Using
$\rho_c(z)$ in its place folds in $E^2(z)$ and overstates it by 34% at $z=0.25$.

---

## 1. Cosmology and background

| Quantity | Symbol | Meaning | Units | Code | Module | |
|---|---|---|---|---|---|---|
| Hubble constant | $H_0$ | Expansion rate today | km/s/Mpc | `H0` | `cosmology.fiducial` | ✅ |
| Reduced Hubble | $h$ | $H_0/100$. **Never** the Einasto scale radius | — | — | — | ✅ |
| Matter density | $\Omega_{m,0}$ | Present-day matter density parameter | — | `Om0` | `cosmology.fiducial` | ✅ |
| Critical density | $\rho_{c,0}$ | Closure density today | $M_\odot\,{\rm Mpc}^{-3}$ | `critical_density0` | astropy | ✅ |
| Mean matter density | $\rho_m$ | $\Omega_{m,0}\rho_{c,0}$, **comoving**, no $z$ dependence | $M_\odot\,{\rm Mpc}^{-3}$ | `mean_matter_density()` | `cosmology.fiducial` | ✅ |
| Comoving distance | $\chi(z)$ | Line-of-sight comoving distance | Mpc | `comoving_distance` | astropy | ✅ |
| Angular diameter distance | $D_A(z)$ | $\chi/(1+z)$ for flat | Mpc | `angular_diameter_distance` | astropy | ✅ |
| Lens–source distance | $D_A(z_l,z_s)$ | $D_A(z_s)-\frac{1+z_l}{1+z_s}D_A(z_l)$ — **not** $D_A(z_s)-D_A(z_l)$ | Mpc | `angular_diameter_distance_z1z2` | astropy | ✅ |
| Volume element | $\dfrac{dV}{d\Omega\,dz}$ | Comoving volume per steradian per unit $z$ | ${\rm Mpc}^3\,{\rm sr}^{-1}$ | ⬜ | `cosmology` | ⬜ |
| Linear power spectrum | $P(k,z)$ | Matter power spectrum | ${\rm Mpc}^{3}$ | `PkGrid` | `cosmology.pkgrid` | ✅ |
| Nonlinear correlation | $\xi_{\rm NL}(r,z)$ | Nonlinear matter 2-point function | — | `TwoHaloTerm.xi` | `halo.twohalo` | ✅ |
| Speed of light | $c$ | | km/s | `C_LIGHT` | `utils.constants` | ✅ |
| Newton constant | $G$ | | ${\rm Mpc}\,M_\odot^{-1}({\rm km/s})^2$ | `G_NEWTON` | `utils.constants` | ✅ |

## 2. Halo profile

| Quantity | Symbol | Meaning | Units | Code | Module | |
|---|---|---|---|---|---|---|
| Halo mass | $M_{200}$ | Mass inside $r_{200}$. Definition set by `rho_ref` | $M_\odot$ | `m200` | `halo.nfw` | ✅ |
| Reference density | $\rho_{\rm ref}$ | Closes $M_{200}=200\rho_{\rm ref}\frac{4}{3}\pi r_{200}^3$. **Carries the mass definition**: $\rho_m\Rightarrow M_{200m}$, $\rho_c\Rightarrow M_{200c}$ | $M_\odot\,{\rm Mpc}^{-3}$ | `rho_ref` | `halo.nfw` | ✅ |
| Halo radius | $r_{200}$ | Overdensity radius | Mpc | `r200` | `halo.nfw` | ✅ |
| Concentration | $c_{200}$ | $r_{200}/r_s$ | — | `c200` | `halo.nfw` | ✅ |
| Scale radius | $r_s$ | NFW scale radius | Mpc | `rs` | `halo.nfw` | ✅ |
| Characteristic density | $\rho_s$ | NFW normalisation | $M_\odot\,{\rm Mpc}^{-3}$ | `rho_s` | `halo.nfw` | ✅ |
| 3D density | $\rho(r)$ | $\rho_s/[x(1+x)^2]$, $x=r/r_s$ | $M_\odot\,{\rm Mpc}^{-3}$ | `density(r)` | `halo.nfw` | ✅ |
| Fourier profile | $u(k\mid M)$ | Normalised FT of $\rho$ | — | `fourier(k)` | `halo.nfw` | ✅ |
| Einasto index | $n$ | $\rho=\rho_0\exp[-(r/r_s)^{1/n}]$; $\alpha=1/n$ | — | `n`, `alpha` | `halo.einasto` | ✅ |
| Einasto scale radius | $h$ *(notes)* | **The notes write $h$; the code says `r_s`.** Not $H_0/100$ | Mpc | `r_s` | `halo.einasto` | ✅ |
| Einasto normalisation | $\rho_0$ | Central density scale | $M_\odot\,{\rm Mpc}^{-3}$ | `rho_0` | `halo.einasto` | ✅ |
| Halo mass function | $n(M,z)$ | $dn/dM$ (papers) — check against $dn/d\ln M$ | ${\rm Mpc}^{-3}M_\odot^{-1}$ | ⬜ | `cosmology` | ⬜ |
| Linear halo bias | $b(M,z)$ | Tinker et al. 2010 | — | `BiasModel.bias(M)` | `halo.bias` | ✅ |
| Peak height | $\nu$ | $\delta_c/\sigma(M)$, $\delta_c=1.686$ | — | `nu_at_mass(M)` | `halo.bias` | ✅ |
| Mass variance | $\sigma(M)$ | Top-hat filtered $\sigma$ | — | `sigma_tophat(M)` | `halo.bias` | ✅ |

## 3. Lensing observables

| Quantity | Symbol | Meaning | Units | Code | Module | |
|---|---|---|---|---|---|---|
| Projected radius | $R$ | Comoving transverse separation | Mpc | `R` | all | ✅ |
| Surface density | $\Sigma(R)$ | Projected mass density | $M_\odot\,{\rm Mpc}^{-2}$ | `sigma(R)` | `halo.nfw`, `lensing` | ✅ |
| Mean interior $\Sigma$ | $\bar\Sigma(<R)$ | $\frac{2}{R^2}\int_0^R \Sigma(R')R'dR'$ | $M_\odot\,{\rm Mpc}^{-2}$ | `mean_sigma(R)` | `lensing.miscentering` | ✅ |
| Excess surface density | $\Delta\Sigma(R)$ | $\bar\Sigma(<R)-\Sigma(R)$ | $M_\odot\,{\rm Mpc}^{-2}$ | `deltasigma(R)` | `halo.nfw`, `lensing` | ✅ |
| Critical surface density | $\Sigma_{\rm crit}$ | $\frac{c^2}{4\pi G}\frac{D_s}{D_l D_{ls}}$ | $M_\odot\,{\rm Mpc}^{-2}$ | `sigma_critical()` | `cosmology.utils` | ✅ |
| **Mean inverse** $\Sigma_{\rm crit}$ | $\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)$ | $h_0\!\int\!dz_s\,p(z_s{+}\Delta z)\frac{4\pi G}{c^2}\frac{D_A(z_l)D_A(z_l,z_s)}{D_A(z_s)}$, clamped $\ge 0$. **Average the inverse, never invert the average** | ${\rm Mpc}^2 M_\odot^{-1}$ | ⬜ | `kernels` | ⬜ |
| Tangential shear | $\gamma_t$ | $\Delta\Sigma\cdot\langle\Sigma_{\rm crit}^{-1}\rangle$. Setting the average to 1 emits $\Delta\Sigma$ | — | `shear(R)` | `halo.einasto` | 🔶 |
| Convergence | $\kappa$ | $\Sigma/\Sigma_{\rm crit}$ | — | `convergence(R)` | `halo.einasto` | 🔶 |
| One-halo term | $\Sigma^{1h}$, $\Delta\Sigma^{1h}$ | The cluster's own halo | $M_\odot\,{\rm Mpc}^{-2}$ | `halo_profile` | `lensing.profile` | ✅ |
| Two-halo / projection | $\Sigma^{\rm prj}$, $\Delta\Sigma^{\rm prj}$ | Correlated LSS along the line of sight (E26 eq. Sprj) | $M_\odot\,{\rm Mpc}^{-2}$ | `TwoHaloTerm` | `halo.twohalo` | 🔶 |
| Source redshift | $z_s$ | | — | `z_source` | `lensing.profile` | 🔶 |
| Lens / cluster redshift | $z_l$, $z_{\rm cls}$ | | — | `z_cluster` | `lensing.profile` | ✅ |

> 🔶 `TwoHaloTerm` implements the linear-bias $b(M)\rho_m\xi(r)$ form, **not** the
> full $\Sigma^{\rm prj}$ of E26 eq. `Sprj` (no $\bar b_{\rm sel}$, no exclusion,
> no $\theta$ integral). See §5 and `refactor-plan.md` errata E.3.
>
> 🔶 `lensing.miscentering` computes the azimuthal average at **one fixed
> offset** $R_{\rm mis}$. The Gamma-distributed offset population and the
> $f_{\rm mis}$ mixture of E26 eq. `miscentering_model` are not implemented.
> The same single-offset average serves the two-halo term of E26 eq. `Smis`,
> where the "offset" is the halo–halo separation $R_\theta$ and carries no
> nuisance parameter.

## 4. Richness, selection, miscentering

| Quantity | Symbol | Meaning | Units | Code | Module | |
|---|---|---|---|---|---|---|
| True richness | $\lambda^{\rm tr}$ | Latent halo galaxy content | — | `lam_tr` | ⬜ | ⬜ |
| Observed richness | $\lambda^{\rm ob}$ | redMaPPer measured richness | — | `lam_ob` | ⬜ | ⬜ |
| True redshift | $z^{\rm tr}$ | | — | `z_tr` | ⬜ | ⬜ |
| Observed redshift | $z^{\rm ob}$ | Cluster photo-z | — | `z_ob` | ⬜ | ⬜ |
| Richness bin | $\Delta\lambda_i=[\lambda_i^{\min},\lambda_i^{\max}]$ | $i$-th richness bin | — | `RichnessBin` | `utils.binning` | ⬜ |
| Redshift bin | $\Delta z_j=[z_j^{\min},z_j^{\max}]$ | $j$-th photo-z bin | — | `RichnessBin` | `utils.binning` | ⬜ |
| Binned counts | $\langle N_{ij}\rangle$ | Expected clusters in bin $(i,j)$ (RSF eq. Nij_2D) | — | ⬜ | ⬜ | ⬜ |
| Survey solid angle | $\Omega(z)$ | Effective footprint. **In counts; cancels in shear** | sr | ⬜ | `survey` | ⬜ |
| Selection function | $\mathcal S_{ij}(\lambda^{\rm tr},z^{\rm tr})$ | Prob. of scattering into bin $(i,j)$; factorises as $\mathcal S_i\mathcal S_j$ | — | ⬜ | `selection` | ⬜ |
| Richness kernel | $\mathcal S_i$, $\mathcal K_i$ | $(1-f^{\rm prj})\Phi\vert_{\Delta\lambda_i}+f^{\rm prj}F_{\rm EMG}\vert_{\Delta\lambda_i}$ (RSF eq. Ki_final) | — | ⬜ | `selection` | ⬜ |
| Photo-z kernel (counts) | $\mathcal S_j(z^{\rm tr})$ | $\Phi\big(\frac{z^{\rm ob}-z^{\rm tr}}{\sigma_z}\big)\big\vert_{\Delta z_j}$ — **Gaussian** | — | ⬜ | `selection` | ⬜ |
| Photo-z weight (projection) | $w_{pz}(z;z^{\rm ob})$ | $1-u^2$ for $\vert u\vert<1$, $u=(z-z^{\rm ob})/\sigma_z(z)$ — **parabolic, a different kernel** | — | ⬜ | `selection` | ⬜ |
| Photo-z scatter | $\sigma_z(\Delta\lambda_i)$ | Bin-dependent | — | ⬜ | `selection` | ⬜ |
| Photo-z bias | $\Delta z$ | Source $p(z)$ shift, marginalised | — | ⬜ | `survey` | ⬜ |
| EMG CDF | $F_{\rm EMG}(x;\mu,\sigma,\tau)$ | $\Phi(\frac{x-\mu}{\sigma})-e^{-\tau(x-\mu)+\frac{1}{2}\tau^2\sigma^2}\Phi(\frac{x-\mu}{\sigma}-\tau\sigma)$ (RSF eq. exg_cdf) | — | ⬜ | `selection` | ⬜ |
| Kernel mean | $\mu=\lambda^{\rm tr}+\Delta\mu$ | $\Delta\mu<0$: background subtraction biases low | — | ⬜ | `selection` | ⬜ |
| Kernel width | $\sigma(\lambda^{\rm tr},z)$ | Photometric + background noise | — | ⬜ | `selection` | ⬜ |
| Projection fraction | $f^{\rm prj}(\lambda^{\rm tr},z)$ | Fraction with a projection boost, $\in[0,1]$ | — | ⬜ | `selection` | ⬜ |
| Projection tail rate | $\tau(\lambda^{\rm tr},z)$ | Inverse scale; **smaller $\tau$ = longer tail**. Distinct from miscentering $\tau$ | — | ⬜ | `selection` | ⬜ |
| MOR mean (log-normal) | $\langle\ln\lambda\rangle$ | $\ln A_\lambda+B_\lambda\ln\frac{M}{M_p}+C_\lambda\ln\frac{1+z}{1+z_p}$; $M_p=3\times10^{14}h^{-1}M_\odot$, $z_p=0.45$ | — | ⬜ | `selection` | ⬜ |
| MOR scatter | $\sigma_{\ln\lambda}$ | $D_\lambda^2+\frac{\langle\lambda\rangle-1}{\langle\lambda\rangle^2}$ | — | ⬜ | `selection` | ⬜ |
| HOD satellite mean | $\langle\lambda^{\rm sat}\rangle$ | $\big(\frac{M-M_{\min}}{M_1-M_{\min}}\big)^\alpha\big(\frac{1+z}{1+z_\star}\big)^\epsilon$, $z_\star=0.45$ | — | ⬜ | `selection` | ⬜ |
| HOD scatter | $\sigma_{\rm intr}$ | Super-Poisson halo-to-halo term | — | ⬜ | `selection` | ⬜ |
| Shifted-Poisson rate | $\nu=\langle\lambda^{\rm sat}\rangle+\delta$, $\delta=(\sigma_{\rm intr}\langle\lambda^{\rm sat}\rangle)^2$ | RSF eq. cont_poisson_shifted | — | ⬜ | `selection` | ⬜ |
| Cluster radius | $R_\lambda$ | $(\lambda^{\rm ob}/100)^{0.2}$ | ${\rm Mpc}/h$ | ⬜ | `selection` | ⬜ |
| Miscentering offset | $R_{\rm mis}$ | Projected centring error, **a single fixed offset** | Mpc | `r_mis` | `lensing.miscentering` | ✅ |
| Halo-centric radius | $R_h$ | $\sqrt{R^2+R_{\rm mis}^2-2RR_{\rm mis}\cos\varphi}$ — law of cosines, in the half-angle form $\sqrt{(R-R_{\rm mis})^2+4RR_{\rm mis}\sin^2\frac{\varphi}{2}}$ | Mpc | `_halo_centric_radii` | `lensing.miscentering` | ✅ |
| Miscentered fraction | $f_{\rm mis}$ | $0.25\pm0.08$ (Kelly 2024); $\Delta\Sigma^{1h}=(1-f_{\rm mis})\Delta\Sigma_{\rm cen}+f_{\rm mis}\Delta\Sigma_{\rm mis}$ | — | ⬜ | `selection` | ⬜ |
| Miscentering scale | $\tau_{\rm mis}$ | $0.17\pm0.04$; $p(R_{\rm mis})=\frac{R_{\rm mis}}{(\tau R_\lambda)^2}e^{-R_{\rm mis}/(\tau R_\lambda)}$. **Not the EMG $\tau$** | — | ⬜ | `selection` | ⬜ |
| Boost factor | $\mathcal B(R)$ | Member dilution of the source sample (McC19) | — | `boost_factor_nfw` | `lensing.boost` | ✅ |

## 5. Projection two-halo (E26 §4.1) — not yet implemented

| Quantity | Symbol | Meaning | Units | |
|---|---|---|---|---|
| Angular separation | $\theta$ | LOS neighbour angular offset. Measure is $2\pi\sin\theta\,d\theta$ — spherical, **no Limber** | rad | ⬜ |
| Projected offset | $R_\theta=\theta D_A(z^{\rm ob})$ | Neighbour transverse separation | Mpc | ⬜ |
| Selection bias | $\bar b_{\rm sel}(\lambda^{\rm ob},z^{\rm ob},\theta)$ | $\bar b_{\rm small}[1-\sigma(\theta)]+\bar b_{\rm large}\sigma(\theta)$; multiplies the **correlated channel only** | — | ⬜ |
| LOS separation | $d\chi$ | $\sqrt{\chi_z^2+\chi_o^2-2\chi_z\chi_o\cos\theta}$ — law of cosines | Mpc | ⬜ |
| Exclusion angle | $\theta_{\rm excl}(z)$ | Law of cosines from $R_\lambda$; masks $\theta\le\theta_{\rm excl}$ | rad | ⬜ |
| Random channel | $w_{\rm rnd}(M)$ | $\int dz\,{\rm common}(z)\,n(M,z)$ | | ⬜ |
| Correlated channel | $w_{\rm cl}(\theta,M)$ | $\int dz\,{\rm common}(z)\,\xi_{\rm NL}\,n\,b\,\mathbb 1[\theta>\theta_{\rm excl}]$ | | ⬜ |
| Common weight | ${\rm common}(z)$ | $\frac{dV}{d\Omega dz}w_{pz}(z;z^{\rm ob})w_z^{\rm GL}$ — **no $\Omega(z)$** | | ⬜ |

Keep `rnd` and `cl` stored separately and sum at the end; the scientific
argument is about which dominates where.

---

## Collisions to watch

Four symbols mean two things. Each has bitten this codebase or its references.

| Symbol | Meaning A | Meaning B | Resolution |
|---|---|---|---|
| $h$ | Reduced Hubble $H_0/100$ | Einasto scale radius (EIN notes) | Code always says `r_s` for the radius; `h` is reserved for Hubble |
| $\tau$ | EMG projection tail rate (C19a) | Miscentering offset scale (McC19) | `tau_prj` vs `tau_mis` — never bare `tau` |
| $M_{200}$ | $M_{200m}$ (mean) | $M_{200c}$ (critical) | Carried by `rho_ref`; ~30% mass error if confused |
| $\sigma$ | Richness-kernel width | Surface density $\Sigma$ / sigmoid $\sigma(\theta)$ | Capitalisation is load-bearing: `Sigma` vs `sigma` |
| $\alpha$ | Einasto shape $1/n$ | HOD mass slope | Module scope keeps them apart |

Two more that are not symbol collisions but produce the same class of error:

- **$\Omega(z)$ belongs to counts and cancels in shear.** Never apply one weight
  builder's footprint to both.
- **The photo-z kernel differs by observable**: Gaussian CDF for counts,
  parabolic for the projection. Two implementations, deliberately.
