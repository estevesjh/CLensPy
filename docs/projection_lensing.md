# Projection Lensing: Σ_prj and ΔΣ_prj

A cluster selected on observed richness does not sit in a random patch of
sky. The same line-of-sight structure that boosted its richness also lenses
the sources behind it, so the stacked surface density around a
$\lambda^{\rm ob}$-selected sample carries a *projected two-halo* term over
and above the cluster's own halo. `SigmaPrj` computes that term — the
Costanzi et al. (2026) Eq. 13 observable — from the pieces the rest of the
package already owns: the offset-NFW kernel ({doc}`miscentering`), the
selection-affected bias $b_{\rm sel}(\theta)$ ({doc}`selection_bias`), and
the halo model ({doc}`mass_function`, {doc}`halo_bias`,
{doc}`power_spectrum`).

```{figure} _static/img/projection_lensing.png
:alt: Sigma_prj channel decomposition and the selection-bias ratio
:width: 95%
:align: center

Left: the observable $\Sigma^{\rm prj}$ (correlated excess — the only
place $b_{\rm sel}$ enters), the near-uniform background $\Sigma_{\rm
bkg}$, and their sum $\Sigma_{\rm tot}$ (the raw mass-map quantity), for
the Buzzard-mock configuration. Right: the selection-bias observable,
the selected stack over the $b_{\rm eff}$-weighted random stack.
```

## The master equation

The two-halo term is the correlated excess *above* the mean matter column
— there is no background in it. This is cluster_toolkit's $\Sigma_{2h}$
convention, and it is what a random-point-subtracted measurement contains.
Around a cluster observed at $(\lambda^{\rm ob}, z^{\rm ob})$, every
neighbour halo of mass $M$ at angular offset $\theta$ contributes its own
**mass shell** $M_\theta(R \mid M)$ — *not* an aperture mass "inside $R$":
$R$ is the fixed point where the surface density is evaluated, and
$M_\theta$ is a shell in the neighbour's own offset
$R_\theta=\theta\chi_o$, converted from a mass into a density,
$$
M_\theta(R\mid M) = \frac{\sin\bar\theta/\bar\theta}{\chi_o^2}
  \int_{\theta\text{-shell}} 2\pi s\,\Sigma_{\rm mis}(R,s\mid M)\,ds,
\qquad s\equiv\theta\chi_o,
$$
using the single-offset kernel of {doc}`miscentering` (the offset now a
physical separation rather than a centring error). The $1/\chi_o^2$ turns
the raw shell mass (Msun) into a density (Msun/Mpc$^2$) — required for
the master equation below to come out in the right units, since
$n_{\rm cl}(\theta,M)$ is a number per unit mass, not per unit area — and
$\sin\bar\theta/\bar\theta$ corrects the shell (built on the flat
$\theta$ measure via `theta_edges`/`theta_grid`) to the exact
$2\pi\sin\theta\,d\theta$ measure used below. See "Numerics" for the
shell-mass integral itself. That weight, $n_{\rm cl}(\theta, M)$,
is the correlated excess above the uncorrelated background rate
$n_{\rm rnd}(\theta, M) = \int dz\;{\rm common}(z)\, n(M, z)$
($n$ the mass function, ${\rm common}(z) = \tfrac{dV}{d\Omega\,dz}\,
w_{pz}(z; z^{\rm ob})$):

$$
n_{\rm cl}(\theta, M) = \int dz\;{\rm common}(z)\, n(M, z)\, b(M, z)\,
  b_{\rm sel}(\theta)\, \xi_{\rm NL}(|d\chi|, z^{\rm ob}),
\qquad |d\chi| > R_{\rm excl},
$$

with $b$ the halo bias, $b_{\rm sel}(\theta)$ the selection-affected bias,
and $|d\chi|$ the exact law-of-cosines chord $d\chi^2 = \chi_z^2 +
\chi_o^2 - 2\chi_z\chi_o\cos\theta$ — near the exclusion ring the
transverse leg dominates and the $|\chi_z - \chi_o|$ shortcut is wrong by
orders of magnitude. A halo inside the exclusion ball, $|d\chi| \le
R_{\rm excl}$, *is* the cluster: its entire neighbour count must vanish,
not merely its clustering excess, so there $n_{\rm cl}(\theta, M) =
-n_{\rm rnd}(\theta, M)$ — certainty of absence, carrying no bias and no
$b_{\rm sel}$ (see "Exclusion" below). The master equation sums the mass
shell against the correlated weight over every offset and neighbour mass,

$$
\Sigma_{\rm prj}(R) = \int d\theta\, 2\pi\sin\theta \int dM\;
  n_{\rm cl}(\theta, M)\, M_\theta(R \mid M).
$$

A raw projected *mass map* — the Costanzi mock's per-halo columns, or any
stack that has not been random-point subtracted — additionally contains
the mean background column, the uncorrelated $n_{\rm rnd}(\theta, M)$
weight defined above. That piece is kept available as the separate `rnd`
channel,

$$
\Sigma_{\rm bkg}(R) = \int d\theta\, 2\pi\sin\theta \int dM\;
  n_{\rm rnd}(\theta, M)\, M_\theta(R \mid M),
$$

near-uniform in $R$ and blind to the selection. `sigma_prj` and
`deltasigma_prj` return the correlated piece by default; pass
`channel="sum"` only when comparing against a raw mass map. Both channels
are always stored on the object (`components()`), because the scientific
argument is about which dominates where.

Three named conventions, all of which have bitten a pipeline before: the
measure is $2\pi\sin\theta\,d\theta$ — an exact angular integral on the
sphere, no Limber approximation and no Bessel transform; the photo-z
weight $w_{pz}$ is the **parabolic** projection kernel of
{doc}`lensing_kernel`'s sibling `photoz_projection`, never the Gaussian
counts kernel; and the survey footprint $\Omega(z)$ does **not** appear —
it cancels in the surface density, and folding it in is a silent
normalisation error.

## ΔΣ_prj is its own integral

The lensing observable is the excess $\Delta\Sigma_{\rm prj} =
\bar\Sigma_{\rm prj}(<R) - \Sigma_{\rm prj}(R)$. Because the excess
functional acts only on the radial argument, it commutes with the outer
$(\theta, M)$ integral, and

$$
\Delta\Sigma_{\rm prj}(R) = \int d\theta\, 2\pi\sin\theta \int dM\;
  n_{\rm cl}(\theta, M)\, \Delta M_\theta(R \mid M)
$$

is the *same* master equation with the mass shell swapped for its signed
excess, $M_\theta \to \Delta M_\theta$ — never a numerical reconstruction from a
tabulated $\Sigma_{\rm prj}$. The signed negative lobe of
$\Delta\Sigma_{\rm mis}$ at $R_\theta > R$ ({doc}`miscentering`) is
load-bearing here: mass conservation of the azimuthal average makes
$\int d^2s\, \Delta\Sigma_{\rm mis}(R, s) = 0$ exactly, so the excess
functional annihilates the uniform rnd channel and only the correlated cl
channel survives — the model form of the classical random-point
subtraction (Sheldon 2009). Clamping the lobe would break that
cancellation and leave a spurious mean-field term.

## Exclusion: the counter term

A halo closer to the cluster than its own aperture *is* the cluster, so
its neighbour count is zeroed at $R_{\rm excl} =
R_\lambda(\lambda^{\rm ob})(1 + z^{\rm ob})$ comoving — exclusion acts on
the *total* count, not on the correlated excess alone. The default,
`exclusion="counter"`, books that zero as a counterterm of the master
equation (the Costanzi convention): **outside the ball the correlated
weight is $n_{\rm cl}$; inside it is $-n_{\rm rnd}$** — minus the
background weight there — cancelling the background's own contribution
exactly. The total vanishes in the ball — the same total as deleting the
neighbours, as the mock does — but the bookkeeping matters:
$\Sigma_{\rm bkg}$ stays strictly uniform and the exclusion hole is
carried by $\Sigma^{\rm prj}$, where a random-point-subtracted
measurement keeps it. The ball indicator is evaluated per redshift as an
angular cap, $\cos\theta_{\rm excl}(z) = [\chi_z^2 + \chi_o^2 -
R_{\rm excl}^2]/[2\chi_z\chi_o]$ — the angular slicing of the 3-D
exclusion ball, not a separate line-of-sight prescription.
`exclusion="ball"` books the same hole in the background instead
(identical sum); `exclusion="cl"` is the E.3 legacy slab, which merely
zeroes the correlated term — dropping the counterterm, i.e. no exclusion
of the neighbour count itself ($\lesssim 0.6\%$ of the summed profile at
$R \to 0$, gone by $R \approx 2$ cMpc). Switching is a config change, not
a code change.

## Numerics: each θ shell is integrated exactly

The code separates the master equation into two factors and contracts
them at the end: the line-of-sight integrals (`SigmaPrj.n_los_integral`,
three plain integrand closures handed to `integrate_los` on a
`LosGeometry` chord), the exclusion bookkeeping (`Exclusion.channels`),
and the mass shell (`MassShells`, below); `sigma_prj` /
`deltasigma_prj` sum $\sum_{\theta}\sum_M n_{\rm channel}(\theta, M)\,
M_\theta(R \mid M)$.

$\Sigma_{\rm mis}(R, s)$ as a function of the offset $s$ is a ring of
width $\sim r_s$ at $s \approx R$, and no affordable pointwise $\theta$
rule resolves it. `MassShells` therefore integrates each
log-spaced $\theta$ shell **exactly**: the azimuthal average is symmetric,
$\Sigma_{\rm mis}(R, s) = \Sigma_{\rm mis}(s, R)$, so the shell mass is
an enclosed-mass difference of the halo offset by $R$,

$$
\int_{s_1}^{s_2} 2\pi s\,\Sigma_{\rm mis}(R, s)\, ds
= \pi\Sigma_0\Big[s^2\, \hat m\big(s/r_s,\, R/r_s\big)\Big]_{s_1}^{s_2},
\qquad \hat m = \hat\Sigma_{\rm mis} + \widehat{\Delta\Sigma}_{\rm mis},
$$

a genuine mass (Msun), $s_1,s_2$ the edges of one $\theta$-cell times
$\chi_o$. `MassShells.__call__` then applies the $1/\chi_o^2$ and
$\sin\bar\theta/\bar\theta$ factors of "The master equation" above to
turn this shell mass into the $M_\theta(R\mid M)$ that equation actually
uses — the boxed identity above is the mass, not yet the master
equation's density-valued object. $\hat m$ — the mean enclosed surface
density per $\Sigma_0$, `MassShells.mean_sigma` — is read from the
packaged miscentering table. The
$\Delta\Sigma_{\rm mis}$ shell splits into a smooth aperture-mean term
(per-shell Gauss–Legendre nodes) minus the same exact shell mass. Two
thin-window approximations make the factorisation possible: the profile
offset is evaluated at the cluster's
distance $R_\theta = \theta\,\chi(z^{\rm ob})$ and pulled out of the $z$
integral ($\chi$ varies by a few per cent across the support), and the
neighbour concentration is evaluated at $z^{\rm ob}$
($c \propto (1+z)^{-1.01}$, $\lesssim 2\%$ across a hard $\pm 50\,h^{-1}$
cMpc window). $b_{\rm sel}(\theta)$ is applied at the neighbour's polar
angle about the cluster centre; the Costanzi notebook evaluates it inside
the azimuthal average around the point $R$ — the same double integral in
exchanged polar coordinates.

## Conventions

**Units.** Physical $M_\odot$, comoving Mpc, h-free: `hmf` is $dn/dM$ in
$M_\odot^{-1}{\rm Mpc}^{-3}$ comoving at physical mass, and
$\Sigma_{\rm prj}$ comes out in $M_\odot\,{\rm Mpc}^{-2}$ **comoving**.
Mock catalogues in h-scaled units convert at the caller's boundary.

**The comoving/physical $(1+z)^2$ enters through the miscentering table's
density scale.** The table is dimensionless; its runtime prefactor is
$\Sigma_0 = 2 r_s \rho_s$ with $\rho_s \propto \rho_{\rm def}$. `SigmaPrj`
passes the comoving $\bar\rho_m = \Omega_m\rho_{c,0}$ (no $(1+z)^3$), so
$r_s$ and $\Sigma$ are comoving — matching the Costanzi mock. The
`y3_cluster_cpp` `nfw_off_center` tables rescale with the physical
$\rho_{\rm crit}(z)$ instead, so their $\Sigma_{\rm mis}$ is physical:
$\Sigma_{\rm phys} = (1+z)^2\,\Sigma_{\rm com}$ — a factor 2.25 at
$z = 0.5$. Compare across conventions with that one visible
multiplication, never by re-deriving either side.

**The rnd channel is the selected-halo background column, not the
cosmological mean-matter column.** It is the mean column of the modelled
halo population — mass-restricted to `min_mass`..`log10_M_max`, dressed
with untruncated NFW wings unless `r_trunc` is set — so it carries only
the halo-budget share of $\bar\rho_m \times 2\,{\rm depth}$ (≈ 0.2–0.4
for the default mass cut). That is exactly the mock's background (mock
matter *is* those halos); a full-matter closure is a separate,
not-yet-implemented mode.

## Validation

`validation/validate_sigma_prj_mock.py` compares the full chain — `b_eff
= N[b]/N[1]` from {doc}`number_counts`, $b_{\rm sel}(\theta)$ from
{doc}`selection_bias` with that b_eff, and `SigmaPrj` — against the
Costanzi mock catalogue, bin by bin. In the two-halo regime
($R > 3\,h^{-1}$cMpc) the model tracks the mock's selected-to-random ratio
to better than 0.04 in all 12 $(\lambda^{\rm ob}, z)$ bins, and the
closure's own prediction of the mean richness boost,
$\Delta_{\rm RND} = P_1 + b_{\rm eff} I_2$, matches the mock's measured
$\langle\lambda^{\rm ob} - \lambda^{\rm tr}\rangle$ to 3–13%. Inside
$\sim 2R_\lambda$ the ratio is set by the closure's $b_{\rm small}$ — the
linear inversion the {doc}`selection_bias` NOTE flags — and is reported
unscored. See {doc}`validation`.

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"projection-lensing\"]"
:end-before: "%% [markdown]"
:language: python
```

```
Sigma_prj(R | lob=20, zob=0.5) [Msun/Mpc^2 comoving]:
  R=  0.5  rnd=3.583e+12  cl=3.997e+13  sum=4.355e+13
  R=  2.0  rnd=3.579e+12  cl=2.390e+13  sum=2.748e+13
  R=  8.0  rnd=3.576e+12  cl=2.963e+12  sum=6.539e+12
  R= 25.0  rnd=3.538e+12  cl=8.588e+11  sum=4.397e+12
DeltaSigma_prj: [9.037e+11 8.272e+12 4.705e+12 1.169e+12]
  rnd/cl at R=8: +0.0001 (boundary term only)
```

See also: {doc}`api/index` for the full `clenspy.lensing` reference,
{doc}`notation` for the symbol table, {doc}`selection_bias` for the
$b_{\rm sel}(\theta)$ that feeds the cl channel, {doc}`miscentering` for
the offset-NFW kernel, and {doc}`shear_proj` for how this term adds to
the cluster's own one-halo profile in the full stacked observable.
