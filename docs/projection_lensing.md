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
Around a cluster observed at $(\lambda^{\rm ob}, z^{\rm ob})$, with
$\theta$ the neighbour's angular offset and $\chi_o = \chi(z^{\rm ob})$,

$$
\Sigma_{\rm prj}(R) = \int d\theta\, 2\pi\sin\theta\;
  b_{\rm sel}(\theta) \sum_M w_{\rm cl}(\theta, M)\;
  \Sigma_{\rm mis}(R,\, \theta\chi_o \mid M),
$$

where $\Sigma_{\rm mis}$ is the azimuth-averaged surface density of a halo
offset by $R_\theta = \theta\chi_o$ — the same single-offset kernel as
{doc}`miscentering`, with the offset now a real transverse separation
rather than a centring error — and the per-slice redshift weight is

$$
w_{\rm cl}(\theta, M) = \int dz\;{\rm common}(z)\,
    \xi_{\rm NL}\big(|d\chi|(z,\theta),\, z^{\rm ob}\big)\,
    n(M, z)\, b(M, z)\, m_{\rm cl}(\theta, z), \qquad
{\rm common}(z) = \frac{dV}{d\Omega\,dz}\, w_{pz}(z; z^{\rm ob}),
$$

with $n$ the mass function, $b$ the halo bias, and $|d\chi|$ the exact
law-of-cosines chord $d\chi^2 = \chi_z^2 + \chi_o^2 -
2\chi_z\chi_o\cos\theta$ — near the exclusion ring the transverse leg
dominates and the $|\chi_z - \chi_o|$ shortcut is wrong by orders of
magnitude. $b_{\rm sel}(\theta)$ multiplies this correlated integrand
*alone*.

A raw projected *mass map* — the Costanzi mock's per-halo columns, or any
stack that has not been random-point subtracted — additionally contains
the mean background column, the `1` of the halo-model bracket
$[1 + b\,b_{\rm sel}\,\xi_{\rm NL}]$. That piece is kept available as the
separate `rnd` channel,

$$
\Sigma_{\rm bkg}(R) = \int d\theta\, 2\pi\sin\theta
  \sum_M w_{\rm rnd}(\theta, M)\, \Sigma_{\rm mis}(R,\, \theta\chi_o \mid M),
\qquad
w_{\rm rnd} = \int dz\;{\rm common}(z)\, n(M, z)\, m_{\rm rnd},
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
$(\theta, z, M)$ integrals, and

$$
\Delta\Sigma_{\rm prj}(R) \propto \int dz \int dM \int d\theta \;\ldots\;
\Delta\Sigma_{\rm mis}(R,\, \theta\chi_o \mid M)
$$

is the *same* operator with the kernel swap $\Sigma_{\rm mis} \to
\Delta\Sigma_{\rm mis}$ — never a numerical reconstruction from a
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
its volume is excised at $R_{\rm excl} =
R_\lambda(\lambda^{\rm ob})(1 + z^{\rm ob})$ comoving. The default,
`exclusion="counter"`, is the Costanzi convention: inside the chord ball
the correlated integrand is set to $-1$, a **counter term** cancelling
the background's $+1$ exactly. The total vanishes in the ball — the same
total as deleting the neighbours, as the mock does — but the bookkeeping
matters: $\Sigma_{\rm bkg}$ stays strictly uniform and the exclusion
hole is carried by $\Sigma^{\rm prj}$, where a random-point-subtracted
measurement keeps it. `exclusion="ball"` books the same hole in the
background instead (identical sum); `exclusion="cl"` is the E.3
production slab, which merely zeroes the correlated term. Switching is a
config change, not a code change.

## Numerics: the ring is integrated exactly

$\Sigma_{\rm mis}(R, s)$ as a function of the offset $s$ is a ring of
width $\sim r_s$ at $s \approx R$, and no affordable pointwise $\theta$
rule resolves it. `SigmaPrj.kernel` therefore integrates each log-spaced
$\theta$ cell **exactly**: the azimuthal average is symmetric,
$\Sigma_{\rm mis}(R, s) = \Sigma_{\rm mis}(s, R)$, so the cell integral is
an annulus-mass difference of the halo offset by $R$,

$$
\int_{s_1}^{s_2} 2\pi s\,\Sigma_{\rm mis}(R, s)\, ds
= \pi\Sigma_0\Big[s^2\, \hat m\big(s/r_s,\, R/r_s\big)\Big]_{s_1}^{s_2},
\qquad \hat m = \hat\Sigma_{\rm mis} + \widehat{\Delta\Sigma}_{\rm mis},
$$

read from the packaged miscentering table. The $\Delta\Sigma_{\rm mis}$
kernel splits into a smooth aperture-mean term (trapezoid on the cell
edges) minus the same exact ring. Two thin-window approximations are
named in the class docstring: the kernel offset is evaluated at
$\chi(z^{\rm ob})$, and the neighbour concentration at $z^{\rm ob}$.

## Validation

`validation/validate_sigma_prj_mock.py` compares the full chain — `b_eff
= N[b]/N[1]` from {doc}`observables`, $b_{\rm sel}(\theta)$ from
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
  R=  0.5  rnd=3.566e+12  cl=4.130e+13  sum=4.486e+13
  R=  2.0  rnd=3.576e+12  cl=2.413e+13  sum=2.770e+13
  R=  8.0  rnd=3.575e+12  cl=2.980e+12  sum=6.555e+12
  R= 25.0  rnd=3.538e+12  cl=8.777e+11  sum=4.415e+12
DeltaSigma_prj: [9.427e+11 8.597e+12 4.784e+12 1.146e+12]
  rnd/cl at R=8: +0.0044 (boundary term only)
```

See also: {doc}`api/index` for the full `clenspy.lensing` reference,
{doc}`notation` for the symbol table, {doc}`selection_bias` for the
$b_{\rm sel}(\theta)$ that feeds the cl channel, and {doc}`miscentering`
for the offset-NFW kernel.
