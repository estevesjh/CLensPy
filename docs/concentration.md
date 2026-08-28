# Concentration

The NFW profile needs two numbers, mass and concentration, and the halo
mass function only supplies the first. $c(M,z)$ is a structure-formation
result — calibrated on N-body simulations at fixed cosmology, exactly like
{doc}`mass_function` and {doc}`halo_bias` — so it lives in
`clenspy.cosmology` rather than `clenspy.halo`.

```{figure} _static/img/concentration.png
:alt: Concentration-mass relations at z=0.3 -- child18, child18_powerlaw, duffy08
:width: 85%
:align: center

$c_{200c}(M)$ at $z=0.3$: concentration falls with mass, since more
massive haloes collapsed more recently and had less time to grow their
inner density. `child18` sits above `duffy08` at cluster scales, tracking
the trend into the Eq. 18 plateau.
```

## Child et al. (2018): the $c$–$M/M_\star$ form

Scaling mass by the nonlinear mass $M_\star(z)$ collapses the whole
redshift family onto one curve (Eq. 18):

$$
c_{200c} = A\left[
    \left(\frac{M_{200c}/M_\star}{b}\right)^{m}
    \left(1 + \frac{M_{200c}/M_\star}{b}\right)^{-m} - 1
\right] + c_0,
$$

a power law in $M/M_\star$ below the threshold $M_T = b\,M_\star$,
flattening to a plateau $c_0 \sim 3$–4 above it. All of $z$'s dependence
runs through $M_\star(z)$, not through $A$, $b$, $m$, or $c_0$.
Coefficients are `CHILD18_TABLE1`; the y3 production choice is the
`"individual_all"` row.

$M_\star(z)$ is defined by $\sigma(R_\star, z) = \delta_c = 1.686$,
$M_\star = \tfrac{4\pi}{3}\bar\rho_m R_\star^3$, and Child et al. quote it
at four anchors, $\log_{10}(M_\star/h^{-1}M_\odot) = 12.5, 11, 9.5, 8$ at
$z = 0,1,2,3$ — exactly linear in $z$, so `m_star_hinv` is that line, not a
fit. `m_star_from_sigma` solves the definition exactly against any
$\sigma(R,z)$, for cosmologies away from Child et al.'s WMAP-7.

A power-law alternative, Eq. 19, needs no $M_\star$ but is only valid for
$0 \le z \le 1$:

$$
c_{200c} = A\,(1+z)^{d}\,M^{m}.
$$

## Duffy et al. (2008): the older power law

A plain power law, kept because `pyccl` and much of the literature default
to it:

$$
c = A\left(\frac{M}{2\times10^{12}\,h^{-1}M_\odot}\right)^{B}(1+z)^{C}.
$$

Its WMAP-5 $\sigma_8 = 0.796$ is low, so it sits below Child et al. at
cluster scales. `duffy08` takes a `mass_def` — `"vir"`, `"200m"`, or `"200c"` — since Duffy
et al. tabulate all three as three different halo boundaries, not three
fits to one; `clenspy`'s default is `"200m"`, matching `NfwProfile`, **not**
`pyccl`'s default of `"200c"`.

```{note}
These relations were calibrated in $h^{-1}M_\odot$ and on $M_{200c}$ — the
one place in `clenspy` where the h-free convention breaks, and a different
halo boundary than `NfwProfile`/`TinkerMassFunction`'s $M_{200m}$. Every
mass argument here carries the unit and definition in its name
(`m200c_hinv`), and `clenspy` does not vendor an $M_{200m}\to M_{200c}$
conversion — the caller owns it.
```

## $\delta_c$ and scatter

The NFW characteristic overdensity, so that $\rho_s = \delta_c\,\rho_{\rm
ref}$:

$$
\delta_c(c) = \frac{200\,c^3/3}{\ln(1+c) - c/(1+c)}.
$$

This is unrelated to $\delta_c = 1.686$ above (spherical-collapse
threshold) despite the shared name — one is $O(1)$, the other $O(10^4)$.
Both Child et al. tables also carry a scatter note, $\sigma_c =
c_{200c}/3$ — a 33% halo-to-halo spread large enough that a stacked profile
is not the profile at $\langle c\rangle$; a stacked analysis wanting one
number should use the `"stacked_nfw"` row instead.

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"concentration\"]"
:end-before: "%% [markdown]"
:language: python
```

```
M_star(z=0.3) = 1.122e+12 h^-1 Msun,  M/M_star = 89.1
c_200c: child18 = 3.853, child18_powerlaw = 3.831, duffy08 = 3.634
child18 / duffy08 = 1.060  (> 1, as expected)
NFW delta_c(c=3.853) = 4854.4, scatter sigma_c = 1.284
```

See also: {doc}`api/index` for the full `clenspy.cosmology` reference,
{doc}`notation` for the symbol table.
