# Halo Mass Function

The comoving number density of haloes per log-mass, $dn/d\ln M$, from Tinker
et al. (2008).

## Peak height $\nu$

Both this mass function and the Tinker (2010) bias ({doc}`halo_bias`) are
fits to the same peak height, built from {doc}`power_spectrum`'s $\sigma^2(R)$,

$$
\nu = \frac{\delta_c^{2}}{\sigma^{2}(R)},
$$

with $\delta_c$ the critical linear-collapse overdensity. In the Tinker
(2008) mass function $\delta_c$ cancels identically — the fit is written in
$\sigma$, and every conversion $\sigma \to \nu \to \sigma$ round-trips the
same constant — so `TinkerMassFunction` carries no $\delta_c$ at all. (The
Fortran ancestor carried $\delta_c = 1.6865$ here against the bias's 1.686;
both were pass-throughs for $dn/d\ln M$.)

`TinkerMassFunction`'s constructor is a chain, not three separate steps:
`cosmo` → {doc}`power_spectrum`'s `PkGrid` (z=0, physical units) →
`SigmaGrid` → `dndlnm_grid`, all built lazily, on first use. Pass `cosmo=`
and the chain runs itself; pass `k=`/`pk=` to override just the `PkGrid`
step with a custom spectrum, or `sigma_grid=` to inject a prebuilt
`SigmaGrid` — the same instance the {doc}`halo_bias` `BiasModel` should
share, since the two fits must read one $\sigma(M)$.

## The fitting function

$f(\sigma)$ is the fraction of mass in collapsed haloes per
$\ln\sigma^{-1}$ — its defining property is that it depends on cosmology
and redshift **only through $\sigma$** itself, not on any other combination
of them, which is why Tinker et al. (2008) calibrate a single $f(\sigma)$
against N-body simulations spanning several cosmologies at once. That
universality is only approximate: they find the amplitude and shape still
drift by tens of percent with redshift at fixed $\sigma$, which is why
$(A,a,b,c)$ below carry their own explicit $z$-dependence rather than
being held fixed. Tinker et al. (2008) fit $f(\sigma)$ with a
five-parameter form,

$$
f(\sigma) = A\left[\left(\frac{\sigma}{b}\right)^{-a} + 1\right]
            \exp\!\left(-\frac{c}{\sigma^{2}}\right),
$$

with $(A,a,b,c)$ interpolated in $\log_{10}\Delta$ from Tinker (2008) Table 2
and evolved with redshift by their Eqs. 5–8. The mass function itself follows
from the fraction of mass in collapsed haloes,

$$
\frac{dn}{d\ln M} = -\frac{\bar\rho_m}{6M}\,f(\sigma)\,
    \frac{d\ln\sigma^{2}}{d\ln R},
\qquad
R(M) = \left(\frac{3M}{4\pi\bar\rho_m}\right)^{1/3},
$$

with $\bar\rho_m = \Omega_{m,0}\,\rho_{c,0}$ the comoving mean matter
density and $R(M)$ the Lagrangian radius — an intermediate of the variance
integral, not a projected or halo radius.

$\sigma(M,z) = D(z)\,\sigma(M,0)$ is standard linear theory
({doc}`cosmology`) — `dndlnm_grid` applies it internally, together with the
Tinker $(A,a,b,c)$ coefficients' own separate evolution (their Eqs. 5–8, a
residual fit correction of order tens of percent). But **`dndlnm_grid`
covers only the $z$ it was built on** — the `zvec` grid the instance was
built with, `linspace(0, 1.5, 31)` by default. Querying outside it
extrapolates rather than recomputing; pass `zvec=` covering the redshifts
you need.

Query shapes follow one rule, shared with `BiasModel.bias`:
`dndlnm(1e14, z=0.3)` is a float, `dndlnm(Mvec, z=0.3)` is `(nM,)`,
vector-x plus vector-z always returns the outer `(nM, nz)` grid,
and no arguments returns the full grid.

```{admonition} Units are physical, h-free
:class: note
$M$ in $M_\odot$, $R$ in Mpc, $k$ in Mpc$^{-1}$, $P$ in Mpc$^3$,
$dn/d\ln M$ in ${\rm Mpc}^{-3}$ — the same convention as the rest of the
package. The Fortran ancestor worked in $R$ [Mpc$/h$], mass
[$\Omega_m h^{-1}M_\odot$], $dn/d\ln M$ [$h^3\,{\rm Mpc}^{-3}$], with the
rounded $\rho_{c,0}/h^2 = 2.775\times10^{11}\,M_\odot/{\rm Mpc}^3$ and a
$\pi$ literal truncated at the 11th digit; replacing both with exact
values shifts the mass axis by $1.3\times10^{-4}$ relative — far below
the $\sim$5% calibration accuracy of the fit.
```

## $\sigma^2(R)$: conventions of the variance evaluator

`SigmaGrid` is transcribed from `y3_cluster_cpp`
(`mf_tinker_cpp/python/tinker_core.py`, itself a port of the Fortran
`sigma.f90`/`linearpk.f90`, Komatsu CRL). Its Gauss–Legendre panel
evaluator agrees with arbitrary-precision quadrature to $4.4\times10^{-16}$
and with `cluster_toolkit` to $1.0\times10^{-7}$. Three conventions are
part of the algorithm, not numerical choices:

1. **The integration limits are $k \in [10^{-4}\,{\rm Mpc}^{-1},\,20/R]$.**
   The upper limit is the dimensionless cut $kR \le 20$ the production
   $dn/dM$ was calibrated against — which is why `sigma2` takes `truncate`
   explicitly. The fixed lower limit binds only when the tabulated $k$
   range extends below it.

2. **FFTLog cannot reproduce that truncation.** An FFTLog transform
   integrates the whole sampled $k$ range at every $R$ at once, so the fast
   path `sigma2_fftlog` computes the `truncate=False` quantity and must be
   validated against `truncate=False`. The difference it cannot capture,
   propagated to $dn/d\ln M$: $7.0\times10^{-3}$ over $0 \le z \le 2$,
   $8.2\times10^{-4}$ for $z \le 0.8$.

3. **The derivative is taken under the integral sign**, and when the
   truncation is active the moving boundary contributes a Leibniz term,

   $$
   \frac{d\sigma^2}{d\ln R} =
     \int d\ln k\,\frac{k^3 P}{2\pi^2}\,2W(x)W'(x)\,x
     \;-\; \left.\frac{k^3P}{2\pi^2}W^2\right|_{k = 20/R},
   $$

   because $\ln k_{\rm up} = \ln 20 - \ln R$ has $d/d\ln R = -1$. Dropping
   the boundary term is a silent bias in $dn/d\ln M$.

The input $P(k)$ policy is part of the definition too: a **natural** cubic
spline of $\ln P$ against $\ln k$ (the Fortran's `spline_cubic_set` with
`ibcbeg = ibcend = 2`), and $P \equiv 0$ strictly outside the tabulated
range — not clamping, not power-law extrapolation. Substituting a different
policy changes $\sigma^2$ at the ends of the $R$ grid. The panel edges of
the quadrature are the spline's own knots (the integrand is analytic within
a knot interval); a 24-point rule per panel reaches $\sim10^{-14}$ relative
and 48 points changes nothing at $10^{-16}$.

```{figure} _static/img/mass_function.png
:alt: Tinker (2008) dn/dlnM vs M at z = 0, 0.5, 1, growth-scaled
:width: 85%
:align: center

$dn/d\ln M$ at $z=0,0.5,1$, with $P(k)$ scaled by $D(z)^2$ at each
redshift — massive haloes are exponentially rarer as $z$ increases.
```

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"mass-function\"]"
:end-before: "%% [markdown]"
:language: python
```

```
M [Msun]           = [1.e+13 1.e+14 5.e+14 1.e+15]
dn/dlnM [Mpc^-3]   = [2.44745267e-04 2.35982439e-05 2.30147108e-06 5.33981067e-07]
```

See also: {doc}`api/index` for the full `clenspy.cosmology` reference,
{doc}`notation` for the symbol table.
