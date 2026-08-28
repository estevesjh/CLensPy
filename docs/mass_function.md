# Halo Mass Function

The comoving number density of haloes per log-mass, $dn/d\ln M$, from Tinker
et al. (2008).

## Peak height $\nu$

Both this mass function and the Tinker (2010) bias ({doc}`halo_bias`) are
fits to the same peak height, built from {doc}`power_spectrum`'s $\sigma^2(R)$,

$$
\nu = \frac{\delta_c^{2}}{\sigma^{2}(R)},
$$

with $\delta_c$ the critical linear-collapse overdensity. `TinkerMassFunction`'s constructor is a chain, not three
separate steps: `cosmo` → {doc}`power_spectrum`'s `PkGrid` (h-free, z=0) →
converted to the h-scaled convention below → `SigmaGrid` → `dndlnm_grid`,
all built lazily, on first use. You never construct a `SigmaGrid` by hand;
pass `cosmo=` and the chain runs itself — pass `k_h=`/`pk_h3=` instead to
override just the `PkGrid` step with a custom spectrum (a toy power law, a
cached grid from elsewhere).

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
from the fraction of mass in peaks of height $\nu$,

$$
\frac{dn}{d\ln R} = \frac{3}{4\pi}\,\frac{d\ln\nu}{d\ln R}\,
    \frac{f(\sigma)}{2}\,\frac{1}{R^{3}},
\qquad
\frac{dn}{d\ln M} = \frac13\,\frac{dn}{d\ln R}.
$$

$\sigma(M,z) = D(z)\,\sigma(M,0)$ is standard linear theory
({doc}`cosmology`) — `dndlnm_grid` applies it internally, together with the
Tinker $(A,a,b,c)$ coefficients' own separate evolution (their Eqs. 5–8, a
residual fit correction of order tens of percent). But **`dndlnm_grid`
covers only the $z$ it was built on** — the `zvec` grid the instance was
built with, `[0.0]` by default. Querying `dndlnm(M, z=1.0)` on a default
instance extrapolates rather than recomputing; pass `zvec=` covering the
redshifts you need.

```{admonition} Units are h-scaled here, not h-free
:class: note
`TinkerMassFunction` inherits its convention from `SigmaGrid`: $R$ in
Mpc$/h$, mass in $\Omega_m h^{-1}M_\odot$ (no $\Omega_m$ folded in — the
caller applies it), $dn/d\ln M$ in $h^3\,{\rm Mpc}^{-3}$. This is the one
place the package's h-free convention breaks, inherited from the Fortran
reference it reproduces.
```

```{admonition} Mass, not radius, is the query variable
:class: note
$R(M)$ (the Lagrangian radius, `radius_of_mass`) is only the intermediate
the variance integral is evaluated at — it is **not** a projected or
physical radius like the $R$ in {doc}`density_profiles`. `dndlnm(M_vals,
z)` is the query method; `mass_of_radius`/`radius_of_mass` convert when
you have a radius instead.
```

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
M_h [Omega_m h^-1 Msun] = [1.16238928e+12 5.95143312e+14 9.29911425e+15]
dn/dlnM [h^3 Mpc^-3]    = [1.02125521e-02 2.01939709e-05 1.18440628e-08]
```

See also: {doc}`api/index` for the full `clenspy.cosmology` reference,
{doc}`notation` for the symbol table.
