# The Lensing Profile

`LensingProfile` is the composite object a driver actually calls: it
combines a one-halo term ({doc}`density_profiles`/{doc}`projected_profiles`,
via `NfwProfile`) with a linearly-biased two-halo term
({doc}`two_halo_term`) into the single density, surface density, and shear
a stacked-lensing analysis fits.

```{figure} _static/img/lensing_profile.png
:alt: Excess surface density, one-halo term, two-halo term, and their sum
:width: 75%
:align: center

$\Delta\Sigma(R)$ decomposed into its one-halo and two-halo pieces. The
one-halo term dominates inside $\sim$a few Mpc and falls steeply past
$r_{200}$; the two-halo term takes over beyond it and eventually turns
over as the correlation function itself declines.
```

## Combining the two terms

Every observable is the one-halo piece plus the halo-bias-weighted
two-halo piece, with the same comoving mean matter density $\rho_m$
{doc}`two_halo_term` requires to normalize $\Sigma_{\rm 2h}$/
$\Delta\Sigma_{\rm 2h}$:

$$
\Sigma(R) = \Sigma_{\rm 1h}(R) + b(M)\,\rho_m\,\Sigma_{\rm 2h}(R),
\qquad
\Delta\Sigma(R) = \Delta\Sigma_{\rm 1h}(R) + b(M)\,\rho_m\,
\Delta\Sigma_{\rm 2h}(R),
$$

$$
\rho(r) = \rho_{\rm 1h}(r) + \rho_m\big[1+b(M)\,\xi_{\rm 2h}(r)\big],
$$

with $b(M)$ the linear halo bias ({doc}`halo_bias`), evaluated on the same
$P(k)$ the two-halo term uses. From $\Sigma$ and $\Delta\Sigma$, the
lensing observables follow directly:

$$
\gamma_t(R) = \frac{\Delta\Sigma(R)}{\Sigma_{\rm crit}}, \qquad
\kappa(R) = \frac{\Sigma(R)}{\Sigma_{\rm crit}}, \qquad
g_t(R) = \frac{\gamma_t(R)}{1-\kappa(R)},
$$

with $\Sigma_{\rm crit}$ the lens-source critical surface density
({doc}`lensing_kernel`).

```{note}
Masses here are **$M_{200m}$** (200 times the *comoving* mean matter
density), inherited from `NfwProfile` by passing $\rho_m$ as its
`rho_ref` — not $M_{200c}$, and not the concentration relations of
{doc}`concentration`, which are calibrated on $M_{200c}$.
```

## Everything is lazy except validation

The constructor stores $z_{\rm cluster}$, $M_{200}$, concentration, and
cosmology, and validates them immediately — a bad redshift or a negative
mass raises at construction, not pages later. `halo_profile`,
`two_halo_profile`, `bias_model`, `bias`, and `sigma_crit` are
`functools.cached_property`, so constructing a `LensingProfile` costs
nothing; the first property access that needs $P(k)$ is what runs the
Boltzmann solver (CAMB or `pyccl`, via `PkGrid`). Pass a pre-built
`two_halo=`, `bias=`, or `halo_profile=` to reuse one $P(k)$ across many
haloes at the same redshift, rather than rebuilding it per halo.

```{note}
`fourier_profile` — the Fourier-space analog of `sigma`/`deltasigma` — has
a units bug: it adds the *unnormalized* $\tilde\rho_{\rm 1h}(k)$ (see
{doc}`density_profiles`'s note on `NfwProfile.fourier`) to a 2-halo term
already divided by $M_{200}$. The one-halo term dominates the sum at every
$k$ as a result; see the method's own docstring `NOTE` for the fix. Not
yet corrected, since it changes the method's public return value.
```

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"lensing-profile\"]"
:end-before: "%% [markdown]"
:language: python
```

```
LensingProfile(model=nfw, z_cluster=0.300, m200=1.00e+14, c=4.00), include_2halo=True)
DeltaSigma [Msun/Mpc^2] = [6.75689107e+13 3.50179576e+13 1.95470366e+13 3.11057365e+12]
b(M) = 2.425   Sigma_crit = 2.834e+15 Msun/Mpc^2
1-halo only             = [6.74701550e+13 3.47624083e+13 1.91883408e+13 2.52255942e+12]
2-halo fraction         = [0.00146156 0.00729766 0.01835039 0.18903723]
shear(R)         = [0.0238418  0.01235614 0.0068972  0.00109757]
reduced_shear(R) = [0.02542785 0.01252437 0.0069368  0.00109872]
```

See also: {doc}`api/index` for the full `clenspy.lensing` reference,
{doc}`notation` for the symbol table, {doc}`miscentering` for the
mis-centered stacked profile built on top of this one.
