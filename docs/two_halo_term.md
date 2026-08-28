# The Two-Halo Term

At small radius a stacked cluster lensing profile is dominated by mass
bound to the cluster's own halo — {doc}`density_profiles` and
{doc}`projected_profiles`. At large radius it is dominated instead by the
*correlated* matter around the halo: neighboring haloes tracing the same
large-scale overdensity. `TwoHaloTerm` computes that second piece directly
from the linear matter power spectrum, with no halo bias baked in — the
bias factor is applied by the caller ({doc}`lensing_profile`).

```{figure} _static/img/two_halo_term.png
:alt: Two-halo correlation function and excess surface density at three redshifts
:width: 95%
:align: center

$\xi(r,z)$ and $\Delta\Sigma_{\rm 2h}(R,z)$ from a linear CAMB $P(k,z)$.
Both fall with $z$ as the linear growth factor shrinks the correlated
structure; $\Delta\Sigma_{\rm 2h}$ peaks around $R\sim10\,{\rm Mpc}$,
well outside any individual halo's virial radius.
```

## From P(k) to ξ, Σ, and ΔΣ

`TwoHaloTerm` takes a gridded $P(k,z)$ and FFTLog-transforms it to the
matter correlation function,

$$
\xi(r,z) = \frac{1}{2\pi^2}\int dk\, k^2 P(k,z)\,\frac{\sin(kr)}{kr},
$$

then Abel-projects $\xi$ along the line of sight to get the surface
density, and takes its cumulative-trapezoid enclosed mean to get the
excess surface density — the same $\Sigma\to\Delta\Sigma$ relation as
{doc}`projected_profiles`:

$$
\Sigma(R,z) = 2\int_R^\infty \xi(r,z)\,\frac{r\,dr}{\sqrt{r^2-R^2}},
\qquad
\Delta\Sigma(R,z) \equiv \bar\Sigma(<R,z) - \Sigma(R,z).
$$

```{note}
`sigma` and `deltasigma` are **unnormalized** — units of Mpc, not
Msun/Mpc². They are pure projections of $\xi(r,z)$ with no density factor;
multiply by the *comoving* mean matter density $\rho_m=\Omega_{m,0}
\rho_{c,0}$ to get a physical surface density. Using $\rho_c(z)\Omega_m$
instead folds in $E^2(z)$ and overstates the result by ~34% at $z=0.25$ —
$\xi$ here is comoving too, so the two conventions must match.
```

`TwoHaloTerm` accepts $P(k,z)$ directly — no cosmology object, no halo
bias, no mass. It is the pure large-scale-structure piece; a caller
multiplies by $b(M,z)$ ({doc}`halo_bias`) to get the halo-mass-dependent
two-halo contribution to a stacked profile.

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"two-halo-term\"]"
:end-before: "%% [markdown]"
:language: python
```

```
xi(R, z)              = [8.29782038 5.0481562  1.08533824 0.43567554 0.01748727]
Sigma_2h(R)      [Msun/Mpc^2] = [1.88077313e+12 1.66402049e+12 9.28143290e+11 5.78265032e+11
 6.45736495e+10]
DeltaSigma_2h(R) [Msun/Mpc^2] = [1.05373718e+11 1.47918604e+11 2.42494808e+11 2.48179822e+11
 1.23301599e+11]
```

See also: {doc}`api/index` for the full `clenspy.halo` reference,
{doc}`notation` for the symbol table, {doc}`lensing_profile` for how the
one-halo and two-halo terms combine into a full stacked profile.
