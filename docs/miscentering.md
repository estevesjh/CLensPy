# Miscentering

A cluster's assumed center — a brightest-cluster-galaxy position, a
richness-weighted centroid — is not always its true halo center. A fixed
offset $R_{\rm mis}$ smears the stacked profile: azimuthally averaging
around the *wrong* point mixes radii that would otherwise stay separate,
and biases $\Delta\Sigma(R)$ low near $R\sim R_{\rm mis}$.

```{figure} _static/img/miscentering.png
:alt: DeltaSigma, its 1-halo piece, and single-offset vs Gamma-averaged miscentered profiles
:width: 85%
:align: center

$\Delta\Sigma(R)$ against a single fixed offset and the DES Y1/Y3 Gamma
offset law averaged over the same scale ($\lambda=25$,
$\tau_{\rm mis}=0.17$, giving $\theta\approx0.18$ Mpc — the worked example
of {doc}`miscentering_math` Sec. 9.3). The single-offset curve plunges off
the bottom of the log axis where it goes negative — the signed lobe
described below — while the Gamma-law average, a mix of many offsets,
stays positive throughout and rejoins the centered $\Delta\Sigma$ well
outside $\theta$.
```

## The azimuthal average around the wrong center

Averaging the true, centered $\Sigma$ around a point displaced by
$R_{\rm mis}$ mixes in every true radius $u(t)$ on the circle of angle $t$:

$$
\Sigma_{\rm mis}(R\mid R_{\rm mis}) = \frac{1}{\pi}\int_0^\pi
\Sigma\big(u(t)\big)\,dt, \qquad
u(t) = \sqrt{R^2+R_{\rm mis}^2-2RR_{\rm mis}\cos t},
$$

with the aperture mean built the same way, and $\Delta\Sigma_{\rm mis}$
the usual difference of the two:

$$
\bar\Sigma_{\rm mis}(<R\mid R_{\rm mis}) = \frac{1}{2\pi R^2}\int_0^\pi
\big[u(t)^2+R^2-R_{\rm mis}^2\big]\,\bar\Sigma\big(<u(t)\big)\,dt,
\qquad
\Delta\Sigma_{\rm mis} \equiv \bar\Sigma_{\rm mis}(<R) - \Sigma_{\rm mis}(R).
$$

```{note}
$\Delta\Sigma_{\rm mis}$ is **signed** — negative for $R_{\rm mis}\gtrsim
R$. This is a genuine finite-profile effect (a point mass sitting outside
the aperture contributes exactly zero to $\bar\Sigma_{\rm mis}$ but still
pulls the local $\Sigma_{\rm mis}$ up), and the population average
$\int_0^\infty \Delta\Sigma_{\rm mis}\,2\pi R_{\rm mis}\,dR_{\rm mis}$
vanishes only because this negative lobe is there — never clamp it to
zero. See {doc}`miscentering_math` for the full derivation, the
by-parts identity that keeps $\bar\Sigma_{\rm mis}$'s integrand smooth at
$R_{\rm mis}=R$, and the table's validation.
```

## Interpolated, never integrated at call time

`MiscenteringProfile` reads $\Sigma_{\rm mis}$, $\bar\Sigma_{\rm mis}$, and
$\Delta\Sigma_{\rm mis}$ from a packaged lookup table — `clenspy` does not
solve the offset integrals above at evaluation time. The quadrature that
built the table lives in `clenspy.selection.miscentering_kernel`, an
offline generator, not a runtime fallback; a profile with no table (today,
only NFW) raises `MiscenteringTableError` at construction, before any
evaluation is attempted. `MiscenteringProfile` subclasses
{doc}`lensing_profile`'s `LensingProfile` — the centered observables
(`sigma`, `deltasigma`, ...) are inherited unchanged, and only the
`_mis`-suffixed methods consult the table.

## The population-averaged offset: DES Y1/Y3's Gamma law

This is the profile at one fixed offset; a stacked cluster sample is a mix
of many. DES Y1/Y3 (following the redMaPPer calibration of Hoshino et al.
2015, adopted by McClintock et al. 2019) draws the offset from a Gamma
distribution scaled by the richness-based radius $R_\lambda$,

$$
p(R_{\rm mis}\mid\theta) = \frac{R_{\rm mis}}{\theta^2}\,
e^{-R_{\rm mis}/\theta}, \qquad
\theta = \tau_{\rm mis}\,R_\lambda, \qquad
R_\lambda = \left(\frac{\lambda}{100}\right)^{0.2} h^{-1}{\rm Mpc}
$$

(Rykoff et al. 2014), with $\tau_{\rm mis}\approx0.17$ fit to simulations.
`MiscenteringProfile` only evaluates one fixed `r_mis` at a time (see the
class Notes) — averaging over $p(R_{\rm mis}\mid\theta)$ is genuinely the
caller's job, built in the figure above as a plain trapezoid quadrature
over a grid of `deltasigma_mis` calls at different `r_mis`, after
converting $R_\lambda$'s h-scaled convention to `clenspy`'s h-free one by
dividing by $h$ once, visibly.

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"miscentering\"]"
:end-before: "%% [markdown]"
:language: python
```

```
r_mis=0.0 Mpc  DeltaSigma_mis [Msun/Mpc^2] = [8.80580049e+13 6.64058864e+13 3.01267694e+13 8.89043936e+12]
r_mis=0.2 Mpc  DeltaSigma_mis [Msun/Mpc^2] = [-1.92243006e+12  3.41501316e+13  2.90062982e+13  8.86603639e+12]
r_mis=1.0 Mpc  DeltaSigma_mis [Msun/Mpc^2] = [-5.30598152e+10 -5.03125804e+11 -1.22216230e+13  8.20500151e+12]
```

At $r_{\rm mis}=0.2\,{\rm Mpc}$, $\Delta\Sigma_{\rm mis}(R=0.1) < 0$ —
$R=0.1\,{\rm Mpc}$ sits inside the offset, exactly the signed regime
above. At $r_{\rm mis}=1.0\,{\rm Mpc}$, both $R=0.1$ and $R=0.3\,{\rm Mpc}$
are negative for the same reason, while $R=3.0\,{\rm Mpc}$, well outside
every offset shown, is close to its centered value in every row.

See also: {doc}`api/index` for the full `clenspy.lensing` reference,
{doc}`notation` for the symbol table, {doc}`miscentering_math` for the
full derivation and table validation, {doc}`boost_factor` for the other
selection-driven correction to a stacked profile.
