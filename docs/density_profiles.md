# Halo Density Profiles

`clenspy.halo` supplies two 3D density profiles, $\rho(r)$, and their
Fourier transforms — the raw building blocks projected profiles
({doc}`projected_profiles`) and the two-halo term
({doc}`two_halo_term`) are both built from.

```{figure} _static/img/density_profiles.png
:alt: NFW (truncated and untruncated) vs Einasto density profile and Fourier transform, all over M200
:width: 95%
:align: center

$\rho(r)$ and $\tilde\rho(k)/M_{200}$ for an NFW halo and an Einasto
profile sharing the same $r_s=r_{200}/c_{200}$ and the same enclosed mass
at $r_{200}$ — both curves divided by the same $M_{200}$. Truncated NFW
and Einasto track closely inside $r_{200}$ and above $k\sim1\,{\rm
Mpc}^{-1}$; below that, Einasto's $\tilde\rho(k)/M_{200}$ plateaus at a
finite value above 1 rather than at it — it keeps $\sim$87% more mass
beyond $r_{200}$ than the NFW truncation discards, visible as the
shallower fall in $\rho(r)$ past the dotted line. The dotted curve is the
*untruncated* NFW, included to show why NFW cannot be compared to Einasto
the same way: rather than plateauing at some finite value like Einasto's,
it keeps climbing as $k\to0$ with no limit — because, unlike Einasto's,
its total mass is not just larger, it is **infinite** (see the note
below).
```

## NFW: two parameters, closed form everywhere

The NFW profile is fixed by mass and concentration alone,

$$
\rho(r) = \frac{\rho_s}{x(1+x)^2}, \qquad x = \frac{r}{r_s},
\qquad r_s = \frac{r_{200}}{c_{200}},
$$

with $\rho_s$ set by requiring the profile to enclose $M_{200}$ within
$r_{200}$. `NfwProfile` carries no cosmology — the mass definition (200
times *which* density) is entirely the caller's choice, made once by
passing `rho_ref` (default: the comoving mean matter density, i.e.
$M_{200m}$).

The Fourier transform has a closed form (Cooray & Sheth 2002, Eq. 81),
truncated at $r_{200} = c_{200}r_s$ or left infinite in extent:

$$
\tilde\rho(k) = \frac{M}{\ln(1+c) - c/(1+c)} \Big\{
\sin(x)\left[\mathrm{Si}\big((1+c)x\big) - \mathrm{Si}(x)\right]
+ \cos(x)\left[\mathrm{Ci}\big((1+c)x\big) - \mathrm{Ci}(x)\right]
- \frac{\sin(cx)}{(1+c)x} \Big\}, \qquad x = kr_s,
$$

or, for the untruncated (infinite-extent) profile,

$$
\tilde\rho(k) = \frac{M}{\ln(1+c) - c/(1+c)} \left\{
\sin(x)\left[\frac{\pi}{2} - \mathrm{Si}(x)\right] - \cos(x)\,\mathrm{Ci}(x)
\right\}, \qquad x = kr_s,
$$

with $c\equiv c_{200}$ in both.

```{note}
`NfwProfile.fourier` returns $\tilde\rho(k)$ — carrying units of mass, and
going to $M$ (not $1$) as $k\to0$ — not the dimensionless mass-normalized
$u(k\mid M) \equiv \tilde\rho(k)/M$ that most halo-model formulas mean by
"the profile's Fourier transform." Divide by the halo's own `m200` at the
call site to get $u(k\mid M)$; this matches `pyccl`'s
`HaloProfileNFW.fourier` convention exactly.
```

```{note}
NFW's $\rho\propto r^{-3}$ falloff makes its **untruncated** enclosed mass
$M(r) = 4\pi\rho_s r_s^3\left[\ln(1+x) - x/(1+x)\right]$ diverge — only
logarithmically, but it never converges as $r\to\infty$. There is no
finite "NFW total mass" the way there is for Einasto; truncating at
$r_{200}$ isn't a numerical convenience, it's what makes the mass finite
at all. That is why `NfwProfile.fourier(k, truncated=False)` in the figure
above keeps rising as $k\to0$ rather than settling on a plateau like
Einasto's — passing `truncated=False` with no radius at which to stop
integrating leaves nothing for $\tilde\rho(k{=}0)$ to converge to.
```

## Einasto: a curved profile with no fixed inner slope

The Einasto profile trades the NFW's fixed asymptotic slopes for a single
shape parameter $\alpha = 1/n$,

$$
\rho(r) = \rho_0 \exp\!\left[-(r/h)^{1/n}\right],
$$

with $h = r_s/(2n)^n$. Its logarithmic slope,

$$
\frac{d\ln\rho}{d\ln r} = -\frac{1}{n}\left(\frac{r}{h}\right)^{1/n},
$$

goes to $0$ as $r\to0$ — Einasto has **no central cusp**: $\rho(r)$
approaches the finite $\rho_0$ smoothly, unlike NFW's $\rho\propto r^{-1}$
divergence at small $r$. `EinastoProfile` reports the enclosed mass and
total mass in closed form for any $n$ via the incomplete gamma function,

$$
M_{\rm 3D}(r) = 4\pi\rho_0\, n\, h^3\,\gamma\big(3n,\,(r/h)^{1/n}\big),
\qquad
M_{\rm tot} = 4\pi\rho_0\, n\, h^3\,\Gamma(3n).
$$

`power_spectrum` rescales the same Fourier transform,

$$
P(k) = \frac{\tilde\rho(k)}{(4\pi)^2},
$$

evaluated, for $n>1$ (large-$k$ series, converges for all $k$, $\tilde
k\equiv kh$),

$$
P(k) = \frac{\rho_0 h^3}{4\pi\tilde k^3}\sum_{m\ge1} A_m^-\,\tilde k^{-m/n},
\qquad A_m^- = \frac{(-1)^{m+1}}{m!}\Gamma\!\left(2+\frac{m}{n}\right)
\sin\!\left(\frac{\pi m}{2n}\right),
$$

and by an FFTLog transform of `density` for $0<n<1$, away from the $n=1$
and $n=1/2$ anchors' own closed forms. `fourier(k)` is $(4\pi)^2P(k)$, so
the two methods agree exactly. See {doc}`einasto_math` for the full
per-regime dispatch, the closed forms at $n=1,1/2$, and the low-$n$
numerical stability analysis.

```{note}
This `power_spectrum`/$P(k)$ is internal Einasto machinery, not the
cosmological matter power spectrum of {doc}`power_spectrum` — same symbol,
unrelated quantity. Like `NfwProfile.fourier` above, `EinastoProfile.fourier`
is mass-dimensioned and goes to $M_{\rm tot}$ (not $1$) as $k\to0$; it is
**not** the dimensionless, mass-normalized $u(k\mid M)$ that halo-model
formulas expect. Divide by `total_mass` at the call site to get $u(k\mid
M)$, exactly as for `NfwProfile`.
```

```{note}
`alpha` here is the Einasto shape index, unrelated to the HOD satellite
power-law index or a source-redshift-distribution slope elsewhere in
`clenspy` — same Greek letter, three unrelated physical quantities.
```

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"density-profiles\"]"
:end-before: "%% [markdown]"
:language: python
```

```
r200 = 1.4303 Mpc, rs = 0.2861 Mpc, rho_s = 3.547e+14 Msun/Mpc^3
rho_NFW(r)     [Msun/Mpc^3] = [5.57109391e+14 2.68756655e+13 5.02012592e+12 7.94381981e+11]
rho_Einasto(r) [Msun/Mpc^3] = [5.83600877e+14 2.77183496e+13 4.86300049e+12 6.13799586e+11]
rho_tilde_NFW(k)     [Msun] = [9.98998564e+13 9.05346969e+13 8.91993930e+12]
rho_tilde_Einasto(k) [Msun] = [1.83204136e+14 1.06342641e+14 9.27420738e+12]
```

See also: {doc}`api/index` for the full `clenspy.halo` reference,
{doc}`notation` for the symbol table, {doc}`einasto_math` for the Einasto
profile's derivations.
