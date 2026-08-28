# Projected Density Profiles

Weak lensing never sees $\rho(r)$ directly — it sees mass projected along
the line of sight. Both `NfwProfile` and `EinastoProfile` implement the
same two projected quantities from {doc}`density_profiles`'s $\rho(r)$:
the surface density and the excess surface density that a lensing survey
actually measures.

```{figure} _static/img/projected_profiles.png
:alt: NFW vs Einasto surface density and excess surface density
:width: 95%
:align: center

$\Sigma(R)$ and $\Delta\Sigma(R)$ for the same mass-matched pair as
{doc}`density_profiles`. Both track closely inside $r_{200}$; Einasto's
extra mass beyond $r_{200}$ shows up as a shallower $\Sigma(R)$ tail and a
$\Delta\Sigma(R)$ that has not yet turned over at $r_{200}$, unlike the
NFW's sharp truncation there.
```

## The projection

The surface density is the line-of-sight integral of $\rho$, and the
excess surface density compares it to its own interior mean —
$\gamma_t = \Delta\Sigma/\Sigma_{\rm crit}$ is what a lensing survey
actually fits:

$$
\Sigma(R) = \int_{-\infty}^{\infty} \rho\!\left(\sqrt{R^2+\chi^2}\right)
d\chi,
\qquad
\Delta\Sigma(R) \equiv \bar\Sigma(<R) - \Sigma(R),
\qquad
\bar\Sigma(<R) = \frac{2}{R^2}\int_0^R \Sigma(R')\,R'\,dR'.
$$

Substituting $r=\sqrt{R^2+\chi^2}$ turns the $\chi$-integral above into
the radial form it actually is — the **Abel transform** of $\rho(r)$,

$$
\Sigma(R) = 2\int_R^\infty \frac{\rho(r)\,r}{\sqrt{r^2-R^2}}\,dr.
$$

$\Sigma(R)$ is a single, local integral. $\bar\Sigma(<R)$ is not: it is a
*cumulative* integral over $\Sigma$ from $0$ to $R$, so getting
$\Delta\Sigma(R)$ at one radius numerically means having already
integrated $\Sigma$ over every smaller radius first — and doing that
accurately on a tabulated grid is genuinely delicate.
`clenspy.utils.sigma_to_deltasigma_cumtrapz` (the fallback used where no
closed form exists, e.g. Einasto's $n\le3/2$ below) does this by
cumulative trapezoidal quadrature in $\ln R$, which implicitly assumes
$\Sigma$ is constant inward of the grid's first point — accurate only
once that first point is small enough relative to the profile's scale,
and capable of giving exactly $0$ for a cored profile if it isn't.

NFW and Einasto sidestep all of this: both classes have full **analytical**
$\Sigma$ and $\Delta\Sigma$ (Wright & Brainerd 2000 for NFW; the Catalan
series below for Einasto), so the cumulative numerical route above is
never on the hot path for either — it exists only as the last-resort
fallback. Both classes also expose `mean_sigma`, $\bar\Sigma(<R)$, from
its own closed form rather than reconstructed as $\Sigma+\Delta\Sigma$ —
the two routes agree to machine precision, checked in the example below.

## NFW: the Wright & Brainerd (2000) closed forms

Both the truncated and untruncated Abel transform above have a closed
form for NFW, so `NfwProfile.sigma`/`deltasigma` never touch the
numerical fallback described above:

$$
\Sigma(R) = 2r_s\rho_s\,f(x), \qquad
\Delta\Sigma(R) = r_s\rho_s\,g(x), \qquad x = \frac{R}{r_s},
$$

with $f$, $g$ the piecewise (inside/outside $r_s$) closed forms of Wright
& Brainerd (2000). Both kernels are 0/0 at the $x=1$ kink in their direct
form; `NfwProfile` switches to a Taylor series in $|x-1|$ there, and to a
small-$x$ series below $x\sim3\times10^{-3}$ where the direct branches
cancel catastrophically (both traps documented inline in `_fNfw`/`_gNfw`).

## Einasto: a closed form, just not an elementary one

$\Sigma$ and $\Delta\Sigma$ *do* have a closed form for general $n$ — it
is just not the single elementary function {doc}`density_profiles`'s
$M_{\rm 3D}(r)$ gets from the incomplete gamma function. Closing a
Mellin-Barnes contour integral of the line-of-sight projection instead
gives an infinite residue series that is **exact at every radius**, for
every $n>0$ (proved by contour closure, checked against an independent
numerical Mellin-Barnes integral and a brute-force Abel quadrature to
$10^{-40}$ — see {doc}`einasto_math`) — not a truncated approximation
standing in for a missing closed form:

$$
\Sigma(x) = \sqrt{\pi}\rho_0 h \Big[\sum_{k\ge1}A_k\,x^{k/n+1}
+\sum_{j\ge0}S_j\,x^{2j}\Big], \qquad
\Delta\Sigma(x) = \sqrt{\pi}\rho_0 h \Big[\sum_{k\ge1}D_k\,x^{k/n+1}
+\sum_{j\ge1}T_j\,x^{2j}\Big], \qquad x=\frac{R}{h},
$$

$$
A_k = \frac{(-1)^k}{k!}\frac{\Gamma\!\big(-\tfrac12-\tfrac{k}{2n}\big)}
{\Gamma\!\big(-\tfrac{k}{2n}\big)}, \quad
D_k = -\frac{n+k}{3n+k}A_k, \qquad
S_j = \frac{2n(-1)^j}{j!}\frac{\Gamma(n-2nj)}{\Gamma(\tfrac12-j)}, \quad
T_j = -\frac{j}{j+1}S_j\;\;(T_0=0).
$$

This is `EinastoProfile.sigma`/`deltasigma`'s dispatch for every
non-anchor $n$ (both above and below $3/2$), via `EinastoLowN`: the
Retana-Montenegro et al. (2012) case-1 residue series above with resonance
pairing where the two coefficient families collide, switching to an
all-positive $E_{\nu_k}$ representation only at $z=(R/h)^{1/n}$ far beyond
any physical radius for $n>3/2$. $n=1$ and $n=1/2$ collapse this to
genuinely elementary closed forms in modified Bessel functions.

```{note}
An older, plain Catalan $c_k E_{\nu_k}(x)$ series used to compute
$\Sigma$/$\Delta\Sigma$ directly for $n>3/2$ — it has been **removed**
(`_build`, `self._ck`, `self._nu_k`, `_E_nu`, `order_for_tol` are all gone)
rather than left as dead code: its `DeltaSigma` truncation error was
$O(K^{-1/2})$ *absolute*, i.e. 30-200% relative, and nothing read it any
more once the residue series above took over for every $n$ — not
`sigma`/`deltasigma`, and not `power_spectrum` either, whose own $n>3/2$
branch is a wholly separate analytic cascade (`clenspy.halo.einasto_series`)
that never touched it. Only the bare integer `self.order` survives,
sizing `power_spectrum`'s explicit `small_k`/`large_k` branches' own
unrelated $A_m^\pm$ series — not its default `"auto"` dispatch.
```

See {doc}`einasto_math` for the full per-regime dispatch and the small-$R$
asymptotic used where the native
series cancels catastrophically.

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"projected-profiles\"]"
:end-before: "%% [markdown]"
:language: python
```

```
Sigma_NFW(R)          [Msun/Mpc^2] = [1.91291224e+14 3.25021969e+13 1.11712030e+13 3.36400736e+12]
Sigma_Einasto(R)      [Msun/Mpc^2] = [1.97446111e+14 3.17987731e+13 9.66245761e+12 2.13438607e+12]
DeltaSigma_NFW(R)     [Msun/Mpc^2] = [8.51802198e+13 3.87203748e+13 2.00729059e+13 8.74034644e+12]
DeltaSigma_Einasto(R) [Msun/Mpc^2] = [8.89031956e+13 4.03108101e+13 2.08324780e+13 8.73234570e+12]
NFW Sigmabar consistency, max|rel| = 4.218847493575595e-15
```

See also: {doc}`api/index` for the full `clenspy.halo` reference,
{doc}`notation` for the symbol table, {doc}`einasto_math` for the Einasto
profile's derivations.
