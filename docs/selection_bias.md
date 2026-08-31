# The Selection-Affected Bias b_sel

A cluster selected at observed richness $\lambda^{\rm ob}$ preferentially
sits behind extra line-of-sight structure — that is what pushed its
richness up in the first place. Its two-halo term therefore does not carry
the ordinary halo bias $b(M,z)$ ({doc}`halo_bias`); it carries a
$\theta$-dependent $b_{\rm sel}(\theta)$ that this module gives a
closed-form alternative to calibrating on Buzzard light-cones.

```{figure} _static/img/selection_bias.png
:alt: b_sel(theta) sigmoid between b_small and b_large plateaus
:width: 75%
:align: center

$b_{\rm sel}(\theta)$ from `SelBiasEngine`'s own toy `hmf`/`bias`/$\xi_{\rm
NL}$ stand-ins — chosen to demonstrate the *shape*, not a calibrated
amplitude (see the note below on why $b_{\rm small}$ is unphysically large
here). The sigmoid sits halfway between the plateaus at
$\theta=\theta_\lambda$ by construction.
```

## Two plateaus and a sigmoid

A cluster's own aperture and the field far outside it see two different
effective biases, and $b_{\rm sel}(\theta)$ is the smooth interpolation
between them:

$$
b_{\rm sel}(\theta) = b_{\rm small}\big[1-\sigma(\theta)\big]
+ b_{\rm large}\,\sigma(\theta)
$$

with $b_{\rm small}$ the bias inside the cluster's own aperture
($\theta\lesssim\theta_\lambda$) and $b_{\rm large}$ the field value well
outside it. Both come from a single **closure**: averaging any quantity
$X$ against the line-of-sight projection kernel defines an operator
$\mathcal P[X]$, specialized three ways —
$P_1=\mathcal P[1]$, $I_2=\mathcal P[b\,\xi_{\rm NL}]$,
$I_1=\mathcal P[b\,\xi_{\rm NL}\,\sigma(\theta)]$ — and then

$$
b_{\rm large} = b_{\rm eff}\big[1+0.13\,\delta^{\rm prj}\big], \qquad
b_{\rm small} = \frac{(\lambda^{\rm ob}-\lambda^{\rm tr}) - P_1
- b_{\rm large}I_1}{I_2-I_1},
$$

with $b_{\rm eff}=\langle b(M,z)\rangle$ the unselected aggregate bias and
$\delta^{\rm prj}$ the fractional excess of observed over true richness
relative to the closure's own prediction $\Delta^{\rm prj}_{\rm RND}
=P_1+b_{\rm eff}I_2$.

```{note}
**The deliverable is two scalars per bin.** The $\lambda^{\rm tr}$
marginalization commutes with the sigmoid — $\sigma(\theta)$ carries no
$\lambda^{\rm tr}$ dependence — so averaging the plateaus first and
building the sigmoid after is exact, not an approximation. `SelBiasEngine`
never stores a $\theta$ grid; `SelectionBiasTable` is two columns wide,
matching the `y3_cluster_cpp` wall contract.
```

```{note}
**The 0.13 is the one non-closed-form number** — a Buzzard-calibrated
amplitude, exposed as `SelBiasEngine.boost_slope` so it can be varied
rather than hidden. $b_{\rm small}$ comes from a **linear inversion** and
is the output that can go unstable: when $I_2\to I_1$ the denominator
vanishes, and `_closure` falls back to $b_{\rm large}$ there — a named
degradation, not a silent one. The figure's $b_{\rm small}\approx19$ is this
instability made visible on purpose: `HodMor.des_y1()` is not calibrated
against the demo's Tinker(2008)+Tinker(2010)+CAMB halo model at this
$(\lambda^{\rm ob},z^{\rm ob})$, so $\Delta^{\rm prj}_{\rm RND}$ under-predicts the true
richness excess and the division by the small $I_2-I_1$ inflates
$b_{\rm small}$. The *shape* of $b_{\rm sel}(\theta)$ is still correct;
only the amplitude needs a mutually calibrated MOR, e.g.
`HodMor.from_lognormal()` or `HodMor.buzzard()`.
```

## Units and numerics

Masses are **physical** $M_\odot$ here, not $h^{-1}M_\odot$: `SelBiasEngine`
shares its `hmf`/`bias`/`xi_nl` with {doc}`projection_lensing`'s `SigmaPrj`
(pass a built one, or let the engine build a default), all in that
convention. $R_\lambda$ is physical Mpc, $\chi$ and $\xi_{\rm NL}$ are
comoving Mpc, angles are radians, $b_{\rm sel}$ is dimensionless. This
differs from `clenspy.selection.scaling_relation`, which is h-scaled;
`PhysicalMassMor` converts at the boundary — the engine wraps the raw MOR
in it internally, so the caller passes e.g. `HodMor.des_y1()` directly.

The photo-$z$ weight is the **exact tabulated window**
(`clenspy.kernels.photoz.y3_photoz_window`), passed with `n_sigma=1.0`
because that table already *is* the $n_\sigma\sigma_z$ half-width. Its
support is asymmetric by 17% at $z^{\rm ob}=0.4$, so the line-of-sight
bounds come from `photoz_projection_support` and not a symmetric
$z^{\rm ob}\pm$ width.

The $\mathcal P[X]$ operator's line-of-sight integral is the same
cosh–Abel `LosGeometry`/`integrate_los` machinery as `SigmaPrj`'s own
(see {doc}`projection_lensing`, "Exclusion"): halo exclusion is an exact
interval boundary, never a mask or a per-$z$ grid split. One named
approximation remains: $\xi_{\rm NL}$ is **clipped at zero**, discarding
the BAO trough (measured at $O(10^{-4})$ in a $w_z$-suppressed region),
which keeps $I_1,I_2$ positive so the closure cannot divide by a
sign-indefinite denominator. $I_2-I_1$ itself — the $b_{\rm small}$
denominator — is computed as its own direct quadrature
($\mathcal P[b\,\xi_{\rm NL}(1-\sigma)]$, by linearity) rather than a
float subtraction of $I_2$ and $I_1$, removing the cancellation error at
the one place this closure can go unstable.

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"selection-bias\"]"
:end-before: "%% [markdown]"
:language: python
```

```
theta_lambda = 0.001073 rad, b_small = 18.666, b_large = 5.068
theta/theta_lambda=0.00  b_sel=15.6376
theta/theta_lambda=0.50  b_sel=11.8669
theta/theta_lambda=1.00  b_sel=8.0962
theta/theta_lambda=2.00  b_sel=5.3804
theta/theta_lambda=5.00  b_sel=5.0681
```

See also: {doc}`api/index` for the full `clenspy.selection` reference,
{doc}`notation` for the symbol table, {doc}`selection_function` for the
$S_i$ this same engine's richness marginalization builds on.
