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
NL}$ stand-ins — chosen to demonstrate the *shape*; the amplitude here
still runs a bit high because `HodMor.des_y1()` is not mutually calibrated
against this toy halo model (see the note below), but it is no longer the
order-of-magnitude blowup a naive closure gives. The sigmoid sits halfway
between the plateaus at $\theta=\theta_\lambda$ by construction.
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
$I_1=\mathcal P[b\,\xi_{\rm NL}\,\sigma(\theta)]$, $D=I_2-I_1$ — plus the
variance analogs $P_1^{(2)}$, $I_2^{(2)}$ (`operators_var`, squared
weights). Writing $\Delta_{\rm RND}=P_1+b_{\rm eff}I_2$ (the mean
projected-richness excess for a *random* line of sight) and
$\delta=\langle\lambda^{\rm ob}-\lambda^{\rm tr}\rangle/\Delta_{\rm RND}-1$,
the closure reduces exactly (`excess_delta`, `b_small_large`) to a
one-parameter linear response:

$$
b_{\rm large} = b_{\rm eff}\big(1+0.13\,\delta\big), \qquad
b_{\rm small} = b_{\rm eff} + \delta\, A_s, \qquad
A_s = \frac{\Delta_{\rm RND} - 0.13\, b_{\rm eff} I_1}{D},
$$

with $b_{\rm eff}=\langle b(M,z)\rangle$ the unselected aggregate bias and
$A_s$ the closure's **gain** — how strongly $b_{\rm small}$ responds to
$\delta$ (typically 18–40; see {doc}`plan-bsel-stable-closure` for the
derivation). The one physical input, $\delta$, is estimated from the
model's own operators — a first-order Eddington tilt of the *correlated*
part of the projection variance,

$$
\delta = \gamma\,\frac{b_{\rm eff}\,I_2^{(2)}}{\Delta_{\rm RND}}, \qquad
\gamma = -\left.\frac{d\ln n(\lambda^{\rm tr})}{d\lambda^{\rm tr}}
\right|_{\lambda^{\rm ob}}
\quad\text{(`gamma_lambda`)},
$$

rather than marginalized over an externally calibrated
$P(\lambda^{\rm ob}\mid\lambda^{\rm tr})$ kernel — see the note below for
why that used to be the unstable part.

```{note}
**The deliverable is two scalars per bin.** Both plateaus are affine in
$\lambda^{\rm tr}$ (equivalently in $\delta$), so a $\lambda^{\rm tr}$
posterior would contribute only its *mean* to either one — no quadrature
over $\lambda^{\rm tr}$ is needed, or helps. `SelBiasEngine` never stores a
$\theta$ grid; `SelectionBiasTable` is two columns wide, matching the
`y3_cluster_cpp` wall contract.
```

```{note}
**The 0.13 is the one non-closed-form number** — a Buzzard-calibrated
amplitude, exposed as `SelBiasEngine.boost_slope`. $b_{\rm small}$ is a
**gain of 18–40** on $\delta$ (`excess_delta`), not a fragile
cancellation: $D=I_2-I_1$ is computed directly (never as a float
subtraction) and is perfectly well-conditioned on its own. The gain lives
in the $0.13$ itself — pinning $b_{\rm large}$ to move only 13% per unit
$\delta$ forces the *entire* required change in the correlated-channel
integral through the $(1-\sigma)$ (small-scale) piece, so $b_{\rm small}$
is a residual bucket by construction. That gain means $\delta$ has to be
right to a few percent for $b_{\rm small}$ to be right to tens of percent
— an earlier version of this engine got $\delta$ from marginalizing
`_closure` over an externally calibrated $P(\lambda^{\rm ob}\mid
\lambda^{\rm tr})$ kernel, which turned out to overestimate
$\langle\lambda^{\rm ob}-\lambda^{\rm tr}\rangle$ by $1.5$–$2.2\times$
(an exponential-tilt divergence, not a calibration-choice problem — see
{doc}`plan-bsel-stable-closure`) and inflated $b_{\rm small}$ by
$3$–$4\times$ against Costanzi et al. (2026)'s own published curve. The
figure's $b_{\rm small}\approx11$ (down from $\approx19$ under the old
closure) is what is left once that bug is gone: `HodMor.des_y1()` still
isn't calibrated against the demo's toy Tinker+CAMB halo model, so
$\delta$ is not small here, but it no longer gets amplified by a second,
spurious error on top. See {doc}`validation` for the calibrated
(`HodMor.buzzard()`) amplitude against the mock and the published figure.
```

```{warning}
**Open limitation: $\delta$'s redshift dependence has the wrong sign.**
`excess_delta` gets the *mean* amplitude right (validated in
{doc}`validation`), but at fixed richness its value *decreases* with
$z^{\rm ob}$ while the published Fig. 6 curve needs it to *increase* —
confirmed converged (not a quadrature issue) and traced to
$I_2^{(2)}$'s own redshift evolution falling faster than $\Delta_{\rm
RND}$ rises. Not fixed; see {doc}`plan-bsel-stable-closure` §9 for the
full diagnostic. Treat any single-$z$-bin use of this closure as
mean-level-correct but shape-uncertain across $z$.
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
sign-indefinite denominator. $D=I_2-I_1$ itself — the $b_{\rm small}$
gain's denominator — is computed as its own direct quadrature
($\mathcal P[b\,\xi_{\rm NL}(1-\sigma)]$, by linearity) rather than a
float subtraction of $I_2$ and $I_1$, removing the one real cancellation
risk in this operator; it is *not*, however, where $b_{\rm small}$'s
sensitivity actually comes from (see the note above and
{doc}`plan-bsel-stable-closure`).

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"selection-bias\"]"
:end-before: "%% [markdown]"
:language: python
```

```
theta_lambda = 0.001073 rad, b_small = 11.005, b_large = 4.320
theta/theta_lambda=0.00  b_sel=9.5167
theta/theta_lambda=0.50  b_sel=7.6629
theta/theta_lambda=1.00  b_sel=5.8091
theta/theta_lambda=2.00  b_sel=4.4739
theta/theta_lambda=5.00  b_sel=4.3204
```

See also: {doc}`api/index` for the full `clenspy.selection` reference,
{doc}`notation` for the symbol table, {doc}`selection_function` for the
$S_i$ this same engine's richness marginalization builds on, and
{doc}`plan-bsel-stable-closure` for the closure's derivation and its
validation against the mock and the published Costanzi et al. (2026)
figure ({doc}`validation`).
