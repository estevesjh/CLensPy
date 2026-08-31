# Shear Projection: ΔΣ_prj in the Stacked Profile

`SigmaPrj` supplies the second piece of a real stacked-shear measurement.
The halo's own one-halo profile ({doc}`stacked_shear`) sits along a line
of sight that a richness-selected sample has already picked out for being
structure-rich, so the stack also carries a projected two-halo excess on
top of it — two separate models, summed by hand, since no single class
owns both pieces at the binned level:

$$
\Delta\Sigma_{ij}^{\rm tot}(R) = \Delta\Sigma_{ij}^{\rm 1h}(R)
  + \Delta\Sigma_{\rm prj}\bigl(R \mid \lambda^{\rm ob}_{ij}, z^{\rm ob}_{ij}\bigr).
$$

Unlike the one-halo term, $\Delta\Sigma_{\rm prj}$ is not a further
contraction of the counts weight $W_{ij}$ ({doc}`number_counts`): it is
already a richness-conditioned model of the correlated structure around
the cluster (`SigmaPrj.deltasigma_prj`, Costanzi et al. 2026 Eq. 13), so
it is evaluated once at the bin's representative
$(\lambda^{\rm ob}_{ij}, z^{\rm ob}_{ij})$ rather than integrated against
$W_{ij}$ a second time.

## What Σ_prj is

Every neighbour halo of mass $M$ at angular offset $\theta$ contributes
its own mass shell $M_\theta(R \mid M)$ — its projected mass inside
radius $R$, offset by the real transverse separation
$R_\theta = \theta\chi_o$ — weighted by the correlated excess
$n_{\rm cl}(\theta, M)$ of finding such a neighbour there over the
uncorrelated background rate $n_{\rm rnd}(\theta, M)$:

$$
\Sigma_{\rm prj}(R) = \int d\theta\, 2\pi\sin\theta \int dM\;
  n_{\rm cl}(\theta, M)\, M_\theta(R \mid M),
$$

with $\Delta\Sigma_{\rm prj}$ the same sum with the mass shell swapped
for its signed excess. A halo within the exclusion radius
$R_{\rm excl} = R_\lambda(\lambda^{\rm ob})(1+z^{\rm ob})$ *is* the
cluster, so there $n_{\rm cl} = -n_{\rm rnd}$ rather than carrying any
clustering weight — certainty of absence, not clustering. Full
derivation, the background/correlated channel split, exclusion
semantics, and numerics: {doc}`projection_lensing`.

```{note}
The projection excess is not a small correction. In the example below,
at $R = 8\,{\rm Mpc}$ it already carries 79% of the total stacked
$\Delta\Sigma$ at $\lambda^{\rm ob}=20$ — the one-halo term alone
underpredicts the stack past a few Mpc.
```

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"shear-proj\"]"
:end-before: "%% [markdown]"
:language: python
```

```
DeltaSigma(R) [Msun/Mpc^2] at (lambda_ob=20, z_ob=0.5):
  R [Mpc]           1h          prj        total   prj frac
     0.50   3.4762e+13   9.0368e+11   3.5666e+13     0.0253
     2.00   8.8014e+12   8.2721e+12   1.7073e+13     0.4845
     8.00   1.2389e+12   4.7051e+12   5.9440e+12     0.7916
    25.00   1.9426e+11   1.1688e+12   1.3631e+12     0.8575
```

See also: {doc}`api/index` for the full `clenspy.lensing` reference,
{doc}`notation` for the symbol table, {doc}`stacked_shear` for the
one-halo term this adds to, and {doc}`projection_lensing` for the full
derivation, exclusion semantics, and mock validation of $\Sigma_{\rm prj}$.
