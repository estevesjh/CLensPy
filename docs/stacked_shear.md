# Stacked Shear: ΔΣ_ij(R)

The lensing observable in a binned cluster analysis is the stacked excess
surface density $\Delta\Sigma_{ij}(R)$ — the same per-halo weight as
{doc}`number_counts`'s $\langle N_{ij}\rangle$, contracted against the
halo's own profile instead of 1. This is the cluster's own one-halo term;
a real measurement also carries a projected two-halo excess on top of it,
$\Delta\Sigma_{\rm prj}$ — see {doc}`shear_proj` for how the two combine.

```{figure} _static/img/observables.png
:alt: Stacked DeltaSigma_ij(R) for four richness bins
:width: 75%
:align: center

$\Delta\Sigma_{ij}^{\rm 1h}(R)$, the counts weight contracted against the
halo's own profile, for four richness bins at fixed redshift. It rises
with richness at every radius, because $\langle M\rangle_{ij}$ does —
exactly the sanity property a broken weight would violate first.
```

## The one-halo contraction

$$
\Delta\Sigma_{ij}^{\rm 1h} = \frac{\int dM\!\int\!dz\; W_{ij}\,
  \Delta\Sigma(R\mid M,z)}{\langle N_{ij}\rangle}
$$

`StackedDeltaSigma` is exactly this contraction — it takes a
`ClusterCounts` and calls its `average`, owning no weight of its own, so
the stack and the counts cannot disagree about which haloes are in the
bin. That is checkable directly: with $\Delta\Sigma\equiv1$ the stack must
return exactly 1, and with $\Delta\Sigma=M$ it must reproduce
$\langle M\rangle_{ij}$ — both asserted in the tests and the first in the
example below.

```{note}
$\Omega(z)$ multiplies the **counts** and cancels identically in the
`average` ratio — it must never be applied to a stacked profile a second
time, or the footprint is counted twice.
```

The one-halo term is itself a mixture over centring,

$$
\Delta\Sigma^{\rm 1h}(R) = (1-f_{\rm mis})\,\Delta\Sigma_{\rm cen}(R)
+ f_{\rm mis}\,\Delta\Sigma_{\rm mis}(R),
$$

with $f_{\rm mis}=0.25\pm0.08$ from the DES Y3 calibration of Kelly et al.
(2024) — `StackedDeltaSigma.mixture` applies it, and being linear, it
commutes with the stack: mixing then stacking equals stacking then mixing,
so a caller may do either in whichever order is cheaper.

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"stacked-shear\"]"
:end-before: "%% [markdown]"
:language: python
```

```
DeltaSigma_ij^1h(R) [Msun/Mpc^2], lowest redshift bin, rises with richness:
[[6.97978965e+13 5.38288665e+13 3.09880765e+13 1.26630791e+13
  3.88154204e+12 9.77650143e+11]
 [8.00754870e+13 6.36057002e+13 3.82833052e+13 1.63623742e+13
  5.18726305e+12 1.33535666e+12]
 [9.28192399e+13 7.58913343e+13 4.77750289e+13 2.13872569e+13
  7.02293569e+12 1.84960860e+12]
 [1.17797585e+14 1.00464460e+14 6.79638090e+13 3.30144419e+13
  1.15866261e+13 3.19108647e+12]]

stacking DeltaSigma=1, max|result - 1| = 0.00e+00
```

See also: {doc}`api/index` for the full `clenspy.observables` reference,
{doc}`notation` for the symbol table, {doc}`number_counts` for the shared
weight $W_{ij}$, {doc}`shear_proj` for the projected two-halo term this
combines with, and {doc}`covariance` for the uncertainty on this profile.
