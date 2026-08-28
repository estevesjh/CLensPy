# Observables: Counts and Stacked DeltaSigma

A binned cluster analysis compares its model against two numbers per bin:
how many clusters landed there, and what their stacked lensing profile
looks like. Both come from **the same weight**
$W_{ij}(M,z^{\rm tr})$ — this is not two models, but one weight contracted
two ways.

```{figure} _static/img/observables.png
:alt: Stacked DeltaSigma_ij(R) for four richness bins
:width: 75%
:align: center

$\Delta\Sigma_{ij}(R)$, the second contraction of $W_{ij}$, for four
richness bins at fixed redshift. It rises with richness at every radius,
because $\langle M\rangle_{ij}$ does — exactly the sanity property a
broken weight would violate first.
```

## One weight, two contractions

Every binned prediction in this package — counts or stacked profile —
traces back to the same per-halo weight, evaluated once and contracted
two different ways:

$$
W_{ij}(M,z^{\rm tr}) = \Omega(z^{\rm tr})\,\frac{dV}{d\Omega\,dz^{\rm tr}}\,
n(M,z^{\rm tr})\,\mathcal S_{ij}(M,z^{\rm tr}),
$$

with $n(M,z)$ the mass function ({doc}`mass_function`), $\mathcal S_{ij}$
the selection function ({doc}`selection_function`), and
$\Omega(z)\,dV/(d\Omega\,dz)$ the survey's comoving volume element
({doc}`survey`). Contract it against 1 for the counts, or against any
per-halo quantity $X(M,z)$ for that quantity's stacked average:

$$
\langle N_{ij}\rangle = \int dM\!\int\!dz\; W_{ij}, \qquad
\langle X\rangle_{ij} = \frac{\int dM\!\int\!dz\;W_{ij}\,X(M,z)}
{\langle N_{ij}\rangle}.
$$

`StackedDeltaSigma` is exactly this second contraction with
$X=\Delta\Sigma(R\mid M,z)$ — it takes a `ClusterCounts` and calls its
`average`, owning no weight of its own, so the stack and the counts cannot
disagree about which haloes are in the bin. That is checkable directly:
with $\Delta\Sigma\equiv1$ the stack must return exactly 1, and with
$\Delta\Sigma=M$ it must reproduce $\langle M\rangle_{ij}$ — both asserted
in the tests and the first in the example below.

```{note}
$\Omega(z)$ multiplies the **counts** and cancels identically in the
`average` ratio — it must never be applied to a stacked profile a second
time, or the footprint is counted twice. `ClusterCounts` is also where the
h-convention crosses: the mass function and mass-observable relations are
h-scaled, astropy's volume element is not, and `_volume_per_dz` is the one
place that conversion happens, with the powers of $h$ written out.
```

The one-halo piece of a real stacked profile is itself a mixture over
centring,

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
:start-after: "tags=[\"observables\"]"
:end-before: "%% [markdown]"
:language: python
```

```
<N_ij> (richness x redshift bins):
[[345.66171541 641.92848156 772.89742931]
 [228.13035905 427.1340156  518.0046128 ]
 [ 97.1349326  183.1411802  223.43978179]
 [129.85585645 247.75454502 305.22110145]]

DeltaSigma_ij(R) [Msun/Mpc^2], lowest redshift bin, rises with richness:
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
{doc}`notation` for the symbol table, {doc}`covariance` for the
uncertainty on these same two contractions.
