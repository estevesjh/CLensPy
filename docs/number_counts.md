# Cluster Counts: ⟨N_ij⟩

A binned cluster analysis first asks how many clusters landed in each
$(\lambda^{\rm ob}, z)$ bin. `ClusterCounts` answers that from a single
per-halo weight, evaluated once and integrated over mass and redshift.

## The weight and its counts contraction

Every binned prediction in this package — counts or stacked profile
({doc}`stacked_shear`) — traces back to the same per-halo weight,

$$
W_{ij}(M,z^{\rm tr}) = \Omega(z^{\rm tr})\,\frac{dV}{d\Omega\,dz^{\rm tr}}\,
n(M,z^{\rm tr})\,\mathcal S_{ij}(M,z^{\rm tr}),
$$

with $n(M,z)$ the mass function ({doc}`mass_function`), $\mathcal S_{ij}$
the selection function ({doc}`selection_function`), and
$\Omega(z)\,dV/(d\Omega\,dz)$ the survey's comoving volume element
({doc}`survey`). The counts are $W_{ij}$ contracted against 1:

$$
\langle N_{ij}\rangle = \int dM\!\int\!dz\; W_{ij}.
$$

Contracting the same weight against a per-halo quantity instead of 1 gives
that quantity's stacked average — {doc}`stacked_shear` is exactly this,
with $X = \Delta\Sigma(R\mid M,z)$; it takes a `ClusterCounts` and calls
its `average`, owning no weight of its own, so the stack and the counts
cannot disagree about which haloes are in the bin.

```{note}
$\Omega(z)$ multiplies the **counts** and cancels identically in the
`average` ratio used for a stacked profile — it must never be applied to
a stack a second time, or the footprint is counted twice. `ClusterCounts`
is also where the h-convention crosses: the mass function and
mass-observable relations are h-scaled, astropy's volume element is not,
and `_volume_per_dz` is the one place that conversion happens, with the
powers of $h$ written out.
```

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"number-counts\"]"
:end-before: "%% [markdown]"
:language: python
```

```
<N_ij> (richness x redshift bins):
[[345.66171541 641.92848156 772.89742931]
 [228.13035905 427.1340156  518.0046128 ]
 [ 97.1349326  183.1411802  223.43978179]
 [129.85585645 247.75454502 305.22110145]]
```

See also: {doc}`api/index` for the full `clenspy.observables` reference,
{doc}`notation` for the symbol table, {doc}`stacked_shear` for the second
contraction of this same weight, and {doc}`covariance` for the
uncertainty on the counts.
