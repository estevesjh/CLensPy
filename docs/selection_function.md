# The Selection Function

The halo mass function ({doc}`mass_function`) predicts how many haloes
exist; a cluster catalogue only counts the ones that got *observed* into a
richness and redshift bin. $\mathcal S_{ij}(M, z^{\rm tr})$ is the
probability of that happening — the factor that turns $dn/d\ln M$ into a
prediction for $\langle N_{ij}\rangle$.

```{figure} _static/img/selection_function.png
:alt: Selection function S_i(M) for four richness bins plus their sum
:width: 75%
:align: center

$S_i(M,z=0.3)$ for four richness bins. Each peaks near the mass whose mean
richness sits mid-bin and tails into its neighbors from scatter; the sum
over every bin (dashed) is the probability of landing anywhere in
$\lambda\in[20,200)$ at all, rising from 0 to 1 as $M$ grows.
```

## Two exact factors and one quadrature

The full forward model is a five-dimensional integral over true and
observed mass, richness, and redshift. Three of those five collapse
analytically — the observed-redshift integral to a Gaussian CDF difference
({doc}`lensing_kernel`'s photo-z kernel), the observed-richness integral to
the EMG kernel's CDF difference — leaving

$$
\mathcal S_{ij}(M,z^{\rm tr}) = S_i(M,z^{\rm tr})\,\mathcal S_j(z^{\rm tr}),
\qquad
S_i(M,z) = \int_0^\infty d\lambda^{\rm tr}\,
\mathcal S_i(\lambda^{\rm tr},z)\,P(\lambda^{\rm tr}\mid M,z),
$$

with the remaining $\lambda^{\rm tr}$ integral done by one Gauss-Legendre
rule shared across every $(M,z)$ cell. $P(\lambda^{\rm tr}\mid M,z)$ is the
mass-observable relation's PDF (`LogNormalMor` or `HodMor`) — a swappable
collaborator, not a hard-coded law.

```{note}
The factorization $\mathcal S_{ij}=S_i\,\mathcal S_j$ is **exact**, not an
approximation — a property of the photo-z kernel, since
$P(z^{\rm ob}\mid z^{\rm tr})$ depends on the richness bin only through a
per-bin constant $\sigma_z(\Delta\lambda_i)$. The approximation that *does*
bite is the quadrature bracket: $\lambda^{\rm tr}$ runs over
$(0,\infty)$, but the rule only covers $[\max(0,\mu_{\rm eff}-L\sigma_{\rm
eff}),\,\mu_{\rm eff}+L\sigma_{\rm eff}]$. Too narrow an $L$ silently
discards probability — `residual` measures exactly how much, rather than
trusting the default $L=8$.
```

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"selection-function\"]"
:end-before: "%% [markdown]"
:language: python
```

```
SelectionFunction(4 richness x 3 redshift bins, LogNormalMor(A=76.9, B=1.02, C=0.29, D=0.23), L=8, n_quad=64)

S_i(M, z=0.3): probability of landing in each richness bin
M= 1.0e+13 h^-1 Msun  S_i=[0.0239 0.0086 0.0014 0.0003]  sum=0.0342  bracket_miss=2.6e-04
M= 5.0e+13 h^-1 Msun  S_i=[0.1262 0.0372 0.0058 0.0011]  sum=0.1704  bracket_miss=3.0e-05
M= 1.0e+14 h^-1 Msun  S_i=[0.4134 0.2507 0.0448 0.0087]  sum=0.7177  bracket_miss=1.0e-05
M= 3.0e+14 h^-1 Msun  S_i=[6.000e-04 2.950e-02 1.686e-01 8.012e-01]  sum=0.9999  bracket_miss=3.1e-06
M= 1.0e+15 h^-1 Msun  S_i=[0.     0.     0.     0.1618]  sum=0.1618  bracket_miss=1.8e-06
```

The sum is not monotonic in mass for a bounded bin set: at $10^{15}$
$h^{-1}M_\odot$ it falls back to 0.16, since this relation's mean richness
there is far above the last edge (200) — the binning runs out, not the
physics.

See also: {doc}`api/index` for the full `clenspy.selection` reference,
{doc}`notation` for the symbol table, {doc}`selection_bias` for the
selection-affected halo bias this same $S_i$ feeds into.
