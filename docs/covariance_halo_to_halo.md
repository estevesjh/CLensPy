# Halo-to-Halo Covariance

{doc}`covariance`'s Gaussian-field term treats the halo and matter fields
as Gaussian and gives the variance of the *mean* stacked profile. But a
stack is built from $N_{\rm cl}$ individual clusters, each with its own
mass and concentration, each carrying its own $\Delta\Sigma$ — and the
stack of them inherits the covariance of that population. This sixth term
(McClintock et al. 2019; Gruen et al. 2015) is not inside the Gaussian
bracket; it has a different origin — a finite-sample effect of stacking a
heterogeneous population, not a field property — so `clenspy` keeps it
separate and adds it to the total.

```{figure} _static/img/halo_to_halo.png
:alt: Fractional intrinsic covariance vs radius for four richness bins
:width: 75%
:align: center

$\sigma_{\rm intr}(R)/\langle\Delta\Sigma(R)\rangle$ for four richness
bins. It generally rises with richness — a broader mass population means
more per-cluster scatter — and peaks near the one-halo/two-halo
transition, where the Hayashi & White $\max$ composition below is most
sensitive to which term wins.
```

```{figure} _static/img/halo_to_halo_matrix.png
:alt: Halo-to-halo covariance and correlation matrix for one richness/redshift bin
:width: 85%
:align: center

The same term, but as the full $R\times R$ matrix for one richness/
redshift bin rather than a per-bin summary curve — McClintock et al.
(2019) Fig. 6's own presentation of their SAC matrix. Left: actual
values (log scale); right: the correlation matrix, close to 1 along the
diagonal band since broad mass/concentration scatter moves nearby radii
together.
```

## The intrinsic DeltaSigma covariance, Wu et al. (2019) eq. cov_intr

Wu et al. (2019) name this term `eq:cov_intr` and calibrate it from N-body
simulations (Abacus), since the Gaussian-field approximation underestimates
the small-scale covariance once halo concentration, sub-structure, and
orientation scatter enter. `clenspy` instead evaluates it analytically,
following McClintock et al. (2019) and Gruen et al. (2015): the population
covariance of the per-cluster $\Delta\Sigma$ profiles themselves, divided
by the number of clusters in the bin,

$$
C^{\rm intr}_{ij} = \frac{1}{N_{\rm cl}}\Big[
\big\langle\Delta\Sigma(R_i)\,\Delta\Sigma(R_j)\big\rangle_{\rm pop}
-\big\langle\Delta\Sigma(R_i)\big\rangle_{\rm pop}
\big\langle\Delta\Sigma(R_j)\big\rangle_{\rm pop}\Big],
$$

```{note}
This package's $\sigma_{\rm intr}$ is unrelated to {doc}`notation`'s own
$\sigma_{\rm intr}$, the HOD super-Poisson satellite scatter
(`selection.scaling_relation`) — same letters, two unconnected quantities,
one per paper (Wu et al. here; the HOD literature there).
```

averaged over the bin's own selection-weighted mass population — the same
$W_{ij}$-derived $P(M)$ the counts themselves are built from
({doc}`number_counts`), so this term cannot describe a different sample from
the data vector it is attached to. Concentration scatter at fixed mass is
folded in by Gauss-Hermite quadrature over $\ln c$ (8 nodes, exact for a
lognormal to the precision anything else here has — deterministic, unlike
a Monte Carlo draw, since a covariance that changes between runs is not
usable in a likelihood).

Each per-cluster profile itself uses the Hayashi & White $\max$
composition, not a sum:

$$
\Delta\Sigma(R\mid M,c) = \max\Big[\Delta\Sigma_{\rm 1h}(R\mid M,c),\;
b(M)\,\bar\rho_m\,\Delta\Sigma_{hh}(R)\Big],
$$

so mass scatter propagates to *both* the one-halo amplitude at small $R$
(carrying the extra concentration scatter) and the large-scale bias $b(M)$
— scatter in the mass-richness relation causes variance on all scales, not
only where the one-halo term lives.

```{note}
This term scales as $1/N_{\rm cl}$ — the one contribution that does
**not** improve when a survey gets deeper at fixed sample size; only more
clusters help. That is the opposite scaling from shape noise per unit
area, which is why the two are tracked separately rather than folded
together.
```

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"covariance-halo-to-halo\"]"
:language: python
```

```
DeltaSigmaHaloToHaloCovariance(z_eff=0.28, sigma_lnc=0.16, n_c=8)

richness bin 0, redshift bin 0:
<DeltaSigma>  = [5.86470477e+13 3.98339927e+13 2.12403318e+13 8.95993786e+12
 3.13747865e+12 9.65277899e+11]
sigma_intr    = [7.74554573e+11 5.58427676e+11 3.33269945e+11 1.58058081e+11
 6.06036179e+10 1.96243301e+10]
fractional    = [0.01320705 0.01401887 0.01569043 0.01764053 0.01931603 0.02033024]

mean fractional sigma_intr by richness bin (rises with richness):
  bin 0: 0.0167   N_cl = 345.7
  bin 1: 0.0213   N_cl = 228.1
  bin 2: 0.0295   N_cl = 97.1
  bin 3: 0.0268   N_cl = 129.9
```

See also: {doc}`api/index` for the full `clenspy.covariance` reference,
{doc}`notation` for the symbol table, {doc}`covariance` for the five-term
Gaussian-field covariance this term is added to.
