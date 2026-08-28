# Covariance

Both binned observables from {doc}`observables` — the counts and the
stacked profile — carry an uncertainty, and in both cases the interesting
physics is in the *off-diagonal* structure, not just the diagonal error
bar. `clenspy` keeps every contribution separate and sums at the end,
because the argument an analysis makes is almost always about which term
dominates where.

## Counts covariance: Poisson plus a coherent window mode

Two clusters in the same redshift slice both feel the same long-wavelength
density mode, so their counts are correlated even before any measurement
noise enters:

$$
{\rm Cov}[N_{ij},N_{i'j'}] = \underbrace{\delta_{ii'}\delta_{jj'}N_{ij}}
_{\rm Poisson} + \underbrace{\delta_{jj'}\,\bar b_{ij}\bar b_{i'j}\,
N_{ij}N_{i'j}\,\sigma_W^2(z_j)}_{\rm sample\ variance},
$$

with $\sigma_W(z)=\sigma_R(R_{\rm eff})\,D(z)$ — the **linear** growth
factor times the $z=0$ variance ({doc}`mass_function`'s `SigmaGrid`,
{doc}`cosmology`'s `growth_factor`), since the response of the counts to a
long-wavelength mode is a linear-theory statement. The sample-variance
term is a rank-one outer product within each redshift block: every cluster
in the same redshift slice sits in the same realization of that mode, so
richness bins at fixed $z$ are **fully correlated**, while different
redshift slices are independent by construction (an approximation, valid
when the bins are wider than the window mode's correlation length).

```{note}
Treating the counts covariance as diagonal — a common shortcut — discards
exactly the correlation that limits how much a richness-binned abundance
can say about $\sigma_8$, and makes the errors look smaller than they
are. Poisson error falls as $1/\sqrt{N}$ with richness; sample variance
does not fall at all, since it is a coherent mode, not counting noise —
visible directly in the example below.
```

## Gaussian-field DeltaSigma covariance: five terms from one bracket

$\Delta\Sigma_{ij}(R)$'s covariance has two independent contributions,
summed at the end — this section's Gaussian-field term, treating the halo
and matter fields as Gaussian random fields, and {doc}`covariance_halo_to_halo`'s
**intrinsic** term, the population scatter of per-cluster profiles that a
Gaussian field cannot capture. Wu et al. (2019)'s Gaussian piece expands
into five physically distinct terms, kept separate rather than grouped
into "cosmic shear" / "shape noise" / "cross" (a grouping that would have
to choose where the mixed terms go):

$$
{\rm Cov}^{\rm Gauss} \propto
\big(C^{hh}+N_h\big)\big(C^{\Sigma\Sigma}+N_\Sigma\big)+\big(C^{h\Sigma}
\big)^2
= \underbrace{C^{hh}C^{\Sigma\Sigma}}_{\rm lss\_lss}
+\underbrace{C^{hh}N_\Sigma}_{\rm lss\_shape}
+\underbrace{N_hC^{\Sigma\Sigma}}_{\rm shot\_lss}
+\underbrace{N_hN_\Sigma}_{\rm shot\_shape}
+\underbrace{\big(C^{h\Sigma}\big)^2}_{\rm cross},
$$

with $N_h=1/n_h$ the halo shot noise and $N_\Sigma=\langle\Sigma_{\rm
crit}\rangle^2\sigma_\gamma^2/n_s$ the shape noise ({doc}`survey`,
{doc}`lensing_kernel`). `DeltaSigmaGaussianCovariance.components` returns
all five separately, and `cov` sums any subset via a `terms` selector.

```{figure} _static/img/covariance_terms.png
:alt: Fractional contribution of the five Gaussian covariance terms vs projected radius
:width: 75%
:align: center

Shot noise x shape noise dominates at small $r_p$ (few source galaxies
per annulus); the correlated large-scale-structure term (`lss_lss`) and
the cross term take over at large $r_p$. This toy's $C^{h\Sigma}=
\sqrt{C^{hh}C^{\Sigma\Sigma}}$ (exact linear bias) makes `cross` and
`lss_lss` numerically identical by construction — the dashed `cross`
curve traces the solid `lss_lss` one exactly.
```

```{figure} _static/img/covariance_matrix.png
:alt: Full Gaussian-field covariance and correlation matrix for one halo-redshift slice
:width: 85%
:align: center

The same term, but as the full $r_p\times r_p$ matrix rather than a
per-term fraction — McClintock et al. (2019) Fig. 6's own presentation
of their SAC matrix. Left: actual values (log scale), diagonal-dominated
since `shot_shape` is exactly diagonal and dominates at small $r_p$;
right: the correlation matrix, falling off the diagonal as `lss_lss`
and `cross` (broad in $k$, hence in $r_p$) hand off to the sharply
diagonal noise terms.
```

```{note}
The `shot_shape` term — dominant at small $r_p$ — is evaluated in
**closed form**, not by quadrature: its bracket $N_hN_\Sigma$ carries no
$k$-dependence, so the Hankel closure $\int_0^\infty
J_2(ka)J_2(kb)\,k\,dk=\delta(a-b)/a$ applies exactly once binned into
disjoint annuli. The other four terms need a genuine quadrature over $k$,
which is **truncation-limited** ($\epsilon\sim2.5/k_{\max}$), not
node-limited — and an FFTLog engine
(`clenspy.kernels.fftlog_cov.GaussianCovFFTLog`) is ~560x more accurate
than the quadrature at equal cost on the off-diagonal, for geometric
radial bins. See {doc}`covariance_fftlog_math` for the full derivation of
both routes.
```

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"covariance\"]"
:end-before: "%% [markdown]"
:language: python
```

```
fractional error by component (Poisson falls with N; sample
variance does not, since it is a coherent mode shared at fixed z):
Poisson       = [0.02   0.018  0.0192 0.0333 0.0295 0.0316 0.0577 0.0513 0.055  0.0953
 0.0845 0.0913]
sample_var    = [0.0951 0.0922 0.0899 0.1177 0.1132 0.1094 0.1448 0.1383 0.1368 0.1946
 0.1886 0.1876]
```

See also: {doc}`api/index` for the full `clenspy.covariance` reference,
{doc}`notation` for the symbol table, {doc}`covariance_halo_to_halo` for
the sixth term this Gaussian-field covariance does not contain.
