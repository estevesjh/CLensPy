# API Reference

## `clenspy.halo`

```{eval-rst}
.. currentmodule:: clenspy.halo

.. autosummary::
   :toctree: generated

   NfwProfile
   EinastoProfile
   BiasModel
   TwoHaloTerm
```

## `clenspy.lensing`

```{eval-rst}
.. currentmodule:: clenspy.lensing

.. autosummary::
   :toctree: generated

   LensingProfile
```

```{eval-rst}
.. automodule:: clenspy.lensing.miscentering
   :members:
```

## `clenspy.selection`

Systematics between the halo and the observable. This layer sits above
`clenspy.halo` and below `clenspy.lensing`.

```{eval-rst}
.. automodule:: clenspy.selection.boost
   :members:
```

### Miscentering tables

The miscentered profiles are interpolated from a packaged grid, never
integrated at runtime. See {doc}`../miscentering_math` section 9.

```{eval-rst}
.. automodule:: clenspy.selection.miscentering
   :members:

.. automodule:: clenspy.selection.miscentering_kernel
   :members:
```

### The richness selection

The closed-form chain that collapses the five-dimensional forward-model
integral to two dimensions: the $z^{\rm ob}$ integral is a Gaussian CDF
difference, the $\lambda^{\rm ob}$ integral is an EMG CDF difference, and
only $\lambda^{\rm tr}$ needs quadrature.

The EMG CDF is evaluated through `erfcx`, not through the form the
derivation produces — that one is a product of a factor that overflows and
a factor that underflows, giving `inf * 0 = nan` for $\tau\sigma \gtrsim
40$ where the true value is an ordinary number in $[0,1]$.

```{eval-rst}
.. automodule:: clenspy.selection.richness_kernel
   :members:

.. automodule:: clenspy.selection.scaling_relation
   :members:

.. automodule:: clenspy.selection.selection_function
   :members:
```

## `clenspy.cosmology`

```{eval-rst}
.. currentmodule:: clenspy.cosmology

.. autosummary::
   :toctree: generated

   PkGrid
   comoving_to_theta
   theta_to_comoving
   fiducial_cosmology
   mean_matter_density
```

### Concentration--mass relations

$c(M,z)$ lives here rather than in `clenspy.halo` because it is a
structure-formation result calibrated on N-body simulations at a fixed
cosmology, exactly like the mass function and the halo bias.

Two warnings, both load-bearing: these relations are calibrated in
$h^{-1}M_\odot$ (the one place the package's h-free convention breaks, so
every mass argument names its unit), and `child18` is an $M_{200c}$
relation while `NfwProfile` and the Tinker mass function use $M_{200m}$.

```{eval-rst}
.. automodule:: clenspy.cosmology.concentration
   :members:
```

### Growth, variance, and the mass function

$\sigma(M)$ is computed **once**, in `SigmaGrid`, because the Tinker (2008)
mass function and the Tinker (2010) bias are two fits to the *same* peak
height $\nu = \delta_c/\sigma(M)$ — computing it twice from one $P(k)$ is
how they silently drift apart.

Ported from `y3_cluster_cpp`'s in-repo replacement for CosmoSIS's
`MfTinker` (`mf_tinker_cpp/python/tinker_core.py`), whose Gauss–Legendre
panel evaluator agrees with arbitrary-precision mpmath to 4.4e-16.

Three conventions carried across, all of which bite:

- the integration limits are $k \in [10^{-4},\,20/R]$ — the **upper limit
  depends on $R$**, and it is algorithm-defining, not a convergence cutoff;
- **FFTLog cannot express an $R$-dependent limit**, so the fast path
  computes the untruncated quantity and must be validated against
  `truncate=False`;
- $d\sigma^2/d\ln R$ is taken under the integral sign, and the moving
  boundary contributes a Leibniz term that a finite difference of the
  truncated $\sigma^2$ is the only honest way to verify.

```{eval-rst}
.. automodule:: clenspy.cosmology.growth
   :members:

.. automodule:: clenspy.cosmology.sigma
   :members:

.. automodule:: clenspy.cosmology.mass_function
   :members:
```

## `clenspy.survey`

What the dataset is, as distinct from what the universe is. Three separate
concerns on purpose: $\Omega(z)$ appears in $\langle N_{ij}\rangle$ and
**cancels** in the shear projection, so it must never be applied to both as
an ambient survey property.

$\Omega(z)$ is code (a polynomial transcribed from `y3_cluster_cpp`); bin
edges, $\sigma_z$, $\sigma_\gamma$, $n_{\rm src}$ and the $p(z_s)$
parameters are analysis choices and live in `clenspy/configs/<survey>.json`.

```{eval-rst}
.. automodule:: clenspy.survey.survey
   :members:
```

## `clenspy.kernels`

Line-of-sight windows and the geometry that weights them. $\Sigma_{\rm crit}$
is here rather than in `clenspy.cosmology` because it depends on the
cosmology *and* on two redshifts: it is lens--source geometry, not a
property of the universe.

```{eval-rst}
.. automodule:: clenspy.kernels.sigma_crit
   :members:

.. automodule:: clenspy.kernels.lensing_kernel
   :members:

.. automodule:: clenspy.kernels.photoz
   :members:

.. automodule:: clenspy.kernels.limber
   :members:
```

### The Bessel kernel and the FFTLog engine

$\hat J_2$ has **one** copy in the package, here, because two consumers
need it and `kernels` is the lowest layer both may import: the direct
quadrature in `clenspy.covariance.deltasigma` and the Mellin kernel in
`fftlog_cov`. Two implementations of a kernel with a delicate cancellation
branch is how they drift apart — and they had.

`GaussianCovFFTLog` evaluates the bin-averaged **double**-Bessel covariance
integral as one FFTLog per diagonal offset. That is possible because for
**geometric** bins the pair ratio $\alpha_d = \rho^d$ depends only on the
offset, so the product kernel is a function of $u = \ell\theta$ alone —
the reason the geometric check in its constructor is a precondition, not a
convenience. The 16 Mellin coefficients are summed *before* the inverse
FFT, so the $K_d \sim u^4$ cancellation happens in analytic continuation
rather than in floating point.

Measured against the direct quadrature on matched geometry: comparable on
the diagonal, and **~560× more accurate off-diagonal at equal cost** (5.3e-6
with 4096 nodes against 3.0e-3 with 8192; the quadrature needs 262144 nodes
to match). Derivation in {doc}`../covariance_fftlog_math`.

```{eval-rst}
.. automodule:: clenspy.kernels.bessel
   :members:

.. automodule:: clenspy.kernels.fftlog_cov
   :members:
```

## `clenspy.utils`

```{eval-rst}
.. currentmodule:: clenspy.utils

.. autosummary::
   :toctree: generated

   LogGridInterpolator
   default_rvals_z
   time_method
   scalar_array_output
   compute_sigma_grid
   compute_sigma_leggauss
   compute_sigma_trapz_vectorized
   compute_sigma_quadvec
   sigma_to_deltasigma_cumtrapz
   pk_to_xi_fftlog
   RichnessBin
   BinCollection
```

```{eval-rst}
.. automodule:: clenspy.utils.constants
   :members:

.. automodule:: clenspy.utils.special
   :members:
```

## `clenspy.observables`

The binned observables, and the one idea that organises them: everything a
cluster analysis predicts is a contraction of the **same** weight

$$W_{ij}(M,z) = \Omega(z)\,\frac{dV}{d\Omega\,dz}\,n(M,z)\,\mathcal S_{ij}(M,z)$$

against either 1 (the counts) or a per-halo quantity (its stacked
average). $\Delta\Sigma_{ij}$ is therefore not a second model, and
`StackedDeltaSigma` owns no weight of its own — it cannot disagree with
the counts about which haloes are in the bin.

$\Omega(z)$ cancels identically in any average and must **not** be applied
to a lensing profile as well.

```{eval-rst}
.. automodule:: clenspy.observables.abundance
   :members:

.. automodule:: clenspy.observables.deltasigma
   :members:
```

## `clenspy.covariance`

The `Estimator` layer. In both blocks the physical components are stored
**separately** and summed at the end, with switches to isolate each one —
the scientific argument is almost always about which term dominates where.

**Counts**: Poisson plus sample variance. The sample-variance term is
**rank one** within each redshift slice, because every cluster in the slice
sees the same window mode, and exactly zero between slices. Dropping it
understates the error by 4–10×.

**$\Delta\Sigma$**: the Gaussian-field expression of Wu et al. (2019),
whose bracket expands into five terms — `lss_lss`, `lss_shape`,
`shot_lss`, `shot_shape`, `cross`. Grouping them into three would require
choosing where the mixed terms go, so all five are kept and `cov` takes a
`terms` selector.

Two things worth knowing before using it:

- it is valid for a **thin** halo-redshift slice only, since
  $\theta = r_p/\chi_h$ and $\ell = k\chi_h$ are evaluated at a single
  $\chi_h$;
- there is **no FFTLog**, deliberately. The integral is a bilinear form,
  $\hat J_2(kr_p)\hat J_2(kr_p')$ under one $k$ integral, not a Hankel
  transform of a single function — so it does not factorise into anything
  FFTLog could accelerate. As a matrix product it costs
  $O(n_k n_r^2)$ and is already negligible.

Survey area appears twice meaning two different things: $\Omega(z)$
normalises the counts, while $f_{\rm sky}$ sets the number of independent
modes. Conflating them is a factor of $4\pi$.

```{eval-rst}
.. automodule:: clenspy.covariance.counts
   :members:

.. automodule:: clenspy.covariance.deltasigma
   :members:
```

### The sixth term: intrinsic profile variance

Not a Gaussian-field contribution. Each cluster in a stack carries its own
$\Delta\Sigma$, so the stack inherits the **population** covariance of
those profiles over the bin's selection-weighted mass distribution,
convolved with lognormal concentration scatter. It scales as
$1/N_{\rm cl}$ — the term that does *not* improve with survey depth, only
with more clusters, which is the opposite scaling to shape noise and why
the two are tracked separately.

Per-cluster profiles use the Hayashi & White **max** composition, so mass
scatter propagates both to the one-halo amplitude and to $b(M)$ — variance
on all scales, not only where the one-halo term lives.

```{eval-rst}
.. automodule:: clenspy.covariance.intrinsic
   :members:
```

## Protocols

The structural contracts the sibling classes conform to. Nothing in the
science modules imports these at runtime, and no class inherits from them --
conformance is by shape, and `tests/test_protocols.py` checks it.

```{eval-rst}
.. automodule:: clenspy.protocols
   :members:
```
