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

## Protocols

The structural contracts the sibling classes conform to. Nothing in the
science modules imports these at runtime, and no class inherits from them --
conformance is by shape, and `tests/test_protocols.py` checks it.

```{eval-rst}
.. automodule:: clenspy.protocols
   :members:
```
