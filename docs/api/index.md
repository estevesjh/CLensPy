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
.. automodule:: clenspy.lensing.boost
   :members:

.. automodule:: clenspy.lensing.miscentering
   :members:
```

## `clenspy.cosmology`

```{eval-rst}
.. currentmodule:: clenspy.cosmology

.. autosummary::
   :toctree: generated

   PkGrid
   sigma_critical
   comoving_to_theta
   fiducial_cosmology
   mean_matter_density
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
```

## Miscentering tables

The miscentered profiles are interpolated from a packaged grid, never
integrated at runtime. See {doc}`../miscentering_math` section 9.

```{eval-rst}
.. automodule:: clenspy.halo.miscentering_table
   :members:

.. automodule:: clenspy.halo.miscentering_kernel
   :members:
```
