# API Reference

Mechanical reference only — every public class, function, and constant,
one row per name, each linking to its own generated page. For the
physics, prose, and worked examples, see the Theory pages linked from
each section below.

## `clenspy.cosmology`

See {doc}`../cosmology`, {doc}`../power_spectrum`, {doc}`../mass_function`,
{doc}`../halo_bias`, {doc}`../concentration`.

```{eval-rst}
.. currentmodule:: clenspy.cosmology

.. autosummary::
   :toctree: generated

   PkGrid
   BiasModel
   comoving_to_theta
   theta_to_comoving
   fiducial_cosmology
   mean_matter_density
```

```{eval-rst}
.. currentmodule:: clenspy.cosmology.concentration

.. autosummary::
   :toctree: generated

   CHILD18_TABLE1
   CHILD18_TABLE2
   DUFFY08_TABLE1
   DUFFY08_PIVOT_HINV
   Y3_FIXED_CONCENTRATION
   DELTA_COLLAPSE
   child18
   child18_powerlaw
   duffy08
   m_star_hinv
   m_star_from_sigma
   delta_c
   scatter

.. currentmodule:: clenspy.cosmology.growth

.. autosummary::
   :toctree: generated

   growth_factor
   growth_unnormalised

.. currentmodule:: clenspy.cosmology.sigma

.. autosummary::
   :toctree: generated

   LinearPk
   SigmaGrid
   lnr_grid

.. currentmodule:: clenspy.cosmology.halo_mass_function

.. autosummary::
   :toctree: generated

   TinkerMassFunction
   consumed_mask
```

## `clenspy.covariance`

See {doc}`../covariance`, {doc}`../covariance_halo_to_halo`.

```{eval-rst}
.. currentmodule:: clenspy.covariance.counts

.. autosummary::
   :toctree: generated

   CountsCovariance

.. currentmodule:: clenspy.covariance.deltasigma

.. autosummary::
   :toctree: generated

   ALL_TERMS
   DeltaSigmaGaussianCovariance
   j2_bin

.. currentmodule:: clenspy.covariance.halo_to_halo

.. autosummary::
   :toctree: generated

   DeltaSigmaHaloToHaloCovariance
```

## `clenspy.halo`

See {doc}`../density_profiles`, {doc}`../projected_profiles`,
{doc}`../two_halo_term`.

```{eval-rst}
.. currentmodule:: clenspy.halo

.. autosummary::
   :toctree: generated

   NfwProfile
   EinastoProfile
   TwoHaloTerm
```

## `clenspy.kernels`

See {doc}`../lensing_kernel`.

```{eval-rst}
.. currentmodule:: clenspy.kernels.sigma_crit

.. autosummary::
   :toctree: generated

   sigma_critical

.. currentmodule:: clenspy.kernels.lensing_kernel

.. autosummary::
   :toctree: generated

   LensingKernel
   sigma_crit_comoving

.. currentmodule:: clenspy.kernels.photoz

.. autosummary::
   :toctree: generated

   gaussian_cdf
   photoz_counts
   photoz_projection
   photoz_projection_support
   y3_photoz_window

.. currentmodule:: clenspy.kernels.limber

.. autosummary::
   :toctree: generated

   LimberProjector
   limber

.. currentmodule:: clenspy.kernels.bessel

.. autosummary::
   :toctree: generated

   J2_SERIES_CUTOFF
   j2_bin

.. currentmodule:: clenspy.kernels.fftlog_cov

.. autosummary::
   :toctree: generated

   GaussianCovFFTLog
   BinAveragedJ2DoubleBessel
   white_noise_diagonal
```

## `clenspy.lensing`

See {doc}`../lensing_profile`, {doc}`../miscentering`.

```{eval-rst}
.. currentmodule:: clenspy.lensing

.. autosummary::
   :toctree: generated

   LensingProfile

.. currentmodule:: clenspy.lensing.miscentering

.. autosummary::
   :toctree: generated

   MiscenteringProfile
   MiscenteringTableError
```

## `clenspy.observables`

See {doc}`../observables`.

```{eval-rst}
.. currentmodule:: clenspy.observables.number_counts

.. autosummary::
   :toctree: generated

   ClusterCounts

.. currentmodule:: clenspy.observables.deltasigma

.. autosummary::
   :toctree: generated

   StackedDeltaSigma
   F_MIS_Y3
   TAU_MIS_Y3
```

## `clenspy.selection`

See {doc}`../boost_factor`, {doc}`../selection_function`,
{doc}`../selection_bias`, {doc}`../miscentering`.

```{eval-rst}
.. currentmodule:: clenspy.selection.boost

.. autosummary::
   :toctree: generated

   boost_factor_nfw
   load_boost_factor_data
   load_boost_factor_collection

.. currentmodule:: clenspy.selection.miscentering

.. autosummary::
   :toctree: generated

   NfwMiscenteringTable
   load_nfw_miscentering_table
   require_tabulated_profile
   MiscenteringTableError

.. currentmodule:: clenspy.selection.miscentering_kernel

.. autosummary::
   :toctree: generated

   nfw_sigma_hat
   nfw_mean_sigma_hat
   miscentered_sigma
   miscentered_mean_sigma
   miscentered_deltasigma

.. currentmodule:: clenspy.selection.richness_kernel

.. autosummary::
   :toctree: generated

   EmgParams
   emg_cdf
   emg_pdf
   richness_bin_probability
   richness_bin_first_moment
   richness_pdf

.. currentmodule:: clenspy.selection.scaling_relation

.. autosummary::
   :toctree: generated

   LogNormalMor
   HodMor

.. currentmodule:: clenspy.selection.selection_function

.. autosummary::
   :toctree: generated

   SelectionFunction

.. currentmodule:: clenspy.selection.geometry

.. autosummary::
   :toctree: generated

   r_lambda
   theta_lambda
   area_overlap
   sigmoid_theta

.. currentmodule:: clenspy.selection.bsel

.. autosummary::
   :toctree: generated

   SigmoidBias
   SelectionBiasTable
   XiNL
   SelBiasEngine
   PhysicalMassMor
```

## `clenspy.survey`

See {doc}`../survey`.

```{eval-rst}
.. currentmodule:: clenspy.survey.survey

.. autosummary::
   :toctree: generated

   Survey
   survey_area
   survey_bins
   load_config
   available_configs
   deg2
   omega_des_y1
   omega_des_y3
   omega_sdss
   omega_y3xspt
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

.. currentmodule:: clenspy.utils.constants

.. autosummary::
   :toctree: generated

   C_LIGHT
   G_NEWTON

.. currentmodule:: clenspy.utils.special

.. autosummary::
   :toctree: generated

   EULER_GAMMA
   catalan_over_4k
   expint_asymptotic
   expn_fast
   tophat_w
   tophat_dw
```

## Protocols

Structural contracts the sibling classes conform to by shape, not
inheritance — see `tests/test_protocols.py`.

```{eval-rst}
.. currentmodule:: clenspy.protocols

.. autosummary::
   :toctree: generated

   Cosmology
   Profile
   Survey
```
