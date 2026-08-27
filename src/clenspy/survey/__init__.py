r"""What the dataset is: footprint, source population, bin definitions.

Everything that changes when you change the *survey* while holding the
universe and the nuisance model fixed. This layer may import
`clenspy.cosmology` and `clenspy.utils`; nothing here knows about a halo
profile or a lens-source geometry.

One module, `survey`, and the split inside it is between **code** and
**configuration**:

- :math:`\Omega(z)` is a polynomial transcribed from ``y3_cluster_cpp``, so
  it is code -- one mistyped digit is a silent normalisation error.
- Bin edges, :math:`\sigma_z`, :math:`\sigma_\gamma`, :math:`n_{\rm src}`
  and the :math:`p(z_s)` parameters are analysis choices, so they live in
  ``clenspy/configs/<survey>.json`` and are read by `Survey.from_config`
  and `survey_bins`.

See `clenspy.survey.survey` for the units, the E.2 note on why
:math:`\Omega(z)` is not an attribute of `Survey`, and the transcription
provenance.
"""

from .survey import (
    CONFIG_DIR,
    Survey,
    available_configs,
    deg2,
    load_config,
    omega_des_y1,
    omega_des_y3,
    omega_sdss,
    omega_y3xspt,
    survey_area,
    survey_bins,
)

__all__ = [
    "Survey",
    "survey_area",
    "survey_bins",
    "load_config",
    "available_configs",
    "deg2",
    "omega_des_y1",
    "omega_des_y3",
    "omega_sdss",
    "omega_y3xspt",
    "CONFIG_DIR",
]
