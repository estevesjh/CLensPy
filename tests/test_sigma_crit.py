r"""``sigma_critical`` input validation.

The positive path (physical vs. comoving Sigma_crit, the flat subtraction
form, etc.) is already exercised in ``tests/test_lensing_kernel.py`` and
``tests/test_lensing_profile.py``. What is missing is the one guard this
module owns: a source that is not behind the lens must raise rather than
silently returning something negative or infinite.
"""

import pytest

from clenspy.cosmology import fiducial_cosmology
from clenspy.kernels import sigma_critical

COSMO = fiducial_cosmology()


def test_equal_redshifts_raise_with_both_values_in_the_message():
    z = 0.4
    with pytest.raises(ValueError, match=f"{z}.*{z}"):
        sigma_critical(z, z, COSMO)


def test_source_in_front_of_the_lens_raises_with_both_values_in_the_message():
    z_lens, z_source = 0.5, 0.3
    with pytest.raises(ValueError, match=f"{z_source}.*{z_lens}"):
        sigma_critical(z_lens, z_source, COSMO)
