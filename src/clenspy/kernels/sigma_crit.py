r"""The lensing efficiency: :math:`\Sigma_{\rm crit}` for one lens-source pair.

:math:`\Sigma_{\rm crit}` is not a property of the universe and not a
property of the halo -- it is a property of the *geometry* of one lens and
one source. That is why it lives in `clenspy.kernels` and receives a
cosmology rather than being a member of `clenspy.cosmology`.

NOTE: units are h-free absolute -- redshifts dimensionless, distances in
Mpc, and :math:`\Sigma_{\rm crit}` in Msun/Mpc^2.

NOTE: **thin-lens approximation.** The deflector is treated as a single
sheet at :math:`z_l`; the formula has no notion of the halo's line-of-sight
extent. Valid whenever the halo is small compared with
:math:`D_A(z_l, z_s)`, which for clusters it always is.

NOTE: this returns the **physical** :math:`\Sigma_{\rm crit}`, from
angular diameter distances. `clenspy.kernels.sigma_crit_comoving` returns
the **comoving** one, and the two differ by exactly :math:`(1+z_l)^2`. The
comoving form is the one to use with `clenspy`'s comoving
:math:`\Delta\Sigma`, since :math:`\gamma_t` has to be dimensionless;
this one is for a single pair quoted in physical units.

The source-averaged inverse, :math:`\langle\Sigma_{\rm crit}^{-1}\rangle
(z_l)`, is the quantity an observable actually needs and is a *different*
function -- average the inverse, never invert the average. It is
`clenspy.kernels.LensingKernel.mean_inverse_sigma_crit`; see
``docs/refactor-plan.md`` errata E.1.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from ..utils.constants import C_LIGHT, G_NEWTON

__all__ = ["sigma_critical"]


def sigma_critical(
    z_lens: float, z_source: float, cosmology: FlatLambdaCDM
) -> float:
    r"""Critical surface density for a lens at ``z_lens``, source at ``z_source``.

    .. math::
        \Sigma_{\rm crit} = \frac{c^2}{4\pi G}\,
            \frac{D_A(z_s)}{D_A(z_l)\, D_A(z_l, z_s)}

    NOTE: :math:`D_A(z_l, z_s)` is the flat subtraction form

    .. math::
        D_A(z_l, z_s) = D_A(z_s) - \frac{1+z_l}{1+z_s}\, D_A(z_l),

    **not** :math:`D_A(z_s) - D_A(z_l)`. `astropy` supplies it as
    `angular_diameter_distance_z1z2`, which is why that method is called
    here instead of differencing two distances.

    Parameters
    ----------
    z_lens : float
        Lens (cluster) redshift.
    z_source : float
        Source redshift. Must exceed ``z_lens``.
    cosmology : astropy.cosmology.FlatLambdaCDM
        The world model. Only its angular diameter distances are used.

    Returns
    -------
    float
        :math:`\Sigma_{\rm crit}` in Msun/Mpc^2.

    Raises
    ------
    ValueError
        If ``z_source <= z_lens``. A source in front of the lens is not
        lensed by it; returning a negative or infinite number here would
        propagate silently, so it is refused. The population average handles
        foreground sources by clamping the *integrand* at zero instead --
        see errata E.1.
    """
    if z_source <= z_lens:
        msg = f"Source redshift ({z_source}) must be greater than"
        msg += f" lens redshift ({z_lens})."
        raise ValueError(msg)

    D_l = cosmology.angular_diameter_distance(z_lens)
    D_s = cosmology.angular_diameter_distance(z_source)
    D_ls = cosmology.angular_diameter_distance_z1z2(z_lens, z_source)

    # Physical constants, from the one place they are defined
    c = C_LIGHT * u.km / u.s
    G = G_NEWTON * u.Mpc / u.Msun * (u.km / u.s) ** 2

    sigma_crit = (c**2 / (4 * np.pi * G)) * (D_s / (D_l * D_ls))

    # the one unit conversion, applied once at the boundary
    return sigma_crit.to(u.Msun / u.Mpc**2).value


if __name__ == "__main__":
    from ..cosmology import fiducial_cosmology

    cosmo = fiducial_cosmology()
    z_l = 0.35
    print(f"lens z_l = {z_l}")
    for z_s in (0.6, 1.0, 1.5, 2.0):
        print(f"  z_s = {z_s:.2f}  Sigma_crit = "
              f"{sigma_critical(z_l, z_s, cosmo):.4e} Msun/Mpc^2")
    # Sigma_crit diverges as z_s -> z_l and flattens for distant sources:
    # both limits are visible in the four numbers above.
