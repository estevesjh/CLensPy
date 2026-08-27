r"""Structural contracts for the package.

NOTE: nothing in the science modules imports this file at runtime. It exists
for static checking, for documentation, and for ``tests/test_protocols.py``.
Implementing classes do **not** inherit from these Protocols -- a class
conforms by having the right methods. Record conformance with one line in the
class docstring::

    \"\"\"implements the Profile protocol\"\"\"

``@runtime_checkable`` verifies that the methods exist, not their signatures.
That is the intended level: it catches a sibling class that forgot a method,
without pretending to check the physics.

NOTE: units follow the package convention -- h-free absolute, with mass in
Msun, lengths in Mpc, densities in Msun/Mpc^3, surface densities in
Msun/Mpc^2 and wavenumbers in 1/Mpc.

The contracts here are the ones CLensPy actually has. The `cosmology-code`
skill lists six; `Selection` and `Kernel` are still absent because the
package has no such layer yet, and writing a contract before an
implementation would be inventing rather than transcribing. See
``docs/refactor-plan.md`` for what they will need to be.

NOTE: `Survey` is the **source population** -- :math:`p(z_s)` and the shape
noise. It is *not*
:math:`\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)`, which is built from a
source population but is lens-source geometry and belongs to
`clenspy.kernels`; nor is it :math:`\Omega(z)`, which is a footprint, is
consumed only by the counts, and cancels in the shear projection. All three
are in `clenspy.survey` or `clenspy.kernels` as separate objects on purpose
(errata E.1, E.2).
"""

from typing import Protocol, runtime_checkable

__all__ = ["Cosmology", "Profile", "Survey"]


@runtime_checkable
class Cosmology(Protocol):
    r"""The world model: parameters and the background quantities from them.

    Satisfied by ``astropy.cosmology.FlatLambdaCDM``, which is what
    `clenspy.cosmology.fiducial_cosmology` returns. The contract names only
    what CLensPy actually calls, so a hand-rolled cosmology can stand in.

    NOTE: knows nothing about the survey, the tracer, or the observable.
    """

    Om0: float

    def comoving_distance(self, z):
        """Line-of-sight comoving distance [Mpc]."""
        ...

    def angular_diameter_distance(self, z):
        """Angular diameter distance [Mpc]."""
        ...

    def critical_density(self, z):
        """Critical density at z, as an astropy Quantity."""
        ...


@runtime_checkable
class Profile(Protocol):
    r"""A spherically symmetric halo profile and its projections.

    The halo is fixed in the constructor; the radius is an argument to the
    method. Every method is vectorised over its radius argument.

    Satisfied by `clenspy.halo.NfwProfile` and
    `clenspy.halo.EinastoProfile`. The four projections are related by

    .. math::
        \bar\Sigma(<R) = \frac{2}{R^2}\int_0^R \Sigma(R')R'\,dR',
        \qquad
        \Delta\Sigma(R) = \bar\Sigma(<R) - \Sigma(R),

    but each is declared separately because each has a closed form worth
    evaluating directly -- forming one from the others cancels badly at small
    :math:`R/r_s` (see `NfwProfile._gbarNfw`).

    NOTE: `convergence` and `shear` are deliberately **not** in this
    contract. They need :math:`\Sigma_{\rm crit}`, which is a property of the
    lens-source geometry rather than of the halo, so they belong to the
    lensing layer. `EinastoProfile` carries them with a
    ``sigma_crit=1.0`` default; that is a historical anomaly, not the
    contract.
    """

    def density(self, r):
        r"""3D density :math:`\rho(r)` [Msun/Mpc^3]."""
        ...

    def sigma(self, R):
        r"""Projected surface density :math:`\Sigma(R)` [Msun/Mpc^2]."""
        ...

    def mean_sigma(self, R):
        r"""Mean interior surface density :math:`\bar\Sigma(<R)` [Msun/Mpc^2]."""
        ...

    def deltasigma(self, R):
        r"""Excess surface density :math:`\Delta\Sigma(R)` [Msun/Mpc^2]."""
        ...

    def fourier(self, k):
        r"""Fourier transform of :math:`\rho`, :math:`u(k)` [dimensionless]."""
        ...


@runtime_checkable
class Survey(Protocol):
    r"""A shear catalogue: its redshift distribution and its noise.

    Satisfied by `clenspy.survey.Survey`. The contract is the
    four things a lensing weight or a covariance actually asks for, and
    nothing else.

    NOTE: units -- :math:`p(z_s)` is a density in redshift and integrates
    to 1 over :math:`[z_s^{\min}, z_s^{\max}]`; ``sigma_gamma`` is
    dimensionless; ``n_src_arcmin`` is a **sky surface density in
    arcmin^-2**, the one non-Mpc unit in the package, which is why the unit
    is in the name.

    NOTE: deliberately excluded -- :math:`\Omega(z)` (a footprint, in the
    counts only) and :math:`\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)`
    (lens-source geometry, `clenspy.kernels`). A `Survey` that carried
    either would let a counts consumer and a shear consumer pick up each
    other's factors.
    """

    sigma_gamma: float
    n_src_arcmin: float
    zs_min: float
    zs_max: float

    def pz_src(self, z):
        r"""Normalised source redshift density :math:`p(z_s)` [1/z]."""
        ...


if __name__ == "__main__":
    # Sanity: the Protocols are runtime_checkable and reject an empty class.
    class NotAProfile:
        pass

    print("empty class satisfies Profile?", isinstance(NotAProfile(), Profile))

    from clenspy.cosmology import fiducial_cosmology
    from clenspy.halo import EinastoProfile, NfwProfile

    print("FlatLambdaCDM satisfies Cosmology?",
          isinstance(fiducial_cosmology(), Cosmology))
    print("NfwProfile satisfies Profile?  ",
          isinstance(NfwProfile(m200=1e14), Profile))
    print("EinastoProfile satisfies Profile?",
          isinstance(EinastoProfile(alpha=0.2, rho_0=1e15, r_s=0.3), Profile))

    # NOTE: the concrete class and the protocol share the name Survey, so
    # one must be aliased -- importing both as `Survey` makes the check
    # compare the class against itself and pass vacuously.
    from clenspy.survey import Survey as SurveyImpl

    print("Survey impl satisfies Survey protocol?",
          isinstance(SurveyImpl.from_config("des_y1"), Survey))
