r"""The stacked lensing profile :math:`\Delta\Sigma_{ij}(R)`.

.. math::
    \Delta\Sigma_{ij}(R) = \frac{\int dM\!\int\! dz\;W_{ij}(M,z)\,
        \Delta\Sigma(R\mid M, z)}{\int dM\!\int\! dz\;W_{ij}(M,z)}

**This is not a second model.** It is the same weight
:math:`W_{ij}` that gives :math:`\langle N_{ij}\rangle`, contracted
against a different per-halo quantity. `StackedDeltaSigma` therefore takes
a `~clenspy.observables.ClusterAbundance` and calls its `average`; it owns
no weight of its own, and cannot disagree with the counts about which
haloes are in the bin.

That is the point of the decomposition, and it is checkable: with
:math:`\Delta\Sigma \equiv 1` the stack must return exactly 1, and with
:math:`\Delta\Sigma = M` it must return :math:`\langle M\rangle_{ij}`.
Both are asserted in the tests.

The one-halo term is a mixture over centring (the paper's Eq. for
:math:`\Delta\Sigma^{1h}`),

.. math::
    \Delta\Sigma^{1h}(R) = (1 - f_{\rm mis})\,\Delta\Sigma_{\rm cen}(R)
                         + f_{\rm mis}\,\Delta\Sigma_{\rm mis}(R)

with :math:`f_{\rm mis} = 0.25 \pm 0.08` and
:math:`\tau = 0.17 \pm 0.04` from the DES Y3 calibration of Kelly et al.
(2024). `mixture` applies it; the miscentered profile itself comes from
`clenspy.selection.miscentering`, which reads a packaged table rather than
integrating at runtime.

NOTE: **units.** :math:`R` in Mpc (h-free) and :math:`\Delta\Sigma` in
:math:`M_\odot/{\rm Mpc}^2`, matching `clenspy.halo.NfwProfile` -- **not**
the h-scaled convention of the mass function. The profile callable is
handed :math:`\ln M` in :math:`h^{-1}M_\odot` because that is what the
weight's grid carries, so the callable is responsible for the conversion
and `StackedDeltaSigma` documents that rather than guessing. Passing
``h`` lets `from_profile` do it for you, visibly.

NOTE: :math:`\Omega(z)` cancels in this ratio identically, so a footprint
must **not** be applied here as well -- see
`clenspy.observables.abundance`. The counts carry it; the profile does not.

NOTE: the stack is over the **same** :math:`(M, z)` grid as the counts.
A profile that is expensive per :math:`(M, z)` therefore costs
``n_m * n_z`` evaluations per radius, which is why `from_profile` evaluates
on the grid once and caches, rather than being called inside the
quadrature.
"""

from __future__ import annotations

import numpy as np

__all__ = ["StackedDeltaSigma", "F_MIS_Y3", "TAU_MIS_Y3"]

#: DES Y3 redMaPPer miscentring fraction, Kelly et al. (2024).
F_MIS_Y3 = 0.25

#: DES Y3 redMaPPer miscentring scale, in units of :math:`R_\lambda`.
TAU_MIS_Y3 = 0.17


class StackedDeltaSigma:
    r""":math:`\Delta\Sigma_{ij}(R)` as a contraction of the counts weight.

    NOTE: units -- ``R`` in Mpc, :math:`\Delta\Sigma` in
    :math:`M_\odot/{\rm Mpc}^2`. See the module NOTE for the h boundary.

    Parameters
    ----------
    abundance : clenspy.observables.ClusterAbundance
        Stored verbatim. Supplies the weight and its normalisation, so the
        stack and the counts cannot disagree about the sample.
    profile_grid : array-like, shape (n_m, n_z, n_r)
        :math:`\Delta\Sigma(R\mid M, z)` already evaluated on the
        abundance's :math:`(\ln M, z)` grid and the radii below.
        Pre-evaluated on purpose: see the module NOTE on cost.
    radii : array-like, shape (n_r,)
        Projected radii [Mpc].
    """

    def __init__(self, abundance, profile_grid, radii):
        self.abundance = abundance
        self.radii = np.asarray(radii, dtype=float)
        self.profile_grid = np.asarray(profile_grid, dtype=float)
        expected = (abundance.ln_mass.size, abundance.z.size,
                    self.radii.size)
        if self.profile_grid.shape != expected:
            raise ValueError(
                f"profile_grid must have shape {expected} "
                f"(n_m, n_z, n_r), got {self.profile_grid.shape}"
            )

    @classmethod
    def from_profile(cls, abundance, profile, radii, h=None):
        r"""Evaluate a profile callable on the abundance's grid.

        NOTE: the h division is applied here, once, and is the reason this
        constructor exists rather than leaving the caller to build the grid.
        Getting it backwards scales every mass by :math:`h^2 \approx 2`.

        Parameters
        ----------
        abundance : ClusterAbundance
            The weight to stack against.
        profile : callable
            ``profile(radii, mass, z) -> array of shape (n_r,)``, with
            ``mass`` in **h-free** :math:`M_\odot` and ``radii`` in Mpc.
        radii : array-like
            Projected radii [Mpc].
        h : float, optional
            Used for the one conversion
            :math:`M[M_\odot] = M[h^{-1}M_\odot]/h`, since the abundance's
            grid is h-scaled and `clenspy.halo` profiles are not. Defaults
            to ``abundance.h``.

        Returns
        -------
        StackedDeltaSigma
            With ``profile_grid`` filled on the abundance's grid.
        """
        h = float(abundance.h if h is None else h)
        radii = np.asarray(radii, dtype=float)
        grid = np.empty((abundance.ln_mass.size, abundance.z.size,
                         radii.size))
        for i, lnm in enumerate(abundance.ln_mass):
            # one visible conversion: h^-1 Msun -> Msun
            mass = np.exp(lnm) / h
            for j, z in enumerate(abundance.z):
                grid[i, j] = np.ravel(profile(radii, mass, z))
        return cls(abundance, grid, radii)

    def profile(self):
        r""":math:`\Delta\Sigma_{ij}(R)`, shape
        ``(n_lambda, n_z_bins, n_r)``."""
        return self.abundance.average(self.profile_grid)

    def mixture(self, centred, miscentered, f_mis=F_MIS_Y3):
        r"""The centring mixture, applied to two already-stacked profiles.

        .. math::
            \Delta\Sigma^{1h} = (1-f_{\rm mis})\,\Delta\Sigma_{\rm cen}
                              + f_{\rm mis}\,\Delta\Sigma_{\rm mis}

        NOTE: a **linear** mixture, so it commutes with the stack: mixing
        then stacking equals stacking then mixing. That is asserted in the
        tests, and it is why this is a free function on results rather than
        something buried in the weight -- the caller may do it in whichever
        order is cheaper.

        NOTE: :math:`f_{\rm mis}` is a nuisance parameter with a prior
        (:math:`0.25 \pm 0.08`), not a constant. The default is the DES Y3
        central value; a chain must vary it.
        """
        if not 0.0 <= f_mis <= 1.0:
            raise ValueError(f"f_mis must lie in [0, 1], got {f_mis}")
        centred = np.asarray(centred, dtype=float)
        miscentered = np.asarray(miscentered, dtype=float)
        if centred.shape != miscentered.shape:
            raise ValueError(
                f"the two profiles must have the same shape, got "
                f"{centred.shape} and {miscentered.shape}"
            )
        return (1.0 - f_mis) * centred + f_mis * miscentered

    def __repr__(self):
        return (f"StackedDeltaSigma({self.abundance!r}, "
                f"n_r={self.radii.size})")


if __name__ == "__main__":
    from ..cosmology.fiducial import fiducial_cosmology
    from ..halo.nfw import NfwProfile
    from ..selection import EmgParams, LogNormalMor, SelectionFunction
    from ..survey.survey import omega_des_y1
    from .abundance import ClusterAbundance

    cosmo = fiducial_cosmology()

    # a smooth analytic stand-in for dn/dlnM: exponential in mass, so the
    # demo needs neither CAMB nor a sigma grid
    def mass_function(ln_mass, z):
        lnm, zz = np.broadcast_arrays(np.asarray(ln_mass, float),
                                      np.asarray(z, float))
        m = np.exp(lnm)
        return 1e-5 * (m / 1e14) ** -1.0 * np.exp(-m / 5e14) / (1.0 + zz)

    sel = SelectionFunction(
        np.array([20.0, 30.0, 45.0, 60.0, 200.0]),
        np.array([0.20, 0.35, 0.50, 0.65]),
        LogNormalMor(), EmgParams(-1.5, 3.0, 0.3, 0.12), sigma_z=0.01,
    )
    ln_mass = np.log(np.logspace(13.5, 15.3, 24))
    z = np.linspace(0.16, 0.70, 32)
    abundance = ClusterAbundance(ln_mass, z, mass_function, sel, cosmo,
                                 omega_des_y1)

    radii = np.logspace(-1.0, 1.0, 6)

    def nfw_deltasigma(r, mass, z_cluster):
        """Centred NFW DeltaSigma at fixed concentration, in Msun/Mpc^2."""
        rho_m = (cosmo.critical_density0.to_value("Msun/Mpc^3") * cosmo.Om0)
        return NfwProfile(m200=mass, c200=4.0, rho_ref=rho_m).deltasigma(r)

    stack = StackedDeltaSigma.from_profile(abundance, nfw_deltasigma, radii)
    print(stack, "\n")

    ds = stack.profile()
    print("DeltaSigma_ij(R) [Msun/Mpc^2], lowest redshift bin:")
    print(f"{'lambda bin':>14s}  " + "  ".join(f"{r:11.3f}" for r in radii))
    for i, (a, b) in enumerate(zip(sel.lambda_edges[:-1],
                                   sel.lambda_edges[1:])):
        print(f"{f'[{a:.0f}, {b:.0f})':>14s}  " + "  ".join(
            f"{v:11.4e}" for v in ds[i, 0]))
    print("  <- rises with richness at every radius, because <M> does.")

    # the identities that prove the stack is the counts' own weight
    ones = np.ones_like(stack.profile_grid)
    print(f"\nstacking DeltaSigma = 1 gives "
          f"{np.max(np.abs(abundance.average(ones) - 1.0)):.2e} from 1")
    mass_grid = np.broadcast_to(
        np.exp(ln_mass)[:, None], (ln_mass.size, z.size)
    )
    mass_residual = np.max(
        np.abs(abundance.average(mass_grid) / abundance.mean_mass() - 1.0)
    )
    print(f"stacking DeltaSigma = M reproduces <M>_ij to {mass_residual:.2e}")

    # the mixture is linear, so it commutes with the stack
    mis_grid = 0.4 * stack.profile_grid          # a stand-in miscentered set
    mixed_then_stacked = abundance.average(
        (1 - F_MIS_Y3) * stack.profile_grid + F_MIS_Y3 * mis_grid
    )
    stacked_then_mixed = stack.mixture(ds, abundance.average(mis_grid))
    # relative, not absolute: DeltaSigma is O(1e13), so an absolute
    # residual of 1e-2 would read as a failure when it is fp64 round-off
    relative = np.max(np.abs(mixed_then_stacked / stacked_then_mixed - 1.0))
    print(f"\nmixture commutes with the stack: max relative diff = "
          f"{relative:.2e}")
    print(f"  (f_mis = {F_MIS_Y3}, tau = {TAU_MIS_Y3}, Kelly et al. 2024)")

    print("\nthe h boundary, made explicit:")
    print(f"  grid mass at the top node: {np.exp(ln_mass[-1]):.4e} h^-1 Msun")
    print(f"  handed to the profile as : "
          f"{np.exp(ln_mass[-1]) / abundance.h:.4e} Msun   (h = "
          f"{abundance.h:g})")
    print("  getting that backwards scales every mass by h^2 ~ 2.")
