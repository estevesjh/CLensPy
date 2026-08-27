r"""Cluster abundance: the weight :math:`W_{ij}(M, z)` and its contractions.

Everything a binned cluster analysis predicts is one of two contractions of
the *same* weight. That is the organising idea of this module, and the
reason `ClusterAbundance` is a single object rather than one class per
observable.

.. math::
    W_{ij}(M, z^{\rm tr}) = \Omega(z^{\rm tr})\,
        \frac{dV}{d\Omega\,dz^{\rm tr}}\,
        n(M, z^{\rm tr})\,
        \mathcal S_{ij}(M, z^{\rm tr})

Contract it against 1 and you get the counts; contract it against any
per-halo quantity and you get that quantity's stacked average:

.. math::
    \langle N_{ij}\rangle = \int dM\!\int\! dz^{\rm tr}\;W_{ij},
    \qquad
    \langle X\rangle_{ij} = \frac{\int dM\!\int\! dz^{\rm tr}\;W_{ij}\,
        X(M, z^{\rm tr})}{\langle N_{ij}\rangle}

This is the paper's Eq. for :math:`\langle N_{ij}\rangle` after the
five-dimensional integral has been collapsed to two by
`clenspy.selection.SelectionFunction`. The stacked lensing profile
:math:`\Delta\Sigma_{ij}(R)` is the second contraction with
:math:`X = \Delta\Sigma(R\mid M, z)`, and lives in
`clenspy.observables.deltasigma` -- it is the same weight, not a second
model.

NOTE: **units.** :math:`\Omega` in steradians, :math:`dV/(d\Omega\,dz)` in
Mpc^3/sr, so their product is a comoving volume per unit redshift in Mpc^3.
The mass function must therefore be in :math:`{\rm Mpc}^{-3}` per unit
mass, and the integration is over :math:`\ln M` with the Jacobian applied
explicitly. :math:`\langle N_{ij}\rangle` is a dimensionless count.

NOTE: **the h-convention crosses here, and this is the boundary.** The
mass function and the mass--observable relations are h-scaled
(:math:`{\rm Mpc}/h`, :math:`h^{-1}M_\odot`); astropy's volume element is
h-free (Mpc). `ClusterAbundance` takes ``h`` explicitly and applies the
conversion in **one visible place**, `_volume_per_dz`, with the powers
written out. Nothing else in this module touches h.

NOTE: :math:`\Omega(z)` multiplies the **counts** and cancels in the
lensing projection -- it is a property of the abundance, not an ambient
survey property. `average` therefore normalises by
:math:`\langle N_{ij}\rangle`, in which :math:`\Omega` divides out
identically. Applying it to both would count the footprint twice; the
cancellation is asserted in the tests.

NOTE: the :math:`(\ln M, z)` grid is the module's named approximation. Both
integrals are trapezoid rules on grids the caller chooses, because the
integrand is a product of things with different natural scales -- the mass
function falls exponentially, :math:`\mathcal S_{ij}` is a bump in
redshift a few :math:`\sigma_z` wide. `convergence` measures the grid
error by halving, rather than asserting it.
"""

from __future__ import annotations

import numpy as np

from ..cosmology.distances import comoving_volume_element

__all__ = ["ClusterAbundance"]


class ClusterAbundance:
    r""":math:`W_{ij}(M,z)` on a grid, plus the two contractions.

    NOTE: units -- see the module NOTE. ``ln_mass`` is :math:`\ln M` with
    M in :math:`h^{-1}M_\odot`; counts are dimensionless.

    Parameters
    ----------
    ln_mass : array-like, shape (n_m,)
        :math:`\ln M` grid, M in :math:`h^{-1}M_\odot`. Ascending.
    z : array-like, shape (n_z,)
        True-redshift grid. Ascending.
    mass_function : callable
        ``mass_function(ln_mass, z) -> dn/dlnM`` in :math:`h^3
        {\rm Mpc}^{-3}`, broadcasting over the grid. Stored verbatim.
    selection : clenspy.selection.SelectionFunction
        Supplies :math:`\mathcal S_{ij}`. Stored verbatim.
    cosmology : astropy.cosmology.Cosmology
        For the volume element.
    omega : callable
        ``omega(z) -> steradians``, e.g.
        `clenspy.survey.survey.omega_des_y1`.
    h : float, optional
        Hubble parameter, for the single h conversion. Defaults to
        ``cosmology.h``.
    """

    def __init__(self, ln_mass, z, mass_function, selection, cosmology,
                 omega, h=None):
        self.ln_mass = np.asarray(ln_mass, dtype=float)
        self.z = np.asarray(z, dtype=float)
        for name, grid in (("ln_mass", self.ln_mass), ("z", self.z)):
            if grid.ndim != 1 or grid.size < 2:
                raise ValueError(f"{name} must be 1-D with >= 2 points")
            if np.any(np.diff(grid) <= 0.0):
                raise ValueError(f"{name} must be strictly ascending")
        if np.any(self.z <= 0.0):
            raise ValueError("z must be positive (the volume element needs it)")

        self.mass_function = mass_function
        self.selection = selection
        self.cosmology = cosmology
        self.omega = omega
        self.h = float(cosmology.h if h is None else h)

    # -- the pieces, each separately inspectable ------------------------

    def _volume_per_dz(self):
        r""":math:`\Omega(z)\,dV/(d\Omega\,dz)` in
        :math:`(h^{-1}{\rm Mpc})^3`, shape ``(n_z,)``.

        NOTE: **the one h conversion in this module.** astropy returns
        :math:`dV/(d\Omega\,dz)` in Mpc^3/sr; the mass function is per
        :math:`(h^{-1}{\rm Mpc})^3`. Multiplying by :math:`h^3` converts
        Mpc^3 to :math:`(h^{-1}{\rm Mpc})^3` so the two cancel and the
        count is dimensionless. Written here, once, and nowhere else.
        """
        dv_dz_domega = comoving_volume_element(self.z, self.cosmology)
        omega_sr = np.asarray(self.omega(self.z), dtype=float)
        # Mpc^3 -> (Mpc/h)^3 : one visible multiplication
        return omega_sr * dv_dz_domega * self.h**3

    def weight(self):
        r""":math:`W_{ij}(\ln M, z)`, shape
        ``(n_m, n_z, n_lambda, n_z_bins)``.

        NOTE: carries the :math:`d\ln M` Jacobian implicitly by holding
        :math:`dn/d\ln M` rather than :math:`dn/dM` -- so the mass integral
        below is over :math:`\ln M` with no extra factor of M. Applying one
        anyway is the classic way to be wrong by :math:`\ln`-decades.
        """
        lnm = self.ln_mass[:, None]
        z = self.z[None, :]
        # dn/dlnM * Omega * dV/dOmega/dz  ->  (n_m, n_z)
        dndlnm = np.asarray(self.mass_function(lnm, z), dtype=float)
        volume = self._volume_per_dz()[None, :]
        # S_ij(lnM, z) -> (n_m, n_z, n_lambda, n_zbins)
        s_ij = self.selection.S_ij(
            np.broadcast_to(lnm, (self.ln_mass.size, self.z.size)),
            np.broadcast_to(z, (self.ln_mass.size, self.z.size)),
        )
        return (dndlnm * volume)[..., None, None] * s_ij

    def _integrate(self, integrand):
        """Trapezoid over lnM then z, leaving the trailing bin axes."""
        over_mass = np.trapezoid(integrand, x=self.ln_mass, axis=0)
        return np.trapezoid(over_mass, x=self.z, axis=0)

    # -- the two contractions ------------------------------------------

    def counts(self):
        r""":math:`\langle N_{ij}\rangle`, shape ``(n_lambda, n_z_bins)``."""
        return self._integrate(self.weight())

    def average(self, values):
        r""":math:`\langle X\rangle_{ij}`, the weight-normalised stack.

        .. math::
            \langle X\rangle_{ij} = \frac{\int W_{ij}\,X}{\int W_{ij}}

        NOTE: :math:`\Omega(z)` cancels identically in this ratio, which is
        why it may appear in the counts and must **not** be applied again to
        a lensing profile. See the module NOTE.

        Parameters
        ----------
        values : array-like
            :math:`X(\ln M, z)` with shape ``(n_m, n_z)``, or
            ``(n_m, n_z, n_extra)`` to average several quantities (e.g. a
            radial profile) at once.

        Returns
        -------
        np.ndarray
            Shape ``(n_lambda, n_z_bins)``, or
            ``(n_lambda, n_z_bins, n_extra)``.
        """
        values = np.asarray(values, dtype=float)
        weight = self.weight()
        expected = (self.ln_mass.size, self.z.size)
        if values.shape[:2] != expected:
            raise ValueError(
                f"values must start with shape {expected}, got {values.shape}"
            )
        numerator = self._integrate(
            weight[..., None] * values[:, :, None, None, ...]
            if values.ndim == 3 else weight * values[..., None, None]
        )
        denominator = self._integrate(weight)
        if values.ndim == 3:
            return numerator / denominator[..., None]
        return numerator / denominator

    def mean_mass(self):
        r""":math:`\langle M\rangle_{ij}` in :math:`h^{-1}M_\odot`.

        The single most useful contraction after the counts -- it is what a
        mass-calibration analysis compares against, and it has an obvious
        sanity property (it must rise with richness bin) that a wrong
        weight breaks immediately.
        """
        mass = np.exp(self.ln_mass)[:, None] * np.ones_like(self.z)[None, :]
        return self.average(mass)

    def mean_redshift(self):
        r""":math:`\langle z\rangle_{ij}`, dimensionless."""
        z = np.ones_like(self.ln_mass)[:, None] * self.z[None, :]
        return self.average(z)

    def convergence(self):
        r"""Relative change in the counts when both grids are halved.

        NOTE: the grid is this module's named approximation, so it is
        measured rather than asserted. Returns the max relative difference
        between the full-grid counts and the every-other-point counts, which
        for a trapezoid rule is a 4x bound on the true error.
        """
        coarse = ClusterAbundance(
            self.ln_mass[::2], self.z[::2], self.mass_function,
            self.selection, self.cosmology, self.omega, h=self.h,
        )
        fine, crude = self.counts(), coarse.counts()
        nonzero = fine > 0.0
        return float(np.max(np.abs(crude[nonzero] / fine[nonzero] - 1.0)))

    def __repr__(self):
        return (f"ClusterAbundance({self.ln_mass.size} lnM x "
                f"{self.z.size} z -> {self.selection.n_lambda_bins} x "
                f"{self.selection.n_z_bins} bins, h={self.h:g})")


if __name__ == "__main__":
    from ..cosmology.fiducial import fiducial_cosmology
    from ..cosmology.mass_function import TinkerMassFunction
    from ..cosmology.sigma import LinearPk, SigmaGrid
    from ..selection import EmgParams, LogNormalMor, SelectionFunction
    from ..survey.survey import deg2, omega_des_y1

    cosmo = fiducial_cosmology()

    # a scale-free P(k) so the demo needs no CAMB call
    k = np.logspace(-4.0, 3.0, 600)
    grid = SigmaGrid(LinearPk(k, 2.0e4 * k**-1.5 * np.exp(-((k / 60.0) ** 2))))
    hmf = TinkerMassFunction(grid)

    def mass_function(ln_mass, z):
        """dn/dlnM at (lnM, z), via the Tinker grid walk."""
        lnm = np.broadcast_arrays(ln_mass, z)[0]
        r = np.cbrt(np.exp(lnm) / ((4 * np.pi / 3) * 2.775e11))
        ln_s2 = np.array([np.log(grid.sigma2(ri)) for ri in np.ravel(r)])
        dln = np.array([grid.dlnsigma2_dlnr(ri) for ri in np.ravel(r)])
        z_flat = np.ravel(np.broadcast_arrays(ln_mass, z)[1])
        out = np.empty_like(ln_s2)
        for i in range(ln_s2.size):
            out[i] = hmf.outputs(np.log(np.ravel(r)[i]), ln_s2[i], dln[i],
                                 z_flat[i])["dndlnmh"]
        return out.reshape(np.broadcast(ln_mass, z).shape)

    sel = SelectionFunction(
        np.array([20.0, 30.0, 45.0, 60.0, 200.0]),
        np.array([0.20, 0.35, 0.50, 0.65]),
        LogNormalMor(), EmgParams(-1.5, 3.0, 0.3, 0.12), sigma_z=0.01,
    )

    ln_mass = np.log(np.logspace(13.0, 15.5, 40))
    z = np.linspace(0.15, 0.72, 60)
    abundance = ClusterAbundance(ln_mass, z, mass_function, sel, cosmo,
                                 omega_des_y1)
    print(abundance)
    print(f"Omega(0.3) = {float(deg2(omega_des_y1(0.3)).item()):.1f} deg^2\n")

    n_ij = abundance.counts()
    print("N_ij (richness bin x redshift bin):")
    print(f"{'lambda bin':>14s}  " + "  ".join(
        f"{f'z {a:.2f}-{b:.2f}':>13s}"
        for a, b in zip(sel.z_edges[:-1], sel.z_edges[1:])))
    for i, (a, b) in enumerate(zip(sel.lambda_edges[:-1],
                                   sel.lambda_edges[1:])):
        print(f"{f'[{a:.0f}, {b:.0f})':>14s}  " + "  ".join(
            f"{v:13.2f}" for v in n_ij[i]))
    print(f"{'total':>14s}  " + "  ".join(f"{v:13.2f}"
                                          for v in n_ij.sum(axis=0)))

    print(f"\ngrid convergence (halving both axes): "
          f"{abundance.convergence():.2e}")

    print("\n<M>_ij [h^-1 Msun] -- must rise with richness:")
    m_ij = abundance.mean_mass()
    for i, (a, b) in enumerate(zip(sel.lambda_edges[:-1],
                                   sel.lambda_edges[1:])):
        print(f"{f'[{a:.0f}, {b:.0f})':>14s}  " + "  ".join(
            f"{v:13.4e}" for v in m_ij[i]))

    print("\n<z>_ij -- must sit inside each redshift bin:")
    z_ij = abundance.mean_redshift()
    for i, (a, b) in enumerate(zip(sel.lambda_edges[:-1],
                                   sel.lambda_edges[1:])):
        print(f"{f'[{a:.0f}, {b:.0f})':>14s}  " + "  ".join(
            f"{v:13.4f}" for v in z_ij[i]))

    # the cancellation that says Omega belongs to the counts alone
    print("\nOmega(z) cancels in `average` but not in `counts`:")
    doubled = ClusterAbundance(ln_mass, z, mass_function, sel, cosmo,
                               lambda zz: 2.0 * omega_des_y1(zz))
    print(f"  counts ratio  (2 Omega / Omega) = "
          f"{np.max(doubled.counts() / n_ij):.6f}   <- 2, as it must be")
    print(f"  <M> ratio                       = "
          f"{np.max(np.abs(doubled.mean_mass() / m_ij - 1.0)):.2e}"
          "   <- 0, identically")
    print("  applying Omega to a lensing profile as well would count the")
    print("  footprint twice.")
