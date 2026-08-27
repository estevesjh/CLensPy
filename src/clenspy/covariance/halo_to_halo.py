r"""Halo-to-halo covariance of a stacked profile.

The term the Gaussian covariance does not contain. Wu et al. (2019)
`eq:cov_DS` treats the halo field and the matter field as Gaussian and
gives the variance of the *mean* profile from sample and shape noise. But
each cluster in a stack carries its own :math:`\Delta\Sigma`, and the
stack of :math:`N_{\rm cl}` of them inherits the population covariance of
those profiles (McClintock et al. 2019; Gruen et al. 2015):

.. math::
    C^{\rm intr}_{ij} = \frac{1}{N_{\rm cl}}\left[
        \left\langle \Delta\Sigma(R_i)\,\Delta\Sigma(R_j)
        \right\rangle_{\rm pop}
        - \left\langle \Delta\Sigma(R_i) \right\rangle_{\rm pop}
          \left\langle \Delta\Sigma(R_j) \right\rangle_{\rm pop}\right]

The population is the bin's own selection-weighted mass distribution --
the :math:`\mathcal S_{ij}`-weighted :math:`P(M)`, which is the analytic
counterpart of McClintock's Monte-Carlo draw through an inverted
mass--richness relation -- convolved with lognormal concentration scatter
at fixed mass (Diemer & Kravtsov 2015).

Per-cluster profiles use the Hayashi & White max composition,

.. math::
    \Delta\Sigma(R\mid M, c) = \max\left[
        \Delta\Sigma_{1h}(R\mid M, c),\;
        b(M)\,\bar\rho_m\,\Delta\Sigma_{hh}(R)\right]

so mass scatter propagates **both** to the one-halo amplitude at small
:math:`R` (with the extra :math:`c` scatter) *and* to the large-scale bias
:math:`b(M)`. That is the point McClintock et al. make: scatter in the
:math:`M`--:math:`\lambda` relation causes variance on *all* scales, not
only where the one-halo term lives.

NOTE: this term does **not** go inside
`clenspy.covariance.DeltaSigmaGaussianCovariance`'s five-term bracket. It is a
sixth, independent contribution with a different origin -- a finite-sample
effect of stacking a heterogeneous population, not a Gaussian field
property -- so it is added to the total, and kept separately so it can be
switched off.

NOTE: it scales as :math:`1/N_{\rm cl}`, so it is the term that does
**not** improve when the survey gets deeper at fixed sample; only more
clusters help. That is the opposite scaling to shape noise per unit area,
which is why the two must be tracked separately.

NOTE: **units.** ``R`` in Mpc, the result in
:math:`(M_\odot/{\rm Mpc}^2)^2`, matching
`clenspy.halo.NfwProfile`. The abundance grid is h-scaled
(:math:`h^{-1}M_\odot`) while the profiles are h-free, so this module
divides by ``h`` in exactly one place, `_masses_hfree`, and says so.

NOTE: miscentering stochasticity -- the third component of McClintock's
Monte Carlo -- is **not** modelled. It adds small-scale covariance. The
hook is the profile node set: extend it with
`clenspy.selection.miscentering` draws at each mass node. Named rather
than silently omitted.

NOTE: the concentration scatter is integrated by **Gauss--Hermite**
quadrature over :math:`\ln c`, not Monte Carlo. Eight nodes are exact for
a lognormal to the precision anything else here has, and the result is
deterministic -- a covariance that changes between runs is not usable in a
likelihood.
"""

from __future__ import annotations

import numpy as np

from ..halo.nfw import NfwProfile

__all__ = ["DeltaSigmaHaloToHaloCovariance"]

#: Diemer & Kravtsov (2015): ~0.16 for relaxed samples, ~0.25 spans the
#: full population. The DES Y1 config carries 0.16.
SIGMA_LNC_DEFAULT = 0.16

#: Gauss--Hermite nodes over ln c. Eight is enough for a lognormal.
N_C_DEFAULT = 8


class DeltaSigmaHaloToHaloCovariance:
    r"""Population covariance of per-cluster max-model profiles.

    NOTE: units -- ``R`` in Mpc, result in
    :math:`(M_\odot/{\rm Mpc}^2)^2`. See the module NOTE for the h
    boundary.

    Parameters
    ----------
    abundance : clenspy.observables.ClusterAbundance
        Stored verbatim. Supplies the bin's mass population (its
        :math:`z`-contracted weight) and :math:`N_{\rm cl}`, so this term
        and the counts cannot disagree about the sample.
    twohalo : clenspy.halo.TwoHaloTerm
        The matter :math:`\Delta\Sigma_{hh}` engine, **not** premultiplied
        by the mean density -- that factor is applied here, once, so it
        cannot be applied twice.
    bias : clenspy.halo.BiasModel
        Supplies :math:`b(M)`. Takes **h-free** mass.
    rho_m0 : float
        Comoving mean matter density :math:`\Omega_{m,0}\rho_{c,0}`
        [Msun/Mpc^3].
    z_eff : float
        Representative redshift of the bin, for the two-halo term and the
        concentration pivot.
    concentration : float or callable, optional
        Median :math:`c`, or ``c(m_hfree, z)`` (default: 4.0).
    sigma_lnc : float, optional
        Lognormal scatter of :math:`c` at fixed mass (default: 0.16).
    n_c : int, optional
        Gauss--Hermite nodes over :math:`\ln c` (default: 8).
    """

    def __init__(self, abundance, twohalo, bias, rho_m0, z_eff,
                 concentration=4.0, sigma_lnc=SIGMA_LNC_DEFAULT,
                 n_c=N_C_DEFAULT):
        self.abundance = abundance
        self.twohalo = twohalo
        self.bias = bias
        self.rho_m0 = float(rho_m0)
        self.z_eff = float(z_eff)
        self.sigma_lnc = float(sigma_lnc)
        self.n_c = int(n_c)
        if self.sigma_lnc < 0.0:
            raise ValueError("sigma_lnc must be non-negative")
        if self.n_c < 1:
            raise ValueError("n_c must be at least 1")

        # Gauss-Hermite for the lognormal: ln c = ln c_med + sqrt(2) sigma t
        nodes, weights = np.polynomial.hermite.hermgauss(self.n_c)
        self._c_shift = np.exp(np.sqrt(2.0) * self.sigma_lnc * nodes)
        self._c_weight = weights / np.sqrt(np.pi)

        masses = self._masses_hfree()
        if callable(concentration):
            c_median = np.asarray(concentration(masses, self.z_eff),
                                  dtype=float)
            c_median = np.broadcast_to(c_median, masses.shape)
        else:
            c_median = np.full_like(masses, float(concentration))
        # one halo axis over (mass node, c node), flattened
        self._nfw = NfwProfile(
            m200=np.repeat(masses, self.n_c),
            c200=(c_median[:, None] * self._c_shift[None, :]).ravel(),
            rho_ref=self.rho_m0,
        )
        self._bias_of_mass = np.asarray(self.bias.bias(masses), dtype=float)

    def _masses_hfree(self):
        r"""The abundance's mass nodes in **h-free** :math:`M_\odot`.

        NOTE: the one h conversion in this module. The abundance grid is
        :math:`\ln M` with :math:`M` in :math:`h^{-1}M_\odot`; `NfwProfile`
        and `BiasModel` are h-free. Getting it backwards scales every mass
        by :math:`h^2 \approx 2`.
        """
        return np.exp(self.abundance.ln_mass) / self.abundance.h

    def mass_population(self, i, j):
        r"""The bin's normalised mass distribution, shape ``(n_m,)``.

        The :math:`z`-contracted selection weight,
        :math:`p(\ln M) \propto \int dz\,W_{ij}(\ln M, z)`, times the
        :math:`\ln M` trapezoid measure and normalised to sum to one.

        NOTE: taken from the **same** weight that gives the counts, so the
        population this term averages over is by construction the
        population the counts counted. A separately-built :math:`P(M)` is
        how an intrinsic-variance term ends up describing a different
        sample from the data vector it is attached to.
        """
        weight = self.abundance.weight()[:, :, i, j]
        over_z = np.trapezoid(weight, x=self.abundance.z, axis=1)
        measure = np.gradient(self.abundance.ln_mass)
        population = over_z * measure
        total = population.sum()
        if total <= 0.0:
            raise ValueError(
                f"bin ({i}, {j}) has no selection weight, so it has no mass "
                "population and no intrinsic variance"
            )
        return population / total

    def profiles(self, R):
        r"""Per-cluster max-model profiles, shape ``(n_m, n_c, n_R)``.

        NOTE: the ``max`` is the Hayashi & White composition, taken in
        configuration space at each :math:`(M, c)` -- **not** a sum. Summing
        double-counts the transition region, which is exactly where the
        intrinsic variance peaks.
        """
        R = np.atleast_1d(np.asarray(R, dtype=float))
        one_halo = self._nfw.deltasigma(R).reshape(
            self.abundance.ln_mass.size, self.n_c, R.size
        )
        # the mean-density factor, applied here and only here
        two_halo = self.rho_m0 * np.ravel(
            np.asarray(self.twohalo.deltasigma(R, self.z_eff), dtype=float)
        )
        return np.maximum(
            one_halo,
            self._bias_of_mass[:, None, None] * two_halo[None, None, :],
        )

    def cov(self, R, i, j):
        r""":math:`C^{\rm intr}` for bin ``(i, j)``, shape ``(n_R, n_R)``."""
        R = np.atleast_1d(np.asarray(R, dtype=float))
        profiles = self.profiles(R)

        mass_weight = self.mass_population(i, j)
        joint = mass_weight[:, None] * self._c_weight[None, :]

        mean = np.einsum("kc,kcr->r", joint, profiles)
        second = np.einsum("kc,kcr,kcs->rs", joint, profiles, profiles)
        population_cov = second - np.outer(mean, mean)

        n_cl = float(self.abundance.counts()[i, j])
        if n_cl <= 0.0:
            raise ValueError(f"bin ({i}, {j}) contains no clusters")
        # symmetrise: the outer-product subtraction is not bit-symmetric
        population_cov = 0.5 * (population_cov + population_cov.T)
        return population_cov / n_cl

    def mean_profile(self, R, i, j):
        r"""The population mean :math:`\langle\Delta\Sigma\rangle`.

        Returned alongside `cov` because the two must come from the same
        population weights -- comparing a covariance built here against a
        mean built elsewhere is how an inconsistency hides.
        """
        R = np.atleast_1d(np.asarray(R, dtype=float))
        joint = (self.mass_population(i, j)[:, None]
                 * self._c_weight[None, :])
        return np.einsum("kc,kcr->r", joint, self.profiles(R))

    def __repr__(self):
        return (f"DeltaSigmaHaloToHaloCovariance(z_eff={self.z_eff:g}, "
                f"sigma_lnc={self.sigma_lnc:g}, n_c={self.n_c})")


if __name__ == "__main__":
    from ..cosmology.fiducial import fiducial_cosmology, mean_matter_density
    from ..halo.bias import BiasModel
    from ..halo.twohalo import TwoHaloTerm
    from ..observables import ClusterAbundance
    from ..selection import EmgParams, LogNormalMor, SelectionFunction

    cosmo = fiducial_cosmology()
    rho_m0 = mean_matter_density(cosmo)

    # a normalised power-law P(k), so the demo needs no CAMB
    k = np.logspace(-4.0, 2.0, 400)
    pk = 2.0e4 * k**-1.5 * np.exp(-((k / 30.0) ** 2))

    def toy_mass_function(ln_mass, z):
        lnm, zz = np.broadcast_arrays(np.asarray(ln_mass, float),
                                     np.asarray(z, float))
        m = np.exp(lnm)
        return 1e-5 * (m / 1e14) ** -1.0 * np.exp(-m / 5e14) / (1.0 + zz)

    sel = SelectionFunction(
        np.array([20.0, 30.0, 45.0, 60.0, 200.0]),
        np.array([0.20, 0.35, 0.50, 0.65]),
        LogNormalMor(), EmgParams(-1.5, 3.0, 0.3, 0.12), sigma_z=0.01,
    )
    abundance = ClusterAbundance(
        np.log(np.logspace(13.5, 15.3, 14)), np.linspace(0.16, 0.70, 16),
        toy_mass_function, sel, cosmo,
        lambda z: np.full_like(np.asarray(z, float),
                               1500.0 * (np.pi / 180.0) ** 2),
    )
    twohalo = TwoHaloTerm(k, pk, zvec=np.array([0.28]))
    bias = BiasModel(k, pk, cosmo=cosmo)

    intrinsic = DeltaSigmaHaloToHaloCovariance(abundance, twohalo, bias, rho_m0,
                                         z_eff=0.28)
    print(intrinsic, "\n")

    radii = np.logspace(-0.7, 1.0, 6)
    print("richness bin 0, redshift bin 0:")
    c = intrinsic.cov(radii, 0, 0)
    mean = intrinsic.mean_profile(radii, 0, 0)
    print(f"{'R [Mpc]':>9s}  {'<DS>':>12s}  {'sigma_intr':>12s}  "
          f"{'frac':>8s}")
    for r, m, s in zip(radii, mean, np.sqrt(np.diag(c))):
        print(f"{r:9.3f}  {m:12.4e}  {s:12.4e}  {s / m:8.4f}")
    print("  <- NOTE this is a toy P(k) and a toy mass function, so the")
    print("     magnitude means nothing; the shape does. The fractional")
    print("     scatter falls with R here by ~30x, because in this setup")
    print("     the one-halo term (which carries the c-scatter as well as")
    print("     the mass scatter) dominates the max at small R while the")
    print("     large-R value is set by b(M) scatter alone.")

    print(f"\nN_cl in this bin = {abundance.counts()[0, 0]:.1f}; the term")
    print("scales as 1/N_cl, so it is the one that only more clusters fix.")

    print("\nit rises with richness, because the population is broader:")
    for i in range(sel.n_lambda_bins):
        s = np.sqrt(np.diag(intrinsic.cov(radii, i, 0)))
        m = intrinsic.mean_profile(radii, i, 0)
        print(f"  bin {i}: mean fractional sigma_intr = "
              f"{np.mean(s / m):.4f}   N_cl = {abundance.counts()[i, 0]:8.1f}")

    print("\nconcentration scatter is what the Gauss-Hermite nodes buy:")
    for sig in (0.0, 0.16, 0.25):
        iv = DeltaSigmaHaloToHaloCovariance(abundance, twohalo, bias, rho_m0,
                                      z_eff=0.28, sigma_lnc=sig)
        s = np.sqrt(np.diag(iv.cov(radii, 0, 0)))
        print(f"  sigma_lnc = {sig:4.2f}: sigma_intr at R = "
              f"{radii[0]:.2f} Mpc is {s[0]:.4e}")
    print("  <- c-scatter only matters at small R, where the one-halo term")
    print("     wins the max; the large-R value is set by b(M) scatter.")

    ev = np.linalg.eigvalsh(c)
    print(f"\neigenvalues span {ev.min():.3e} to {ev.max():.3e}")
    print(f"  most negative, relative to the largest: "
          f"{ev.min() / ev.max():.2e}")
    print(f"  positive semi-definite to numerical tolerance: "
          f"{ev.min() >= -1e-8 * ev.max()}")
    print("  <- the small negative eigenvalues are rank deficiency, not a")
    print(f"     bug: the population has {abundance.ln_mass.size} x "
          f"{intrinsic.n_c} nodes but the matrix is {len(radii)}x{len(radii)},")
    print("     and `second - outer(mean, mean)` loses digits where the")
    print("     two nearly cancel. A likelihood should regularise it.")
