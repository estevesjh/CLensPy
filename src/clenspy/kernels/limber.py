r"""Limber projection: angular power spectra from line-of-sight windows.

Transcribed from **Wu et al. (2019), MNRAS 490, 2606**, section 5 and
appendix A. The paper writes all three spectra the covariance needs as *one*
formula with two windows,

.. math::
    C_\ell^{AB} = \int_{\chi_{\min}}^{\chi_{\max}} \!\! d\chi\;
        \left(\frac{F_A(\chi)}{\chi}\right)
        \left(\frac{F_B(\chi)}{\chi}\right)
        P_{AB}\!\left(k = \frac{\ell + 1/2}{\chi}\right),

so that is how it is written here: `limber` once, and `F_h` / `F_Sigma`
passed to it. The three near-duplicate slab loops it replaces differed only
in which windows they carried.

Paper symbol, this module, and the equation it comes from:

* :math:`C_\ell^{\rm hh}` -- `C_ell_hh`, eq. ``clhh``
* :math:`C_\ell^{\Sigma\Sigma}` -- `C_ell_SS`, eq. ``clSS``
* :math:`C_\ell^{\rm h\Sigma}` -- `C_ell_hS`, eq. ``clhS``
* :math:`F_{\rm h}` -- `F_h`, eq. ``F_h``
* :math:`F_\Sigma` -- `F_Sigma`, eq. ``F_Sigma``
* :math:`1/n_{\rm h}^{(2D)}` -- `shot_noise_h`, eq. ``cov_DS``
* :math:`\sigma_\gamma^2 \langle\Sigma_{\rm crit}\rangle^2 /
  n_{\rm s}^{(2D)}` -- `shape_noise_Sigma`, eq. ``cov_DS``

The windows, verbatim from the paper:

.. math::
    F_{\rm h}(\chi_h) = \frac{\chi_h^2}{V},
    \qquad V = \int_{\chi_{\min}}^{\chi_{\max}} \chi_h^2\, d\chi_h ,

.. math::
    F_\Sigma(\chi_{\rm lss}, \chi_h) = \bar\rho
      \int_{\chi_{\rm lss}}^{\infty} \!\! d\chi_s\; p_{\rm src}(\chi_s)\,
      \frac{\Sigma_{\rm crit}(z_s, z_h)}{\Sigma_{\rm crit}(z_s, z_{\rm lss})}

NOTE: that second integral runs from :math:`\chi_{\rm lss}` **to
infinity** -- it is keyed on the line-of-sight structure, not on the halo.
So :math:`F_\Sigma = \bar\rho\, q_\Sigma(z_{\rm lss}, z_h)` exactly, with
:math:`q_\Sigma` as `clenspy.kernels.LensingKernel.q_sigma` computes it.
The paper's own range settles a choice that looked arbitrary in the code:
:math:`q_\Sigma` keys its source range on the lens and therefore keeps a
sign, and that is Wu et al.'s definition, not an implementation accident.

NOTE: the :math:`\Sigma_{\rm crit}(z_s, z_h)` in the numerator is *not* a
lensing kernel. It is there because the covariance interprets all
line-of-sight structure as noise on a profile at the halo's redshift, so
the LSS contribution is converted to a :math:`\Sigma` at :math:`z_h`
(paper, section 6). This is why :math:`C_\ell^{\Sigma\Sigma}` carries units
of :math:`\Sigma^2` and :math:`C_\ell^{\rm h\Sigma}` units of
:math:`\Sigma`.

NOTE: units are h-free absolute -- :math:`\chi` in Mpc, :math:`P` in
Mpc^3, :math:`\bar\rho` in Msun/Mpc^3, so :math:`C_\ell^{\Sigma\Sigma}` is
in (Msun/Mpc^2)^2 and :math:`C_\ell^{\rm h\Sigma}` in Msun/Mpc^2.
:math:`n_{\rm s}` enters per **steradian**; ``n_src_arcmin2`` is converted
by `ARCMIN_TO_RAD` in one visible multiplication.

NOTE: **Limber, with the** :math:`\ell + 1/2` **prescription**
(LoVerde & Afshordi 2008), and the flat-sky, thin-lens-bin identification
:math:`\theta = r_p/\chi(z_{\rm mid})`, :math:`\ell = k\,\chi`. Valid for a
*thin* slice of halo redshift; the paper says so explicitly under eq.
``cov_DS``.

NOTE: one deviation from the paper, deliberate. Its shape-noise term is
:math:`\sigma_\gamma^2/n_{\rm s}^{(2D)}`; `shape_noise_Sigma` divides by
:math:`n_{\rm s} f_{\rm src}(z_h)` instead, counting only sources behind
the lens. That is the refinement the `cluster-lensing-cov` implementation
carries and what the frozen reference was built with. Pass
``f_src_behind=lambda z: 1.0`` for the paper's form exactly.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["ARCMIN_TO_RAD", "LimberProjector", "limber"]

#: Radians per arcminute. The one unit crossing in this module.
ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)

#: Redshift width of the slabs the :math:`\chi` integrals are summed over.
#:
#: NOTE: 0.1 in redshift. The windows and :math:`P(k, z)` vary slowly on
#: that scale, while the bins themselves are 0.15 wide, so a lens bin gets
#: one or two slabs -- which is the point: eq. ``cov_DS`` is only valid for
#: a thin lens bin anyway, so refining this past the bin width would be
#: refining inside an approximation that has already been made.
DZ_SLAB = 0.1

#: Default :math:`\ell` grid: 1000 points per decade over
#: :math:`[10^{-1}, 2\times10^{7}]`.
#:
#: NOTE: the extent is set by the radii, not by taste.
#: :math:`\ell \sim 1/\theta = \chi/r_p`; at :math:`z_h = 0.25`
#: (:math:`\chi \approx 10^3` Mpc) and :math:`r_p = 0.03` Mpc that is
#: :math:`\ell \sim 3\times10^4`, and the :math:`\bar J_2` kernel needs
#: two decades beyond its first peak to damp. The lower end must reach
#: below :math:`\chi/r_{p,\max}`. This is the Wu et al. (2019) convention.
N_ELL_DEFAULT = 8000
ELL_RANGE_DEFAULT = (1e-1, 2e7)


def limber(ell, chi_nodes, dchi, window_a, window_b, pk):
    r"""The Limber integral, once, for any pair of windows.

    .. math::
        C_\ell^{AB} = \sum_{\rm slabs} \Delta\chi\;
            \frac{F_A(\chi)}{\chi}\, \frac{F_B(\chi)}{\chi}\;
            P_{AB}\!\left(k = \frac{\ell + 1/2}{\chi}\right)

    Parameters
    ----------
    ell : np.ndarray
        Multipoles.
    chi_nodes : np.ndarray
        Slab-centre comoving distances [Mpc].
    dchi : np.ndarray
        Slab widths [Mpc], same length as ``chi_nodes``.
    window_a, window_b : np.ndarray
        :math:`F_A`, :math:`F_B` evaluated at ``chi_nodes``. Pass the same
        array twice for an auto spectrum.
    pk : sequence of np.ndarray
        :math:`P_{AB}(k = (\ell + 1/2)/\chi_i)` for each slab ``i``, each
        the shape of ``ell``. Supplied by the caller because only it knows
        which :math:`P` belongs to which pair.

    Returns
    -------
    np.ndarray
        :math:`C_\ell^{AB}`, the shape of ``ell``.
    """
    out = np.zeros_like(np.asarray(ell, dtype=float))
    for chi_i, dchi_i, fa, fb, p_i in zip(chi_nodes, dchi, window_a,
                                          window_b, pk):
        out += dchi_i * (fa / chi_i) * (fb / chi_i) * p_i
    return out


class LimberProjector:
    r"""Wu et al. (2019) angular power spectra on a shared log-:math:`\ell` grid.

    NOTE: units and approximations are the module's -- h-free absolute,
    Limber with :math:`\ell + 1/2`, thin lens bin.

    Every input is a plain callable, so a live `clenspy` object and a frozen
    snapshot table drive it identically.

    Parameters
    ----------
    chi : callable
        Comoving distance :math:`\chi(z)` [Mpc].
    pk_lin : callable
        Linear matter power :math:`P_{\rm lin}(k, z)` [Mpc^3], vectorised
        in ``k``.
    rho_mean0 : float
        Comoving mean matter density :math:`\bar\rho` [Msun/Mpc^3]. The
        :math:`\bar\rho` of eq. ``F_Sigma``; comoving, so no :math:`z`.
    q_sigma : callable
        :math:`q_\Sigma(z_{\rm lss}, z_h)`, so that
        :math:`F_\Sigma = \bar\rho\, q_\Sigma` -- see the module NOTE.
    mean_sigma_crit : callable
        :math:`\langle\Sigma_{\rm crit}\rangle(z_h)` [Msun/Mpc^2].
    f_src_behind : callable
        Fraction of sources behind :math:`z_h`. See the module's note on
        the one deviation from the paper.
    sigma_gamma : float
        Per-galaxy shape noise.
    n_src_arcmin2 : float
        Source surface density [arcmin^-2].
    n_ell : int, optional
        Points on the :math:`\ell` grid (default `N_ELL_DEFAULT`).
    ell_range : tuple of float, optional
        :math:`\ell` extent (default `ELL_RANGE_DEFAULT`); see the note
        there for why.
    """

    def __init__(
        self,
        *,
        chi: Callable,
        pk_lin: Callable,
        rho_mean0: float,
        q_sigma: Callable,
        mean_sigma_crit: Callable,
        f_src_behind: Callable,
        sigma_gamma: float,
        n_src_arcmin2: float,
        n_ell: int = N_ELL_DEFAULT,
        ell_range: tuple[float, float] = ELL_RANGE_DEFAULT,
    ) -> None:
        # store the collaborators verbatim
        self.chi = chi
        self.pk_lin = pk_lin
        self.rho_mean0 = float(rho_mean0)
        self.q_sigma = q_sigma
        self.mean_sigma_crit = mean_sigma_crit
        self.f_src_behind = f_src_behind
        self.sigma_gamma = float(sigma_gamma)
        self.n_src_arcmin2 = float(n_src_arcmin2)
        self.ell = np.exp(
            np.linspace(np.log(ell_range[0]), np.log(ell_range[1]), n_ell)
        )

    # -- the slab decomposition, shared by all three spectra --------------

    def _slabs(self, z_min, z_max):
        r"""Slab centres, widths and :math:`\chi` over ``[z_min, z_max]``.

        Returns ``(z_mid, chi_mid, dchi, volume)`` where
        :math:`V = \sum \Delta\chi\, \chi^2` is the :math:`V` of eq.
        ``F_h`` -- formed on the *same* slabs the spectra are summed on, so
        that :math:`\int F_{\rm h}\,d\chi / \chi^2 = 1` holds discretely and
        not just in the continuum.
        """
        n = max(int((z_max - z_min) / DZ_SLAB), 1)
        edges = np.linspace(z_min, z_max, n + 1)
        z_mid = 0.5 * (edges[:-1] + edges[1:])
        chi_mid = np.array([float(self.chi(z)) for z in z_mid])
        dchi = np.array([float(self.chi(hi) - self.chi(lo))
                         for lo, hi in zip(edges[:-1], edges[1:])])
        volume = float(np.sum(dchi * chi_mid**2))
        return z_mid, chi_mid, dchi, volume

    def _pk_at_slabs(self, chi_mid, z_mid, pk):
        r""":math:`P(k = (\ell + 1/2)/\chi_i, z_i)` for each slab."""
        return [pk((self.ell + 0.5) / chi_i, z_i)
                for chi_i, z_i in zip(chi_mid, z_mid)]

    # -- the two windows, eq. F_h and eq. F_Sigma --------------------------

    @staticmethod
    def F_h(chi_mid, volume):
        r""":math:`F_{\rm h}(\chi_h) = \chi_h^2 / V` (eq. ``F_h``).

        The halo-sample window: a normalised comoving-volume weight, so it
        carries units of 1/Mpc and :math:`C_\ell^{\rm hh}` comes out
        dimensionless.
        """
        return chi_mid**2 / volume

    def F_Sigma(self, z_mid, z_halo):
        r""":math:`F_\Sigma(\chi_{\rm lss}, \chi_h) = \bar\rho\,
        q_\Sigma(z_{\rm lss}, z_h)` (eq. ``F_Sigma``).

        NOTE: the identification with :math:`q_\Sigma` is exact, not an
        approximation -- the paper's :math:`\int_{\chi_{\rm lss}}^\infty
        d\chi_s\, p_{\rm src}\,\Sigma_{\rm crit}(z_s,z_h) /
        \Sigma_{\rm crit}(z_s,z_{\rm lss})` *is*
        `LensingKernel.q_sigma`. Units: Msun/Mpc^3.
        """
        # q_sigma is vectorised in the lens redshift, so one call
        return self.rho_mean0 * np.ravel(
            np.asarray(self.q_sigma(z_mid, z_halo), dtype=float)
        )

    # -- the three spectra ------------------------------------------------

    def C_ell_SS(self, z_lss_min, z_lss_max, z_halo):
        r""":math:`C_\ell^{\Sigma\Sigma}(z_h)` [(Msun/Mpc^2)^2], eq. ``clSS``.

        .. math::
            C_\ell^{\Sigma\Sigma} = \int_0^\infty \!\! d\chi_{\rm lss}
              \left(\frac{F_\Sigma}{\chi_{\rm lss}}\right)^2
              P_{\rm mm}\!\left(k=\frac{\ell+1/2}{\chi_{\rm lss}}\right)

        NOTE: the paper integrates over **all** line-of-sight structure,
        :math:`0` to :math:`\infty`. Here the range is an argument, because
        :math:`q_\Sigma` vanishes once no sources lie behind
        :math:`z_{\rm lss}` and the caller knows where that is -- the
        covariance uses :math:`[0.1, \min(2, z_s^{\max} - 0.1)]`. Truncating
        it is the caller's stated approximation, not a hidden one.
        """
        z_mid, chi_mid, dchi, _ = self._slabs(z_lss_min, z_lss_max)
        window = self.F_Sigma(z_mid, z_halo)
        return limber(self.ell, chi_mid, dchi, window, window,
                      self._pk_at_slabs(chi_mid, z_mid, self.pk_lin))

    def C_ell_hh(self, z_min, z_max, bias, pk_hh: Callable | None = None):
        r""":math:`C_\ell^{\rm hh}` (dimensionless), eq. ``clhh``.

        .. math::
            C_\ell^{\rm hh} = \int_{\chi_{\min}}^{\chi_{\max}} \!\! d\chi_h
              \left(\frac{F_{\rm h}}{\chi_h}\right)^2
              P_{\rm hh}\!\left(k=\frac{\ell+1/2}{\chi_h}\right)

        Parameters
        ----------
        z_min, z_max : float
            The cluster redshift bin.
        bias : float
            Linear halo bias, used only for the default
            :math:`P_{\rm hh} = b^2 P_{\rm lin}` (the paper's choice, with
            the Tinker et al. 2010 bias).
        pk_hh : callable, optional
            Full :math:`P_{\rm hh}(k, z)`, overriding the linear-bias form.
        """
        z_mid, chi_mid, dchi, volume = self._slabs(z_min, z_max)
        window = self.F_h(chi_mid, volume)
        pk = pk_hh if pk_hh is not None else (
            lambda k, z: bias**2 * self.pk_lin(k, z)
        )
        return limber(self.ell, chi_mid, dchi, window, window,
                      self._pk_at_slabs(chi_mid, z_mid, pk))

    def C_ell_hS(self, z_min, z_max, bias, z_halo,
                 pk_hm: Callable | None = None):
        r""":math:`C_\ell^{\rm h\Sigma}` [Msun/Mpc^2], eq. ``clhS``.

        .. math::
            C_\ell^{\rm h\Sigma} = \int_{\chi_{\min}}^{\chi_{\max}}
              \!\! d\chi \left(\frac{F_{\rm h}}{\chi}\right)
              \left(\frac{F_\Sigma}{\chi}\right)
              P_{\rm hm}\!\left(k=\frac{\ell+1/2}{\chi}\right)

        NOTE: integrated over the **halo** bin, where the two fields
        overlap -- not over all LSS as :math:`C_\ell^{\Sigma\Sigma}` is.

        Parameters
        ----------
        pk_hm : callable, optional
            Full :math:`P_{\rm hm}(k, z)`. The paper uses the halo model:
            two-halo (:math:`b P_{\rm lin}`) plus a one-halo NFW term. The
            default here is the two-halo part only, :math:`b P_{\rm lin}`,
            so pass ``pk_hm`` for the paper's full form.
        """
        z_mid, chi_mid, dchi, volume = self._slabs(z_min, z_max)
        pk = pk_hm if pk_hm is not None else (
            lambda k, z: bias * self.pk_lin(k, z)
        )
        return limber(self.ell, chi_mid, dchi,
                      self.F_h(chi_mid, volume),
                      self.F_Sigma(z_mid, z_halo),
                      self._pk_at_slabs(chi_mid, z_mid, pk))

    # -- the two noise terms ----------------------------------------------

    @staticmethod
    def shot_noise_h(counts, area_sr):
        r"""Halo shot noise :math:`1/n_{\rm h}^{(2D)}` [sr], eq. ``cov_DS``.

        :math:`n_{\rm h}^{(2D)}` is the halo surface density per steradian,
        so this is ``area_sr / counts``. It adds to
        :math:`C_\ell^{\rm hh}`, which is why both are dimensionless.
        """
        return float(area_sr) / float(counts)

    def shape_noise_Sigma(self, z_halo):
        r"""Shape noise on :math:`\Sigma` [(Msun/Mpc^2)^2], eq. ``cov_DS``.

        .. math::
            N^{\Sigma} = \langle\Sigma_{\rm crit}\rangle^2(z_h)\,
                \frac{\sigma_\gamma^2}{n_{\rm s}^{(2D)} f_{\rm src}(z_h)}

        NOTE: the :math:`f_{\rm src}` is the one deviation from the paper --
        see the module NOTE. It counts only sources behind the lens, which
        *raises* the noise relative to eq. ``cov_DS`` by
        :math:`1/f_{\rm src}`.
        """
        f_src = float(np.ravel(self.f_src_behind(z_halo))[0])
        # the one unit crossing: arcmin^-2 -> sr^-1
        n_src_sr = self.n_src_arcmin2 * f_src / ARCMIN_TO_RAD**2
        if n_src_sr <= 0.0:
            return np.inf  # no sources behind the lens: infinite noise
        sigma_crit = float(np.ravel(self.mean_sigma_crit(z_halo))[0])
        return self.sigma_gamma**2 / n_src_sr * sigma_crit**2

    # -- deprecated aliases, one release ----------------------------------
    #
    # The names above follow Wu et al. (2019); these are what
    # `cluster-lensing-cov` calls today.

    def c_ell_sigma(self, zl_min, zl_max, z_h):
        """Deprecated alias for `C_ell_SS`."""
        return self.C_ell_SS(zl_min, zl_max, z_h)

    def c_ell_h(self, z_min, z_max, bias, counts, area_sr,
                pk_hh: Callable | None = None):
        """Deprecated alias returning ``(C_ell_hh, shot_noise_h)``."""
        return (self.C_ell_hh(z_min, z_max, bias, pk_hh=pk_hh),
                self.shot_noise_h(counts, area_sr))

    def c_ell_h_sigma(self, z_min, z_max, bias, z_h,
                      pk_hm: Callable | None = None):
        """Deprecated alias for `C_ell_hS`."""
        return self.C_ell_hS(z_min, z_max, bias, z_h, pk_hm=pk_hm)

    def shape_noise_sigma(self, z_h):
        """Deprecated alias for `shape_noise_Sigma`."""
        return self.shape_noise_Sigma(z_h)

    def __repr__(self) -> str:
        return (f"LimberProjector(n_ell={self.ell.size}, "
                f"ell=[{self.ell[0]:.1e}, {self.ell[-1]:.1e}])")


if __name__ == "__main__":
    from ..cosmology import fiducial_cosmology, mean_matter_density
    from ..survey import Survey
    from .lensing_kernel import LensingKernel

    cosmo = fiducial_cosmology()
    survey = Survey.from_config("des_y1")
    lk = LensingKernel(survey, cosmo)

    # a smooth stand-in for P_lin, so the demo needs no Boltzmann solver
    def pk_lin(k, z):
        k = np.asarray(k, dtype=float)
        return 2e4 * k**-1.5 / (1.0 + (k / 0.2) ** 2) / (1.0 + z) ** 2

    proj = LimberProjector(
        chi=lambda z: cosmo.comoving_distance(z).value,
        pk_lin=pk_lin,
        rho_mean0=mean_matter_density(cosmo),
        q_sigma=lk.q_sigma,
        mean_sigma_crit=lk.mean_sigma_crit,
        f_src_behind=lk.f_src_behind,
        sigma_gamma=survey.sigma_gamma,
        n_src_arcmin2=survey.n_src_arcmin,
        n_ell=200,  # coarse, for the demo only
    )
    print(proj)

    z_min, z_max, z_h, bias = 0.35, 0.50, 0.425, 3.5
    c_hh = proj.C_ell_hh(z_min, z_max, bias)
    c_SS = proj.C_ell_SS(0.1, 2.0, z_h)
    c_hS = proj.C_ell_hS(z_min, z_max, bias, z_h)
    shot = proj.shot_noise_h(counts=1500.0, area_sr=0.455)
    n_shape = proj.shape_noise_Sigma(z_h)

    print(f"\nlens bin z = [{z_min}, {z_max}], z_h = {z_h}, b = {bias}")
    print(f"{'ell':>10s}  {'C_hh':>11s}  {'C_SS':>11s}  {'C_hS':>11s}")
    for i in (0, 60, 120, 199):
        print(f"{proj.ell[i]:10.3e}  {c_hh[i]:11.4e}  {c_SS[i]:11.4e}  "
              f"{c_hS[i]:11.4e}")
    print(f"\nshot noise 1/n_h    = {shot:.4e} sr")
    print(f"shape noise N^Sigma = {n_shape:.4e} (Msun/Mpc^2)^2")

    # The correlation the paper's structure implies -- but only when all
    # three spectra span the SAME range.
    ratio = c_hS**2 / (c_hh * c_SS)
    print(f"\n(C_hS)^2 / (C_hh C_SS) over the wide C_SS range: "
          f"min={ratio.min():.4f} max={ratio.max():.4f}")
    print("  < 1 because C_SS integrates ALL line-of-sight structure")
    print("  (0.1-2.0) while C_hS only integrates the lens bin -- the extra")
    print("  LSS adds variance that does not correlate with the haloes.")

    # restrict C_SS to the lens bin and the ratio becomes exactly 1: with
    # linear bias, P_hm^2 = P_hh P_mm identically, so the two fields are
    # perfectly correlated. This is the real consistency check on `limber`.
    c_SS_bin = proj.C_ell_SS(z_min, z_max, z_h)
    ratio_bin = c_hS**2 / (c_hh * c_SS_bin)
    print(f"\nsame range, linear bias: min={ratio_bin.min():.10f} "
          f"max={ratio_bin.max():.10f}")
    print("  exactly 1 -- P_hm^2 = P_hh P_mm for linear bias, so this is a")
    print("  closed-form check that the three windows are wired correctly.")
