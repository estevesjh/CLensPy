r"""The lensing kernel: source-averaged :math:`\Sigma_{\rm crit}` weights.

Four quantities, all integrals of the source distribution over
:math:`z_s`, all functions of a **lens** redshift:

.. math::
    \langle\Sigma_{\rm crit}^{-1}\rangle(z_l)
      = \int\! dz_s\; p(z_s + \Delta z)\,
        \Sigma_{\rm crit}^{-1}(z_l, z_s),
    \qquad
    \gamma_t = \Delta\Sigma \cdot
        \langle\Sigma_{\rm crit}^{-1}\rangle(z_l),

.. math::
    q_\Sigma(z_l; z_h)
      = \int\! dz_s\; p(z_s)\,
        \frac{\Sigma_{\rm crit}(z_h, z_s)}{\Sigma_{\rm crit}(z_l, z_s)},
    \qquad
    \langle\Sigma_{\rm crit}\rangle(z_h)
      = \int_{z_h}\! dz_s\; p(z_s)\, \Sigma_{\rm crit}(z_h, z_s),

.. math::
    f_{\rm src}(z_h) = \int_{z_h}^{z_s^{\max}}\! dz_s\; p(z_s).

The first is what turns :math:`\Delta\Sigma` into a shear. The other three
are exactly what the covariance consumes (``clens.covariance.inputs
.SourceInputs``: ``q_sigma``, ``mean_sigma_crit``, ``f_src_behind``).

NOTE: **comoving** :math:`\Sigma_{\rm crit}`, in Msun/Mpc^2 comoving:

.. math::
    \Sigma_{\rm crit}^{\rm com}(z_l, z_s)
      = \frac{c^2}{4\pi G}\,
        \frac{\chi_s}{\chi_l\,(\chi_s - \chi_l)\,(1 + z_l)} .

This is not the same as `clenspy.kernels.sigma_critical`, which returns the
**physical** value from angular diameter distances; the two differ by
exactly :math:`(1 + z_l)^2` (verified in ``tests/test_lensing_kernel.py``).
Comoving is the right one here because `clenspy`'s :math:`\Delta\Sigma` is
comoving, and :math:`\gamma_t` has to come out dimensionless. It is also
the convention of the exemplar's `LensingKernel`, which is what the frozen
covariance contract was built against.

NOTE: the y3 module ``average_sigma_crit_inv.py`` uses a *third*
convention -- physical, and multiplied by :math:`h_0` -- so its
``sci_average`` is neither of the above. Converting between them is two
factors of :math:`(1+z_l)` and one of :math:`h`, and getting it wrong is
the single easiest way to be quietly wrong here.

NOTE: units are h-free absolute otherwise -- distances in Mpc, surface
densities in Msun/Mpc^2, redshifts dimensionless. `Survey.n_src_arcmin` is
not used by this module.

Four things a naive implementation gets wrong (``docs/refactor-plan.md``
errata E.1), all of them here:

1. **Average the inverse, never invert the average.**
   :math:`\langle\Sigma_{\rm crit}^{-1}\rangle \neq
   1/\langle\Sigma_{\rm crit}\rangle`, and the difference *is* the source
   weighting.
2. **Clamp the integrand at zero.** Sources in front of the lens are not
   lensed by it; they must contribute nothing rather than something
   negative.
3. **Use the flat subtraction form** for the lens-source distance,
   :math:`D_A(z_l, z_s) = D_A(z_s) - \frac{1+z_l}{1+z_s} D_A(z_l)`. In
   comoving terms that is :math:`\chi_s - \chi_l`, which is why this module
   works in :math:`\chi`. The naive :math:`D_A(z_s) - D_A(z_l)` is wrong by
   34% at :math:`z_l = 0.35`, :math:`z_s = 1`.
4. **Carry the photo-z bias** :math:`\Delta z` in the signature. It shifts
   the source :math:`p(z)` and is marginalised over, so it is an argument,
   never a stored constant.

NOTE: two of the four quantities are **logarithmically divergent** and only
exist relative to a convention -- see `MIN_LENS_SOURCE_SEPARATION`. This
module reproduces the frozen covariance reference
(``cluster-lensing-cov/validation/frozen_inputs/kernels.npz``) to 0.14% on
:math:`\langle\Sigma_{\rm crit}\rangle` and
:math:`\langle\Sigma_{\rm crit}^{-1}\rangle`, and exactly on
:math:`f_{\rm src}`. That 0.14% is not an error on either side: the
reference uses :math:`c = 3\times10^5` km/s where `clenspy` uses the exact
299792.458, and :math:`(299792.458/3\times10^5)^2 = 0.9986169` accounts for
all of it. ``validation/validate_lensing_kernel.py`` checks this.
"""

from __future__ import annotations

import numpy as np

from ..utils.constants import C_LIGHT, G_NEWTON

__all__ = ["LensingKernel", "sigma_crit_comoving"]

#: :math:`c^2 / (4\pi G)` in Msun/Mpc -- the amplitude of every
#: :math:`\Sigma_{\rm crit}` below. Formed once from the constants module so
#: the two definitions in this package cannot drift.
_SIGMA_CRIT_AMPLITUDE = C_LIGHT**2 / (4.0 * np.pi * G_NEWTON)

#: Minimum lens-source separation in redshift. **Part of the definition,
#: not a numerical tolerance, and never smaller than this.**
#:
#: NOTE: :math:`\Sigma_{\rm crit} \propto 1/(\chi_s - \chi_l)` diverges as
#: :math:`z_s \to z_l`, so :math:`\langle\Sigma_{\rm crit}\rangle` and
#: :math:`q_\Sigma` are **logarithmically divergent** integrals: their value
#: is set by where the source integral starts. A pair separated by less than
#: this in redshift is not treated as a lens-source pair at all. The
#: exemplar's `LensingKernel` uses 0.01 and the frozen covariance reference
#: was built with it, so it is the definition here too.
#:
#: :math:`\langle\Sigma_{\rm crit}^{-1}\rangle` is the one that does *not*
#: care: its integrand vanishes at the edge. That is the deeper reason
#: errata E.1 says to average the inverse -- the other average does not
#: exist without a convention.
MIN_LENS_SOURCE_SEPARATION = 0.01

#: Nodes for the :math:`z_s` integrals.
#:
#: NOTE: 100, matching the exemplar, and this is **also part of the
#: definition** for the divergent quantities. With a floor at
#: `MIN_LENS_SOURCE_SEPARATION` the integral is finite, but the first
#: trapezoid interval still carries the spike, and its weight is half that
#: interval's width -- so refining the grid *lowers*
#: :math:`\langle\Sigma_{\rm crit}\rangle` rather than converging it
#: (100 -> 200 nodes moves it 4%). Both are arguments on the methods that
#: need them, so a caller can reproduce the reference or refine
#: deliberately.
#:
#: :math:`\langle\Sigma_{\rm crit}^{-1}\rangle` and
#: :math:`f_{\rm src}` *are* convergent: 100 -> 800 nodes moves them by
#: less than 1e-4.
N_ZS_NODES = 100

#: Nodes for the :math:`z_l` grid the interpolants are built on.
_N_ZL_NODES = 100


def sigma_crit_comoving(z_lens, z_source, cosmology, signed=False):
    r"""Comoving :math:`\Sigma_{\rm crit}(z_l, z_s)` [Msun/Mpc^2 comoving].

    .. math::
        \Sigma_{\rm crit}^{\rm com}
          = \frac{c^2}{4\pi G}\,
            \frac{\chi_s}{\chi_l (\chi_s - \chi_l)(1 + z_l)}

    NOTE: :math:`(\chi_s - \chi_l)` *is* the flat lens-source distance --
    the :math:`(1+z_l)/(1+z_s)` of the angular-diameter subtraction form is
    exactly what disappears when the same expression is written in comoving
    distance. Working in :math:`\chi` makes the trap in errata E.1 item 3
    unavailable rather than merely documented.

    NOTE: returns ``+inf`` where :math:`z_s \le z_l`, so that the inverse
    is zero and an unlensed source drops out of an average on its own.
    `LensingKernel` also clamps, because a mixture of ``inf`` and finite
    values in a product is easy to get wrong.

    NOTE: ``signed=True`` skips that guard and returns the bare expression,
    which is **negative** for :math:`z_s < z_l` because
    :math:`\chi_s - \chi_l < 0`. That is not physical on its own -- it is
    what `LensingKernel.q_sigma` needs, because the frozen covariance
    definition of :math:`q_\Sigma` keeps the sign. See that method.

    Parameters
    ----------
    z_lens : float
        Lens redshift. Scalar: the geometry below assumes one lens plane.
    z_source : float or array-like
        Source redshift(s).
    cosmology : astropy.cosmology.Cosmology
        Supplies ``comoving_distance``.

    Returns
    -------
    np.ndarray
        Comoving :math:`\Sigma_{\rm crit}`, broadcast over ``z_source``.
    """
    z_source = np.atleast_1d(np.asarray(z_source, dtype=float))
    chi_l = float(cosmology.comoving_distance(z_lens).value)
    chi_s = np.asarray(cosmology.comoving_distance(z_source).value, dtype=float)

    # A lens at z = 0 has chi_l = 0 and lenses nothing: Sigma_crit is
    # infinite there, so return that rather than dividing by zero.
    if chi_l <= 0.0:
        return np.full(z_source.shape, np.inf)

    delta_chi = chi_s - chi_l
    if signed:
        with np.errstate(divide="ignore", invalid="ignore"):
            return (_SIGMA_CRIT_AMPLITUDE * chi_s
                    / (chi_l * delta_chi * (1.0 + z_lens)))
    out = np.full(z_source.shape, np.inf)
    behind = delta_chi > 0.0
    out[behind] = (
        _SIGMA_CRIT_AMPLITUDE
        * chi_s[behind]
        / (chi_l * delta_chi[behind] * (1.0 + z_lens))
    )
    return out


class LensingKernel:
    r"""Source-averaged :math:`\Sigma_{\rm crit}` weights for one survey.

    Parameters
    ----------
    survey : object
        Anything satisfying the `~clenspy.protocols.Survey` protocol --
        `clenspy.survey.Survey` in practice. Stored verbatim.
    cosmology : astropy.cosmology.Cosmology
        The world model. Stored verbatim.
    unity : bool, optional
        If True, `mean_inverse_sigma_crit` returns 1 everywhere, so every
        downstream consumer emits :math:`\Delta\Sigma` instead of
        :math:`\gamma_t` (default: False).

        NOTE: this is the clean protocol seam, and it is what the y3
        pipeline's ``unity=T`` option does -- it is how a
        :math:`\Delta\Sigma` data vector is compared against a model
        written for shear, with no second code path.

    NOTE: units and conventions are the module's -- **comoving**
    :math:`\Sigma_{\rm crit}`, h-free, Msun/Mpc^2.

    NOTE: the constructor stores and computes nothing. The interpolant over
    :math:`z_l` is built on first use of `kernel_z`, so constructing this
    object is free.
    """

    def __init__(self, survey, cosmology, unity: bool = False) -> None:
        self.survey = survey
        self.cosmo = cosmology
        self.unity = bool(unity)
        self._kernel_z = None  # built lazily by `kernel_z`

    # -- the source grid, shared by every integral ------------------------

    def _zs_nodes(self, z_from, min_separation=None, n_nodes=None):
        r"""Source nodes from ``z_from + min_separation`` to :math:`z_s^{\max}`.

        Returns an empty array if nothing is behind ``z_from``, which is
        what makes `f_src_behind` go to zero at the top of the source
        distribution instead of raising.

        NOTE: ``min_separation`` is floored at `MIN_LENS_SOURCE_SEPARATION`,
        never below it -- it is a definition of what counts as a
        lens-source pair, not a tolerance to be tightened. Passing a
        smaller value raises.
        """
        if min_separation is None:
            min_separation = MIN_LENS_SOURCE_SEPARATION
        if min_separation < MIN_LENS_SOURCE_SEPARATION:
            raise ValueError(
                f"min_separation={min_separation} is below "
                f"MIN_LENS_SOURCE_SEPARATION={MIN_LENS_SOURCE_SEPARATION}. "
                "That floor is part of the definition of a lens-source "
                "pair, not a numerical tolerance: <Sigma_crit> and q_sigma "
                "diverge logarithmically as the separation goes to zero, so "
                "a smaller value does not converge, it just grows."
            )
        n_nodes = N_ZS_NODES if n_nodes is None else int(n_nodes)
        lo = max(z_from + min_separation, self.survey.zs_min)
        hi = self.survey.zs_max
        if not hi > lo:
            return np.empty(0)
        return np.linspace(lo, hi, n_nodes)

    # -- gamma_t = DeltaSigma * <Sigma_crit^-1> ---------------------------

    def mean_inverse_sigma_crit(self, z_lens, delta_z: float = 0.0):
        r""":math:`\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)` [Mpc^2/Msun].

        .. math::
            \langle\Sigma_{\rm crit}^{-1}\rangle(z_l)
              = \int\! dz_s\; p(z_s + \Delta z)\,
                \max\!\left[0,\,
                  \Sigma_{\rm crit}^{-1}(z_l, z_s)\right]

        The **inverse is averaged**, which is not the inverse of
        `mean_sigma_crit` and differs from it by the source weighting.

        Parameters
        ----------
        z_lens : float or array-like
            Lens redshift(s).
        delta_z : float, optional
            Photo-z bias shifting the source :math:`p(z)`, as
            :math:`p(z_s + \Delta z)`. A nuisance parameter, marginalised
            over, so it is an argument (default: 0).

        Returns
        -------
        np.ndarray
            The average, or ones if ``unity`` -- see the class NOTE.
        """
        z_lens = np.atleast_1d(np.asarray(z_lens, dtype=float))
        if self.unity:
            return np.ones(z_lens.shape)

        out = np.zeros(z_lens.shape)
        for i, zl in enumerate(z_lens):
            zs = self._zs_nodes(float(zl))
            if zs.size == 0:
                continue
            pz = self.survey.pz_src(zs + delta_z)
            inv = 1.0 / sigma_crit_comoving(float(zl), zs, self.cosmo)
            # clamp: a source in front of the lens contributes nothing, and
            # must not contribute negatively (errata E.1 item 2)
            out[i] = np.trapezoid(np.maximum(0.0, pz * inv), x=zs)
        return out

    def kernel_z(self, z_lens, delta_z: float = 0.0):
        r"""`mean_inverse_sigma_crit`, interpolated over :math:`z_l`.

        Same quantity, evaluated on a `_N_ZL_NODES` grid once and
        interpolated after -- which is what a Limber integral over
        :math:`\chi` wants, since it evaluates the kernel at every node.

        NOTE: the grid spans :math:`[z_s^{\min}, z_s^{\max}]`, and the
        interpolant is **not** extrapolated: outside it the result is zero,
        because a lens behind every source is unlensed.

        NOTE: cached on the instance keyed by ``delta_z``, so sweeping a
        photo-z bias does not silently reuse the wrong grid.
        """
        z_lens = np.atleast_1d(np.asarray(z_lens, dtype=float))
        if self._kernel_z is None or self._kernel_z[0] != delta_z:
            grid = np.linspace(
                self.survey.zs_min,
                self.survey.zs_max - MIN_LENS_SOURCE_SEPARATION,
                _N_ZL_NODES,
            )
            self._kernel_z = (delta_z, grid,
                              self.mean_inverse_sigma_crit(grid, delta_z))
        _, grid, values = self._kernel_z
        return np.interp(z_lens, grid, values, left=0.0, right=0.0)

    # -- the three the covariance consumes --------------------------------

    def q_sigma(self, z_lens, z_halo, min_separation=None,
                n_nodes=None):
        r"""The :math:`\Sigma_{\rm crit}`-weighted kernel :math:`q_\Sigma(z_l; z_h)`.

        .. math::
            q_\Sigma(z_l; z_h) = \int\! dz_s\; p(z_s)\,
                \frac{\Sigma_{\rm crit}(z_h, z_s)}
                     {\Sigma_{\rm crit}(z_l, z_s)}

        NOTE: **cutoff-defined.** The integrand diverges as
        :math:`z_s \to z_h`, so the value depends on ``min_separation``
        (floored at `MIN_LENS_SOURCE_SEPARATION`) and, through the
        endpoint's trapezoid weight, on ``n_nodes``. Both default to the
        values the frozen covariance reference was built with. The frozen
        reference is itself **signed**: for :math:`z_s` between
        :math:`z_l` and :math:`z_h` the ratio changes sign, and its
        ``q_sigma`` runs from -2.29 to +3.91. That is the definition, not a
        bug -- do not clamp it.

        Dimensionless. It is the weight with which structure at
        :math:`z_l` contributes to a :math:`\Sigma` measured around a halo
        at :math:`z_h` -- the ratio, not either factor alone, which is why
        the :math:`c^2/4\pi G` cancels and only geometry survives.

        NOTE: the source range starts at :math:`z_l` + ``min_separation``,
        keyed on the **lens**, not on the halo. So when
        :math:`z_l < z_h` the range includes sources *in front of the
        halo*, where :math:`\chi_s - \chi_h < 0` and
        :math:`\Sigma_{\rm crit}(z_h, z_s)` is negative. The frozen
        reference keeps that sign -- its ``q_sigma`` runs from -2.29 to
        +3.91 -- so this is the definition and the integrand is **not**
        clamped. Clamping it, or keying the range on
        :math:`\max(z_l, z_h)`, changes the covariance.

        NOTE: the same range choice puts the :math:`z_s = z_h` pole
        *inside* the integral whenever :math:`z_l < z_h`. The trapezoid
        straddles it and returns a finite but grid-dependent number, which
        is where the :math:`\pm 4` excursions come from. That is inherited
        from the reference definition, not introduced here.

        Parameters
        ----------
        z_lens : float or array-like
            Redshift of the lensing structure.
        z_halo : float
            Redshift of the halo the profile is measured around.
        min_separation, n_nodes : float, int, optional
            The two conventions the value depends on; see
            `MIN_LENS_SOURCE_SEPARATION`.
        """
        z_lens = np.atleast_1d(np.asarray(z_lens, dtype=float))
        out = np.zeros(z_lens.shape)
        for i, zl in enumerate(z_lens):
            # keyed on the lens, per the frozen definition above
            zs = self._zs_nodes(float(zl), min_separation, n_nodes)
            if zs.size == 0:
                continue
            pz = self.survey.pz_src(zs)
            sc_halo = sigma_crit_comoving(z_halo, zs, self.cosmo, signed=True)
            sc_lens = sigma_crit_comoving(float(zl), zs, self.cosmo)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = sc_halo / sc_lens
            # signed, not clamped -- only genuine non-finites are dropped
            ratio = np.where(np.isfinite(ratio), ratio, 0.0)
            out[i] = np.trapezoid(pz * ratio, x=zs)
        return out

    def mean_sigma_crit(self, z_halo, min_separation=None,
                        n_nodes=None):
        r""":math:`\langle\Sigma_{\rm crit}\rangle(z_h)` [Msun/Mpc^2 comoving].

        .. math::
            \langle\Sigma_{\rm crit}\rangle(z_h)
              = \int_{z_h}\! dz_s\; p(z_s)\,\Sigma_{\rm crit}(z_h, z_s)

        NOTE: **cutoff-defined, and formally divergent.** Sources just
        behind the lens have unbounded :math:`\Sigma_{\rm crit}`, so the
        true average does not exist; this returns the value under the
        conventions of `MIN_LENS_SOURCE_SEPARATION` and ``n_nodes``, which
        are what the frozen covariance reference used. Refining ``n_nodes``
        *lowers* the answer rather than converging it -- 100 to 200 moves
        it 4%.

        NOTE: **not** the reciprocal of `mean_inverse_sigma_crit`. Both are
        needed and they are different averages; see errata E.1 item 1. That
        one is convergent, which is the deeper reason to prefer it.

        NOTE: not normalised by `f_src_behind`. It is the average as the
        covariance defines it -- the integral over the *whole* source
        distribution, with sources in front contributing zero -- so it
        carries the behind-fraction implicitly. Dividing by
        `f_src_behind` would give the average over lensed sources only,
        which is a different quantity.
        """
        z_halo = np.atleast_1d(np.asarray(z_halo, dtype=float))
        out = np.zeros(z_halo.shape)
        for i, zh in enumerate(z_halo):
            zs = self._zs_nodes(float(zh), min_separation, n_nodes)
            if zs.size == 0:
                continue
            sc = sigma_crit_comoving(float(zh), zs, self.cosmo)
            integrand = np.where(np.isfinite(sc), self.survey.pz_src(zs) * sc,
                                 0.0)
            out[i] = np.trapezoid(integrand, x=zs)
        return out

    def f_src_behind(self, z_halo):
        r"""Fraction of sources behind :math:`z_h`, dimensionless.

        .. math::
            f_{\rm src}(z_h) = \int_{z_h}^{z_s^{\max}}\! dz_s\; p(z_s)

        Falls to zero at the top of the source distribution and is 1 below
        its bottom, since :math:`p(z_s)` is normalised.
        """
        z_halo = np.atleast_1d(np.asarray(z_halo, dtype=float))
        out = np.zeros(z_halo.shape)
        for i, zh in enumerate(z_halo):
            zs = self._zs_nodes(float(zh))
            if zs.size == 0:
                continue
            out[i] = np.trapezoid(self.survey.pz_src(zs), x=zs)
        return out

    def __repr__(self) -> str:
        return (f"LensingKernel(survey={getattr(self.survey, 'name', '?')!r}, "
                f"unity={self.unity})")


if __name__ == "__main__":
    from ..cosmology import fiducial_cosmology
    from ..survey import Survey

    cosmo = fiducial_cosmology()
    lk = LensingKernel(survey=Survey.from_config("des_y1"), cosmology=cosmo)
    print(lk)

    z_l = np.array([0.2, 0.35, 0.5, 0.65])
    print(f"\n{'z_l':>6s}  {'<Sc^-1>':>12s}  {'1/<Sc>':>12s}"
          f"  {'ratio':>7s}  {'f_behind':>9s}")
    inv = lk.mean_inverse_sigma_crit(z_l)
    mean = lk.mean_sigma_crit(z_l)
    fb = lk.f_src_behind(z_l)
    for zl, a, b, f in zip(z_l, inv, mean, fb):
        print(f"{zl:6.2f}  {a:12.5e}  {1 / b:12.5e}  {a * b:7.4f}  {f:9.4f}")
    print("\n  ratio != 1 is the source weighting: averaging the inverse is")
    print("  not inverting the average (errata E.1 item 1).")

    print(f"\nq_sigma(z_l; z_h=0.35) at z_l = {z_l}:")
    print("  ", np.array2string(lk.q_sigma(z_l, 0.35), precision=4))

    # the protocol seam: unity=True turns gamma_t into DeltaSigma
    unity = LensingKernel(survey=Survey.from_config("des_y1"),
                          cosmology=cosmo, unity=True)
    print(f"\nunity=True -> <Sc^-1> = {unity.mean_inverse_sigma_crit(z_l)}")

    # the photo-z bias moves the answer, which is why it is marginalised
    for dz in (-0.02, 0.0, 0.02):
        v = lk.mean_inverse_sigma_crit(0.35, delta_z=dz).item()
        print(f"  delta_z = {dz:+.2f}: <Sc^-1>(0.35) = {v:.5e}")
