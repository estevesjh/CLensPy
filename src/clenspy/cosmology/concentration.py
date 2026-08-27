r"""Concentration--mass relations, :math:`c(M, z)`.

The NFW profile needs two numbers, and the halo mass function supplies only
one. This module supplies the other. It sits in `cosmology` rather than in
`halo` because :math:`c(M,z)` is a *structure-formation* result -- it is
calibrated on N-body simulations at a fixed cosmology, exactly like the mass
function and the halo bias -- and because a referee varies it while holding
the profile shape fixed.

Three relations, and one non-relation:

- `child18` -- Child et al. (2018), Eq. 18 and Table 1. The
  :math:`c`--:math:`M/M_\star` form, redshift-independent once mass is
  scaled by the nonlinear mass :math:`M_\star(z)`. This is the production
  relation in ``y3_cluster_cpp``
  (``src/modules/deltasigma/massconcen.py::c_from_m200``).
- `child18_powerlaw` -- Child et al. (2018), Eq. 19 and Table 2. A plain
  power law in :math:`M` and :math:`1+z`, valid only for
  :math:`0 \le z \le 1`, needing no :math:`M_\star`.
- `duffy08` -- Duffy et al. (2008), Table 1. The older power law, kept
  because `pyccl` and much of the literature default to it, so it is what
  a cross-check will be against.
- `Y3_FIXED_CONCENTRATION` -- the constant :math:`c = 4` that the y3
  offline miscentering tables were *built at* (``nfw_dsigma_mis.hh``:
  ``double const CONC = 4.0;``). Not a relation, and not adjustable
  downstream of the table: see the NOTE on it.

NOTE: **units.** These relations were calibrated in :math:`h^{-1}M_\odot`
and are the one place in `clenspy` where the h-free convention breaks, so
every mass argument carries the unit in its name (``m200c_hinv``,
``m_star_hinv``, ``m_hinv``). Convert at the call site, visibly:
``m200c_hinv = m200c / h``. Concentration itself is dimensionless.

NOTE: **mass definition.** `child18` is a :math:`M_{200c}` relation -- mass
inside :math:`\Delta = 200` times the *critical* density -- while
`clenspy.halo.NfwProfile` and the Tinker mass function use
:math:`M_{200m}`, referred to the *mean matter* density. They are not the
same halo. ``y3_cluster_cpp`` converts :math:`M_{200m} \to M_{200c}` with
the fitting functions of Ragagnin et al. (2021) (their ``hydro_mc``
package) before evaluating Eq. 18, and notes that :math:`M_{200c}` is
strictly the smaller of the two. `clenspy` does not vendor that conversion,
so `child18` takes :math:`M_{200c}` in its signature and the caller owns
the conversion. `duffy08` takes a ``mass_def`` because Duffy et al.
tabulate all three.

NOTE: **cosmology dependence.** Child et al. state plainly that the
:math:`c`--:math:`M/M_\star` form "is not fully universal in the sense of
being approximately cosmology-independent" -- concentrations are higher in
high-:math:`\sigma_8` cosmologies. The coefficients here are their Table 1
verbatim, as adopted for DES Y3; the cosmology enters through
:math:`M_\star(z)`, which must be recomputed per cosmology. That is why
`child18` takes ``m_star_hinv`` as an argument rather than computing it.

NOTE: **scatter.** Both Child et al. tables carry the same note: "use ...
with variance :math:`\sigma_c = c_{200c}/3`". A 33% halo-to-halo spread is
large, and a stacked profile is not the profile at the mean concentration.
`scatter` returns it.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

__all__ = [
    "CHILD18_TABLE1",
    "CHILD18_TABLE2",
    "DUFFY08_TABLE1",
    "DUFFY08_PIVOT_HINV",
    "Y3_FIXED_CONCENTRATION",
    "DELTA_COLLAPSE",
    "child18",
    "child18_powerlaw",
    "duffy08",
    "m_star_hinv",
    "m_star_from_sigma",
    "delta_c",
    "scatter",
]

#: Child et al. (2018) Table 1 -- ``fit -> (m, A, b, c_0)``, where
#: :math:`b \equiv M_T/M_\star` is the threshold mass in units of the
#: nonlinear mass. For Eq. 18, valid :math:`0 \le z \le 4`.
#:
#: ``individual_all`` is the row ``y3_cluster_cpp`` uses.
CHILD18_TABLE1 = {
    "individual_all": (-0.10, 3.44, 430.49, 3.19),
    "individual_relaxed": (-0.09, 2.88, 1644.53, 3.54),
    "stacked_nfw": (-0.07, 4.61, 638.65, 3.59),
    "stacked_einasto": (-0.01, 63.2, 431.48, 3.36),
}

#: Child et al. (2018) Table 2 -- ``fit -> (A, d, m)`` for Eq. 19. Valid
#: only for :math:`0 \le z \le 1`.
CHILD18_TABLE2 = {
    "individual_all": (75.4, -0.422, -0.089),
    "individual_relaxed": (68.4, -0.347, -0.083),
    "stacked_nfw": (57.6, -0.376, -0.078),
    "stacked_einasto": (122.0, -0.446, -0.101),
}

#: Duffy et al. (2008) Table 1, full sample, :math:`0 \le z \le 2` --
#: ``mass_def -> (A, B, C)``. Keyed by mass definition because the three
#: rows are three different halo boundaries, not three fits to one.
DUFFY08_TABLE1 = {
    "vir": (7.85, -0.081, -0.71),
    "200m": (10.14, -0.081, -1.01),
    "200c": (5.71, -0.084, -0.47),
}

#: Duffy et al. (2008) pivot mass, :math:`2\times10^{12}\,h^{-1}M_\odot`.
DUFFY08_PIVOT_HINV = 2.0e12

#: The fixed concentration the y3 offline miscentering tables were built
#: at (``nfw_dsigma_mis.hh``).
#:
#: NOTE: this is a *table-generation* constant, not a default you may
#: override. The tabulated
#: :math:`\log\Delta\Sigma(\log x, \log x_{\rm mis})` was integrated at
#: :math:`c = 4`; passing a different :math:`c` to the interpolator
#: rescales :math:`r_s` but leaves the tabulated *shape* at :math:`c = 4`,
#: which is wrong and silent. Use `child18` only where the profile is
#: computed, not interpolated.
Y3_FIXED_CONCENTRATION = 4.0

#: Linear critical overdensity for spherical collapse, used in the
#: :math:`M_\star` definition :math:`\sigma(R_\star, z) = \delta_c`.
#: Child et al. (2018) Sec. 4 uses this value; so does their Eq. for
#: :math:`\nu`.
DELTA_COLLAPSE = 1.686

#: Child et al.'s quoted anchors for the nonlinear mass, in
#: :math:`\log_{10}(M_\star/h^{-1}M_\odot)` at :math:`z = 0, 1, 2, 3`.
_M_STAR_ANCHORS_Z = (0.0, 1.0, 2.0, 3.0)
_M_STAR_ANCHORS_LOG10 = (12.5, 11.0, 9.5, 8.0)


def child18(m200c_hinv, z, m_star_hinv, fit: str = "individual_all"):
    r"""Child et al. (2018) Eq. 18: :math:`c_{200c}(M_{200c}/M_\star)`.

    .. math::
        c_{200c} = A\left[
            \left(\frac{M_{200c}/M_\star}{b}\right)^{m}
            \left(1 + \frac{M_{200c}/M_\star}{b}\right)^{-m} - 1
        \right] + c_0

    A power law in :math:`M/M_\star` below the threshold
    :math:`M_T = b M_\star`, flattening to the plateau :math:`c_0 \sim 3`
    above it. Coefficients: `CHILD18_TABLE1`.

    NOTE: domain of validity :math:`0 \le z \le 4`, and
    :math:`M_{200c} \gtrsim 2\times10^{11}\,M_\odot` -- the paper warns the
    fit "should not be naively extrapolated to masses smaller than those
    considered here". Cluster masses at :math:`z \lesssim 1` sit near
    :math:`M/M_\star \sim 10^2`, i.e. just *below* the threshold, on the
    sloped part of the relation, so the result is sensitive to
    :math:`M_\star` and the mass definition both.

    NOTE: :math:`M_{200c}`, not :math:`M_{200m}` -- see the module NOTE.

    Parameters
    ----------
    m200c_hinv : float or array-like
        :math:`M_{200c}` in :math:`h^{-1}M_\odot`.
    z : float or array-like
        Redshift. Enters *only* through ``m_star_hinv``; the relation has
        no explicit :math:`z` dependence, which is its whole point. Kept in
        the signature so a caller cannot silently pair a mass at one
        redshift with :math:`M_\star` at another, and validated against the
        stated domain.
    m_star_hinv : float or array-like
        Nonlinear mass :math:`M_\star(z)` in :math:`h^{-1}M_\odot`, from
        `m_star_hinv` (approximate) or `m_star_from_sigma` (exact).
    fit : str, optional
        Row of `CHILD18_TABLE1` (default: ``"individual_all"``, the y3
        choice).

    Returns
    -------
    np.ndarray
        :math:`c_{200c}`, dimensionless.
    """
    if fit not in CHILD18_TABLE1:
        raise ValueError(
            f"fit must be one of {sorted(CHILD18_TABLE1)}, got {fit!r}"
        )
    m200c_hinv = np.asarray(m200c_hinv, dtype=float)
    m_star = np.asarray(m_star_hinv, dtype=float)
    z = np.asarray(z, dtype=float)
    if np.any(m200c_hinv <= 0.0) or np.any(m_star <= 0.0):
        raise ValueError("masses must be positive")
    if np.any(z < 0.0) or np.any(z > 4.0):
        raise ValueError(f"Eq. 18 is calibrated for 0 <= z <= 4, got {z}")

    m, A, b, c0 = CHILD18_TABLE1[fit]
    # eq. 18, with mmb = (M/M_star)/b -- exactly the y3 grouping
    mmb = m200c_hinv / m_star / b
    return A * (mmb**m * (1.0 + mmb) ** (-m) - 1.0) + c0


def child18_powerlaw(m_hinv, z, fit: str = "individual_all"):
    r"""Child et al. (2018) Eq. 19: the power-law :math:`c`--:math:`M` fit.

    .. math::
        c_{200c} = A\,(1+z)^{d}\,M^{m}

    NOTE: domain of validity :math:`0 \le z \le 1` only -- the paper states
    the power law "is only valid for redshifts :math:`0 \le z \le 1`, where
    a power-law dependence on mass and redshift is a reasonable description
    of our results". Above :math:`z = 1` the true relation flattens and this
    form keeps falling.

    NOTE: :math:`M` is dimensionful inside a power law here, so :math:`A`
    carries the units implicitly; :math:`M` must be in
    :math:`h^{-1}M_\odot` for the tabulated :math:`A` to apply. This is why
    `child18` is preferable where :math:`M_\star` is available -- there the
    ratio is dimensionless.

    Parameters
    ----------
    m_hinv : float or array-like
        :math:`M_{200c}` in :math:`h^{-1}M_\odot`.
    z : float or array-like
        Redshift, :math:`0 \le z \le 1`.
    fit : str, optional
        Row of `CHILD18_TABLE2` (default: ``"individual_all"``).

    Returns
    -------
    np.ndarray
        :math:`c_{200c}`, dimensionless.
    """
    if fit not in CHILD18_TABLE2:
        raise ValueError(
            f"fit must be one of {sorted(CHILD18_TABLE2)}, got {fit!r}"
        )
    m_hinv = np.asarray(m_hinv, dtype=float)
    z = np.asarray(z, dtype=float)
    if np.any(m_hinv <= 0.0):
        raise ValueError("masses must be positive")
    if np.any(z < 0.0) or np.any(z > 1.0):
        raise ValueError(f"Eq. 19 is calibrated for 0 <= z <= 1, got {z}")

    A, d, m = CHILD18_TABLE2[fit]
    # eq. 19
    return A * (1.0 + z) ** d * m_hinv**m


def duffy08(m_hinv, z, mass_def: str = "200m"):
    r"""Duffy et al. (2008) Table 1: :math:`c(M, z)` as a power law.

    .. math::
        c = A\left(\frac{M}{2\times10^{12}\,h^{-1}M_\odot}\right)^{B}
            (1+z)^{C}

    NOTE: domain of validity :math:`0 \le z \le 2`, full (relaxed *and*
    unrelaxed) sample, WMAP-5 cosmology. Being WMAP-5, its
    :math:`\sigma_8 = 0.796` is low, and Child et al. show it sits below
    their relation at cluster scales.

    NOTE: the default here is ``"200m"``, matching
    `clenspy.halo.NfwProfile`'s mass definition -- **not** `pyccl`'s
    default, which is ``"200c"``. Comparing the two at the same mass
    without matching ``mass_def`` differs by tens of percent, since
    :math:`A` runs 5.71 to 10.14 across the three rows.

    Parameters
    ----------
    m_hinv : float or array-like
        Halo mass in :math:`h^{-1}M_\odot`, in the definition named by
        ``mass_def``.
    z : float or array-like
        Redshift, :math:`0 \le z \le 2`.
    mass_def : str, optional
        One of ``"vir"``, ``"200m"``, ``"200c"`` (default: ``"200m"``).

    Returns
    -------
    np.ndarray
        Concentration, dimensionless.
    """
    if mass_def not in DUFFY08_TABLE1:
        raise ValueError(
            f"mass_def must be one of {sorted(DUFFY08_TABLE1)}, "
            f"got {mass_def!r}"
        )
    m_hinv = np.asarray(m_hinv, dtype=float)
    z = np.asarray(z, dtype=float)
    if np.any(m_hinv <= 0.0):
        raise ValueError("masses must be positive")

    A, B, C = DUFFY08_TABLE1[mass_def]
    return A * (m_hinv / DUFFY08_PIVOT_HINV) ** B * (1.0 + z) ** C


def m_star_hinv(z):
    r"""The nonlinear mass :math:`M_\star(z)`, interpolated from Child et al.

    .. math::
        \log_{10}\!\left(M_\star/h^{-1}M_\odot\right) \simeq 12.5 - 1.5\,z

    NOTE: **this is the named approximation.** Child et al. quote
    :math:`\log_{10}(M_\star/h^{-1}M_\odot) = 12.5, 11, 9.5, 8` at
    :math:`z = 0, 1, 2, 3` for their WMAP-7 cosmology; those four anchors
    are exactly linear in :math:`z` with slope :math:`-1.5`, so the line
    above reproduces all four to the digits given. It is *their* cosmology,
    though: :math:`M_\star` "depends weakly on cosmology", but Eq. 18 is
    sensitive to it at cluster masses. For a different cosmology, or for
    better than ~0.1 dex, use `m_star_from_sigma`.

    NOTE: ``y3_cluster_cpp`` carries a different approximation,
    :math:`M_\star \approx 10^{14.76 \times 0.808^{z}}`, commented out in
    favour of a tabulated :math:`M_\star` from its own ``mstar.py``. It is
    rightly commented out: it gives :math:`\log_{10} M_\star = 14.76` at
    :math:`z = 0` against Child's 12.5, a factor of 180, and only converges
    onto the anchors for :math:`z \gtrsim 2`. Recorded here so nobody
    revives it for the low-redshift cluster regime, where it is worst.

    Parameters
    ----------
    z : float or array-like
        Redshift. Extrapolates linearly outside :math:`[0, 3]`.

    Returns
    -------
    np.ndarray
        :math:`M_\star` in :math:`h^{-1}M_\odot`.
    """
    # the four anchors are exactly collinear, so the line *is* the
    # interpolant -- no table lookup, and it extrapolates the same slope
    return 10.0 ** (12.5 - 1.5 * np.asarray(z, dtype=float))


def m_star_from_sigma(sigma_of_r, rho_m_comoving, r_bracket=(0.01, 50.0)):
    r"""The exact :math:`M_\star`: solve :math:`\sigma(R_\star, z) = \delta_c`.

    Child et al. (2018) Sec. 4 defines the nonlinear mass by

    .. math::
        \sigma(R_\star, z) = \delta_c,
        \qquad
        M_\star = \frac{4\pi}{3}\,\bar\rho_m\,R_\star^{3}

    with :math:`\delta_c = 1.686` and :math:`\sigma(R,z)` the top-hat
    variance of the linear power spectrum. This is the definition; use it
    once step 13's :math:`\sigma(R,z)` grid exists, in place of
    `m_star_hinv`.

    NOTE: **comoving.** The paper writes the prefactor as
    :math:`\rho_c(z)\,\omega_m(z)`, which reads as the *physical* mean
    matter density and would carry an extra :math:`(1+z)^3`. It cannot be:
    :math:`\sigma(R,z)` is defined on a **comoving** top-hat radius, so the
    enclosed Lagrangian mass must use the **comoving** mean matter density
    :math:`\Omega_{m,0}\rho_{c,0}`, which is redshift-independent. Passing
    :math:`\rho_m(z)` instead inflates :math:`M_\star` by
    :math:`(1+z)^3` -- a factor 2 by :math:`z = 0.26` -- and, through
    Eq. 18's sloped branch, biases :math:`c` low. Same trap as
    `clenspy.halo.TwoHaloTerm`.

    Parameters
    ----------
    sigma_of_r : callable
        ``sigma_of_r(R) -> float``, the top-hat variance amplitude at
        comoving radius ``R``, already evaluated at the redshift of
        interest. Must be monotonically decreasing in ``R``.
    rho_m_comoving : float
        Comoving mean matter density :math:`\Omega_{m,0}\rho_{c,0}`, in
        mass units per comoving volume cubed. :math:`M_\star` comes back in
        whatever mass unit this carries -- pass
        :math:`h^{-1}M_\odot/({\rm Mpc}/h)^3` to get
        :math:`h^{-1}M_\odot`, matching `child18`.
    r_bracket : tuple of float, optional
        Bracket for the root, in the same length unit as ``sigma_of_r``
        takes (default: 0.01 to 50).

    Returns
    -------
    tuple of float
        ``(m_star, r_star)``.
    """
    lo, hi = r_bracket
    f_lo, f_hi = (sigma_of_r(lo) - DELTA_COLLAPSE,
                  sigma_of_r(hi) - DELTA_COLLAPSE)
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"sigma(R) - delta_c does not change sign on {r_bracket}: "
            f"sigma({lo}) = {f_lo + DELTA_COLLAPSE:.4f}, "
            f"sigma({hi}) = {f_hi + DELTA_COLLAPSE:.4f}. At high z, "
            "sigma may not reach delta_c anywhere -- M_star is then below "
            "the grid."
        )
    r_star = brentq(lambda r: sigma_of_r(r) - DELTA_COLLAPSE, lo, hi)
    return (4.0 / 3.0) * np.pi * rho_m_comoving * r_star**3, r_star


def delta_c(c):
    r"""The NFW characteristic overdensity :math:`\delta_c(c)`.

    .. math::
        \delta_c = \frac{200\,c^3/3}{\ln(1+c) - c/(1+c)}

    So that :math:`\rho_s = \delta_c\,\rho_{\rm ref}`. Transcribed from
    ``y3_cluster_cpp/src/models/nfw_dsigma_mis.hh``, which spells it
    inline. The 200 is the overdensity :math:`\Delta`; this form is
    :math:`\Delta`-specific, not general.

    NOTE: named ``delta_c`` after the y3 source, and it collides with the
    *other* :math:`\delta_c` in this module -- `DELTA_COLLAPSE`, the 1.686
    of spherical collapse. Unrelated quantities, unfortunately spelled the
    same in the literature. This one is :math:`O(10^4)`.

    Parameters
    ----------
    c : float or array-like
        Concentration.

    Returns
    -------
    np.ndarray
        :math:`\delta_c`, dimensionless.
    """
    c = np.asarray(c, dtype=float)
    if np.any(c <= 0.0):
        raise ValueError("concentration must be positive")
    return (200.0 * c**3 / 3.0) / (np.log(1.0 + c) - c / (1.0 + c))


def scatter(c):
    r"""Halo-to-halo concentration scatter, :math:`\sigma_c = c_{200c}/3`.

    The note under both Child et al. (2018) tables: "use Equation ... with
    variance :math:`\sigma_c = c_{200c}/3`."

    NOTE: this is a scatter in :math:`c`, not in :math:`\ln c`, and it is
    33% -- large enough that the stacked profile is *not* the profile at
    :math:`\langle c\rangle`. A stacked analysis wanting one number should
    use the ``stacked_nfw`` row of `CHILD18_TABLE1`, which was fit to
    stacks and absorbs the averaging.
    """
    return np.asarray(c, dtype=float) / 3.0


if __name__ == "__main__":
    # the claim that licenses the closed form in m_star_hinv: Child et al.'s
    # four quoted anchors are collinear, so the line is not a fit to them
    line = [12.5 - 1.5 * z for z in _M_STAR_ANCHORS_Z]
    assert line == list(_M_STAR_ANCHORS_LOG10), (line, _M_STAR_ANCHORS_LOG10)
    print(f"M_star anchors {_M_STAR_ANCHORS_LOG10} at z = "
          f"{_M_STAR_ANCHORS_Z}: exactly 12.5 - 1.5 z.\n")

    print("Child et al. (2018) Eq. 18 + Table 1, with M_star from the")
    print("paper's own anchors (log10 M_star = 12.5 - 1.5 z):\n")
    print(f"{'z':>6s} {'log10 M*':>9s} {'M200c/M*':>10s} "
          f"{'c (eq.18)':>10s} {'c (eq.19)':>10s} {'c Duffy08':>10s}")
    m200c_hinv = 1.0e14  # h^-1 Msun, the y3 comment's test mass
    for z in (0.01, 0.33, 0.66, 0.99):
        ms = m_star_hinv(z)
        c18 = child18(m200c_hinv, z, ms)
        c19 = child18_powerlaw(m200c_hinv, z)
        cd8 = duffy08(m200c_hinv, z, mass_def="200c")
        print(f"{z:6.2f} {np.log10(ms):9.3f} {m200c_hinv / ms:10.1f} "
              f"{c18:10.3f} {c19:10.3f} {cd8:10.3f}")

    # The y3 source records its own values at the same mass and redshifts,
    # as inline comments in c_from_m200. They are NOT reproduced above,
    # and the reason is named rather than papered over.
    print("\ny3_cluster_cpp records, at log10 M200m = 14.0:")
    print("     z:  0.01   0.33   0.66   0.99")
    print("     c:  5.02   4.60   4.20   3.90")
    print("Two differences, both stated in massconcen.py itself:")
    print("  1. that mass is M200m, converted to M200c via Ragagnin+2021")
    print("     before Eq. 18 (M200c < M200m, so c comes out higher);")
    print("  2. its M_star is tabulated from Child Eq. 13, not the")
    print("     12.5 - 1.5z anchor line used here.")
    print("Neither conversion is vendored in clenspy -- see module NOTE.")

    print("\nThe three mass definitions of Duffy et al. Table 1, at "
          "M = 1e14 h^-1 Msun, z = 0.3:")
    for md in ("vir", "200m", "200c"):
        print(f"  {md:>5s}: c = {duffy08(1e14, 0.3, mass_def=md).item():.3f}")
    print("  <- spread is the mass definition, not the fit: never compare")
    print("     across rows at fixed M.")

    print("\nThe four Table 1 rows at M200c = 1e14 h^-1 Msun, z = 0.3:")
    ms = m_star_hinv(0.3)
    for fit in CHILD18_TABLE1:
        c = child18(1e14, 0.3, ms, fit=fit).item()
        print(f"  {fit:>19s}: c = {c:.3f}   sigma_c = {scatter(c):.3f}")

    print(f"\nThe y3 table-generation constant: c = {Y3_FIXED_CONCENTRATION}, "
          f"delta_c = {delta_c(Y3_FIXED_CONCENTRATION):.2f}")
    print(f"Child18 at 1e14 h^-1 Msun, z=0.3 instead gives "
          f"delta_c = {delta_c(child18(1e14, 0.3, ms)).item():.2f} "
          f"-- a {delta_c(child18(1e14, 0.3, ms)).item() / delta_c(4.0) - 1:+.1%} "
          "shift in rho_s.")

    # the exact M_star route, on a power-law sigma(R) with a known root
    print("\nm_star_from_sigma on sigma(R) = (R / 8 Mpc/h)^-0.6 * 0.82:")
    ms_exact, r_star = m_star_from_sigma(
        lambda r: 0.82 * (r / 8.0) ** -0.6,
        rho_m_comoving=0.286 * 2.775e11,   # Omega_m,0 * rho_c,0, h-free-in-h
    )
    print(f"  R_star = {r_star:.4f} Mpc/h  ->  "
          f"log10 M_star = {np.log10(ms_exact):.3f}")
    print(f"  (the 12.5 - 1.5z line gives {np.log10(m_star_hinv(0.0)):.3f} "
          "at z = 0; the toy sigma above is not a real P(k))")
