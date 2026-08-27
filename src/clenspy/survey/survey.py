r"""The survey: where it looked, what its sources look like, how it binned.

One module, three kinds of thing, and the distinction between them is the
point:

**The footprint fits are code.** :math:`\Omega(z)` is a polynomial
transcribed coefficient-by-coefficient from ``y3_cluster_cpp``. A fit is
not a number a user should retype -- getting one digit wrong is a silent
normalisation error -- so it lives here and only the *choice* of which fit
to use is configurable.

**The bins and the source properties are configuration.** Richness edges,
redshift edges, :math:`\sigma_z`, :math:`\sigma_\gamma`,
:math:`n_{\rm src}`, the :math:`p(z_s)` parameters: these are analysis
choices. They live in ``clenspy/configs/<survey>.json`` with a
``_provenance`` string on every group, and `Survey.from_config` reads them.
Changing an analysis means editing a config, not editing this file.

**The `Survey` class is the source population.** :math:`p(z_s)`,
:math:`\sigma_\gamma`, :math:`n_{\rm src}`, and the redshift support --
exactly the four things a lensing weight asks for. The name follows the
exemplar, ``clens/util/survey.py::Survey``, which holds the same four and
no footprint.

.. math::
    \langle N_{ij}\rangle = \int\! dz \int\! d\ln M
      \int\! d\lambda^{\rm tr}\;
      n(M, z)\, \frac{dV}{d\Omega\, dz}\, \boldsymbol{\Omega(z)}\,
      K_j(z)\, \mathcal{S}_i(\lambda^{\rm tr}, z)\,
      P(\lambda^{\rm tr} \mid M, z)

NOTE: :math:`\Omega(z)` appears in the **counts** and **cancels** in the
shear projection -- it divides out of the surface density, and the exact
C++ core hard-excludes it there. Folding the footprint into a lensing
weight is a silent normalisation error, so a shared weight builder must
take :math:`\Omega(z)` as an explicit per-observable argument rather than
reading it off a survey object. That is why `Survey` does **not** carry it.
See ``docs/refactor-plan.md`` errata E.2.

NOTE: units. :math:`\Omega(z)` is in **steradians** (rad^2), matching the
C++; `deg2` converts. :math:`p(z_s)` is a density in redshift, integrating
to 1, so it carries :math:`1/z`. :math:`\sigma_\gamma` is dimensionless.
:math:`n_{\rm src}` is a **sky surface density in arcmin^-2** -- the one
non-Mpc unit in the package, which is why it is named ``n_src_arcmin``.
Richness and redshift are dimensionless; :math:`\sigma_z` is in redshift
units.

NOTE: this layer knows nothing about a halo, a profile, or a lens redshift.
:math:`\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)` is built *from* a
`Survey` but is lens-source geometry, so it belongs to `clenspy.kernels`
(errata E.1).

NOTE: the DES footprint header names its coefficient arrays ``SDSS_fit``,
``SDSS_fit2``, ``SDSS_fit3``. That is a copy-paste artifact -- the C++
flags it itself ("A+ naming SDSS_fit for DES =P") -- and the numbers are
DES, not SDSS. Renamed on transcription.

Coverage is uneven on purpose: `clenspy` transcribes survey definitions
rather than estimating them, so a survey has only the pieces a source
records. SDSS has an :math:`\Omega(z)` fit and no config, so
``Survey.from_config("sdss")`` raises.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..utils.binning import BinCollection

__all__ = [
    "CONFIG_DIR",
    "Survey",
    "available_configs",
    "deg2",
    "load_config",
    "omega_des_y1",
    "omega_des_y3",
    "omega_sdss",
    "omega_y3xspt",
    "survey_area",
    "survey_bins",
]

#: Where the per-analysis configs live. Packaged with the wheel.
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


# ==========================================================================
# Omega(z): the footprint fits. Code, not configuration.
# ==========================================================================

#: rad^2 per deg^2, for reporting a footprint in the units papers quote.
_RAD2_PER_DEG2 = (np.pi / 180.0) ** 2

# -- DES Y1: three pieces, breaking at z = 0.504 and z = 0.700 ------------
#
# NOTE: this is the fit the C++ calls ``OMEGA_Z_DES``, and it is **DES Y1**,
# not Y3: it gives 1494 deg^2 at z = 0.2, against the published Y1
# footprint of 1437 deg^2. The y3 repo's own python transcription
# (``test/make_hod_norm_impact.py``) names it ``omega_z_des_y1``, which
# settles it.
_DES_Y1_LOW = (0.0, 0.0, 0.0, -0.00262353, 0.01940118, 0.45133063)
_DES_Y1_MID = (1.33647377e4, 1.35291046e3, -1.26204891e2,
               -2.83454918e1, -2.26465905, 3.84958753e-1)
_DES_Y1_HIGH = (0.0, 0.0, -1.88101967, 4.8071839, -4.11424324, 1.18196785)

#: Where the DES Y1 pieces meet. The fit is **discontinuous** at both: by
#: -0.37% at 0.504 and by -30.6% at 0.700 (58.3 -> 40.5 deg^2). Both jumps
#: are in the C++ too; they sit outside the analysis range below.
_DES_Y1_BREAKS = (0.504, 0.700)

#: Redshift range the DES Y1 cluster analysis actually uses -- the bin
#: edges are 0.20/0.35/0.50/0.65. Outside it the polynomial is an
#: extrapolation: it crosses **zero at z = 0.9378** and is negative above,
#: which is why `omega_des_y1` clamps at zero.
DES_Y1_Z_RANGE = (0.20, 0.65)

# -- SDSS: one degree-11 fit in (z - 0.2) --------------------------------
_SDSS = (-1.14293122e05, 5.96846869e04, 9.24239180e03, -2.23118813e03,
         -4.52580713e03, 1.18404878e03, 1.27951911e02, -5.05716847e01,
         1.01744577e00, -3.11253383e-01, 5.48481084e-03, 3.12629987e00)

#: Redshift range the SDSS redMaPPer cluster analysis uses. The fit peaks
#: at 10263 deg^2 at z = 0.2 and is smooth across this range; a degree-11
#: polynomial diverges fast outside it, so `omega_sdss` clamps at zero.
SDSS_Z_RANGE = (0.10, 0.33)

#: DES Y3, as a flat effective area [rad^2]: the **gold** footprint,
#: 4143 deg^2. This is the area of the data.
#:
#: NOTE: **no z-dependent DES Y3 fit exists.** There is none in
#: ``y3_cluster_cpp``, so rather than invent one this is a constant. The
#: precedent is the repo's own ``OMEGA_Z_Y3XSPT``, which does exactly this
#: for Y3 x SPT ("These fits will need to be computed by Eli") with a
#: constant 2500 deg^2. Replace `omega_des_y3` the moment a real fit lands.
#:
#: NOTE: ``cluster-lensing-cov/configs/des_y3.json`` carries 5000 deg^2
#: instead. That file is a **forecast** -- its own provenance says the
#: counts are "DES Y1 counts scaled by 5000/1437" -- so the two numbers
#: describe different things and must not be reconciled. 4143 is the gold
#: footprint; use it for the data. A forecast that wants 5000 should say so
#: at the call site.
_DES_Y3_AREA_DEG2 = 4143.0

#: Y3 x SPT-SZ, as the C++ has it: a flat 2500 deg^2 placeholder.
_Y3XSPT_AREA_RAD2 = 0.7615435494667714


def deg2(omega_rad2):
    r"""Convert :math:`\Omega` from rad^2 to deg^2.

    The one unit conversion in this module, applied by the caller when it
    wants to compare a footprint against a published number.
    """
    return np.asarray(omega_rad2, dtype=float) / _RAD2_PER_DEG2


def omega_des_y1(z):
    r"""DES Y1 effective survey area :math:`\Omega(z)` [rad^2].

    Three polynomial pieces (``y3_cluster::OMEGA_Z_DES``):

    .. math::
        \Omega(z) = \begin{cases}
          p_1(z),        & z < 0.504 \\
          p_2(z - 0.6),  & 0.504 \le z < 0.700 \\
          p_3(z),        & z \ge 0.700
        \end{cases}

    NOTE: **valid on** :math:`z \in [0.20, 0.65]` (`DES_Y1_Z_RANGE`), the
    range the analysis bins span. Outside it the fit is an extrapolation
    with two known pathologies, both present in the C++: it is
    discontinuous at each break (-0.37% at 0.504, -30.6% at 0.700), and it
    crosses zero at :math:`z = 0.9378`. The result is clamped at zero so an
    integral that strays past that cannot pick up negative area -- the C++
    does not clamp, so beyond z = 0.94 the two differ deliberately.

    Parameters
    ----------
    z : float or array-like
        Cluster (true) redshift.

    Returns
    -------
    np.ndarray
        :math:`\Omega(z)` in rad^2, never negative.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    lo, hi = _DES_Y1_BREAKS
    out = np.where(
        z < lo,
        np.polyval(_DES_Y1_LOW, z),
        np.where(z < hi,
                 np.polyval(_DES_Y1_MID, z - 0.6),
                 np.polyval(_DES_Y1_HIGH, z)),
    )
    return np.maximum(out, 0.0)  # the fit goes negative above z = 0.9378


def omega_des_y3(z):
    r"""DES Y3 effective survey area :math:`\Omega(z)` [rad^2].

    NOTE: **flat in z** at the published 4143 deg^2. No redshift-dependent
    Y3 fit has been computed -- see the note on ``_DES_Y3_AREA_DEG2``. This
    is a stated approximation, not a transcription, and it is the one
    function in this module that is not taken from a source. For counts it
    biases the redshift *shape* of :math:`\langle N_{ij}\rangle`, not its
    normalisation; do not use it to compare bins at different z until a real
    fit replaces it.

    Parameters
    ----------
    z : float or array-like
        Cluster redshift. Used only for its shape.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return np.full(z.shape, _DES_Y3_AREA_DEG2 * _RAD2_PER_DEG2)


def omega_sdss(z):
    r"""SDSS effective survey area :math:`\Omega(z)` [rad^2].

    A single degree-11 polynomial in :math:`(z - 0.2)`
    (``y3_cluster::OMEGA_Z_SDSS``), peaking at 10263 deg^2 at
    :math:`z = 0.2`.

    NOTE: **valid on** :math:`z \in [0.10, 0.33]` (`SDSS_Z_RANGE`). A
    degree-11 fit diverges quickly outside the range it was fit on, so the
    result is clamped at zero.

    Parameters
    ----------
    z : float or array-like
        Cluster (true) redshift.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return np.maximum(np.polyval(_SDSS, z - 0.2), 0.0)


def omega_y3xspt(z):
    r"""DES Y3 x SPT-SZ area :math:`\Omega(z)` [rad^2]: flat 2500 deg^2.

    NOTE: a placeholder in the C++ as well
    (``y3_cluster::OMEGA_Z_Y3XSPT``), which carries the comment "These fits
    will need to be computed by Eli (+Lindsey)". Transcribed as-is,
    including the constant, so that swapping it for a real fit is a
    one-function change here.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return np.full(z.shape, _Y3XSPT_AREA_RAD2)


#: The fits, addressed by the name a config's ``omega_z`` field carries.
_OMEGA = {
    "des_y1": omega_des_y1,
    "des_y3": omega_des_y3,
    "sdss": omega_sdss,
    "y3xspt": omega_y3xspt,
}


def survey_area(name):
    r"""The :math:`\Omega(z)` callable for ``name``.

    Parameters
    ----------
    name : {"des_y1", "des_y3", "sdss", "y3xspt"}
        Fit identifier, case-insensitive. This is what a config's
        ``omega_z`` field names.

    Returns
    -------
    callable
        ``omega(z) -> np.ndarray`` in rad^2.

    Raises
    ------
    KeyError
        If ``name`` is unknown. The known ones are listed in the message,
        because a typo here silently changes a normalisation.
    """
    key = str(name).lower()
    try:
        return _OMEGA[key]
    except KeyError:
        raise KeyError(
            f"unknown Omega(z) fit {name!r}; have {sorted(_OMEGA)}"
        ) from None


# ==========================================================================
# The Survey: the source population. Its numbers come from a config.
# ==========================================================================

#: Nodes used to normalise :math:`p(z_s)` by trapezoid.
#:
#: NOTE: 601 nodes over the support gives a normalisation stable to better
#: than 1e-6 for every shape here -- the integrand is smooth and
#: single-peaked, so the trapezoid error falls as h^2 and 601 nodes over
#: [0, 3] is dz = 0.005 against a peak width of ~0.5. The exemplar used
#: dz = 0.01 on the same integral.
_N_NORM_NODES = 601


def available_configs():
    """The analysis configs shipped with the package, by name."""
    return sorted(p.stem for p in CONFIG_DIR.glob("*.json"))


def load_config(name):
    """Read ``clenspy/configs/<name>.json``.

    Parameters
    ----------
    name : str
        Config stem, case-insensitive, e.g. ``"des_y1"``.

    Raises
    ------
    FileNotFoundError
        If there is no such config. `clenspy` does not fabricate analysis
        choices -- a bin edge is an integration limit, so a guessed one is
        a wrong number that looks right. The message lists what exists.
    """
    path = CONFIG_DIR / f"{str(name).lower()}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no analysis config for {name!r} at {path}. Available: "
            f"{available_configs()}. clenspy transcribes analysis choices "
            "rather than reconstructing them -- add a config file rather "
            "than hardcoding edges at a call site."
        )
    with open(path) as fh:
        return json.load(fh)


def survey_bins(name_or_config) -> BinCollection:
    r"""The :math:`(\Delta\lambda_i, \Delta z_j)` grid from a config.

    Parameters
    ----------
    name_or_config : str or dict
        A config name, or an already-loaded config dict.

    Returns
    -------
    BinCollection
        ``n_lam * n_z`` bins in richness-outer order.

    NOTE: bins are **observed** richness and **observed** photometric
    redshift. The photo-z kernel that maps a true redshift into
    :math:`\Delta z_j` uses :math:`\sigma_z`, which is why the scatter
    travels on the bin (`~clenspy.utils.RichnessBin`) and not on the
    `Survey`.
    """
    cfg = (load_config(name_or_config) if isinstance(name_or_config, str)
           else name_or_config)
    try:
        b = cfg["bins"]
    except KeyError:
        raise KeyError(
            f"config {cfg.get('name', '?')!r} has no 'bins' section"
        ) from None
    return BinCollection.from_edges(
        lam_edges=b["lam_edges"], z_edges=b["z_edges"], sigma_z=b["sigma_z"]
    )


class Survey:
    r"""A survey's source population: :math:`p(z_s)` and its noise.

    implements the Survey protocol

    Build one with `from_config` for a shipped analysis, or with `smail`,
    `top_hat` or `tabulated` to specify a shape directly.

    NOTE: units are those of the module docstring; in particular
    ``n_src_arcmin`` is arcmin^-2, not Mpc^-2.

    NOTE: carries **no** :math:`\Omega(z)` and **no** bin grid. Those are
    reached through `survey_area` and `survey_bins`, because the counts need
    the footprint and the shear does not -- see the module NOTE on E.2.

    Parameters
    ----------
    pz_shape : callable
        Unnormalised :math:`p(z_s)`, vectorised over ``z``. Stored
        verbatim; the normalisation is computed here and applied in
        `pz_src`, so the caller need not normalise.
    sigma_gamma : float
        Per-galaxy shape noise (dimensionless).
    n_src_arcmin : float
        Effective source surface density [arcmin^-2].
    zs_min, zs_max : float
        Support of :math:`p(z_s)`. Outside it `pz_src` returns zero.
    name : str, optional
        Label for `__repr__` and for output paths.

    Attributes
    ----------
    norm : float
        :math:`\int p_{\rm shape}\,dz` over the support, computed once.
    """

    def __init__(
        self,
        pz_shape,
        sigma_gamma: float,
        n_src_arcmin: float,
        zs_min: float = 0.0,
        zs_max: float = 3.0,
        name: str = "custom",
    ) -> None:
        if not zs_max > zs_min:
            raise ValueError(
                f"source redshift range must increase: [{zs_min}, {zs_max}]"
            )
        if sigma_gamma <= 0.0:
            raise ValueError(f"sigma_gamma must be positive: {sigma_gamma}")
        if n_src_arcmin <= 0.0:
            raise ValueError(f"n_src_arcmin must be positive: {n_src_arcmin}")

        # store the collaborator verbatim
        self.pz_shape = pz_shape
        self.sigma_gamma = float(sigma_gamma)
        self.n_src_arcmin = float(n_src_arcmin)
        self.zs_min = float(zs_min)
        self.zs_max = float(zs_max)
        self.name = name

        nodes = np.linspace(self.zs_min, self.zs_max, _N_NORM_NODES)
        self.norm = float(np.trapezoid(pz_shape(nodes), x=nodes))
        if not self.norm > 0.0:
            raise ValueError(
                f"p(z_s) integrates to {self.norm} on "
                f"[{self.zs_min}, {self.zs_max}]; it must be positive"
            )

    # -- the protocol ----------------------------------------------------

    def pz_src(self, z):
        r"""Normalised source redshift density :math:`p(z_s)` [1/z].

        Zero outside :math:`[z_s^{\min}, z_s^{\max}]`, so a caller may
        integrate over any wider range without picking up tail weight the
        catalogue does not have.
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        inside = (z >= self.zs_min) & (z <= self.zs_max)
        # Evaluate the shape only on the support. Masking afterwards is not
        # enough: the eq. 14 form is z**m with fractional m, so a negative
        # query would produce a nan (and a RuntimeWarning) before the mask
        # could discard it.
        out = np.zeros(z.shape, dtype=float)
        if np.any(inside):
            out[inside] = np.asarray(
                self.pz_shape(z[inside]), dtype=float) / self.norm
        return out

    def zs_range(self) -> tuple[float, float]:
        """The support, as the pair a quadrature wants."""
        return (self.zs_min, self.zs_max)

    # -- from a config ---------------------------------------------------

    @classmethod
    def from_config(cls, name_or_config) -> "Survey":
        r"""Build from ``clenspy/configs/<name>.json``.

        The ``sources`` section names a ``pz_model`` -- ``"smail"``,
        ``"top_hat"`` or ``"tabulated"`` -- and supplies that shape's
        parameters plus ``sigma_gamma`` and ``n_src_arcmin``.

        Parameters
        ----------
        name_or_config : str or dict
            A config name, or an already-loaded config dict.

        Raises
        ------
        FileNotFoundError
            If no config exists for ``name``. See `load_config`.
        KeyError
            If the config has no ``sources`` section, or names a
            ``pz_model`` this class does not implement.
        """
        cfg = (load_config(name_or_config) if isinstance(name_or_config, str)
               else name_or_config)
        try:
            src = cfg["sources"]
        except KeyError:
            raise KeyError(
                f"config {cfg.get('name', '?')!r} has no 'sources' section"
            ) from None

        # ignore the _provenance / _note_* keys the configs carry
        kw = {k: v for k, v in src.items() if not k.startswith("_")}
        model = kw.pop("pz_model", "smail")
        kw.setdefault("name", cfg.get("name", str(name_or_config)))

        builders = {"smail": cls.smail, "top_hat": cls.top_hat,
                    "tabulated": cls.tabulated}
        try:
            build = builders[model]
        except KeyError:
            raise KeyError(
                f"unknown pz_model {model!r} in config "
                f"{cfg.get('name', '?')!r}; have {sorted(builders)}"
            ) from None
        return build(**kw)

    # -- the three shapes ------------------------------------------------

    @classmethod
    def smail(
        cls,
        z_star: float = 0.74,
        m: float = 1.68,
        beta: float = 2.33,
        sigma_gamma: float = 0.3,
        n_src_arcmin: float = 6.28,
        zs_min: float = 0.0,
        zs_max: float = 3.0,
        name: str = "smail",
    ) -> "Survey":
        r"""The Rozo et al. (2011) eq. 14 shape.

        .. math::
            p(z_s) \propto z_s^{m}\,
                \exp\!\left[-(z_s/z_\star)^{\beta}\right]

        Also called the Smail form, and "whale-shaped" in the exemplar.

        NOTE: :math:`m` and :math:`\beta` are shape parameters of the
        *source* distribution and have nothing to do with the HOD
        :math:`\alpha` or the Einasto index -- see ``docs/notation.md`` for
        the collision table.
        """
        return cls(
            pz_shape=lambda z: z**m * np.exp(-((z / z_star) ** beta)),
            sigma_gamma=sigma_gamma,
            n_src_arcmin=n_src_arcmin,
            zs_min=zs_min,
            zs_max=zs_max,
            name=name,
        )

    @classmethod
    def top_hat(
        cls,
        zs_min: float,
        zs_max: float,
        sigma_gamma: float = 0.3,
        n_src_arcmin: float = 6.28,
        name: str = "top_hat",
    ) -> "Survey":
        """All sources spread uniformly over ``[zs_min, zs_max]``.

        Useful as a limit: a narrow top-hat approaches a single source
        plane, which is the case an analytic check can be written for.
        """
        return cls(
            pz_shape=lambda z: np.ones_like(np.asarray(z, dtype=float)),
            sigma_gamma=sigma_gamma,
            n_src_arcmin=n_src_arcmin,
            zs_min=zs_min,
            zs_max=zs_max,
            name=name,
        )

    @classmethod
    def tabulated(
        cls,
        z,
        dndz,
        sigma_gamma: float = 0.3,
        n_src_arcmin: float = 6.28,
        name: str = "tabulated",
    ) -> "Survey":
        r"""A measured :math:`dn/dz`, linearly interpolated.

        The support is taken from ``z``, so the table's own edges bound the
        integral -- no extrapolation. ``dndz`` need not be normalised.

        Parameters
        ----------
        z, dndz : array-like
            The tabulated distribution. ``z`` must be increasing.
        """
        z = np.asarray(z, dtype=float)
        dndz = np.asarray(dndz, dtype=float)
        if z.ndim != 1 or z.size < 2:
            raise ValueError("z must be a 1-D array with at least two nodes")
        if np.any(np.diff(z) <= 0):
            raise ValueError("z must be strictly increasing")
        if dndz.shape != z.shape:
            raise ValueError(f"dndz has shape {dndz.shape}, z has {z.shape}")
        if np.any(dndz < 0):
            raise ValueError("dndz must be non-negative")
        return cls(
            # left/right = 0 so the table's edges really do bound it
            pz_shape=lambda zq: np.interp(zq, z, dndz, left=0.0, right=0.0),
            sigma_gamma=sigma_gamma,
            n_src_arcmin=n_src_arcmin,
            zs_min=float(z[0]),
            zs_max=float(z[-1]),
            name=name,
        )

    def __repr__(self) -> str:
        return (
            f"Survey({self.name!r}, sigma_gamma={self.sigma_gamma:g}, "
            f"n_src_arcmin={self.n_src_arcmin:g}, "
            f"zs=[{self.zs_min:g}, {self.zs_max:g}])"
        )


if __name__ == "__main__":
    print(f"analysis configs: {available_configs()}\n")

    print("effective survey area Omega(z), in deg^2\n")
    print(f"{'z':>6s}  {'DES Y1':>10s}  {'DES Y3':>10s}  {'SDSS':>10s}"
          f"  {'Y3xSPT':>10s}")
    for z in (0.05, 0.10, 0.20, 0.30, 0.35, 0.50, 0.55, 0.65, 0.70, 0.95):
        row = [float(deg2(f(z)).item()) for f in
               (omega_des_y1, omega_des_y3, omega_sdss, omega_y3xspt)]
        print(f"{z:6.2f}  " + "  ".join(f"{v:10.1f}" for v in row))

    print("\nthe DES Y1 fit's two seams and its zero crossing:")
    for zb in _DES_Y1_BREAKS:
        below = float(deg2(omega_des_y1(zb - 1e-6)).item())
        above = float(deg2(omega_des_y1(zb)).item())
        print(f"  z = {zb:.3f}: {below:9.1f} -> {above:9.1f} deg^2"
              f"   ({100 * (above / below - 1):+.2f}%)")
    print(f"  clamped to zero above z = 0.9378: "
          f"{float(deg2(omega_des_y1(1.2)).item()):.1f} deg^2")

    for cfg_name in available_configs():
        cfg = load_config(cfg_name)
        su = Survey.from_config(cfg)
        bins = survey_bins(cfg)
        omega = survey_area(cfg["omega_z"])
        print(f"\n{cfg_name}: {su}")
        print(f"  Omega(z=0.4) = {float(deg2(omega(0.4)).item()):.1f} deg^2")
        print(f"  {len(bins)} bins, {bins.n_lam} richness x {bins.n_z} z; "
              f"bins.at(3, 0) = {bins.at(3, 0)}")
        z = np.array([0.3, 0.8, 1.5])
        print(f"  p(z_s) at {z} = "
              + " ".join(f"{v:.4f}" for v in su.pz_src(z)))

    try:
        Survey.from_config("sdss")
    except FileNotFoundError as exc:
        print(f"\nSDSS refused, as designed:\n  {str(exc)[:76]}...")
