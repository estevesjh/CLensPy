r"""The source population: :math:`p(z_s)`, :math:`\sigma_\gamma`,
:math:`n_{\rm src}`.

What the shear catalogue looks like, as distinct from where the survey
looks (`clenspy.survey.area`). Implements the `Survey` protocol.

The default shape is the Rozo et al. (2011) eq. 14 form

.. math::
    p(z_s) \propto z_s^{m}\,
        \exp\!\left[-\left(\frac{z_s}{z_\star}\right)^{\beta}\right],

normalised to unity on :math:`[z_s^{\min}, z_s^{\max}]`. A top-hat and a
tabulated :math:`dn/dz` are the other two forms, as named constructors.

NOTE: units. :math:`p(z_s)` is a density in redshift, so it integrates to 1
and carries units of :math:`1/z`. :math:`\sigma_\gamma` is the per-galaxy
shape noise, dimensionless. :math:`n_{\rm src}` is a **surface density on
the sky in arcmin^-2** -- the one quantity in the package that is not in
h-free Mpc units, because that is how shape catalogues are quoted and how
the shape-noise term of a covariance consumes it. It is named
``n_src_arcmin`` so the unit travels with it.

NOTE: this layer knows nothing about a halo, a profile, or a lens
redshift. :math:`\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)` is built *from*
a source population but is lens-source geometry, so it lives in
`clenspy.kernels`. Keeping them apart is what stops a counts consumer and
a shear consumer from picking up each other's factors (errata E.1, E.2).
"""

from __future__ import annotations

import numpy as np

__all__ = ["SourcePopulation"]

#: Nodes used to normalise :math:`p(z_s)` by trapezoid.
#:
#: NOTE: 601 nodes over the full support gives a normalisation stable to
#: better than 1e-6 for every shape here -- the integrand is smooth and
#: single-peaked, so the trapezoid error falls as h^2 and 601 nodes over
#: [0, 3] is dz = 0.005 against a peak width of ~0.5. The exemplar used
#: dz = 0.01 on the same integral.
_N_NORM_NODES = 601


class SourcePopulation:
    r"""A shear catalogue's redshift distribution and noise properties.

    implements the Survey protocol

    Construct with a :math:`p(z_s)` callable, or use one of the named
    constructors -- `smail`, `top_hat`, `tabulated` -- or a survey preset
    such as `des_y1`.

    NOTE: units are those of the module docstring; in particular
    ``n_src_arcmin`` is arcmin^-2, not Mpc^-2.

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
    ) -> "SourcePopulation":
        r"""The Rozo et al. (2011) eq. 14 shape, :math:`z^m e^{-(z/z_\star)^\beta}`.

        Also called the Smail form, and "whale-shaped" in the exemplar
        package. The defaults are the DES Y1 source fit (see `des_y1`).

        NOTE: :math:`m` and :math:`\beta` are shape parameters of the
        *source* distribution and have nothing to do with the HOD
        :math:`\alpha` or the Einasto index -- see ``docs/notation.md`` for
        the full collision table.
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
    ) -> "SourcePopulation":
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
    ) -> "SourcePopulation":
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
            raise ValueError(
                f"dndz has shape {dndz.shape}, z has {z.shape}"
            )
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

    # -- survey presets --------------------------------------------------

    @classmethod
    def des_y1(cls) -> "SourcePopulation":
        r"""DES Y1 metacalibration sources.

        :math:`\sigma_\gamma = 0.3`, :math:`n_{\rm src} = 6.28\,{\rm
        arcmin}^{-2}`, and the eq. 14 shape with
        :math:`(z_\star, m, \beta) = (0.74, 1.68, 2.33)`.

        NOTE: transcribed from ``cluster-lensing-cov/configs/des_y1.json``,
        whose provenance line reads "DES Y1 source-distribution fit used by
        upstream validation commit b7fd6e4". Pair it with
        `~clenspy.survey.omega_des_y1`.
        """
        return cls.smail(
            z_star=0.74, m=1.68, beta=2.33,
            sigma_gamma=0.3, n_src_arcmin=6.28, name="DES Y1",
        )

    @classmethod
    def des_y3(cls) -> "SourcePopulation":
        r"""DES Y3 metacalibration sources.

        :math:`\sigma_\gamma = 0.261`, :math:`n_{\rm src} = 5.59\,{\rm
        arcmin}^{-2}`.

        NOTE: the noise properties are DES Y3, but the :math:`p(z_s)`
        **shape is still the Y1 fit** -- ``configs/des_y3.json`` says so
        itself: "DES Y1 source-distribution shape retained as a
        placeholder; replace with the DES Y3 source n(z)". Use
        `tabulated` with the real Y3 source n(z) when you have it.
        """
        return cls.smail(
            z_star=0.74, m=1.68, beta=2.33,
            sigma_gamma=0.261, n_src_arcmin=5.59, name="DES Y3",
        )

    @classmethod
    def sdss(cls) -> "SourcePopulation":
        r"""Not available -- no SDSS source-population definition to transcribe.

        Raises
        ------
        NotImplementedError
            Always. `clenspy.survey.omega_sdss` exists because the SDSS
            :math:`\Omega(z)` fit is in ``y3_cluster_cpp``; there is no
            corresponding record of the SDSS shear catalogue's
            :math:`p(z_s)`, :math:`\sigma_\gamma` or :math:`n_{\rm src}`
            in any source this package tracks. Inventing plausible numbers
            here would put a normalisation into the covariance that no one
            could trace, so it is refused. Build it explicitly with
            `smail` or `tabulated` once you have the catalogue.
        """
        raise NotImplementedError(
            "no SDSS source population is defined. clenspy transcribes "
            "survey properties rather than estimating them, and no record "
            "of the SDSS shear catalogue's p(z_s), sigma_gamma or n_src is "
            "available here -- unlike omega_sdss, which comes from "
            "y3_cluster_cpp. Construct one explicitly, e.g. "
            "SourcePopulation.smail(z_star=..., m=..., beta=..., "
            "sigma_gamma=..., n_src_arcmin=..., name='SDSS')."
        )

    def __repr__(self) -> str:
        return (
            f"SourcePopulation({self.name!r}, sigma_gamma="
            f"{self.sigma_gamma:g}, n_src_arcmin={self.n_src_arcmin:g}, "
            f"zs=[{self.zs_min:g}, {self.zs_max:g}])"
        )


if __name__ == "__main__":
    pops = [
        SourcePopulation.des_y1(),
        SourcePopulation.des_y3(),
        SourcePopulation.top_hat(zs_min=0.8, zs_max=1.2, name="top-hat"),
        SourcePopulation.tabulated(
            z=np.linspace(0.1, 2.0, 40),
            dndz=np.exp(-((np.linspace(0.1, 2.0, 40) - 0.8) ** 2) / 0.08),
            name="mock dn/dz",
        ),
    ]
    z = np.array([0.2, 0.5, 0.8, 1.2, 2.0])
    print(f"p(z_s) at z = {z}\n")
    for pop in pops:
        print(f"  {pop}")
        print("    p(z_s) =", " ".join(f"{v:7.4f}" for v in pop.pz_src(z)))

    # every shape must integrate to one over its own support
    print("\nnormalisation check (should all be 1):")
    for pop in pops:
        zz = np.linspace(*pop.zs_range(), 2001)
        print(f"  {pop.name:12s} {np.trapezoid(pop.pz_src(zz), x=zz):.8f}")

    try:
        SourcePopulation.sdss()
    except NotImplementedError as exc:
        print(f"\nSDSS refused, as designed:\n  {str(exc)[:72]}...")
