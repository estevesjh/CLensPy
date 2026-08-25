"""Intrinsic (halo-to-halo) profile variance of a stacked DeltaSigma bin.

The term the Gaussian covariance misses (McClintock et al. 2019, Sec. on
the semi-analytic covariance; Gruen et al. 2015): each cluster in the
stack carries its own :math:`\\Delta\\Sigma` profile, and the stack of
:math:`N_{\\rm cl}` clusters inherits the population covariance

.. math::

    C^{\\rm intr}_{ij} = \\frac{1}{N_{\\rm cl}}
        \\Big[ \\big\\langle \\Delta\\Sigma(R_i)\\, \\Delta\\Sigma(R_j)
        \\big\\rangle_{\\rm pop}
        - \\big\\langle \\Delta\\Sigma(R_i) \\big\\rangle_{\\rm pop}
          \\big\\langle \\Delta\\Sigma(R_j) \\big\\rangle_{\\rm pop} \\Big]

where the population is the selection-weighted mass distribution of the
bin (the S_ij-weighted :math:`P(M)` — the analytic counterpart of
McClintock's Monte-Carlo draw through the inverted mass-richness
relation) convolved with lognormal concentration scatter at fixed mass
(Diemer & Kravtsov 2015).  Per-cluster profiles use the Hayashi & White
max composition,

.. math::

    \\Delta\\Sigma(R \\mid M, c) = \\max\\big[
        \\Delta\\Sigma_{\\rm 1h}(R \\mid M, c),\\;
        b(M)\\, \\bar\\rho_m\\, \\Delta\\Sigma_{hh}(R) \\big],

so mass scatter propagates both to the 1-halo amplitude (small scales,
with the extra c-scatter) *and* to the large-scale bias :math:`b(M)` —
"scatter in the M–lambda relation causes variance on all scales"
(McClintock et al. 2019).  Miscentering stochasticity is not included
(extension hook noted in the class docstring).

Deterministic Gauss-Hermite quadrature over :math:`\\ln c` replaces the
Monte Carlo: ~8 nodes suffice for a lognormal.
"""

from __future__ import annotations

import numpy as np

from ..config import DEFAULT_COSMOLOGY
from ..halo.nfw import NfwProfile
from .weights import MassZWeights

__all__ = ["IntrinsicProfileVariance"]


class IntrinsicProfileVariance:
    """Population covariance of per-cluster max-model profiles.

    Parameters
    ----------
    weights : MassZWeights
        z-contracted selection weights (defines the mass population per
        bin; ``norm`` supplies :math:`N_{\\rm cl}`).
    twohalo : clenspy.halo.TwoHaloTerm
        Matter :math:`\\Delta\\Sigma_{hh}` engine (NOT premultiplied by
        the mean density — applied here).
    bias : object with ``at_lnM(lnM, z)``
    rho_m0 : float
        Comoving mean matter density [Msun/Mpc^3].
    z_eff : float
        Representative redshift of the bin (2h term and c-model pivot).
    cosmology : astropy cosmology
    concentration : float or callable ``c(M, z)``
        Median concentration relation.
    sigma_lnc : float
        Lognormal scatter of c at fixed mass (Diemer & Kravtsov 2015:
        ~0.16 for relaxed samples; 0.25 spans the full population).
    n_c : int
        Gauss-Hermite nodes over ln c.

    Notes
    -----
    Miscentering stochasticity (the third component of McClintock's
    intrinsic-variance Monte Carlo) is not modeled here; it adds
    small-scale covariance and can be added by extending the profile
    node set with ``clenspy.lensing.miscentering`` draws.
    """

    def __init__(
        self,
        weights: MassZWeights,
        twohalo,
        bias,
        rho_m0: float,
        z_eff: float,
        cosmology=DEFAULT_COSMOLOGY,
        concentration: float | callable = 4.0,
        sigma_lnc: float = 0.16,
        n_c: int = 8,
    ) -> None:
        self.w = weights
        self.twohalo = twohalo
        self.bias = bias
        self.rho_m0 = float(rho_m0)
        self.z_eff = float(z_eff)
        self.cosmo = cosmology
        self.sigma_lnc = float(sigma_lnc)

        # Gauss-Hermite nodes for the lognormal c distribution:
        # ln c = ln c_med + sqrt(2) sigma t,  weights w/sqrt(pi)
        t, wh = np.polynomial.hermite.hermgauss(int(n_c))
        self._c_shift = np.exp(np.sqrt(2.0) * self.sigma_lnc * t)  # (n_c,)
        self._c_w = wh / np.sqrt(np.pi)

        M = np.exp(weights.lnm_x)
        if callable(concentration):
            c_med = concentration(M, self.z_eff)
        else:
            c_med = np.full_like(M, float(concentration))
        # profile set over (mass node, c node): flatten to one halo axis
        M_grid = np.repeat(M, n_c)
        c_grid = (c_med[:, None] * self._c_shift[None, :]).ravel()
        self._nfw = NfwProfile(m200=M_grid, c200=c_grid, cosmo=cosmology)
        self._n_c = int(n_c)
        self._b_of_M = np.asarray(
            bias.at_lnM(weights.lnm_x, self.z_eff), dtype=float
        )  # (n_lnm,)

    def _profiles(self, R: np.ndarray) -> np.ndarray:
        """Per-cluster max-model profiles, shape (n_lnm, n_c, n_R)."""
        one = self._nfw.deltasigma(R).reshape(
            self.w.lnm_x.size, self._n_c, R.size
        )
        two = self.rho_m0 * np.asarray(
            self.twohalo.deltasigma(R, self.z_eff), dtype=float
        ).reshape(-1)  # (n_R,)
        return np.maximum(one, self._b_of_M[:, None, None] * two[None, None, :])

    def cov(self, R, b: int) -> np.ndarray:
        r""":math:`C^{\rm intr}_{ij}` for bin ``b`` [(Msun/Mpc^2)^2],
        shape (n_R, n_R)."""
        R = np.atleast_1d(np.asarray(R, dtype=float))
        prof = self._profiles(R)  # (n_lnm, n_c, n_R)

        # population weights: mass (selection) x concentration (lognormal)
        wm = self.w.W[b] * self.w.lnm_w  # (n_lnm,)
        wm = wm / wm.sum()
        wkj = wm[:, None] * self._c_w[None, :]  # (n_lnm, n_c)

        mean = np.einsum("kj,kjr->r", wkj, prof)
        second = np.einsum("kj,kjr,kjs->rs", wkj, prof, prof)
        cov_pop = second - np.outer(mean, mean)
        n_cl = float(self.w.norm[b])
        return cov_pop / n_cl
