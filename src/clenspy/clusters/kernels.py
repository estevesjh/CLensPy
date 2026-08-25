"""Closed-form richness and photo-z selection kernels.

Implements the bin-integrated kernels of
``RichnessSelection/docs/richness_selection_function.tex``:

- ``K_i`` — observed-richness kernel: probability that a halo of true
  richness :math:`\\lambda^{\\rm tr}` at redshift ``z`` is assigned an
  observed richness inside :math:`[\\lambda_i^{\\min}, \\lambda_i^{\\max}]`.
  Closed form: Gaussian CDF + exponentially modified Gaussian (EMG) CDF
  differences (Eq. 16), evaluated with ``erfcx`` for tail stability.
- ``K_j`` — Gaussian-CDF observed-redshift (photo-z) kernel (Eq. 12).

The EMG kernel parameters :math:`(\\mu, \\sigma, \\tau, f^{\\rm prj})`
as functions of :math:`(\\lambda^{\\rm tr}, z)` are read from the vendored
DES Y3 projection-effects fit (``PlobLtrParams``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import erf, erfc, erfcx

__all__ = [
    "emg_cdf",
    "PlobLtrParams",
    "RichnessKernel",
    "EmgRichnessKernel",
    "AnalyticLogNormalKernel",
    "K_j",
]

_SQRT2 = np.sqrt(2.0)

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_PRJ_PARAMS_FILE = "prj_params_DESY3_lss_lin_dep_getdist_v1.txt"


def _Phi(x):
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + erf(x / _SQRT2))


def emg_cdf(x, mu, sigma, tau):
    r"""EMG CDF :math:`F_{\rm EMG}(x; \mu, \sigma, \tau)`.

    .. math::

        F_{\rm EMG}(x) = \Phi\!\left(\frac{x-\mu}{\sigma}\right)
            - e^{-\tau(x-\mu) + \frac{1}{2}\tau^2\sigma^2}\,
              \Phi\!\left(\frac{x-\mu}{\sigma} - \tau\sigma\right)

    evaluated through the scaled complementary error function
    ``erfcx(t) = exp(t^2) erfc(t)`` so both tails stay finite.
    Broadcasts ``(x, mu, sigma, tau)``.
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    tau = np.asarray(tau, dtype=float)

    z = (x - mu) / sigma
    u = (tau * sigma - z) / _SQRT2
    neg = u < 0.0
    abs_u = np.where(neg, -u, u)
    exp_mz2 = np.exp(-0.5 * z**2)

    tail_base = 0.5 * erfcx(abs_u) * exp_mz2
    A = -tau * (x - mu) + 0.5 * (tau * sigma) ** 2
    exp_A = np.where(neg, np.exp(np.where(neg, A, 0.0)), 0.0)
    tail = np.where(neg, exp_A - tail_base, tail_base)
    return np.clip(_Phi(z) - tail, 0.0, 1.0)


class PlobLtrParams:
    r"""EMG parameters of :math:`P(\lambda^{\rm ob} \mid \lambda^{\rm tr}, z)`.

    Ten fit coefficients, each linear-interpolated in ``z`` over 15 nodes
    in [0.10, 0.80] (DES Y3 ``lss_lin_dep`` fit):

    .. math::

        \mu = a_\mu + b_\mu \lambda^{\rm tr}, \quad
        \sigma = b_\sigma (\lambda^{\rm tr})^{a_\sigma}, \quad
        \tau = b_\tau / (\lambda^{\rm tr})^{a_\tau}, \quad
        f^{\rm prj} = \min\!\left[1,
            \frac{b_f}{(1 + e^{-\lambda^{\rm tr}})^{a_f}}\right]
    """

    _NAMES = (
        "atau", "btau", "amu", "bmu", "asig",
        "bsig", "afprj", "bfprj", "afmsk", "bfmsk",
    )

    def __init__(self, table: np.ndarray, z_nodes: np.ndarray) -> None:
        self._spl = {
            name: InterpolatedUnivariateSpline(z_nodes, col, k=1, ext=3)
            for name, col in zip(self._NAMES, table)
        }

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> "PlobLtrParams":
        """Load from the vendored DES Y3 fit file (default) or ``path``."""
        if path is None:
            path = _DATA_DIR / _PRJ_PARAMS_FILE
        table = np.loadtxt(path).T
        z_nodes = np.linspace(0.10, 0.80, 15)
        return cls(table, z_nodes)

    def _coeffs(self, z):
        """Evaluate the 8 needed coefficient splines on ``z`` (unique-value
        fast path for large broadcast grids)."""
        z_arr = np.asarray(z, dtype=float)
        flat = z_arr.ravel()
        zu, inv = np.unique(flat, return_inverse=True)
        out = {}
        for name in ("amu", "bmu", "asig", "bsig", "atau", "btau",
                     "afprj", "bfprj"):
            vals = self._spl[name](zu)[inv].reshape(z_arr.shape)
            out[name] = vals if z_arr.shape else float(vals)
        return out

    def at(self, ltr, z):
        """Return ``(mu, sigma, tau, fprj)`` broadcast over ``(ltr, z)``."""
        ltr = np.asarray(ltr, dtype=float)
        c = self._coeffs(z)
        mu = c["amu"] + c["bmu"] * ltr
        sigma = c["bsig"] * ltr ** c["asig"]
        tau = c["btau"] / ltr ** c["atau"]
        fprj = np.minimum(1.0, c["bfprj"] / (1.0 + np.exp(-ltr)) ** c["afprj"])
        return mu, sigma, tau, fprj


class RichnessKernel(ABC):
    r"""Observed-richness kernel protocol.

    ``cdf(lam_edge, ltr, z)`` is :math:`\Pr(\lambda^{\rm ob} \le
    \lambda_{\rm edge} \mid \lambda^{\rm tr}, z)`; the bin kernel is the
    difference of two edge CDFs.
    """

    @abstractmethod
    def cdf(self, lam_edge, ltr, z):
        """CDF of lambda_ob at ``lam_edge`` given ``(ltr, z)``."""

    def K_i(self, ltr, z, lam_min, lam_max):
        r""":math:`\mathcal{K}_i(\lambda^{\rm tr}, z)` for one richness bin."""
        return self.cdf(lam_max, ltr, z) - self.cdf(lam_min, ltr, z)


class EmgRichnessKernel(RichnessKernel):
    r"""Gaussian + EMG mixture kernel (DES Y3 production form, Eq. 16).

    .. math::

        \mathcal{K}_i = (1 - f^{\rm prj})\,
            \Phi\!\left(\frac{\lambda^{\rm ob} - \mu}{\sigma}\right)
            \Big|_{\Delta\lambda_i}
            + f^{\rm prj}\, F_{\rm EMG}(\lambda^{\rm ob}; \mu, \sigma, \tau)
            \Big|_{\Delta\lambda_i}
    """

    def __init__(self, plob: PlobLtrParams | None = None) -> None:
        self.plob = plob if plob is not None else PlobLtrParams.from_file()

    def cdf(self, lam_edge, ltr, z):
        ltr = np.asarray(ltr, dtype=float)
        mu, sigma, tau, fprj = self.plob.at(ltr, z)
        gauss = _Phi((lam_edge - mu) / sigma)
        emg = emg_cdf(lam_edge, mu, sigma, tau)
        return (1.0 - fprj) * gauss + fprj * emg

    def pdf_lob(self, lob, ltr, z):
        r""":math:`P(\lambda^{\rm ob} \mid \lambda^{\rm tr}, z)` density.

        Gaussian + EMG mixture (needed by the selection-bias
        :math:`\lambda^{\rm tr}` marginalisation).
        """
        lob = np.asarray(lob, dtype=float)
        mu, sigma, tau, fprj = self.plob.at(ltr, z)
        gauss = np.exp(-0.5 * ((lob - mu) / sigma) ** 2) / (
            sigma * np.sqrt(2.0 * np.pi)
        )
        exp_arg = 0.5 * tau * (2.0 * mu + tau * sigma**2 - 2.0 * lob)
        erfc_arg = (mu + tau * sigma**2 - lob) / (_SQRT2 * sigma)
        # exp(exp_arg)*erfc(erfc_arg) via erfcx for tail stability:
        # exp(a)*erfc(t) = erfcx(t) * exp(a - t^2) for t >= 0
        pos = erfc_arg >= 0.0
        safe_stable = erfcx(np.where(pos, erfc_arg, 0.0)) * np.exp(
            np.where(pos, exp_arg - erfc_arg**2, 0.0)
        )
        safe_direct = np.exp(np.where(pos, 0.0, exp_arg)) * erfc(
            np.where(pos, 0.0, erfc_arg)
        )
        emg = 0.5 * tau * np.where(pos, safe_stable, safe_direct)
        return (1.0 - fprj) * gauss + fprj * emg


class AnalyticLogNormalKernel(RichnessKernel):
    r"""No-projection limit: :math:`P(\lambda^{\rm ob} \mid \lambda^{\rm tr})
    = \delta_D(\lambda^{\rm ob} - \lambda^{\rm tr})`.

    ``cdf`` becomes an indicator; combined with :class:`LogNormalMOR`
    the resulting S_i reproduces the closed-form erfc lognormal bin
    probability used by ``cluster-lensing-cov`` (validation path).

    ``hard_edges = True`` tells quadrature consumers the kernel is a step
    function: instead of integrating pdf x step with Gauss-Legendre (slow
    O(1/n) convergence across the jump), they clip the integration bracket
    to the bin and integrate the smooth pdf alone (exact treatment).
    """

    hard_edges = True

    def cdf(self, lam_edge, ltr, z):
        ltr = np.asarray(ltr, dtype=float)
        return (ltr <= lam_edge).astype(float)


def K_j(ztr, zob_min, zob_max, sigma_z):
    r"""Gaussian-CDF photo-z kernel (Eq. 12).

    .. math::

        \mathcal{K}_j(z^{\rm tr}) =
            \Phi\!\left(\frac{z_j^{\max} - z^{\rm tr}}{\sigma_z}\right)
            - \Phi\!\left(\frac{z_j^{\min} - z^{\rm tr}}{\sigma_z}\right)

    In the :math:`\sigma_z \to 0` limit this is the top-hat indicator of
    the redshift bin.
    """
    ztr = np.asarray(ztr, dtype=float)
    if sigma_z <= 0.0:
        return ((ztr >= zob_min) & (ztr < zob_max)).astype(float)
    return _Phi((zob_max - ztr) / sigma_z) - _Phi((zob_min - ztr) / sigma_z)
