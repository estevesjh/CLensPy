r"""The observed-richness kernel :math:`P(\lambda^{\rm ob}\mid\lambda^{\rm tr})`,
and its bin integral in closed form.

The kernel is Costanzi et al. (2019a) Eqs. 3, 5, 6, reused unchanged in
Costanzi et al. (2019b) and (2021):
:math:`\lambda^{\rm ob} = \lambda^{\rm tr} + \Delta^{\rm bkg} +
\Delta^{\rm prj}`, with a Gaussian background/measurement fluctuation and a
one-sided projection boost that is a spike at zero plus an exponential
tail,

.. math::
    P(\Delta^{\rm prj}\mid\lambda^{\rm tr},z)
      = (1-f^{\rm prj})\,\delta_{\rm D}(\Delta^{\rm prj})
      + f^{\rm prj}\,\tau\,e^{-\tau\Delta^{\rm prj}}\,
        \Theta(\Delta^{\rm prj})

Convolving the two gives an **exponentially modified Gaussian** (EMG, or
"ex-Gaussian"; Grushka 1972):

.. math::
    P(\lambda^{\rm ob}\mid\lambda^{\rm tr},z)
      = (1-f^{\rm prj})\,\mathcal N(\lambda^{\rm ob};\mu,\sigma)
      + f^{\rm prj}\,\frac{\tau}{2}
        e^{\frac{\tau}{2}(2\mu+\tau\sigma^2-2\lambda^{\rm ob})}
        \operatorname{erfc}\!\left(
          \frac{\mu+\tau\sigma^2-\lambda^{\rm ob}}{\sqrt2\,\sigma}\right)

with :math:`\mu = \lambda^{\rm tr} + \Delta\mu`.

**The whole point of this module is that the bin integral is analytic.**
:math:`\mathcal S_i \equiv \int_{\Delta\lambda_i} d\lambda^{\rm ob}\,P`
needs no quadrature: it is a difference of CDFs at the two bin edges,

.. math::
    \mathcal S_i = (1-f^{\rm prj})
      \left.\Phi\!\left(\frac{\lambda^{\rm ob}-\mu}{\sigma}\right)
      \right|_{\Delta\lambda_i}
      + f^{\rm prj}\left.F_{\rm EMG}(\lambda^{\rm ob};\mu,\sigma,\tau)
      \right|_{\Delta\lambda_i}

so the only numerical integral left in the selection function is the one
over :math:`\lambda^{\rm tr}` (see
`clenspy.selection.selection_function`).

NOTE: :math:`F_{\rm EMG}` is **not** evaluated in the form the derivation
produces. The textbook expression

.. math::
    F_{\rm EMG} = \Phi(z) - e^{-\tau(x-\mu) + \frac12\tau^2\sigma^2}\,
                  \Phi(z - \tau\sigma),
    \qquad z = \frac{x-\mu}{\sigma}

is a product of a factor that **overflows** and a factor that
**underflows**: for :math:`\tau\sigma \gtrsim 30` the exponential exceeds
``1e308`` while :math:`\Phi(z-\tau\sigma)` is denormal, and the product is
``inf * 0 = nan`` even though the true value is a perfectly ordinary number
in :math:`[0,1]`. This module uses the scaled complementary error function
:math:`\operatorname{erfcx}(u) = e^{u^2}\operatorname{erfc}(u)` to absorb
the exponent exactly, following
``y3_cluster_cpp/src/models/richness_kernel_t.hh::F_EMG``. The identity is
derived in the `emg_cdf` docstring and is exact for every argument -- not
an asymptotic approximation.

NOTE: **units.** Richness is a dimensionless galaxy count;
:math:`\Delta\mu`, :math:`\sigma` and the bin edges are all in richness
units, and :math:`\tau` is an inverse richness. :math:`\mathcal S_i` and
:math:`F_{\rm EMG}` are probabilities, dimensionless and in
:math:`[0, 1]`.

NOTE: the four kernel parameters
:math:`\{\Delta\mu, \sigma, f^{\rm prj}, \tau\}` are functions of
:math:`(\lambda^{\rm tr}, z)`, calibrated on synthetic-cluster injections
in SDSS. In the y3 pipeline they come from spline coefficients
(``cosmology/prj_params.py::PrjParams.default().splines()``), which are not
distributed with `clenspy`. `EmgParams` therefore takes them as explicit
arrays or callables: pass constants for a fixed-parameter study, or a
callable reading the y3 splines for exactness. The approximation is named,
not hidden.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfcx

from ..kernels.photoz import gaussian_cdf

__all__ = ["EmgParams", "emg_cdf", "richness_bin_probability"]

_SQRT2 = np.sqrt(2.0)

#: Clip for the ``exp(A)`` branch. ``A`` can legitimately exceed the fp64
#: exponent range while the *result* stays in [0, 1]; clipping there keeps
#: the arithmetic finite and the final `np.clip` restores the bound.
_EXP_CLIP = 700.0


def emg_cdf(x, mu, sigma, tau):
    r"""The EMG CDF :math:`F_{\rm EMG}(x;\mu,\sigma,\tau)`, dimensionless.

    Let :math:`X = G + E` with :math:`G\sim\mathcal N(\mu,\sigma^2)` and
    :math:`E\sim{\rm Exp}(\tau)` independent. Conditioning on :math:`G` and
    completing the square gives

    .. math::
        F_{\rm EMG}(x) = \Phi(z)
          - e^{A}\,\Phi(z - \tau\sigma),
        \qquad
        z = \frac{x-\mu}{\sigma},
        \qquad
        A = -\tau(x-\mu) + \tfrac12\tau^2\sigma^2

    **The evaluated form.** Write
    :math:`u = (\tau\sigma - z)/\sqrt2`, so that
    :math:`\Phi(z-\tau\sigma) = \tfrac12\operatorname{erfc}(u)`. Using
    :math:`\operatorname{erfc}(u) = e^{-u^2}\operatorname{erfcx}(u)` and the
    algebraic identity

    .. math::
        A - u^2 = -\tau\sigma z + \tfrac12\tau^2\sigma^2
                  - \tfrac12(\tau\sigma - z)^2 = -\tfrac12 z^2

    the whole product collapses to

    .. math::
        e^{A}\,\Phi(z-\tau\sigma)
          = \tfrac12\operatorname{erfcx}(u)\,e^{-z^2/2}

    with **no large exponential anywhere**: the divergent
    :math:`e^{A}` and the vanishing :math:`\Phi` have cancelled
    identically. For :math:`u < 0`, where :math:`\operatorname{erfcx}`
    itself overflows, the reflection
    :math:`\operatorname{erfc}(u) = 2 - e^{-u^2}\operatorname{erfcx}(|u|)`
    gives

    .. math::
        e^{A}\Phi(z-\tau\sigma)
          = e^{A} - \tfrac12\operatorname{erfcx}(|u|)\,e^{-z^2/2}

    which is the only branch where an exponential is evaluated at all.

    NOTE: exact for every argument, not asymptotic. The two branches meet
    at :math:`u = 0` where both reduce to
    :math:`\tfrac12 e^{-z^2/2}\operatorname{erfcx}(0) =
    \tfrac12 e^{-z^2/2}`.

    NOTE: the result is clipped to :math:`[0,1]`. It is a CDF, and the
    subtraction can leave :math:`O(\epsilon)` excursions outside the bound
    that would otherwise propagate as a negative probability.

    Parameters
    ----------
    x : float or array-like
        Observed richness at which to evaluate the CDF.
    mu : float or array-like
        Gaussian mean, :math:`\mu = \lambda^{\rm tr} + \Delta\mu`.
    sigma : float or array-like
        Gaussian width, in richness units. Positive.
    tau : float or array-like
        Exponential rate, inverse richness. Positive; the tail mean is
        :math:`1/\tau`.

    Returns
    -------
    np.ndarray
        :math:`F_{\rm EMG} \in [0, 1]`, broadcast over the inputs.
    """
    x, mu, sigma, tau = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x, mu, sigma, tau))
    )
    if np.any(sigma <= 0.0):
        raise ValueError("sigma must be positive")
    if np.any(tau <= 0.0):
        raise ValueError("tau must be positive")

    z = (x - mu) / sigma
    u = (tau * sigma - z) / _SQRT2
    negative = u < 0.0
    exp_mz2 = np.exp(-0.5 * z * z)
    # the branch that never overflows: erfcx absorbs exp(A) exactly
    tail_base = 0.5 * erfcx(np.abs(u)) * exp_mz2
    # ... and the reflected branch, the only place exp() is evaluated
    A = -tau * (x - mu) + 0.5 * (tau * sigma) ** 2
    tail = np.where(negative,
                    np.exp(np.clip(A, -_EXP_CLIP, _EXP_CLIP)) - tail_base,
                    tail_base)
    return np.clip(gaussian_cdf(z) - tail, 0.0, 1.0)


class EmgParams:
    r"""The four kernel parameters as functions of
    :math:`(\lambda^{\rm tr}, z)`.

    Holds :math:`\Delta\mu`, :math:`\sigma`, :math:`f^{\rm prj}` and
    :math:`\tau`, each either a constant or a callable
    ``f(lambda_true, z)``. Constructing this rather than passing four loose
    arrays keeps the *set* together: they are calibrated jointly and mixing
    one study's :math:`f^{\rm prj}` with another's :math:`\tau` is not a
    kernel anybody fitted.

    NOTE: units -- ``delta_mu`` and ``sigma`` in richness, ``tau`` in
    inverse richness, ``f_prj`` dimensionless in :math:`[0,1]`.

    NOTE: :math:`\mu = \lambda^{\rm tr} + \Delta\mu`, and
    :math:`\Delta\mu < 0` typically -- redMaPPer's global background
    subtraction biases :math:`\lambda^{\rm ob}` low. The sign is the
    caller's to supply; nothing here enforces it.

    NOTE: :math:`f^{\rm prj} = 0` recovers the pure-Gaussian "BKG" model of
    Costanzi et al. (2021) exactly, which is the check
    `richness_bin_probability` uses to show the two limits agree.

    Parameters
    ----------
    delta_mu, sigma, f_prj, tau : float or callable
        Each either a scalar or ``f(lambda_true, z) -> array``.
    """

    def __init__(self, delta_mu, sigma, f_prj, tau):
        self.delta_mu = delta_mu
        self.sigma = sigma
        self.f_prj = f_prj
        self.tau = tau

    @staticmethod
    def _evaluate(value, lambda_true, z):
        return np.asarray(
            value(lambda_true, z) if callable(value) else value, dtype=float
        )

    def at(self, lambda_true, z):
        r"""``(mu, sigma, tau, f_prj)`` at :math:`(\lambda^{\rm tr}, z)`.

        Returns :math:`\mu`, not :math:`\Delta\mu` -- the shift is applied
        here, once, so no caller can forget it.
        """
        lambda_true = np.asarray(lambda_true, dtype=float)
        mu = lambda_true + self._evaluate(self.delta_mu, lambda_true, z)
        sigma = self._evaluate(self.sigma, lambda_true, z)
        tau = self._evaluate(self.tau, lambda_true, z)
        f_prj = self._evaluate(self.f_prj, lambda_true, z)
        if np.any(f_prj < 0.0) or np.any(f_prj > 1.0):
            raise ValueError(f"f_prj must lie in [0, 1], got {f_prj}")
        return mu, sigma, tau, f_prj

    def __repr__(self):
        def show(v):
            return "callable" if callable(v) else f"{v:g}"
        return (f"EmgParams(delta_mu={show(self.delta_mu)}, "
                f"sigma={show(self.sigma)}, f_prj={show(self.f_prj)}, "
                f"tau={show(self.tau)})")


def richness_bin_probability(lambda_edges, lambda_true, z, params):
    r""":math:`\mathcal S_i(\lambda^{\rm tr}, z)` for every bin at once.

    .. math::
        \mathcal S_i = (1-f^{\rm prj})
          \left.\Phi\!\left(\frac{\lambda^{\rm ob}-\mu}{\sigma}\right)
          \right|_{\Delta\lambda_i}
          + f^{\rm prj}\left.F_{\rm EMG}\right|_{\Delta\lambda_i}

    NOTE: computed by **differencing one CDF evaluation per shared edge**,
    not two per bin. With contiguous bins the edges are shared, so
    :math:`n+1` evaluations serve :math:`n` bins, and adjacent bins are
    guaranteed to agree at the edge they share -- differencing each bin
    independently lets round-off open a gap or an overlap between them.
    This is the same structure as
    ``sel_function.py::_K_edges_of_bins``, and it is why this function
    returns all bins rather than taking a bin index.

    Parameters
    ----------
    lambda_edges : array-like, shape (n_bins + 1,)
        Observed-richness bin edges, strictly ascending. Contiguous: edge
        ``i+1`` closes bin ``i`` and opens bin ``i+1``.
    lambda_true : float or array-like
        True richness, any shape.
    z : float or array-like
        True redshift, broadcast against ``lambda_true``.
    params : EmgParams
        The four kernel parameters.

    Returns
    -------
    np.ndarray
        Shape ``(*np.shape(lambda_true), n_bins)``. Probabilities in
        :math:`[0,1]`.
    """
    edges = np.asarray(lambda_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("lambda_edges must be 1-D with at least two edges")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("lambda_edges must be strictly ascending")

    mu, sigma, tau, f_prj = params.at(lambda_true, z)
    # one CDF per shared edge, so adjacent bins agree at their common edge
    cdf = np.stack(
        [(1.0 - f_prj) * gaussian_cdf((edge - mu) / sigma)
         + f_prj * emg_cdf(edge, mu, sigma, tau)
         for edge in edges],
        axis=-1,
    )
    return np.diff(cdf, axis=-1)


if __name__ == "__main__":
    # y3-like bin edges and a plausible kernel
    edges = np.array([20.0, 30.0, 45.0, 60.0, 200.0])
    params = EmgParams(delta_mu=-1.5, sigma=3.0, f_prj=0.3, tau=0.12)
    print(params)
    print(f"edges = {edges}\n")

    print("S_i(lambda_tr) -- rows sum to the probability of landing "
          "anywhere in [20, 200]:")
    print(f"{'lam_tr':>7s}  " + "  ".join(f"{f'[{a:.0f},{b:.0f})':>12s}"
                                          for a, b in zip(edges[:-1],
                                                          edges[1:]))
          + f"  {'sum':>8s}")
    for lam in (18.0, 25.0, 35.0, 50.0, 80.0):
        s = richness_bin_probability(edges, lam, 0.3, params)
        print(f"{lam:7.1f}  " + "  ".join(f"{v:12.6f}" for v in s)
              + f"  {s.sum():8.6f}")

    # the identity that pins the edge-differencing: contiguous bins tile
    print("\nthe sum equals CDF(200) - CDF(20) exactly, by construction:")
    lam = 35.0
    mu, sig, tau, f = params.at(lam, 0.3)
    Phi = gaussian_cdf
    direct = ((1 - f) * (Phi((edges[-1] - mu) / sig)
                         - Phi((edges[0] - mu) / sig))
              + f * (emg_cdf(edges[-1], mu, sig, tau)
                     - emg_cdf(edges[0], mu, sig, tau)))
    summed = richness_bin_probability(edges, lam, 0.3, params).sum()
    print(f"  sum over bins = {summed:.15f}")
    print(f"  CDF|_20^200   = {float(np.ravel(direct)[0]):.15f}")

    # the numerical point of the erfcx form
    print("\nwhy erfcx: the textbook form is inf * 0 = nan where this is "
          "fine.")
    print(f"{'tau*sigma':>10s}  {'exp(A)':>12s}  {'Phi(z-tau sig)':>15s}  "
          f"{'product':>10s}  {'emg_cdf':>10s}")
    for tau_sigma in (5.0, 20.0, 30.0, 40.0):
        sig = 3.0
        tau_v = tau_sigma / sig
        x, mu_v = 40.0, 35.0
        z = (x - mu_v) / sig
        A = -tau_v * (x - mu_v) + 0.5 * tau_sigma**2
        # the overflow here is the point being demonstrated, so it is
        # silenced deliberately rather than allowed to look like a bug
        with np.errstate(over="ignore", invalid="ignore"):
            naive_exp = np.exp(A)
            naive_phi = Phi(z - tau_sigma).item()
            product = naive_exp * naive_phi
        print(f"{tau_sigma:10.1f}  {naive_exp:12.3e}  {naive_phi:15.3e}  "
              f"{product:10.3e}  "
              f"{emg_cdf(x, mu_v, sig, tau_v).item():10.6f}")
    print("  <- the product loses all precision (and would be nan without")
    print("     the clip); emg_cdf stays exact because the exponent never")
    print("     forms.")

    # f_prj = 0 must recover the pure Gaussian
    g_only = EmgParams(delta_mu=-1.5, sigma=3.0, f_prj=0.0, tau=0.12)
    s_emg = richness_bin_probability(edges, 35.0, 0.3, g_only)
    s_gauss = np.diff([Phi((e - (35.0 - 1.5)) / 3.0).item() for e in edges])
    print(f"\nf_prj = 0 recovers the Gaussian BKG model: max|diff| = "
          f"{np.max(np.abs(s_emg - s_gauss)):.2e}")

    # the projection tail moves probability UP in richness, never down
    s_no = richness_bin_probability(edges, 25.0, 0.3, g_only)
    s_yes = richness_bin_probability(edges, 25.0, 0.3, params)
    print("\nthe projection boost is one-sided: it can only move clusters "
          "to higher richness.")
    print(f"  f_prj = 0.0: {np.array2string(s_no, precision=5)}")
    print(f"  f_prj = 0.3: {np.array2string(s_yes, precision=5)}")
    print(f"  change     : {np.array2string(s_yes - s_no, precision=5)}")
