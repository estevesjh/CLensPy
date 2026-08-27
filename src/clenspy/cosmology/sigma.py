r"""The variance of the linear density field, :math:`\sigma^2(R)`.

.. math::
    \sigma^2(R) = \frac{1}{2\pi^2}\int dk\,k^{2}\,P_{\rm lin}(k)\,W^{2}(kR)
                = \int d\ln k\;\frac{k^{3}P_{\rm lin}(k)}{2\pi^{2}}\,W^{2}(kR)

One quantity, two consumers: the Tinker (2008) mass function and the
Tinker (2010) bias are two fits to the *same* peak height
:math:`\nu = \delta_c/\sigma(M)`. Computing :math:`\sigma` twice from one
:math:`P(k)` is how they silently drift apart, so it is computed once,
here, and `clenspy.cosmology.TinkerMassFunction` and
`clenspy.halo.BiasModel` both take a `SigmaGrid`.

Transcribed from ``y3_cluster_cpp`` branch ``docs/sphinx-site``,
``src/modules/mf_tinker_cpp/python/tinker_core.py`` -- the in-repo
replacement for CosmoSIS's ``MfTinker``, itself a port of the Fortran
``sigma.f90`` / ``linearpk.f90`` (Komatsu CRL). Its measured accuracy is
the reason to prefer it: the Gauss--Legendre-panel evaluator agrees with
arbitrary-precision mpmath quadrature to **4.4e-16**, and with
``cluster_toolkit.peak_height.sigma2_at_R`` to **1.0e-7**.

**Three things here are conventions, not choices, and all three bite.**

1. **The integration limits are** :math:`k \in [10^{-4},\,20/R]`. The upper
   limit *depends on* :math:`R`. It is part of the algorithm the production
   :math:`dn/dM` was calibrated against, not a convergence cutoff, which is
   why `sigma2` takes ``truncate`` explicitly rather than hiding it.

2. **FFTLog cannot reproduce that truncation.** An FFTLog transform
   integrates the whole sampled :math:`k` range at every :math:`R`
   simultaneously; an :math:`R`-dependent limit is exactly what it cannot
   express. So the fast path (`sigma2_fftlog`) computes the
   ``truncate=False`` quantity and *must* be validated against
   ``truncate=False``. The measured size of the difference it cannot
   capture, propagated to :math:`dn/d\ln M`: **7.0e-3** over
   :math:`0 \le z \le 2`, **8.2e-4** restricted to :math:`z \le 0.8`.

3. **The derivative is taken under the integral sign, not by finite
   differences,** and when the truncation is active the moving boundary
   contributes a Leibniz term:

   .. math::
       \frac{d\sigma^2}{d\ln R} =
         \int d\ln k\,\frac{k^3 P}{2\pi^2}\,2W(x)W'(x)\,x
         \;-\; \left.\frac{k^3P}{2\pi^2}W^2\right|_{k = 20/R}

   because :math:`\ln k_{\rm up} = \ln 20 - \ln R` has
   :math:`d/d\ln R = -1`. Dropping that boundary term is a silent error in
   :math:`dn/d\ln M`, which is proportional to
   :math:`d\ln\nu/d\ln R`.

NOTE: **units are h-scaled here, deliberately** -- ``k`` in h/Mpc, ``P``
in (Mpc/h)^3, ``R`` in Mpc/h. This is the one convention `clenspy`
inherits rather than chooses, because the Tinker calibration, the
production Fortran and `cluster_toolkit` all use it. Every identifier says
so (``r_hinv``, ``k_h``, ``pk_h3``). The rest of the package is h-free
absolute; convert at the boundary, visibly.

NOTE: the input :math:`P(k)` policy is part of the definition, not
plumbing: a **natural** cubic spline of :math:`\ln P` against
:math:`\ln k`, and :math:`P \equiv 0` strictly outside the tabulated
range. `LinearPk` implements exactly that. Substituting a different
extrapolation changes :math:`\sigma^2` at the ends of the :math:`R` grid.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from ..utils.special import tophat_dw, tophat_w

__all__ = [
    "KCUT_COEF",
    "LNK_LO",
    "LNR1",
    "LNR2",
    "NR",
    "STEP",
    "LinearPk",
    "SigmaGrid",
    "lnr_grid",
]

#: Production lnR grid, ``sigma.f90``: 969 points, ``lnR = -5.684 + 0.01 i``,
#: R in Mpc/h. R spans 0.0034 to 54.6 Mpc/h.
LNR1 = -5.684
LNR2 = 4.0
STEP = 0.01
NR = 969

#: Fixed lower quadrature limit of ``sigma.f90``: :math:`k = 10^{-4}` h/Mpc.
LNK_LO = np.log(1.0e-4)

#: Upper quadrature limit coefficient: :math:`k_{\max} = 20/R`. See the
#: module NOTE -- this is algorithm-defining, and FFTLog cannot express it.
KCUT_COEF = 20.0


def lnr_grid():
    """The production 969-point ``lnR`` grid, R in Mpc/h."""
    return LNR1 + STEP * np.arange(NR)


class LinearPk:
    r"""The linear power spectrum, with the production input policy.

    A **natural** cubic spline of :math:`\ln P` against :math:`\ln k`, and
    :math:`P \equiv 0` outside :math:`[k_{\min}, k_{\max}]`.

    NOTE: units -- ``k_h`` in h/Mpc, ``pk_h3`` in (Mpc/h)^3. Returns
    :math:`P` in (Mpc/h)^3.

    NOTE: both halves matter. ``bc_type="natural"`` (zero second derivative
    at the ends) reproduces the Fortran's ``spline_cubic_set`` with
    ``ibcbeg = ibcend = 2``; the hard zero outside the table reproduces
    ``Linear_Pk``, and is *not* the same as clamping or as a power-law
    extrapolation. It makes :math:`\sigma^2` depend on the tabulated
    :math:`k` range, which is why `SigmaGrid` reports that range.

    Parameters
    ----------
    k_h : array-like
        Wavenumbers [h/Mpc], strictly ascending.
    pk_h3 : array-like
        Linear power spectrum [(Mpc/h)^3], positive, same shape.
    """

    def __init__(self, k_h, pk_h3):
        k_h = np.asarray(k_h, dtype=float)
        pk_h3 = np.asarray(pk_h3, dtype=float)
        if k_h.shape != pk_h3.shape:
            raise ValueError(
                f"k and P must have the same shape, got {k_h.shape} and "
                f"{pk_h3.shape}"
            )
        if np.any(np.diff(k_h) <= 0.0):
            raise ValueError("k must be strictly ascending")
        if np.any(pk_h3 <= 0.0):
            raise ValueError("P must be positive (it is splined in log)")
        self.lnk = np.log(k_h)
        self.spline = CubicSpline(self.lnk, np.log(pk_h3), bc_type="natural")
        self.lnk_min = float(self.lnk[0])
        self.lnk_max = float(self.lnk[-1])

    def __call__(self, lnk):
        """:math:`P(k)` at ``lnk``; exactly zero outside the table."""
        lnk = np.asarray(lnk, dtype=float)
        scalar = lnk.ndim == 0
        lnk = np.atleast_1d(lnk)
        out = np.zeros_like(lnk)
        # strict inequalities, matching Linear_Pk: the endpoints are out
        inside = (lnk > self.lnk_min) & (lnk < self.lnk_max)
        out[inside] = np.exp(self.spline(lnk[inside]))
        return out[0] if scalar else out

    def __repr__(self):
        return (f"LinearPk(k = [{np.exp(self.lnk_min):.3g}, "
                f"{np.exp(self.lnk_max):.3g}] h/Mpc, n = {self.lnk.size})")


class SigmaGrid:
    r"""Top-hat variance :math:`\sigma^2(R)` and its :math:`\ln R` derivative.

    Two evaluation routes, and they compute *different quantities*:

    - `sigma2` / `dsigma2_dlnr` -- Gauss--Legendre panels between spline
      knots. The reference. Honours the :math:`k \le 20/R` truncation and
      the Leibniz boundary term.
    - `sigma2_fftlog` -- one FFTLog per call, all :math:`R` at once, ~1000x
      faster and **untruncated**. Compare it to ``truncate=False`` only.

    NOTE: units -- ``r_hinv`` in Mpc/h, dimensionless output. See the
    module NOTE for why this module is h-scaled.

    NOTE: the panel edges are the spline's own knots, because the integrand
    is analytic *within* a knot interval and merely continuous across one.
    A single global rule of the same total order is far worse; a fixed
    24-point rule per panel reaches ~1e-14 relative and 48 points changes
    nothing at 1e-16.

    Parameters
    ----------
    pk : LinearPk
        Stored verbatim.
    nquad : int, optional
        Gauss--Legendre order per panel (default: 24).
    """

    #: default GL order per panel; 24 vs 48 agree to 1e-16
    NQUAD = 24

    def __init__(self, pk, nquad: int = NQUAD):
        if not isinstance(pk, LinearPk):
            raise TypeError(
                "pk must be a LinearPk -- the input spline and zero-outside "
                "policy are part of the definition of sigma^2 here"
            )
        self.pk = pk
        self.nquad = int(nquad)
        self._nodes, self._weights = np.polynomial.legendre.leggauss(
            self.nquad
        )
        self._fftlog_plan = None

    # -- the reference route --------------------------------------------

    def _edges(self, r_hinv, truncate):
        """Panel edges: the spline knots inside the integration range."""
        lo = max(LNK_LO, self.pk.lnk_min)
        up = self.pk.lnk_max
        if truncate:
            # the R-dependent upper limit -- the whole reason for `truncate`
            up = min(up, np.log(KCUT_COEF / r_hinv))
        if up <= lo:
            return None
        inner = self.pk.lnk[(self.pk.lnk > lo) & (self.pk.lnk < up)]
        return np.concatenate(([lo], inner, [up]))

    def _panel_points(self, edges):
        a, b = edges[:-1], edges[1:]
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        pts = (mid[:, None] + half[:, None] * self._nodes[None, :]).ravel()
        wts = (half[:, None] * self._weights[None, :]).ravel()
        return pts, wts

    def _integrand(self, lnk, r_hinv):
        r"""The :math:`\sigma^2` integrand in :math:`\ln k`."""
        k = np.exp(lnk)
        w = tophat_w(k * r_hinv)
        return k**3 * self.pk(lnk) * w * w / (2.0 * np.pi**2)

    def _d_integrand(self, lnk, r_hinv):
        r""":math:`d/d\ln R` of the integrand at **fixed** limits."""
        k = np.exp(lnk)
        x = k * r_hinv
        return (k**3 * self.pk(lnk) * 2.0 * tophat_w(x) * tophat_dw(x) * x
                / (2.0 * np.pi**2))

    def sigma2(self, r_hinv, truncate: bool = True):
        r""":math:`\sigma^2(R)`, dimensionless. ``r_hinv`` scalar, in Mpc/h."""
        r_hinv = float(r_hinv)
        if r_hinv <= 0.0:
            raise ValueError("R must be positive")
        edges = self._edges(r_hinv, truncate)
        if edges is None:
            return 0.0
        pts, wts = self._panel_points(edges)
        return float(np.dot(wts, self._integrand(pts, r_hinv)))

    def dsigma2_dlnr(self, r_hinv, truncate: bool = True):
        r""":math:`d\sigma^2/d\ln R`, by differentiating under the integral.

        NOTE: includes the Leibniz moving-boundary term
        :math:`-\left[k^3P W^2/2\pi^2\right]_{k=20/R}` whenever the
        truncation is active *and* :math:`20/R` lies inside the tabulated
        range. Omitting it biases :math:`dn/d\ln M` directly.
        """
        r_hinv = float(r_hinv)
        edges = self._edges(r_hinv, truncate)
        if edges is None:
            return 0.0
        pts, wts = self._panel_points(edges)
        value = float(np.dot(wts, self._d_integrand(pts, r_hinv)))
        lnk_up = np.log(KCUT_COEF / r_hinv)
        if truncate and lnk_up < self.pk.lnk_max:
            # d(ln k_up)/d(lnR) = -1, hence the minus sign
            value -= float(self._integrand(np.array([lnk_up]), r_hinv)[0])
        return value

    def sigma(self, r_hinv, truncate: bool = True):
        r""":math:`\sigma(R) = \sqrt{\sigma^2(R)}`, dimensionless."""
        return np.sqrt(self.sigma2(r_hinv, truncate=truncate))

    def dlnsigma2_dlnr(self, r_hinv, truncate: bool = True):
        r""":math:`d\ln\sigma^2/d\ln R`. Negative: variance falls with R."""
        s2 = self.sigma2(r_hinv, truncate=truncate)
        if s2 <= 0.0:
            raise ValueError(
                f"sigma^2(R = {r_hinv:g} Mpc/h) is not positive; R is "
                "outside the range the tabulated P(k) supports"
            )
        return self.dsigma2_dlnr(r_hinv, truncate=truncate) / s2

    # -- the FFTLog fast route ------------------------------------------

    def sigma2_fftlog(self, lnr_hinv, n_fine: int = 8192,
                      pad_decades: float = 3.0):
        r"""Untruncated :math:`\ln\sigma^2` and its derivative, by FFTLog.

        Returns ``(ln_sigma2, dlnsigma2_dlnr)`` on ``lnr_hinv``.

        NOTE: this is the ``truncate=False`` quantity and cannot be made
        otherwise -- see module NOTE 2. Validate it against
        ``sigma2(..., truncate=False)``, never against the default.

        NOTE: the input :math:`P(k)` is resampled onto a log grid
        **explicitly zero-padded** by ``pad_decades`` on each side. Two
        reasons, both necessary: the zero-outside policy then belongs to the
        sampled input rather than to mcfit's periodic padding, and the
        output :math:`R` grid is the reciprocal of the input :math:`k`
        range, so without padding it stops near :math:`1/k_{\max}` -- far
        above the grid floor at 0.0034 Mpc/h.
        """
        import mcfit

        lnr_hinv = np.asarray(lnr_hinv, dtype=float)
        key = (n_fine, pad_decades)
        if self._fftlog_plan is None or self._fftlog_plan[0] != key:
            pad = pad_decades * np.log(10.0)
            lnk_fine = np.linspace(self.pk.lnk_min - pad,
                                   self.pk.lnk_max + pad, int(n_fine))
            plan = mcfit.TophatVar(np.exp(lnk_fine), lowring=True)
            self._fftlog_plan = (key, lnk_fine, plan)
        _, lnk_fine, plan = self._fftlog_plan

        r_fft, var = plan(self.pk(lnk_fine), extrap=False)
        ln_r_fft = np.log(r_fft)
        if (ln_r_fft[0] > lnr_hinv.min() or ln_r_fft[-1] < lnr_hinv.max()):
            raise RuntimeError(
                f"FFTLog R grid [{r_fft[0]:.4g}, {r_fft[-1]:.4g}] Mpc/h does "
                f"not cover the requested R; raise pad_decades (currently "
                f"{pad_decades})"
            )
        window = ((ln_r_fft > lnr_hinv.min() - 0.5)
                  & (ln_r_fft < lnr_hinv.max() + 0.5))
        if np.any(var[window] <= 0.0):
            raise RuntimeError(
                "FFTLog sigma^2 is non-positive inside the requested R "
                "range -- ringing, or the zero-padding policy needs "
                "attention"
            )
        spline = CubicSpline(ln_r_fft[window], np.log(var[window]))
        return spline(lnr_hinv), spline.derivative()(lnr_hinv)

    def __repr__(self):
        return (f"SigmaGrid({self.pk!r}, nquad={self.nquad})")


if __name__ == "__main__":
    # a scale-free test spectrum: P = A k^n with a cutoff, so sigma^2 is
    # smooth and the two routes are comparable without a CAMB call
    k = np.logspace(-5.0, 3.0, 600)
    pk = LinearPk(k, 2.0e4 * k**-1.5 * np.exp(-((k / 50.0) ** 2)))
    grid = SigmaGrid(pk)
    print(grid, "\n")

    print(f"{'R [Mpc/h]':>10s}  {'sigma trunc':>12s}  {'sigma full':>11s}  "
          f"{'ratio-1':>10s}  {'dlnsig2/dlnR':>13s}")
    for r in (0.01, 0.1, 1.0, 8.0, 30.0):
        st = grid.sigma(r, truncate=True)
        sf = grid.sigma(r, truncate=False)
        d = grid.dlnsigma2_dlnr(r, truncate=True)
        print(f"{r:10.3f}  {st:12.6f}  {sf:11.6f}  {st / sf - 1:10.2e}  "
              f"{d:13.6f}")
    print("  <- the truncate column is the production quantity. Note the")
    print("     direction: the k <= 20/R limit bites at LARGE R, because")
    print(f"     20/R must fall inside the table (k_max = "
          f"{np.exp(pk.lnk_max):.0f} h/Mpc) to cut anything. At R = 0.01,")
    print("     20/R = 2000 > k_max, so nothing is cut and the two agree")
    print("     exactly.")

    # the Leibniz term, isolated. Its size is not universal: it is set by
    # how much power still sits at k = 20/R, so it is reported, not claimed.
    print("\nthe moving-boundary (Leibniz) term, as a fraction of "
          "dsigma^2/dlnR:")
    for r in (0.01, 0.1, 1.0, 8.0, 30.0):
        total = grid.dsigma2_dlnr(r, truncate=True)
        lnk_up = np.log(KCUT_COEF / r)
        active = lnk_up < pk.lnk_max
        boundary = (-float(grid._integrand(np.array([lnk_up]), r)[0])
                    if active else 0.0)
        print(f"  R = {r:6.3f} Mpc/h:  20/R = {KCUT_COEF / r:9.1f} h/Mpc  "
              f"{'active ' if active else 'inactive'}  "
              f"boundary/total = {boundary / total:9.2e}")
    print("  <- small here (<= 2e-4) only because this toy P(k) carries an")
    print("     exp[-(k/50)^2] cutoff, so there is little power left at")
    print("     k = 20/R. The term scales with P(20/R): on a CAMB spectrum")
    print("     with power out to k_max it is far larger. Never drop it on")
    print("     the grounds that it was small for some other spectrum.")

    # the derivative, checked against finite differences of sigma^2 itself
    print("\ndifferentiation under the integral vs. finite differences "
          "(truncate=False, where no boundary term exists):")
    for r in (0.1, 1.0, 8.0):
        h = 1e-5
        fd = (grid.sigma2(r * np.exp(h), truncate=False)
              - grid.sigma2(r * np.exp(-h), truncate=False)) / (2 * h)
        exact = grid.dsigma2_dlnr(r, truncate=False)
        print(f"  R = {r:5.2f}:  exact {exact:15.8e}  fd {fd:15.8e}  "
              f"rel {abs(exact / fd - 1):.2e}")

    # FFTLog: fast, and untruncated by construction
    lnr = np.log(np.array([0.1, 1.0, 8.0, 30.0]))
    ln_s2, dln_s2 = grid.sigma2_fftlog(lnr)
    print("\nFFTLog fast path against the untruncated reference:")
    print(f"{'R':>7s}  {'ln s2 fftlog':>13s}  {'ln s2 exact':>12s}  "
          f"{'rel':>9s}  {'d fftlog':>10s}  {'d exact':>10s}")
    for lr, ls, dls in zip(lnr, ln_s2, dln_s2):
        r = np.exp(lr)
        ex = np.log(grid.sigma2(r, truncate=False))
        dex = grid.dlnsigma2_dlnr(r, truncate=False)
        print(f"{r:7.2f}  {ls:13.8f}  {ex:12.8f}  {abs(ls / ex - 1):9.2e}  "
              f"{dls:10.5f}  {dex:10.5f}")
    print("  <- compared to truncate=False, as the docstring insists.")
