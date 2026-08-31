r"""The variance of the linear density field, :math:`\sigma^2(R)`.

.. math::
    \sigma^2(R) = \int d\ln k\;\frac{k^{3}P_{\rm lin}(k)}{2\pi^{2}}\,W^{2}(kR)

Computed once, here: `clenspy.cosmology.TinkerMassFunction` and
`clenspy.cosmology.BiasModel` fit the same peak height and both take a
`SigmaGrid`.

NOTE: physical units, h-free -- k in 1/Mpc, P in Mpc^3, R in Mpc; output
dimensionless. Conventions, provenance, and the truncation/Leibniz/FFTLog
details: ``docs/mass_function.md``.
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
    "SigmaGrid",
    "lnr_grid",
]

#: Production lnR grid, ``sigma.f90``: 969 points, ``lnR = -5.684 + 0.01 i``,
#: R in Mpc/h. R spans 0.0034 to 54.6 Mpc/h.
LNR1 = -5.684
LNR2 = 4.0
STEP = 0.01
NR = 969

#: Fixed lower quadrature limit of ``sigma.f90``: :math:`k = 10^{-4}` 1/Mpc.
#: Binds only when the tabulated k range extends below it.
LNK_LO = np.log(1.0e-4)

#: Upper quadrature limit coefficient: :math:`k_{\max} = 20/R`, i.e. the
#: dimensionless cut :math:`kR \le 20`. See the module NOTE -- this is
#: algorithm-defining, and FFTLog cannot express it.
KCUT_COEF = 20.0


def lnr_grid():
    """The production 969-point ``lnR`` grid, R in Mpc/h."""
    return LNR1 + STEP * np.arange(NR)


class SigmaGrid:
    r"""Top-hat variance :math:`\sigma^2(R)` and its :math:`\ln R` derivative.

    `sigma2`/`dsigma2_dlnr` are the Gauss--Legendre reference (with the
    :math:`kR \le 20` truncation and its Leibniz term); `sigma2_fftlog`
    is the fast untruncated route -- compare it to ``truncate=False`` only.

    Parameters
    ----------
    k : array-like
        Wavenumbers [1/Mpc], physical, strictly ascending.
    pk : array-like
        Linear power spectrum at z=0 [Mpc^3], positive, same shape.
    nquad : int, optional
        Gauss--Legendre order per panel (default: 24).
    """

    #: default GL order per panel; 24 vs 48 agree to 1e-16
    NQUAD = 24

    def __init__(self, k, pk, nquad: int = NQUAD):
        k = np.asarray(k, dtype=float)
        pk = np.asarray(pk, dtype=float)
        if k.shape != pk.shape:
            raise ValueError(
                f"k and P must have the same shape, got {k.shape} and "
                f"{pk.shape}"
            )
        if np.any(np.diff(k) <= 0.0):
            raise ValueError("k must be strictly ascending")
        if np.any(pk <= 0.0):
            raise ValueError("P must be positive (it is splined in log)")
        self.lnk = np.log(k)
        self._lnpk_spline = CubicSpline(self.lnk, np.log(pk),
                                        bc_type="natural")
        self.lnk_min = float(self.lnk[0])
        self.lnk_max = float(self.lnk[-1])
        self.nquad = int(nquad)
        self._nodes, self._weights = np.polynomial.legendre.leggauss(
            self.nquad
        )
        self._fftlog_plan = None

    def pk(self, lnk):
        """:math:`P(k)` at ``lnk``; exactly zero outside the table."""
        lnk = np.asarray(lnk, dtype=float)
        scalar = lnk.ndim == 0
        lnk = np.atleast_1d(lnk)
        out = np.zeros_like(lnk)
        # strict inequalities, matching Linear_Pk: the endpoints are out
        inside = (lnk > self.lnk_min) & (lnk < self.lnk_max)
        out[inside] = np.exp(self._lnpk_spline(lnk[inside]))
        return out[0] if scalar else out

    # -- the reference route --------------------------------------------

    def _edges(self, r, truncate):
        """Panel edges: the spline knots inside the integration range."""
        lo = max(LNK_LO, self.lnk_min)
        up = self.lnk_max
        if truncate:
            # the R-dependent upper limit -- the whole reason for `truncate`
            up = min(up, np.log(KCUT_COEF / r))
        if up <= lo:
            return None
        inner = self.lnk[(self.lnk > lo) & (self.lnk < up)]
        return np.concatenate(([lo], inner, [up]))

    def _panel_points(self, edges):
        a, b = edges[:-1], edges[1:]
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        pts = (mid[:, None] + half[:, None] * self._nodes[None, :]).ravel()
        wts = (half[:, None] * self._weights[None, :]).ravel()
        return pts, wts

    def _integrand(self, lnk, r):
        r"""The :math:`\sigma^2` integrand in :math:`\ln k`."""
        k = np.exp(lnk)
        w = tophat_w(k * r)
        return k**3 * self.pk(lnk) * w * w / (2.0 * np.pi**2)

    def _d_integrand(self, lnk, r):
        r""":math:`d/d\ln R` of the integrand at **fixed** limits."""
        k = np.exp(lnk)
        x = k * r
        return (k**3 * self.pk(lnk) * 2.0 * tophat_w(x) * tophat_dw(x) * x
                / (2.0 * np.pi**2))

    def sigma2(self, r, truncate: bool = True):
        r""":math:`\sigma^2(R)`, dimensionless. ``r`` scalar."""
        r = float(r)
        if r <= 0.0:
            raise ValueError("R must be positive")
        edges = self._edges(r, truncate)
        if edges is None:
            return 0.0
        pts, wts = self._panel_points(edges)
        return float(np.dot(wts, self._integrand(pts, r)))

    def dsigma2_dlnr(self, r, truncate: bool = True):
        r""":math:`d\sigma^2/d\ln R`, differentiated under the integral;
        includes the Leibniz moving-boundary term when the truncation is
        active."""
        r = float(r)
        edges = self._edges(r, truncate)
        if edges is None:
            return 0.0
        pts, wts = self._panel_points(edges)
        value = float(np.dot(wts, self._d_integrand(pts, r)))
        lnk_up = np.log(KCUT_COEF / r)
        if truncate and lnk_up < self.lnk_max:
            # d(ln k_up)/d(lnR) = -1, hence the minus sign
            value -= float(self._integrand(np.array([lnk_up]), r)[0])
        return value

    def sigma(self, r, truncate: bool = True):
        r""":math:`\sigma(R) = \sqrt{\sigma^2(R)}`, dimensionless."""
        return np.sqrt(self.sigma2(r, truncate=truncate))

    def dlnsigma2_dlnr(self, r, truncate: bool = True):
        r""":math:`d\ln\sigma^2/d\ln R`. Negative: variance falls with R."""
        s2 = self.sigma2(r, truncate=truncate)
        if s2 <= 0.0:
            raise ValueError(
                f"sigma^2(R = {r:g}) is not positive; R is outside the "
                "range the tabulated P(k) supports"
            )
        return self.dsigma2_dlnr(r, truncate=truncate) / s2

    # -- the FFTLog fast route ------------------------------------------

    def sigma2_fftlog(self, lnr, n_fine: int = 8192,
                      pad_decades: float = 3.0):
        r"""Untruncated :math:`\ln\sigma^2` and its derivative, by FFTLog.

        Returns ``(ln_sigma2, dlnsigma2_dlnr)`` on ``lnr``. This is the
        ``truncate=False`` quantity; validate it against that, never the
        default. The input P(k) is resampled zero-padded by ``pad_decades``
        per side so the output R grid covers the production range.
        """
        import mcfit

        lnr = np.asarray(lnr, dtype=float)
        key = (n_fine, pad_decades)
        if self._fftlog_plan is None or self._fftlog_plan[0] != key:
            pad = pad_decades * np.log(10.0)
            lnk_fine = np.linspace(self.lnk_min - pad,
                                   self.lnk_max + pad, int(n_fine))
            plan = mcfit.TophatVar(np.exp(lnk_fine), lowring=True)
            self._fftlog_plan = (key, lnk_fine, plan)
        _, lnk_fine, plan = self._fftlog_plan

        r_fft, var = plan(self.pk(lnk_fine), extrap=False)
        ln_r_fft = np.log(r_fft)
        if (ln_r_fft[0] > lnr.min() or ln_r_fft[-1] < lnr.max()):
            raise RuntimeError(
                f"FFTLog R grid [{r_fft[0]:.4g}, {r_fft[-1]:.4g}] does "
                f"not cover the requested R; raise pad_decades (currently "
                f"{pad_decades})"
            )
        window = ((ln_r_fft > lnr.min() - 0.5)
                  & (ln_r_fft < lnr.max() + 0.5))
        if np.any(var[window] <= 0.0):
            raise RuntimeError(
                "FFTLog sigma^2 is non-positive inside the requested R "
                "range -- ringing, or the zero-padding policy needs "
                "attention"
            )
        spline = CubicSpline(ln_r_fft[window], np.log(var[window]))
        return spline(lnr), spline.derivative()(lnr)

    def __repr__(self):
        return (f"SigmaGrid(k = [{np.exp(self.lnk_min):.3g}, "
                f"{np.exp(self.lnk_max):.3g}], n = {self.lnk.size}, "
                f"nquad={self.nquad})")


if __name__ == "__main__":
    # a scale-free test spectrum: P = A k^n with a cutoff, so sigma^2 is
    # smooth and the two routes are comparable without a CAMB call
    k = np.logspace(-5.0, 3.0, 600)
    grid = SigmaGrid(k, 2.0e4 * k**-1.5 * np.exp(-((k / 50.0) ** 2)))
    print(grid, "\n")

    print(f"{'R':>10s}  {'sigma trunc':>12s}  {'sigma full':>11s}  "
          f"{'ratio-1':>10s}  {'dlnsig2/dlnR':>13s}")
    for r in (0.01, 0.1, 1.0, 8.0, 30.0):
        st = grid.sigma(r, truncate=True)
        sf = grid.sigma(r, truncate=False)
        d = grid.dlnsigma2_dlnr(r, truncate=True)
        print(f"{r:10.3f}  {st:12.6f}  {sf:11.6f}  {st / sf - 1:10.2e}  "
              f"{d:13.6f}")
    print("  <- the truncate column is the production quantity. Note the")
    print("     direction: the kR <= 20 limit bites at LARGE R, because")
    print(f"     20/R must fall inside the table (k_max = "
          f"{np.exp(grid.lnk_max):.0f}) to cut anything. At R = 0.01,")
    print("     20/R = 2000 > k_max, so nothing is cut and the two agree")
    print("     exactly.")

    # the Leibniz term, isolated. Its size is not universal: it is set by
    # how much power still sits at k = 20/R, so it is reported, not claimed.
    print("\nthe moving-boundary (Leibniz) term, as a fraction of "
          "dsigma^2/dlnR:")
    for r in (0.01, 0.1, 1.0, 8.0, 30.0):
        total = grid.dsigma2_dlnr(r, truncate=True)
        lnk_up = np.log(KCUT_COEF / r)
        active = lnk_up < grid.lnk_max
        boundary = (-float(grid._integrand(np.array([lnk_up]), r)[0])
                    if active else 0.0)
        print(f"  R = {r:6.3f}:  20/R = {KCUT_COEF / r:9.1f}  "
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
