r"""FFTLog engine for bin-averaged double-Bessel covariance integrals.

Evaluates, for log-spaced (geometric) radial bins with edge ratio
:math:`\rho`,

.. math::

    G_d(\theta) = \int_0^\infty d\ell\, \ell\, C(\ell)\,
        \bar J_2(\ell; \theta, \rho\theta)\,
        \bar J_2(\ell; \alpha_d\theta, \rho\alpha_d\theta),
    \qquad \alpha_d = \rho^d

— the :math:`\ell`-integral of the Gaussian :math:`\Delta\Sigma`
covariance for the radial-bin pair :math:`(i, i+d)` — as ONE FFTLog
transform per diagonal offset :math:`d`, replacing the brute-force
trapz over :math:`\ln\ell`.

Method (see ``docs/covariance_fftlog_math.md`` for the full derivation):
with :math:`\psi(x) = 2J_0(x) + xJ_1(x)` the annulus-averaged kernel is

.. math::

    \bar J_2(\ell; a, \rho a) = \frac{2\,[\psi(u) - \psi(\rho u)]}
        {(\rho^2 - 1)\, u^2}, \qquad u = \ell a,

so the product kernel :math:`K_d(u) = \bar J_2(u)\bar J_2(\alpha_d u)`
depends only on :math:`u = \ell\theta` and the fixed ratio
:math:`\alpha_d`.  Expanding :math:`\psi\psi` yields 16 elementary
:math:`u^{p-4} J_\mu(c_1 u) J_\nu(c_2 u)` terms whose Mellin transforms
are closed forms (Gradshteyn–Ryzhik 6.574.1, via
``mcfit.kernels.Mellin_DoubleBesselJ``); they are **summed at the
Mellin-coefficient level** before the single inverse FFT, so the four
orders of small-:math:`u` cancellation (:math:`K_d \sim u^4`) happen
exactly in the analytic continuation rather than in floating point
(cf. Fang, Eifler & Krause 2020).

Usage: pass ``F = ell**2 * C(ell)``; the transform returns
:math:`G_d` on its output :math:`\theta` grid,

.. math::

    G_d(\theta) = \int_0^\infty \frac{dx}{x}\, F(x)\, K_d(x\theta).

Constant (white-noise) :math:`C_\ell` components must be removed first
and added back with :func:`white_noise_diagonal` — orthogonality of the
bin-averaged kernels makes them exactly diagonal:

.. math::

    \int_0^\infty \ell\, d\ell\, \bar J_2^{(i)} \bar J_2^{(j)}
        = \delta_{ij}\, \frac{2}{\theta_{i,\max}^2 - \theta_{i,\min}^2}.

NOTE: **why this exists, and what it corrects.** The direct quadrature in
`clenspy.covariance.deltasigma` is truncation-limited: its error goes as
:math:`2.5/k_{\max}` because :math:`\hat J_2 \sim x^{-3/2}` leaves
:math:`k\hat J_2^2` falling only as :math:`k^{-2}`. It was previously
argued that FFTLog could not help, on the grounds that
:math:`\hat J_2(kr_p)\hat J_2(kr_p')` under one :math:`k` integral is a
*bilinear* form rather than a Hankel transform of a single function. That
argument is wrong for **geometric** bins: the pair ratio
:math:`\alpha_d = \rho^d` then depends only on the diagonal offset
:math:`d`, so the product kernel is a function of :math:`u = \ell\theta`
alone and there are only :math:`n_{\rm bins}` distinct transforms. Hence
one FFTLog per diagonal, and hence the ``theta_edges`` geometric check in
`GaussianCovFFTLog` -- it is not a convenience, it is the condition that
makes the method exist.

NOTE: **units.** Dimensionless kernels throughout. ``ell`` is a multipole
and ``theta_edges`` are radians; the returned covariance carries whatever
:math:`C_\ell` carries, squared. ``f_sky`` is a fraction, not a solid
angle.

NOTE: the ``q`` tilt is a **named approximation with a domain**. The summed
kernel is analytic for :math:`-4 < {\rm Re}\,s < 3`, but the individual
terms have poles at :math:`s \in \{0, 2\}`, so ``q`` must stay well inside
:math:`(0, 2)`; the default 1.0 maximises the distance to both. Moving it
outside that window does not fail loudly -- it rings.

NOTE: `covariance_trapz_reference` is a **test-only** path, kept because
the plan asks for the brute-force version to survive as a reference so an
equivalence test isolates the integration method rather than the physics.
It is slow by construction; do not call it from a sampler.
"""

from __future__ import annotations

import mcfit
import numpy as np
from mcfit.kernels import Mellin_DoubleBesselJ

from .bessel import j2_bin

__all__ = [
    "BinAveragedJ2DoubleBessel",
    "white_noise_diagonal",
    "GaussianCovFFTLog",
]


_MK_CACHE: dict = {}


def _mellin_binavg_j2j2(rho: float, alpha: float):
    r"""Mellin transform :math:`U(s) = \int_0^\infty u^{s-1} K_d(u)\,du`
    of the product kernel

    .. math::

        K_d(u) = \frac{4\,[\psi(u) - \psi(\rho u)]
            [\psi(\alpha u) - \psi(\rho\alpha u)]}
            {(\rho^2-1)^2\, \alpha^2\, u^4}

    as the exact 16-term sum of shifted/rescaled double-Bessel Mellin
    transforms:

    .. math::

        \int_0^\infty u^{\sigma-1} J_\mu(c_1 u) J_\nu(c_2 u)\, du
            = c_1^{-\sigma}\, M_{\mu\nu}^{(c_2/c_1)}(\sigma).

    Summation happens here, in the analytic continuation, so the
    :math:`u^{-4}` tails of the individual terms cancel exactly.
    """
    rho = float(rho)
    alpha = float(alpha)
    A = 4.0 / ((rho**2 - 1.0) ** 2 * alpha**2)

    # (c1, sign1) x (c2, sign2); psi(c u) = 2 J0(c u) + c u J1(c u)
    pieces = []
    for c1, s1 in ((1.0, +1.0), (rho, -1.0)):
        for c2, s2 in ((alpha, +1.0), (rho * alpha, -1.0)):
            beta = c2 / c1
            sign = s1 * s2
            # (mu, nu, p, coeff): u^{p-4} coeff J_mu(c1 u) J_nu(c2 u)
            for mu, nu, p, coeff in (
                (0, 0, 0, 4.0),
                (0, 1, 1, 2.0 * c2),
                (1, 0, 1, 2.0 * c1),
                (1, 1, 2, c1 * c2),
            ):
                pieces.append(
                    (sign * coeff, c1, p, Mellin_DoubleBesselJ(beta, mu, nu))
                )

    def MK(z):
        z = np.asarray(z)
        # disk cache: the mpmath 2F1 evaluations cost ~seconds per call
        # and depend only on (rho, alpha, sampled z line) — never on
        # cosmology — so persist them across processes.
        import hashlib
        import os
        import tempfile

        key = hashlib.md5(
            np.asarray([rho, alpha]).tobytes() + np.ascontiguousarray(z).tobytes()
        ).hexdigest()
        cache_dir = os.environ.get(
            "CLENSPY_FFTLOG_CACHE",
            os.path.join(tempfile.gettempdir(), "clenspy_fftlog_cache"),
        )
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, key + ".npy")
        if os.path.exists(cache_file):
            return np.load(cache_file)

        out = np.zeros_like(z, dtype=complex)
        for coeff, c1, p, M in pieces:
            sigma = z + (p - 4)
            out = out + coeff * c1 ** (-sigma) * np.asarray(
                M(sigma), dtype=complex
            )
        out = A * out
        np.save(cache_file, out)
        return out

    return MK


class BinAveragedJ2DoubleBessel(mcfit.mcfit):
    r"""FFTLog transform with the summed bin-averaged :math:`\bar J_2
    \bar J_2` Mellin kernel.

    Computes :math:`G_d(\theta) = \int_0^\infty F(\ell)\,
    K_d(\ell\theta)\, d\ell/\ell` — pass ``F = ell**2 * C(ell)`` to get
    the covariance :math:`\ell`-integral
    :math:`\int d\ell\,\ell\, C(\ell) \bar J_2^{(i)} \bar J_2^{(i+d)}`
    evaluated at :math:`\theta = \theta_{i,\min}` for every ``i`` at once.

    Parameters
    ----------
    ell : ndarray
        Log-spaced multipole grid.
    rho : float
        Geometric edge ratio of the radial binning
        (:math:`\theta_{\max}/\theta_{\min}` per bin).
    alpha : float
        Bin-pair scale ratio :math:`\rho^d` for diagonal offset ``d``.
    q : float
        FFTLog tilt.  The summed kernel is analytic for
        :math:`-4 < \mathrm{Re}\,s < 3`, but the *individual* terms have
        poles at :math:`s \in \{0, 2\}` — keep ``q`` well inside
        ``(0, 2)``; the default 1.0 maximizes the distance to both.
    lowring : bool
        Low-ringing output-grid condition.
    """

    def __init__(self, ell, rho: float, alpha: float, q: float = 1.0,
                 lowring: bool = True, **kwargs) -> None:
        key = (round(float(rho), 12), round(float(alpha), 12))
        if key not in _MK_CACHE:
            _MK_CACHE[key] = _mellin_binavg_j2j2(rho, alpha)
        MK = _MK_CACHE[key]
        super().__init__(np.asarray(ell, dtype=float), MK, q,
                         lowring=lowring, **kwargs)
        self.rho = float(rho)
        self.alpha = float(alpha)


def white_noise_diagonal(theta_edges, noise: float, f_sky: float):
    r"""Exact diagonal of the constant-:math:`C_\ell` covariance term.

    For :math:`C_\ell = N` (white noise) and disjoint annuli, bin-averaged
    :math:`J_2` orthogonality gives

    .. math::

        {\rm Cov}^{\rm white}_{ij} = \delta_{ij}\,
            \frac{N}{4\pi^2 f_{\rm sky}
            (\theta_{i,\max}^2 - \theta_{i,\min}^2)}
            = \delta_{ij}\, \frac{N}{A_{\rm survey}\, A_i / \pi}

    with :math:`A_i = \pi(\theta_{i,\max}^2 - \theta_{i,\min}^2)` the
    annulus solid angle.  Returns the diagonal vector (length
    ``len(theta_edges) - 1``).
    """
    theta_edges = np.asarray(theta_edges, dtype=float)
    return noise / (
        4.0 * np.pi**2 * f_sky * (theta_edges[1:] ** 2 - theta_edges[:-1] ** 2)
    )


class GaussianCovFFTLog:
    r"""Gaussian covariance matrix over geometric angular bins via FFTLog.

    .. math::

        {\rm Cov}_{ij} = \frac{1}{4\pi f_{\rm sky}} \int
            \frac{\ell\, d\ell}{2\pi}\, C_{\rm smooth}(\ell)\,
            \bar J_2^{(i)}(\ell)\, \bar J_2^{(j)}(\ell)
            \;+\; \delta_{ij}\, {\rm Cov}^{\rm white}_{ii}

    One :class:`BinAveragedJ2DoubleBessel` per diagonal offset ``d``
    (kernels cached per binning geometry — reusable across redshift/
    richness bins and cosmologies); the smooth :math:`C(\ell)` must decay
    at high :math:`\ell` (strip the constant noise terms and pass them via
    ``noise_const``).

    Parameters
    ----------
    ell : ndarray
        Log-spaced grid on which ``C_smooth`` will be provided.
    theta_edges : ndarray
        Geometric angular bin edges [rad]; ``edges[i+1]/edges[i]`` must be
        constant.
    f_sky : float
    q : float
        FFTLog tilt (see :class:`BinAveragedJ2DoubleBessel`).
    """

    def __init__(self, ell, theta_edges, f_sky: float, q: float = 1.0) -> None:
        self.ell = np.asarray(ell, dtype=float)
        self.theta_edges = np.asarray(theta_edges, dtype=float)
        ratios = self.theta_edges[1:] / self.theta_edges[:-1]
        rho = ratios[0]
        if not np.allclose(ratios, rho, rtol=1e-8):
            raise ValueError(
                "theta_edges must be geometric (constant ratio); got "
                f"ratios in [{ratios.min():.6g}, {ratios.max():.6g}]"
            )
        self.rho = float(rho)
        self.n_bins = self.theta_edges.size - 1
        self.f_sky = float(f_sky)
        self.q = float(q)
        self._transforms: dict[int, BinAveragedJ2DoubleBessel] = {}

    def _transform(self, d: int) -> BinAveragedJ2DoubleBessel:
        if d not in self._transforms:
            self._transforms[d] = BinAveragedJ2DoubleBessel(
                self.ell, self.rho, self.rho**d, q=self.q
            )
        return self._transforms[d]

    def covariance(self, C_smooth, noise_const: float = 0.0) -> np.ndarray:
        r"""Assemble the (n_bins, n_bins) covariance.

        Parameters
        ----------
        C_smooth : ndarray
            Decaying part of the total :math:`C(\ell)` on ``self.ell``.
        noise_const : float
            Constant (white) part of :math:`C(\ell)`, added analytically
            on the diagonal.
        """
        C_smooth = np.asarray(C_smooth, dtype=float)
        F = self.ell**2 * C_smooth
        theta_lo = self.theta_edges[:-1]
        cov = np.zeros((self.n_bins, self.n_bins))
        for d in range(self.n_bins):
            tr = self._transform(d)
            y, G = tr(F, extrap=True)
            # log-linear interpolation of G at theta_{i,min}
            g_i = np.interp(np.log(theta_lo[: self.n_bins - d]), np.log(y), G)
            for i, g in enumerate(g_i):
                cov[i, i + d] = g
                cov[i + d, i] = g
        cov /= 8.0 * np.pi**2 * self.f_sky
        if noise_const != 0.0:
            cov[np.diag_indices_from(cov)] += white_noise_diagonal(
                self.theta_edges, noise_const, self.f_sky
            )
        return cov

    def covariance_trapz_reference(
        self, C_total, dlnell: float = 1e-3, ell_range=(1e-1, 1e7)
    ) -> np.ndarray:
        """Legacy brute-force trapz over ln(ell) with the closed-form
        bin-averaged kernels — slow validation reference ONLY."""
        lnell = np.arange(np.log(ell_range[0]), np.log(ell_range[1]), dlnell)
        ell = np.exp(lnell)
        C = np.interp(ell, self.ell, np.asarray(C_total, dtype=float),
                      left=C_total[0], right=0.0)
        kernels = [
            j2_bin(ell, self.theta_edges[i], self.theta_edges[i + 1])
            for i in range(self.n_bins)
        ]
        cov = np.zeros((self.n_bins, self.n_bins))
        for i in range(self.n_bins):
            for j in range(i, self.n_bins):
                integrand = ell**2 * C * kernels[i] * kernels[j]
                val = np.trapezoid(integrand, lnell)
                cov[i, j] = cov[j, i] = val
        return cov / (8.0 * np.pi**2 * self.f_sky)
