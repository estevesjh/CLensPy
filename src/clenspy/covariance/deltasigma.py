r"""Gaussian-field covariance of :math:`\Delta\Sigma`, Wu et al. (2019).

Their equation for :math:`{\rm Cov}^{\rm Gauss}`
(``eq:cov_DS``), transcribed:

.. math::
    {\rm Cov}^{\rm Gauss}\!\left[\Delta\Sigma(r_p),
                                 \Delta\Sigma(r_p')\right] =
    \frac{1}{4\pi f_{\rm sky}}
    \int \frac{k\,dk}{2\pi}\,
    \hat J_2(k r_p)\,\hat J_2(k r_p')
    \left[
      \left(C_\ell^{hh} + \frac{1}{n_h}\right)
      \left(C_\ell^{\Sigma\Sigma}
            + \langle\Sigma_{\rm crit}\rangle^2
              \frac{\sigma_\gamma^2}{n_s}\right)
      + \left(C_\ell^{h\Sigma}\right)^2
    \right]

with the radial-bin-averaged Bessel function (``eq:hJ2``)

.. math::
    \hat J_2(\ell, \theta_{\min}, \theta_{\max}) =
      \frac{2}{\ell^2(\theta_{\max}^2 - \theta_{\min}^2)}
      \Big[
        2\big(J_0(\ell\theta_{\min}) - J_0(\ell\theta_{\max})\big)
        + \ell\big(\theta_{\min}J_1(\ell\theta_{\min})
                   - \theta_{\max}J_1(\ell\theta_{\max})\big)
      \Big]

**The bracket expands into five physically distinct terms**, and they are
stored separately because the scientific argument is almost always about
which one dominates where:

.. math::
    \left(C^{hh} + N_h\right)\left(C^{\Sigma\Sigma} + N_\Sigma\right)
    + \left(C^{h\Sigma}\right)^2
    = \underbrace{C^{hh}C^{\Sigma\Sigma}}_{\rm lss\_lss}
    + \underbrace{C^{hh}N_\Sigma}_{\rm lss\_shape}
    + \underbrace{N_h C^{\Sigma\Sigma}}_{\rm shot\_lss}
    + \underbrace{N_h N_\Sigma}_{\rm shot\_shape}
    + \underbrace{\left(C^{h\Sigma}\right)^2}_{\rm cross}

with :math:`N_h = 1/n_h` the halo shot noise and
:math:`N_\Sigma = \langle\Sigma_{\rm crit}\rangle^2\sigma_\gamma^2/n_s`
the shape noise. Grouping them into three ("cosmic shear", "shape noise",
"cross") requires choosing which of the mixed terms goes where; the five
are kept instead, and `cov` takes a ``terms`` selector so any grouping can
be formed without this module having to pick one.

NOTE: **thin lens slice.** The conversion from :math:`(\ell,\theta)` to
:math:`(k, r_p)` assumes :math:`\theta = r_p/\chi_h` and
:math:`\ell = k\chi_h` at a *single* :math:`\chi_h`, so this expression
applies only to a **thin** halo-redshift bin. Wu et al. say so explicitly.
A wide bin needs the covariance evaluated per slice and averaged, not this
formula at the mean redshift.

NOTE: **the shot_shape term is evaluated in closed form, not by
quadrature** -- and it is the term that dominates at small :math:`r_p`.
Its bracket :math:`N_h N_\Sigma` carries no :math:`k`, so Hankel closure,
:math:`\int_0^\infty J_2(ka)J_2(kb)\,k\,dk = \delta(a-b)/a`, applies
exactly. Bin-averaging over **disjoint contiguous** annuli collapses it:

.. math::
    \int_0^\infty \frac{k\,dk}{2\pi}\,
      \hat J_2(k r_p)\,\hat J_2(k r_p')
    = \frac{\delta_{ij}}{A_{{\rm ann},i}},
    \qquad
    A_{\rm ann} = \pi\left(r_{p,\max}^2 - r_{p,\min}^2\right)

so that term is exact, diagonal, and free. ``exact_shot_shape=False``
forces the quadrature instead, which is how the two are cross-checked.

NOTE: :math:`A_{\rm ann}` is in **Mpc^2, not steradians**. The integration
variable is conjugate to :math:`r_p`, so the closure bin-averages over
:math:`r_p`; using the angular area is wrong by :math:`\chi_h^2`, a factor
of :math:`10^6` at :math:`\chi_h = 1100` Mpc. I made exactly that error
while deriving this, and the test against the closure is what caught it.

NOTE: **the surviving quadrature is truncation-limited, not
node-limited**, measured rather than assumed. Against the closure result
the relative error goes as

.. math::
    \epsilon \simeq \frac{2.5}{k_{\max}\,[{\rm Mpc}^{-1}]}

-- 2.5e-2 at :math:`k_{\max}=10^2`, 2.4e-3 at :math:`10^3`, 2.4e-4 at
:math:`10^4` -- while raising ``n_k`` from 2048 to 32768 at fixed
:math:`k_{\max}` changes nothing at all. The cause is the
:math:`\hat J_2 \sim x^{-3/2}` oscillatory tail, which leaves
:math:`k\hat J_2^2` falling only as :math:`k^{-2}`. Hence the default
:math:`k_{\max} = 10^5`, and `convergence` sweeps **both** axes.

NOTE: **there is still no FFTLog, and now for a sharper reason.** The
integral is a *bilinear* form -- :math:`\hat J_2(kr_p)\hat J_2(kr_p')`
under one :math:`k` integral -- not a Hankel transform of a single
function, so it does not factorise into a transform FFTLog could
accelerate; as :math:`A^{\rm T}{\rm diag}(w_kP_k)A` it costs
:math:`O(n_kn_r^2)`, already negligible. What FFTLog *would* have bought
is **precision** on that oscillatory tail. But the one term where the tail
dominates is exactly the one that is now analytic, so the remaining
quadrature acts only on brackets that fall with :math:`k` and converge
quickly. The gain is taken by the closure identity instead, and taken
exactly rather than approximately.

NOTE: **units.** :math:`r_p` in Mpc (comoving, h-free) and
:math:`\Delta\Sigma` in :math:`M_\odot/{\rm Mpc}^2`, so the covariance is
in :math:`(M_\odot/{\rm Mpc}^2)^2`. :math:`C_\ell^{\Sigma\Sigma}` carries
:math:`\Sigma^2` and :math:`C_\ell^{h\Sigma}` carries :math:`\Sigma`,
exactly as Wu et al. note -- which is the dimensional check that the
bracket's five terms are commensurate.

NOTE: :math:`f_{\rm sky}` enters as :math:`1/(4\pi f_{\rm sky})` and is
the **only** place survey area appears. It is not :math:`\Omega(z)`: the
footprint that normalises the counts and the sky fraction that sets the
number of independent modes are different quantities used differently, and
conflating them is a factor of :math:`4\pi`.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0, j1

__all__ = ["ALL_TERMS", "J2_SERIES_CUTOFF", "DeltaSigmaCovariance", "j2_bin"]

#: The five terms of the expanded bracket, in the order they are summed.
ALL_TERMS = ("lss_lss", "lss_shape", "shot_lss", "shot_shape", "cross")

#: Below :math:`\ell\theta_{\max}` of this, use the Taylor series rather
#: than the closed form. At 1.0 the two agree to 1e-13; the closed form
#: degrades below it and the series diverges above ~6.
J2_SERIES_CUTOFF = 1.0

#: Terms retained in the series branch. 14 is exact to fp64 at the cutoff.
J2_SERIES_TERMS = 14


def j2_bin(ell, theta_min, theta_max):
    r"""The radial-bin-averaged Bessel function :math:`\hat J_2`.

    .. math::
        \hat J_2 = \frac{2}{\ell^2(\theta_{\max}^2-\theta_{\min}^2)}
          \Big[2\big(J_0(\ell\theta_{\min}) - J_0(\ell\theta_{\max})\big)
          + \ell\big(\theta_{\min}J_1(\ell\theta_{\min})
                     - \theta_{\max}J_1(\ell\theta_{\max})\big)\Big]

    the average of :math:`J_2(\ell\theta)` over the annulus
    :math:`\theta_{\min} < \theta < \theta_{\max}` weighted by
    :math:`2\pi\theta\,d\theta`. Wu et al. (2019) ``eq:hJ2``.

    NOTE: **the closed form above is unusable for**
    :math:`\ell\theta_{\max} \lesssim 1`, and this function does not use it
    there. Its bracket is a near-total cancellation: both
    :math:`2(J_0 - J_0)` and :math:`\ell(\theta J_1 - \theta J_1)` are
    :math:`O(x^2)` with *opposite* signs and cancel to :math:`O(x^4)`, then
    get divided by :math:`\ell^2\Delta\theta^2`. At
    :math:`\ell\theta = 10^{-3}` the surviving value is nine orders below
    the terms that produced it, so fp64 returns roughly four correct
    digits -- measured at **4.8e-4** relative error against direct
    quadrature. Below `J2_SERIES_CUTOFF` the Taylor series is used instead,

    .. math::
        \hat J_2 = \sum_{m\ge0}\frac{(-1)^m\,\ell^{2m+2}
          \left(\theta_{\max}^{2m+4} - \theta_{\min}^{2m+4}\right)}
          {2^{2m+2}\,m!\,(m+2)!\,(m+2)
           \left(\theta_{\max}^2-\theta_{\min}^2\right)},

    obtained by averaging :math:`J_2`'s own series term by term. The two
    branches agree to 1e-13 at the cutoff, and the series diverges beyond
    :math:`x \sim 6` while the closed form is exact there -- so neither
    alone is sufficient.

    NOTE: this is an **average, not a sample**. :math:`J_2` peaks at
    :math:`\ell\theta = 2` and the first peak barely moves with bin width,
    but the decay does: a wider bin decays faster in :math:`\ell`. So
    replacing :math:`\hat J_2` by :math:`J_2` at the bin centre is
    accurate for the LSS terms (which fall steeply in :math:`\ell`, so only
    the first peak matters) and *wrong* for the shot- and shape-noise terms
    (which are :math:`\ell`-independent, so the whole tail contributes).
    That asymmetry is why the bin average cannot be skipped.

    NOTE: dimensionless. ``ell`` is dimensionless and the two angles are in
    radians; only their ratio to :math:`1/\ell` matters.

    Parameters
    ----------
    ell : float or array-like
        Multipole (or :math:`k\chi_h`). Must be positive.
    theta_min, theta_max : float or array-like
        Annulus edges in radians, ``theta_max > theta_min >= 0``.

    Returns
    -------
    np.ndarray
        :math:`\hat J_2`, broadcast over the inputs.
    """
    ell, theta_min, theta_max = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (ell, theta_min, theta_max))
    )
    if np.any(ell <= 0.0):
        raise ValueError("ell must be positive")
    if np.any(theta_max <= theta_min) or np.any(theta_min < 0.0):
        raise ValueError("require theta_max > theta_min >= 0")

    delta_sq = theta_max**2 - theta_min**2
    out = np.empty(ell.shape, dtype=float)

    # the closed form, wherever it is well conditioned
    big = ell * theta_max >= J2_SERIES_CUTOFF
    if np.any(big):
        x_lo, x_hi = ell[big] * theta_min[big], ell[big] * theta_max[big]
        out[big] = (2.0 / (ell[big] ** 2 * delta_sq[big])
                    * (2.0 * (j0(x_lo) - j0(x_hi))
                       + x_lo * j1(x_lo) - x_hi * j1(x_hi)))

    # ... and the term-by-term average of J_2's series, where it is not
    small = ~big
    if np.any(small):
        e, lo, hi = ell[small], theta_min[small], theta_max[small]
        total = np.zeros(e.shape, dtype=float)
        coefficient = 1.0
        for m in range(J2_SERIES_TERMS):
            if m:
                # (-1)^m / (2^(2m+2) m! (m+2)!) built by recurrence, so no
                # factorial overflows and no 2^28 literals
                coefficient *= -1.0 / (4.0 * m * (m + 2))
            else:
                coefficient = 1.0 / 8.0
            total += (coefficient / (m + 2)) * e ** (2 * m + 2) * (
                hi ** (2 * m + 4) - lo ** (2 * m + 4)
            ) / delta_sq[small]
        out[small] = total

    return out


class DeltaSigmaCovariance:
    r"""Wu et al. (2019) Gaussian-field covariance for
    :math:`\Delta\Sigma`.

    NOTE: units -- ``r_p`` edges in Mpc, spectra as documented in
    `clenspy.kernels.limber`, result in
    :math:`(M_\odot/{\rm Mpc}^2)^2`.

    NOTE: valid for a **thin** halo-redshift slice only; see the module
    NOTE.

    Parameters
    ----------
    rp_edges : array-like, shape ``(n_rp + 1,)``
        Radial bin edges [Mpc], ascending. Contiguous annuli.
    chi_h : float
        Comoving distance to the halo slice [Mpc]. Sets
        :math:`\theta = r_p/\chi_h` and :math:`\ell = k\chi_h`.
    f_sky : float
        Sky fraction, in :math:`(0, 1]`.
    c_ell_hh, c_ell_SS, c_ell_hS : callable
        ``f(ell) -> array``. The three spectra, e.g. from
        `clenspy.kernels.limber.LimberProjector`. Stored verbatim.
    n_h : float
        Halo surface density [1/sr]; the shot noise is :math:`1/n_h`.
    shape_noise : float
        :math:`\langle\Sigma_{\rm crit}\rangle^2\sigma_\gamma^2/n_s`,
        already assembled, in :math:`(M_\odot/{\rm Mpc}^2)^2` per
        steradian. Passed pre-combined because the three factors come from
        three different layers and combining them here would hide which.
    k_range : tuple of float, optional
        :math:`(k_{\min}, k_{\max})` in 1/Mpc for the quadrature. The
        default :math:`k_{\max} = 10^5` is set by the measured truncation
        scaling in the module NOTE, not by taste.
    n_k : int, optional
        Number of log-spaced :math:`k` nodes (default 8192).
    exact_shot_shape : bool, optional
        Use the closed-form Hankel-closure result for the ``shot_shape``
        term (default True). Set False to evaluate it by the same
        quadrature as the others, which is how the two are compared.
    """

    def __init__(self, rp_edges, chi_h, f_sky, c_ell_hh, c_ell_SS, c_ell_hS,
                 n_h, shape_noise, k_range=(1e-4, 1e5), n_k=8192,
                 exact_shot_shape=True):
        self.rp_edges = np.asarray(rp_edges, dtype=float)
        if self.rp_edges.ndim != 1 or self.rp_edges.size < 2:
            raise ValueError("rp_edges must be 1-D with >= 2 entries")
        if np.any(np.diff(self.rp_edges) <= 0.0):
            raise ValueError("rp_edges must be strictly ascending")
        if np.any(self.rp_edges < 0.0):
            raise ValueError("rp_edges must be non-negative")
        if not 0.0 < f_sky <= 1.0:
            raise ValueError(f"f_sky must lie in (0, 1], got {f_sky}")
        if chi_h <= 0.0:
            raise ValueError("chi_h must be positive")
        if n_h <= 0.0:
            raise ValueError("n_h must be positive (1/n_h is the shot noise)")

        self.chi_h = float(chi_h)
        self.f_sky = float(f_sky)
        self.c_ell_hh = c_ell_hh
        self.c_ell_SS = c_ell_SS
        self.c_ell_hS = c_ell_hS
        self.n_h = float(n_h)
        self.shape_noise = float(shape_noise)
        self.exact_shot_shape = bool(exact_shot_shape)
        self.k = np.logspace(np.log10(k_range[0]), np.log10(k_range[1]),
                             int(n_k))

    @property
    def n_rp(self):
        return self.rp_edges.size - 1

    def _kernel_matrix(self):
        r""":math:`A_{ki} = \hat J_2(k r_{p,i})`, shape ``(n_k, n_rp)``.

        The bilinear form's design matrix -- see the module NOTE on why
        there is no FFTLog.
        """
        # theta = rp / chi_h,  ell = k chi_h,  so ell*theta = k*rp
        ell = self.k[:, None] * self.chi_h
        theta_lo = self.rp_edges[None, :-1] / self.chi_h
        theta_hi = self.rp_edges[None, 1:] / self.chi_h
        return j2_bin(ell, theta_lo, theta_hi)

    def _spectra(self):
        """The five bracket terms, each as a function of k."""
        ell = self.k * self.chi_h
        c_hh = np.asarray(self.c_ell_hh(ell), dtype=float)
        c_ss = np.asarray(self.c_ell_SS(ell), dtype=float)
        c_hs = np.asarray(self.c_ell_hS(ell), dtype=float)
        n_h_term = 1.0 / self.n_h
        return {
            "lss_lss": c_hh * c_ss,
            "lss_shape": c_hh * self.shape_noise,
            "shot_lss": n_h_term * c_ss,
            "shot_shape": n_h_term * self.shape_noise,
            "cross": c_hs**2,
        }

    def _integrate(self, weight_k):
        r"""``A^T diag(k w / 2pi) A / (4 pi f_sky)``, the bilinear form.

        NOTE: trapezoid in :math:`\ln k`, so the measure is
        :math:`k\,dk = k^2\,d\ln k`. Getting that Jacobian wrong is a
        silent factor of :math:`k`.
        """
        a = self._kernel_matrix()
        # k dk / (2 pi)  ->  k^2 dlnk / (2 pi)
        measure = self.k**2 * np.asarray(weight_k, dtype=float) / (2.0 * np.pi)
        ln_k = np.log(self.k)
        # trapezoid weights, applied once and visibly
        trapz_w = np.gradient(ln_k)
        trapz_w[0] *= 0.5
        trapz_w[-1] *= 0.5
        scaled = a * (measure * trapz_w)[:, None]
        out = (a.T @ scaled) / (4.0 * np.pi * self.f_sky)
        # A^T diag(w) A is symmetric in exact arithmetic but not bit-exactly
        # in fp64 (the BLAS accumulation order differs between the (i,j) and
        # (j,i) dot products). Symmetrise, so callers can rely on it and
        # eigenvalue routines get a genuinely symmetric matrix.
        return 0.5 * (out + out.T)

    def annulus_area(self):
        r""":math:`A_{\rm ann} = \pi(r_{p,\max}^2 - r_{p,\min}^2)`, in Mpc^2.

        NOTE: **Mpc^2, not steradians** -- see the module NOTE. Using the
        angular area is wrong by :math:`\chi_h^2`.
        """
        return np.pi * (self.rp_edges[1:] ** 2 - self.rp_edges[:-1] ** 2)

    def _shot_shape_exact(self):
        r"""The ``shot_shape`` term in closed form, by Hankel closure.

        .. math::
            {\rm Cov}^{\rm shot\_shape}_{ij} =
              \frac{N_h N_\Sigma}{4\pi f_{\rm sky}}\,
              \frac{\delta_{ij}}{A_{{\rm ann},i}}

        Exact: no quadrature, hence no truncation error, in the term that
        dominates at small :math:`r_p`.
        """
        bracket = self.shape_noise / self.n_h
        return np.diag(bracket / (4.0 * np.pi * self.f_sky)
                       / self.annulus_area())

    def components(self):
        """``{name: matrix}`` for the five bracket terms."""
        out = {name: self._integrate(w)
               for name, w in self._spectra().items()}
        if self.exact_shot_shape:
            out["shot_shape"] = self._shot_shape_exact()
        return out

    def cov(self, terms=ALL_TERMS):
        r"""The total, or any subset of the five terms.

        Parameters
        ----------
        terms : iterable of str, optional
            Any subset of `ALL_TERMS`. Defaults to all five.
        """
        terms = tuple(terms)
        unknown = set(terms) - set(ALL_TERMS)
        if unknown:
            raise ValueError(
                f"unknown terms {sorted(unknown)}; choose from "
                f"{list(ALL_TERMS)}"
            )
        spectra = self._spectra()
        # the exact term is added separately, not folded into the bracket
        quadrature = [n for n in terms
                      if not (self.exact_shot_shape and n == "shot_shape")]
        total = np.zeros((self.n_rp, self.n_rp))
        if quadrature:
            total += self._integrate(sum(spectra[n] for n in quadrature))
        if self.exact_shot_shape and "shot_shape" in terms:
            total += self._shot_shape_exact()
        return total

    def _variant(self, **kw):
        """A copy of self with some construction arguments replaced."""
        base = dict(
            rp_edges=self.rp_edges, chi_h=self.chi_h, f_sky=self.f_sky,
            c_ell_hh=self.c_ell_hh, c_ell_SS=self.c_ell_SS,
            c_ell_hS=self.c_ell_hS, n_h=self.n_h,
            shape_noise=self.shape_noise,
            exact_shot_shape=self.exact_shot_shape,
            k_range=(self.k[0], self.k[-1]), n_k=self.k.size,
        )
        base.update(kw)
        return DeltaSigmaCovariance(**base)

    def convergence(self):
        r"""Relative change in the diagonal under coarsening **both** axes.

        Returns ``{"n_k": ..., "k_max": ...}``.

        NOTE: an earlier version halved only ``n_k`` and reported 4e-4 when
        the true error against the closure identity was 2.4e-3. This
        quadrature is **truncation**-limited, so :math:`k_{\max}` is the
        axis that matters, and reporting only ``n_k`` is a false
        reassurance. Both are returned so neither can be mistaken for the
        other.
        """
        fine = np.diag(self.cov())
        halved = np.diag(self._variant(n_k=self.k.size // 2).cov())
        shorter = np.diag(
            self._variant(k_range=(self.k[0], self.k[-1] / 10.0)).cov()
        )
        return {
            "n_k": float(np.max(np.abs(halved / fine - 1.0))),
            "k_max": float(np.max(np.abs(shorter / fine - 1.0))),
        }

    def __repr__(self):
        return (f"DeltaSigmaCovariance(n_rp={self.n_rp}, "
                f"chi_h={self.chi_h:.1f} Mpc, f_sky={self.f_sky:.4f}, "
                f"n_k={self.k.size})")


if __name__ == "__main__":
    rp_edges = np.logspace(np.log10(0.2), np.log10(30.0), 9)
    chi_h = 1100.0                       # Mpc, roughly z = 0.4
    f_sky = 1500.0 * (np.pi / 180.0) ** 2 / (4.0 * np.pi)

    # power-law stand-ins, so the demo needs no Limber run
    def c_hh(ell):
        return 1e-5 * (np.asarray(ell, float) / 100.0) ** -1.0

    def c_ss(ell):
        return 4e26 * (np.asarray(ell, float) / 100.0) ** -1.2

    def c_hs(ell):
        # linear bias would give exactly sqrt(c_hh * c_ss)
        return np.sqrt(c_hh(ell) * c_ss(ell))

    n_h = 3.0e5                          # haloes per steradian
    shape_noise = 1.0e26                 # <Sigma_crit>^2 sigma_gamma^2 / n_s

    cov = DeltaSigmaCovariance(rp_edges, chi_h, f_sky, c_hh, c_ss, c_hs,
                               n_h, shape_noise)
    print(cov)
    print(f"f_sky = {f_sky:.5f}  (1500 deg^2)")
    conv = cov.convergence()
    print(f"convergence:  n_k axis {conv['n_k']:.2e}   "
          f"k_max axis {conv['k_max']:.2e}")

    # the closure identity, and the precision it buys
    quad_only = DeltaSigmaCovariance(rp_edges, chi_h, f_sky, c_hh, c_ss,
                                     c_hs, n_h, shape_noise,
                                     exact_shot_shape=False)
    exact_diag = np.diag(cov._shot_shape_exact())
    quad_diag = np.diag(quad_only.components()["shot_shape"])
    print("\nshot_shape: closed form (Hankel closure) vs quadrature")
    print(f"  max |quadrature/exact - 1| = "
          f"{np.max(np.abs(quad_diag / exact_diag - 1.0)):.3e}")
    off = quad_only.components()["shot_shape"]
    off = off - np.diag(np.diag(off))
    print(f"  quadrature off-diagonal leak = "
          f"{np.max(np.abs(off)) / np.max(quad_diag):.3e}   (exact: 0)")
    print("  the closed form has no truncation error at all, in the term")
    print("  that dominates at small rp -- which is the whole gain.\n")

    rp_mid = np.sqrt(rp_edges[:-1] * rp_edges[1:])
    print("fractional contribution of each term to the diagonal:")
    parts = cov.components()
    total = np.diag(cov.cov())
    print(f"{'rp [Mpc]':>9s}  " + "  ".join(f"{n:>11s}" for n in ALL_TERMS))
    for i, r in enumerate(rp_mid):
        print(f"{r:9.3f}  " + "  ".join(
            f"{np.diag(parts[n])[i] / total[i]:11.5f}" for n in ALL_TERMS))
    print("  <- shot_shape dominates at small rp (both noises, no k")
    print("     dependence, so the whole J2 tail contributes); the LSS")
    print("     terms take over at large rp.")

    print("\nsigma(DeltaSigma) [Msun/Mpc^2] and its correlation with the "
          "next bin:")
    c = cov.cov()
    d = np.sqrt(np.diag(c))
    print(f"{'rp [Mpc]':>9s}  {'sigma':>12s}  {'corr(i,i+1)':>12s}")
    for i, r in enumerate(rp_mid):
        nxt = (c[i, i + 1] / (d[i] * d[i + 1])) if i + 1 < len(d) else np.nan
        print(f"{r:9.3f}  {d[i]:12.4e}  {nxt:12.4f}")
    print("  <- neighbouring radial bins are correlated: J2 is broad in k,")
    print("     so a single mode contributes to several annuli.")

    # the terms must sum to the total, and the total must be a covariance
    summed = sum(parts.values())
    print(f"\nthe five terms sum to the total: max rel. diff = "
          f"{np.max(np.abs(summed / c - 1.0)):.2e}")
    ev = np.linalg.eigvalsh(c)
    print(f"positive definite: {ev.min() > 0}  "
          f"(smallest eigenvalue {ev.min():.4e})")

    # switches, as the skill requires
    print("\nisolating terms (diagonal at rp = 1 Mpc):")
    i = int(np.argmin(np.abs(rp_mid - 1.0)))
    for name in ALL_TERMS:
        v = np.diag(cov.cov(terms=(name,)))[i]
        print(f"  {name:>11s}: {v:12.4e}  ({v / total[i]:6.2%})")
    print(f"  {'all':>11s}: {total[i]:12.4e}")
    print("  NOTE: lss_lss and cross are equal here, and it is not a")
    print("  coincidence -- this demo sets C_hS = sqrt(C_hh C_SS), the")
    print("  linear-bias limit, in which (C_hS)^2 = C_hh C_SS identically.")
    print("  With a real halo-matter spectrum they differ.")

    # why the bin average matters, as a number
    print("\nJ2 at the bin centre vs the bin average, at rp = 1 Mpc:")
    theta = rp_mid[i] / chi_h
    from scipy.special import jv
    for k_test in (1e-2, 1e-1, 1.0, 10.0):
        ell = k_test * chi_h
        centre = jv(2, ell * theta)
        averaged = j2_bin(ell, rp_edges[i] / chi_h,
                          rp_edges[i + 1] / chi_h).item()
        print(f"  k = {k_test:6.2f} 1/Mpc:  J2 = {centre:9.5f}   "
              f"J2_bin = {averaged:9.5f}   ratio {averaged / centre:7.3f}")
    print("  <- they agree at small k (the first peak) and diverge at")
    print("     large k, which is exactly where the flat noise terms live.")
