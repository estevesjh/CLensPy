r"""Joint SZ--richness mass--observable relation, :math:`P(\zeta,\lambda\mid M,z)`.

Correlated bivariate log-normal scatter between the unbiased SZ
significance :math:`\zeta` (SPT notation) and richness :math:`\lambda`.
Neither true->observed step is a plain Gaussian convolution, so
`JointMor.pdf_obs` nests two Gauss--Legendre quadratures rather than
convolving analytically, and `ThresholdSelection` needs only one. Full
derivation and the measured precision/speed tradeoffs:
:doc:`/sz_richness`.

NOTE: units -- ``ln_mass`` is :math:`\ln M`, M in :math:`h^{-1}` Msun;
:math:`\zeta,\xi,\lambda` dimensionless. These are `pydantic.BaseModel`,
not the repo's usual ``@dataclass``.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from ..kernels.photoz import gaussian_cdf
from ..utils.integrate import gl_nodes, gl_nodes_batched
from .richness_kernel import EmgParams, emg_cdf, richness_pdf
from .scaling_relation import LogNormalMor

__all__ = ["SzMor", "SzeNoise", "JointMor", "ThresholdSelection",
           "SzRichnessSelectionFunction", "integrate_pdf_obs"]

#: NOTE: NOT `selection_function.py`'s bracket_width=8 -- measured (see
#: `pdf_obs`'s docstring) to under-resolve the NESTED lambda_true/zeta_true
#: quadrature at high mass, where an 8-sigma-wide bracket wastes most of
#: its fixed 64 GL nodes on the near-empty wings instead of the peak. A
#: narrower bracket concentrates nodes where the joint density actually
#: lives; the higher orders below then resolve that peak. Verified against
#: n_quad=(256,192) reference values across low- and high-mass points
#: (worst-case relative error ~3e-2, down from >3e-1 at the old 8/64/48
#: defaults) -- still a known ceiling, not exact; see `pdf_obs` NOTE.
BRACKET_WIDTH = 5.0
N_QUAD = 128
#: Order of the nested zeta_true quadrature inside pdf_obs.
N_QUAD_ZETA = 96


class SzMor(BaseModel):
    r"""A log-normal SZ--mass relation, :math:`P(\zeta^{\rm tr}\mid M,z)`,
    for the *unbiased* (true) SZ significance :math:`\zeta`.

    Same functional family as `LogNormalMor`
    (:mod:`clenspy.selection.scaling_relation`), minus the Poisson floor --
    :math:`\zeta` is a continuous signal, not a discrete galaxy count:

    .. math::
        \langle\ln\zeta\rangle(M,z) = \ln A_\zeta
          + B_\zeta\ln\!\left(\frac{M}{M_p}\right)
          + C_\zeta\ln\!\left(\frac{1+z}{1+z_p}\right),
        \qquad
        \sigma^2_{\ln\zeta} = D_\zeta^2

    NOTE: no published SZ calibration was given for this task; the
    defaults below are placeholders (unit amplitude, unit mass slope, no
    redshift evolution, 20% scatter) meant to be overridden, not trusted.

    Parameters
    ----------
    A_zeta, B_zeta, C_zeta, D_zeta : float
        Amplitude, mass slope, redshift evolution, intrinsic scatter.
    m_pivot_hinv, z_pivot : float, optional
        Pivots; default to `LogNormalMor`'s for consistency.
    """

    model_config = ConfigDict(frozen=True)

    A_zeta: float = 1.0
    B_zeta: float = 1.0
    C_zeta: float = 0.0
    D_zeta: float = 0.2
    m_pivot_hinv: float = 3.0e14
    z_pivot: float = 0.45

    def mean_ln_zeta(self, ln_mass, z):
        r""":math:`\langle\ln\zeta\rangle`, dimensionless."""
        ln_mass = np.asarray(ln_mass, dtype=float)
        z = np.asarray(z, dtype=float)
        return (np.log(self.A_zeta)
                + self.B_zeta * (ln_mass - np.log(self.m_pivot_hinv))
                + self.C_zeta * np.log((1.0 + z) / (1.0 + self.z_pivot)))

    def var_ln_zeta(self, ln_mass, z):
        r""":math:`\sigma^2_{\ln\zeta} = D_\zeta^2`, constant in :math:`(M,z)`."""
        ln_mass, z = np.broadcast_arrays(
            np.asarray(ln_mass, dtype=float), np.asarray(z, dtype=float)
        )
        return np.full(ln_mass.shape, self.D_zeta**2)

    def mean(self, ln_mass, z):
        r""":math:`\langle\zeta^{\rm tr}\rangle = e^{\mu + \sigma^2/2}`."""
        return np.exp(self.mean_ln_zeta(ln_mass, z)
                      + 0.5 * self.var_ln_zeta(ln_mass, z))

    def std(self, ln_mass, z):
        r"""Standard deviation of :math:`\zeta^{\rm tr}` (not of its log)."""
        var_ln = self.var_ln_zeta(ln_mass, z)
        return self.mean(ln_mass, z) * np.sqrt(np.expm1(var_ln))

    def pdf(self, zeta_true, ln_mass, z):
        r""":math:`P(\zeta^{\rm tr}\mid M,z)`, a density per unit :math:`\zeta`."""
        zeta_true, ln_mass, z = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (zeta_true, ln_mass, z))
        )
        mu = self.mean_ln_zeta(ln_mass, z)
        var = self.var_ln_zeta(ln_mass, z)
        positive = zeta_true > 0.0
        safe = np.where(positive, zeta_true, 1.0)
        density = (np.exp(-0.5 * (np.log(safe) - mu) ** 2 / var)
                   / (safe * np.sqrt(2.0 * np.pi * var)))
        return np.where(positive, density, 0.0)

    def __repr__(self):
        return (f"SzMor(A={self.A_zeta:g}, B={self.B_zeta:g}, "
                f"C={self.C_zeta:g}, D={self.D_zeta:g})")


class SzeNoise(BaseModel):
    r"""The SPT SZ-significance noise model (module docstring):

    .. math::
        P(\xi\mid\zeta) = \mathcal N\!\left(\xi \,\middle|\,
            \sqrt{\zeta^2 + {\rm dof\_offset}},\; \sigma_\xi\right)

    Parameters
    ----------
    dof_offset : float, optional
        The matched-filter bias term (default 3: Vanderlinde et al. 2010's
        2 sky-position + 1 core-radius nuisance parameters).
    sigma_xi : float, optional
        Width of the matched-filter S/N (default 1, fixed by construction
        in the SPT papers; exposed rather than hardcoded).
    """

    model_config = ConfigDict(frozen=True)

    dof_offset: float = 3.0
    sigma_xi: float = 1.0

    @model_validator(mode="after")
    def _check_positive(self):
        if self.dof_offset < 0.0:
            raise ValueError("dof_offset must be non-negative")
        if self.sigma_xi <= 0.0:
            raise ValueError("sigma_xi must be positive")
        return self

    def xi_mean(self, zeta_true):
        r""":math:`\sqrt{\zeta^2 + {\rm dof\_offset}}`."""
        zeta_true = np.asarray(zeta_true, dtype=float)
        return np.sqrt(zeta_true**2 + self.dof_offset)

    def pdf_xi(self, xi_obs, zeta_true):
        r""":math:`P(\xi^{\rm ob}\mid\zeta^{\rm tr})`, a plain Gaussian
        density in :math:`\xi^{\rm ob}` (not log-normal: no Jacobian)."""
        xi_obs = np.asarray(xi_obs, dtype=float)
        mu = self.xi_mean(zeta_true)
        return (np.exp(-0.5 * ((xi_obs - mu) / self.sigma_xi) ** 2)
                / (self.sigma_xi * np.sqrt(2.0 * np.pi)))


class JointMor(BaseModel):
    r"""The correlated joint SZ--richness relation.

    Composes an `SzMor` and a `LogNormalMor` (reused unmodified from
    `clenspy.selection.scaling_relation`) with a single intrinsic
    correlation coefficient. See the module docstring for why this is
    two named fields rather than a generic two-Mor composer: there is
    exactly one concrete joint pair in scope, and
    `clenspy.protocols` already documents the house rule against writing a
    contract before a second implementation exists.

    Parameters
    ----------
    sz_mor : SzMor
    richness_mor : LogNormalMor
    rho : float
        Intrinsic correlation between :math:`\ln\zeta^{\rm tr}` and
        :math:`\ln\lambda^{\rm tr}`, in :math:`(-1, 1)`.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    sz_mor: SzMor
    richness_mor: LogNormalMor
    rho: float

    @model_validator(mode="after")
    def _check_rho(self):
        if not (-1.0 < self.rho < 1.0):
            raise ValueError(f"rho must lie strictly in (-1, 1), got {self.rho}")
        return self

    def mean_log(self, ln_mass, z):
        r"""``(mu_zeta, mu_lambda)``, both :math:`\ln`-space means."""
        return (self.sz_mor.mean_ln_zeta(ln_mass, z),
                self.richness_mor.mean_ln_lambda(ln_mass, z))

    def cov_log(self, ln_mass, z):
        r"""``(var_zeta, var_lambda, cov_zeta_lambda)``, the intrinsic
        :math:`\bar\Sigma` entries (module docstring)."""
        var_zeta = self.sz_mor.var_ln_zeta(ln_mass, z)
        var_lambda = self.richness_mor.var_ln_lambda(ln_mass, z)
        cov = self.rho * np.sqrt(var_zeta * var_lambda)
        return var_zeta, var_lambda, cov

    def pdf_true(self, zeta_true, lambda_true, ln_mass, z):
        r""":math:`P(\zeta^{\rm tr},\lambda^{\rm tr}\mid M,z)`, the bivariate
        log-normal density (Jacobian :math:`1/(\zeta\lambda)` included)."""
        zeta_true, lambda_true, ln_mass, z = np.broadcast_arrays(
            *(np.asarray(v, dtype=float)
              for v in (zeta_true, lambda_true, ln_mass, z))
        )
        mu_zeta, mu_lambda = self.mean_log(ln_mass, z)
        var_zeta, var_lambda, cov = self.cov_log(ln_mass, z)

        positive = (zeta_true > 0.0) & (lambda_true > 0.0)
        safe_zeta = np.where(positive, zeta_true, 1.0)
        safe_lambda = np.where(positive, lambda_true, 1.0)
        x = np.log(safe_zeta) - mu_zeta
        y = np.log(safe_lambda) - mu_lambda

        det = var_zeta * var_lambda - cov**2
        quad = (x**2 / var_zeta - 2.0 * cov * x * y / (var_zeta * var_lambda)
                + y**2 / var_lambda) / (1.0 - cov**2 / (var_zeta * var_lambda))
        norm = (2.0 * np.pi * np.sqrt(det) * safe_zeta * safe_lambda)
        density = np.exp(-0.5 * quad) / norm
        return np.where(positive, density, 0.0)

    def conditional_zeta_given_lambda(self, ln_lambda_true, ln_mass, z):
        r"""``(mu_cond, var_cond)`` of :math:`\ln\zeta^{\rm tr}\mid
        \ln\lambda^{\rm tr}`, standard bivariate-normal conditioning:

        .. math::
            \mu_{\rm cond} = \mu_\zeta + \rho\frac{\sigma_\zeta}{\sigma_\lambda}
              (\ln\lambda^{\rm tr} - \mu_\lambda),
            \qquad
            \sigma^2_{\rm cond} = \sigma_\zeta^2(1 - \rho^2)
        """
        mu_zeta, mu_lambda = self.mean_log(ln_mass, z)
        var_zeta, var_lambda, _ = self.cov_log(ln_mass, z)
        sigma_zeta, sigma_lambda = np.sqrt(var_zeta), np.sqrt(var_lambda)
        mu_cond = mu_zeta + self.rho * (sigma_zeta / sigma_lambda) * (
            np.asarray(ln_lambda_true, dtype=float) - mu_lambda
        )
        var_cond = var_zeta * (1.0 - self.rho**2)
        return mu_cond, var_cond

    def _lambda_true_nodes(self, ln_mass, z, n_quad, bracket_width):
        """GL bracket over lambda_true, same idiom as
        `selection_function.SelectionFunction.bracket`/`_nodes_for`."""
        ln_mass, z = np.broadcast_arrays(
            np.asarray(ln_mass, dtype=float), np.asarray(z, dtype=float)
        )
        mu = np.asarray(self.richness_mor.mean(ln_mass, z), dtype=float)
        sd = np.asarray(self.richness_mor.std(ln_mass, z), dtype=float)
        half = bracket_width * sd
        a, b = np.maximum(mu - half, 0.0), mu + half
        if ln_mass.shape == ():
            lam, wts = gl_nodes(float(a), float(b), n_quad)
        else:
            lam, wts = gl_nodes_batched(a.ravel(), b.ravel(), n_quad)
            lam = lam.reshape(*ln_mass.shape, n_quad)
            wts = wts.reshape(*ln_mass.shape, n_quad)
        return ln_mass, z, lam, wts

    def _zeta_true_nodes(self, mu_cond, var_cond, zeta_min, n_quad,
                          bracket_width):
        r"""GL bracket over zeta_true, clipped below at ``zeta_min`` --
        the :math:`\Theta(\zeta-\zeta_{\min})` cut (module docstring). An
        inverted bracket (``zeta_min`` above the whole bracket) integrates
        to zero via `gl_nodes_batched`'s own documented behaviour: exactly
        the "undetectable halo" limit, no special-casing needed."""
        sd = np.sqrt(var_cond)
        lower = np.maximum(zeta_min, np.exp(mu_cond - bracket_width * sd))
        upper = np.exp(mu_cond + bracket_width * sd)
        shape = mu_cond.shape
        zeta, wts = gl_nodes_batched(lower.ravel(), upper.ravel(), n_quad)
        return zeta.reshape(*shape, n_quad), wts.reshape(*shape, n_quad)

    def pdf_obs(self, xi_obs, lambda_obs, ln_mass, z, sze_noise: SzeNoise,
                zeta_min: float, richness_kernel_params: EmgParams,
                n_quad_lambda: int = N_QUAD, n_quad_zeta: int = N_QUAD_ZETA,
                bracket_width: float = BRACKET_WIDTH):
        r""":math:`P(\xi^{\rm ob},\lambda^{\rm ob}\mid M,z)`: a nested
        Gauss--Legendre quadrature, outer over :math:`\lambda^{\rm tr}`,
        inner over :math:`\zeta^{\rm tr}` (module docstring). Both
        observables need marginalizing this way because :math:`P(\xi\mid
        \zeta)` has a nonlinear mean (:math:`\sqrt{\zeta^2+{\rm
        dof\_offset}}`), so it is not Gaussian-plus-Gaussian analytic the
        way a plain lognormal noise model would be.

        NOTE: **grid, not pairwise.** ``xi_obs`` and ``lambda_obs``
        broadcast against each other via plain NumPy rules -- pass
        ``xi_obs`` shaped ``(nx, 1)`` and ``lambda_obs`` shaped ``(1, ny)``
        for the full outer-product grid in **one call**, returning shape
        ``(nx, ny)``; no Python-level loop over grid points is needed or
        should be added by a caller. (Same convention as
        `clenspy.utils.integrate`'s ``xi_func(r_vec, z_scalar)`` grids.)
        Verified bit-identical against the equivalent scalar-call loop.
        `integrate_pdf_obs` below builds exactly this grid to do a
        Gauss--Legendre double integral over the observables.

        NOTE: **known precision ceiling.** This is a fixed-order product
        rule over a fixed bracket, not adaptive quadrature. At high mass
        (large absolute :math:`\lambda^{\rm tr}` scatter) the joint density
        can be sharply peaked relative to the bracket width, and the
        default order/bracket resolve it only to ~a few percent (measured
        against an :math:`n=(256,192)` reference) -- see `BRACKET_WIDTH`'s
        module-level NOTE for the measurement. For higher precision, raise
        ``n_quad_lambda``/``n_quad_zeta`` (cost scales roughly with their
        product) rather than ``bracket_width``, which was already
        re-tuned; widening it further reintroduces the resolution problem
        it was narrowed to fix. Extremely improbable (far double-tail,
        anti-correlated) ``(xi_obs, lambda_obs, M)`` combinations are
        inherently ill-conditioned for *any* quadrature scheme, fixed or
        adaptive -- this is not specific to this implementation."""
        xi_obs = np.asarray(xi_obs, dtype=float)
        lambda_obs = np.asarray(lambda_obs, dtype=float)
        ln_mass, z, lam, wts_lam = self._lambda_true_nodes(
            ln_mass, z, n_quad_lambda, bracket_width
        )
        mu_cond, var_cond = self.conditional_zeta_given_lambda(
            np.log(lam), ln_mass[..., None], z[..., None]
        )
        zeta, wts_zeta = self._zeta_true_nodes(
            mu_cond, var_cond, zeta_min, n_quad_zeta, bracket_width
        )

        # P(zeta_true | lambda_true node): lognormal with (mu_cond, var_cond)
        zeta_pdf = (np.exp(-0.5 * (np.log(zeta) - mu_cond[..., None]) ** 2
                            / var_cond[..., None])
                    / (zeta * np.sqrt(2.0 * np.pi * var_cond[..., None])))
        xi_ob = xi_obs[..., None, None] if xi_obs.shape else xi_obs
        xi_pdf = sze_noise.pdf_xi(xi_ob, zeta)
        # inner integral over zeta_true -> P(xi_obs | lambda_true node, M, z)
        p_xi_given_lambda = np.einsum("...q,...q,...q->...",
                                       wts_zeta, xi_pdf, zeta_pdf)

        lam_ob = lambda_obs[..., None] if lambda_obs.shape else lambda_obs
        lam_kernel = richness_pdf(lam_ob, lam, z[..., None],
                                   richness_kernel_params)
        lam_true_pdf = self.richness_mor.pdf(lam, ln_mass[..., None],
                                              z[..., None])
        # outer integral over lambda_true
        return np.einsum("...q,...q,...q,...q->...",
                          wts_lam, p_xi_given_lambda, lam_kernel, lam_true_pdf)

    def __repr__(self):
        return f"JointMor({self.sz_mor!r}, {self.richness_mor!r}, rho={self.rho:g})"


class ThresholdSelection(BaseModel):
    r"""Joint selection completeness
    :math:`P(\zeta^{\rm tr} > \zeta_{\min}\ {\rm AND}\ \lambda^{\rm ob} >
    \lambda_{\rm th}\mid M,z)`.

    NOTE: the SZ cut is on the **true** :math:`\zeta`, not the observed
    :math:`\xi` -- see the module docstring. Because :math:`\int d\xi\,
    P(\xi\mid\zeta) = 1` identically, no :math:`\xi` (and hence no
    `SzeNoise`) is needed here: only the single :math:`\lambda^{\rm tr}`
    quadrature `JointMor.pdf_obs` also uses, with two analytic tail
    factors at each node.

    Parameters
    ----------
    zeta_min : float
        The true-significance detection threshold, :math:`\zeta^{\rm tr} >
        \zeta_{\min}`.
    lambda_th : float
        The observed-richness threshold, :math:`\lambda^{\rm ob} >
        \lambda_{\rm th}`.
    """

    model_config = ConfigDict(frozen=True)

    zeta_min: float
    lambda_th: float

    def completeness(self, joint_mor: JointMor, ln_mass, z,
                      richness_kernel_params: EmgParams,
                      n_quad: int = N_QUAD, bracket_width: float = BRACKET_WIDTH):
        ln_mass, z, lam, wts = joint_mor._lambda_true_nodes(
            ln_mass, z, n_quad, bracket_width
        )
        mu_cond, var_cond = joint_mor.conditional_zeta_given_lambda(
            np.log(lam), ln_mass[..., None], z[..., None]
        )
        # analytic lognormal survival: P(zeta_true > zeta_min | lambda_true)
        zeta_tail = 1.0 - gaussian_cdf(
            (np.log(self.zeta_min) - mu_cond) / np.sqrt(var_cond)
        )

        mu, sigma, tau, f_prj = richness_kernel_params.at(lam, z[..., None])
        lam_cdf = ((1.0 - f_prj) * gaussian_cdf((self.lambda_th - mu) / sigma)
                   + f_prj * emg_cdf(self.lambda_th, mu, sigma, tau))
        lam_tail = 1.0 - lam_cdf

        lam_true_pdf = joint_mor.richness_mor.pdf(lam, ln_mass[..., None],
                                                   z[..., None])
        return np.einsum("...q,...q,...q,...q->...",
                          wts, zeta_tail, lam_tail, lam_true_pdf)

    def __repr__(self):
        return (f"ThresholdSelection(zeta_min={self.zeta_min:g}, "
                f"lambda_th={self.lambda_th:g})")


class SzRichnessSelectionFunction(BaseModel):
    r""":math:`\mathcal S(M,z)`: mor + error model + selection, composed.

    `ThresholdSelection.completeness` needs three collaborators passed in
    on every call (`JointMor`, `EmgParams`, and the threshold itself, which
    already holds its own state). This wraps them once so a caller just
    does ``S(ln_mass, z)`` -- the same shape as
    `clenspy.selection.selection_function.SelectionFunction`, generalized
    to the joint SZ--richness case.

    NOTE: **no `SzeNoise` field.** The SZ selection cut is on the *true*
    :math:`\zeta`, not the observed :math:`\xi` (module docstring), and
    :math:`\int d\xi\,P(\xi\mid\zeta)=1` identically -- so :math:`\xi`'s
    error model cancels out of a pure completeness and would be a field
    that never affects the answer. The one error model that *does* matter
    here is `richness_kernel_params` (the EMG kernel), which is why it is
    a field and `SzeNoise` is not. A caller that also wants the joint
    *density* at a specific :math:`\xi^{\rm ob}` uses `JointMor.pdf_obs`
    directly, which does take an `SzeNoise`.

    Parameters
    ----------
    mor : JointMor
        The mass--observable relation (correlated :math:`\zeta,\lambda`).
    selection : ThresholdSelection
        The two thresholds, :math:`\zeta_{\min}` and :math:`\lambda_{\rm th}`.
    richness_kernel_params : EmgParams
        The richness observed-given-true kernel (the error model).
    n_quad, bracket_width : optional
        Quadrature settings passed through to `ThresholdSelection.completeness`.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    mor: JointMor
    selection: ThresholdSelection
    richness_kernel_params: EmgParams
    n_quad: int = N_QUAD
    bracket_width: float = BRACKET_WIDTH

    def __call__(self, ln_mass, z):
        r""":math:`\mathcal S(M,z)`."""
        return self.selection.completeness(
            self.mor, ln_mass, z, self.richness_kernel_params,
            n_quad=self.n_quad, bracket_width=self.bracket_width,
        )

    def __repr__(self):
        return (f"SzRichnessSelectionFunction({self.mor!r}, {self.selection!r})")


def integrate_pdf_obs(joint_mor: JointMor, ln_mass, z, sze_noise: SzeNoise,
                       zeta_min: float, richness_kernel_params: EmgParams,
                       xi_bracket, lambda_bracket,
                       n_quad_xi: int = N_QUAD, n_quad_lambda_obs: int = N_QUAD,
                       **pdf_obs_kwargs):
    r"""Double Gauss--Legendre integral of `JointMor.pdf_obs` over
    :math:`(\xi^{\rm ob}, \lambda^{\rm ob})` inside the given brackets.

    Builds one GL rule per observable (`clenspy.utils.integrate.gl_nodes`),
    evaluates `pdf_obs` on their outer-product grid in a **single**
    vectorized call (`pdf_obs`'s own grid, not pairwise, broadcasting --
    see its docstring), and contracts against the tensor-product weights.
    No Python-level loop over grid points, no ``trapz``.

    Parameters
    ----------
    xi_bracket, lambda_bracket : (float, float)
        Integration limits ``(lo, hi)`` for :math:`\xi^{\rm ob}` and
        :math:`\lambda^{\rm ob}`. The caller's choice -- not computed here
        -- e.g. mean +/- L*std of the relevant marginal, the same bracket
        idiom `JointMor._lambda_true_nodes` uses.
    n_quad_xi, n_quad_lambda_obs : int, optional
        Gauss--Legendre order for each observable's integral.
    **pdf_obs_kwargs
        Passed through to `JointMor.pdf_obs` (e.g. ``n_quad_lambda``,
        ``n_quad_zeta``, ``bracket_width``).

    Returns
    -------
    float
        :math:`\iint d\xi^{\rm ob}\,d\lambda^{\rm ob}\,P(\xi^{\rm
        ob},\lambda^{\rm ob}\mid M,z)`.
    """
    xi_nodes, xi_wts = gl_nodes(*xi_bracket, n_quad_xi)
    lam_nodes, lam_wts = gl_nodes(*lambda_bracket, n_quad_lambda_obs)
    grid = joint_mor.pdf_obs(xi_nodes[:, None], lam_nodes[None, :], ln_mass, z,
                              sze_noise, zeta_min, richness_kernel_params,
                              **pdf_obs_kwargs)
    return float(np.einsum("i,j,ij->", xi_wts, lam_wts, grid))


if __name__ == "__main__":
    from .richness_kernel import EmgParams as _EmgParams

    sz = SzMor(A_zeta=5.0, B_zeta=0.8, C_zeta=0.5, D_zeta=0.2)
    richness = LogNormalMor()
    joint = JointMor(sz_mor=sz, richness_mor=richness, rho=0.5)
    noise = SzeNoise()
    # f_prj = 0: pure-Gaussian richness kernel limit (richness_kernel.py's
    # own documented check), so this run is directly comparable to a plain
    # Gaussian-noise sanity check.
    emg_params = _EmgParams(delta_mu=0.0, sigma=3.0, f_prj=0.0, tau=0.12)

    ln_m = np.log(3e14)
    z = 0.3
    print(joint)
    print(noise)

    print("\nSzMor.pdf integrates to ~1 over zeta:")
    zeta_grid = np.linspace(1e-3, 100.0, 200001)
    total = np.trapezoid(sz.pdf(zeta_grid, ln_m, z), x=zeta_grid)
    print(f"  integral = {total:.6f}")

    print("\npdf_true's zeta-marginal matches SzMor.pdf directly (rho only")
    print("enters the joint, not either marginal):")
    lam_grid = np.linspace(1e-3, 400.0, 4001)
    zeta_test = np.array([sz.mean(ln_m, z).item()])
    marginal = np.trapezoid(
        joint.pdf_true(zeta_test[:, None], lam_grid[None, :], ln_m, z),
        x=lam_grid, axis=-1,
    )
    direct = sz.pdf(zeta_test, ln_m, z)
    print(f"  marginal = {marginal[0]:.6e}   direct = {direct[0]:.6e}   "
          f"rel diff = {abs(marginal[0] / direct[0] - 1):.2e}")

    print("\nrho ~ 0 reduces the joint true pdf to the product of marginals:")
    joint0 = JointMor(sz_mor=sz, richness_mor=richness, rho=1e-9)
    lhs = joint0.pdf_true(zeta_test[0], 30.0, ln_m, z)
    rhs = sz.pdf(zeta_test[0], ln_m, z) * richness.pdf(30.0, ln_m, z)
    print(f"  joint(rho~0) = {float(lhs):.6e}   product = {float(rhs):.6e}")

    print("\npdf_obs, zeta_min -> 0, f_prj = 0: integrating over xi_obs and")
    print("lambda_obs should recover ~1 (same mass point). One vectorized")
    print("pdf_obs call over the (xi, lambda_obs) outer grid -- no Python")
    print("loop over grid points -- contracted with integrate_pdf_obs's")
    print("Gauss-Legendre double integral, not trapz:")
    xi_bracket = (1e-6, sz.mean(ln_m, z).item() + BRACKET_WIDTH * sz.std(ln_m, z).item())
    lambda_bracket = (1e-6,
                       richness.mean(ln_m, z).item()
                       + BRACKET_WIDTH * richness.std(ln_m, z).item())
    total_obs = integrate_pdf_obs(joint, ln_m, z, noise, 1e-6, emg_params,
                                   xi_bracket, lambda_bracket,
                                   n_quad_xi=48, n_quad_lambda_obs=48)
    print(f"  integral = {total_obs:.6f}  (bracket = {xi_bracket} x "
          f"{tuple(round(v, 1) for v in lambda_bracket)}; ~1 is the check)")

    print("\nThresholdSelection.completeness -> 1 as thresholds -> 0,")
    print("-> 0 as thresholds -> infinity:")
    for zeta_min, lam_th in ((1e-6, 1e-6), (2.0, 20.0), (1e6, 1e6)):
        sel = ThresholdSelection(zeta_min=zeta_min, lambda_th=lam_th)
        c = sel.completeness(joint, ln_m, z, emg_params)
        print(f"  (zeta_min={zeta_min:g}, lambda_th={lam_th:g}): "
              f"completeness = {float(c):.6f}")

    print("\nSzRichnessSelectionFunction: mor + error model + selection ->")
    print("S(M,z), matches ThresholdSelection.completeness called directly:")
    sel = ThresholdSelection(zeta_min=2.0, lambda_th=20.0)
    S = SzRichnessSelectionFunction(mor=joint, selection=sel,
                                     richness_kernel_params=emg_params)
    direct = sel.completeness(joint, ln_m, z, emg_params)
    print(f"  S(M,z) = {float(S(ln_m, z)):.6f}   direct = {float(direct):.6f}")

    print("\nln_mass is a vector too (grid, not pairwise): S(M,z) over a")
    print("mass grid in one call matches a scalar-call loop exactly:")
    ln_m_grid = np.log(np.array([1e13, 5e13, 1e14, 3e14, 1e15]))
    vec = S(ln_m_grid, z)
    loop = np.array([float(S(float(lm), z)) for lm in ln_m_grid])
    print(f"  vector = {np.array2string(vec, precision=6)}")
    print(f"  max|vector - loop| = {np.max(np.abs(vec - loop)):.2e}")

    print("\nzeta_min undetectable-halo limit: a threshold far above the")
    print("whole zeta bracket drives pdf_obs to zero (inverted-bracket")
    print("behaviour of gl_nodes_batched), not to a spurious finite value:")
    p_lo = joint.pdf_obs(10.0, 30.0, ln_m, z, noise, 1e-6, emg_params,
                          n_quad_lambda=32, n_quad_zeta=24)
    p_hi = joint.pdf_obs(10.0, 30.0, ln_m, z, noise, 1e6, emg_params,
                          n_quad_lambda=32, n_quad_zeta=24)
    print(f"  zeta_min=1e-6: pdf_obs = {float(p_lo):.6e}")
    print(f"  zeta_min=1e6:  pdf_obs = {float(p_hi):.6e}")
