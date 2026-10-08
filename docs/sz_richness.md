# Joint SZ--Richness Scaling Relation

Two correlated observables at fixed mass: the *unbiased* SZ significance
$\zeta$ (SPT notation; Vanderlinde et al. 2010; Bocquet et al. 2019, 2024)
and the optical richness $\lambda$. The mass--observable correlation
follows the bivariate log-normal formalism Ghirardini et al. (2024,
arXiv:2402.08458) build for (X-ray count rate, richness) in their Sect. 3
"Scaling Relations":

$$
P(\zeta,\lambda\mid M,z) = \mathcal{LN}(\bar\mu, \bar\Sigma),
\qquad
\bar\Sigma = \begin{pmatrix}
    \sigma_\zeta^2 & \rho\,\sigma_\zeta\sigma_\lambda \\
    \rho\,\sigma_\zeta\sigma_\lambda & \sigma_\lambda^2 + \sigma_P^2
\end{pmatrix},
$$

with a single intrinsic correlation coefficient $\rho$ and $\sigma_P^2$ the
richness Poisson floor `LogNormalMor` ({doc}`selection_bias`) already
carries. Neither observable's own true-to-observed step is Gaussian in
log-space the same way, though, which is why `JointMor.pdf_obs` needs a
nested quadrature rather than one analytic convolution.

## The richness step is EMG, not Gaussian

Costanzi et al. (2019a, 2021) model $\lambda^{\rm ob} = \lambda^{\rm tr} +
\Delta^{\rm bkg} + \Delta^{\rm prj}$, where $\Delta^{\rm prj}$ is a
one-sided projection boost — a spike at zero plus an exponential tail of
rate $\tau$, mixed in with weight $f^{\rm prj}$ — giving an **exponentially
modified Gaussian** in *linear* richness space. Already implemented,
unmodified, in `clenspy.selection.richness_kernel` (`EmgParams`,
`richness_pdf`, `emg_cdf`). It has no closed form once convolved with the
lognormal $\lambda^{\rm tr}$, which is why {doc}`selection_function`
integrates over $\lambda^{\rm tr}$ by Gauss--Legendre quadrature rather
than analytically, and why `sz_richness.py` reuses that same bracket idiom
(nodes from `clenspy.utils.integrate.gl_nodes`).

## The SZ step is Gaussian in $\xi$ with a nonlinear mean, and a hard cut on $\zeta$

The Vanderlinde et al. (2010)/Bocquet et al. (2019) SZ-significance model:

$$
P(\xi\mid\zeta) = \mathcal N\!\left(\xi \,\middle|\,
    \sqrt{\zeta^2 + 3},\; 1\right)\,\Theta(\zeta - \zeta_{\min}).
$$

The "+3" reflects the matched filter's noise-free significance being
biased by optimizing over 3 nuisance parameters (2 sky position, 1 core
radius) — exposed as `SzeNoise.dof_offset` rather than hardcoded, in case a
different filter geometry is ever used. $\Theta(\zeta-\zeta_{\min})$ is a
**hard cut on the true, unbiased significance** — structurally different
from the richness cut, which lands on the *observed* $\lambda^{\rm ob}$
(Ghirardini 2024's "affects only the integration limits" note, and
Bocquet's "faint halos are undetectable" here). Because
$\sqrt{\zeta^2+3}$ is nonlinear, this convolution has no closed form
either: `JointMor.pdf_obs` nests a **second** Gauss--Legendre quadrature
over $\zeta^{\rm tr}$ inside the $\lambda^{\rm tr}$ one, at each node using
the intrinsic correlation to get the conditional lognormal law of
$\zeta^{\rm tr}\mid\lambda^{\rm tr}$
(`JointMor.conditional_zeta_given_lambda`) and clipping its lower
integration bound at $\zeta_{\min}$ — `gl_nodes_batched`'s own documented
behaviour makes an inverted bracket integrate to zero, exactly the
"undetectable halo" limit, with no special-casing needed.

## Selection thresholds need only the one lambda_true quadrature

`ThresholdSelection` computes $P(\zeta^{\rm tr} > \zeta_{\min}\ {\rm AND}\
\lambda^{\rm ob} > \lambda_{\rm th}\mid M,z)$ with the single
$\lambda^{\rm tr}$ quadrature `pdf_obs` also uses, because $\int d\xi\,
P(\xi\mid\zeta) = 1$ for any $\zeta>\zeta_{\min}$ — the SZ side of a
completeness calculation collapses to an analytic lognormal survival
probability, no $\xi$ (and hence no `SzeNoise`) needed. The richness side
is the same analytic EMG/Gaussian tail `richness_bin_probability` uses for
a bin edge, evaluated at the single edge $\lambda_{\rm th}$.

## Numerics: bracket width, quadrature order, and a measured precision ceiling

`pdf_obs`'s nested quadrature is a fixed-order product rule over a fixed
bracket, not adaptive quadrature. Profiling found the naive choice —
`selection_function.py`'s own `bracket_width=8` (tuned for that module's
different job, summing a *binned* indicator over the full richness range)
— wastes most of a 64-node rule on near-empty wings when evaluating
`pdf_obs` pointwise at a specific, possibly off-center, observed pair,
especially at high mass where $\lambda^{\rm tr}$'s absolute scatter is
large. Narrowing the bracket to `BRACKET_WIDTH = 5.0` and raising the
default orders to `N_QUAD = 128` / `N_QUAD_ZETA = 96` cuts the worst-case
relative error (measured against an $n=(256,192)$ reference across several
mass/redshift/observable combinations) from over 30% down to roughly a
few percent. That residual is a real, known ceiling: extremely improbable
(far double-tail, anti-correlated) `(xi_obs, lambda_obs, M)` combinations
are inherently ill-conditioned for *any* quadrature scheme, adaptive or
fixed — confirmed by cross-checking against `scipy.integrate.quad`, whose
own adaptive error estimate is comparable to the integral value itself in
that regime. Raise `n_quad_lambda`/`n_quad_zeta` for more precision at a
given mass point; widening `bracket_width` back up reintroduces the
resolution problem it was narrowed to fix.

`integrate_pdf_obs` is a small orchestrator: it builds one Gauss--Legendre
rule per observable, evaluates `pdf_obs` once on their outer-product grid
(`pdf_obs` broadcasts `xi_obs`/`lambda_obs` against each other the same
"grid, not pairwise" way {doc}`selection_function`'s own mass/redshift
grids do — no Python loop over grid points needed or wanted), and
contracts against the tensor-product weights. Measured: a $61\times61$
grid evaluated as a Python loop of scalar `pdf_obs` calls took ~445 ms;
the single vectorized call taking the same grid shape took ~0.6 ms.
