# Open investigation: Sigma_prj's transition sits ~2x too far out in R

**Status:** unresolved, actively suspected mechanism identified but not
confirmed or fixed. Picks up where {doc}`plan-bsel-stable-closure` (the
`b_small`/`excess_delta` closure fix) left off -- that fix is implemented,
tested, and documented; **this** is a separate, still-open problem found
while validating it visually against the digitized Costanzi et al. (2026)
Fig. 6 curve.

## The finding

Plotting the (already-fixed) model against the digitized curve for
$\lambda\in[20,30)$ (not just reading the aggregate residual) shows the
curve's shape doesn't match: the transition from the small-scale plateau
to the large-scale plateau happens at a different $R$ than the data show.

Quantified two ways (steepest pairwise slope, and midpoint-crossing
between the two plateaus -- both agree, so it isn't digitization noise),
comparing to $R_\lambda(\lambda^{\rm ob})$ converted to the digitized
data's own convention (comoving Mpc/h, i.e. $(\lambda^{\rm ob}/100)^{0.2}
(1+z)$):

| $z$ | model transition | digitized-data transition |
|---|---|---|
| 0.20 | $R=0.93$ cMpc/h ($0.97\times R_\lambda$) | $R=0.46$ cMpc/h ($0.48\times R_\lambda$) |
| 0.35 | $R=0.95$ cMpc/h ($0.89\times R_\lambda$) | $R=0.42$ cMpc/h ($0.39\times R_\lambda$) |
| 0.50 | $R=1.02$ cMpc/h ($0.86\times R_\lambda$) | $R=0.40$ cMpc/h ($0.34\times R_\lambda$) |

The **model** transitions right around $R_\lambda(\lambda^{\rm ob})$
itself. The **digitized data** (the real benchmark) transitions at
roughly **half** that -- matching the paper's own sigmoid centre
$\theta_0=\theta_\lambda/2$ almost exactly (that predicts ratio $=0.5$).
So the bare sigmoid (`SigmoidBias`/`sigmoid_theta`) is centred correctly;
by the time its signal comes out the other end of `SigmaPrj`'s
line-of-sight integral, the transition has been pushed outward by
roughly $2\times$.

## What's been ruled out / narrowed down

1. **Not a `SigmoidBias`/`sigmoid_theta` bug.** Formula is
   `k=2.5/theta_lambda`, `theta0=theta_lambda/2`, matching the paper
   exactly, and `marginalised_bias` wires `damping`/`theta0_frac`
   through correctly.
2. **Not `rnd`/`cl` channel dilution in the ratio.** Isolated the `cl`
   channel alone (`SigmaPrj.sigma_prj(..., channel="cl")`, dividing by
   the constant-$b_{\rm eff}$ `cl` channel rather than the full `sum`)
   -- transition is at the same $R\approx0.94$ cMpc/h, ratio $\approx0.97$.
   So the shift is inside the correlated channel's own construction, not
   a mixing artefact with the uncorrelated background term.
3. **Narrowed to `MassShells`** (`src/clenspy/lensing/projection.py`),
   specifically `shell_masses`/`tail_masses`
   (`src/clenspy/utils/los_integrals.py`): the machinery that converts
   each theta-shell's `b_sel(theta)`-weighted LOS contribution into a
   per-$R$ mass via the neighbour halo's own (extended, NFW) profile,
   using a by-parts enclosed-mass identity shared with (and probably
   copied from) `clenspy.selection.miscentering`.

## The open question: is `shell_masses`'s argument order right?

`shell_masses` calls (inside the `im` loop, for `which="sigma"`):

```python
m_edges = (s_edges[:, None] ** 2) * mean_sigma(
    s_edges[:, None] / rs[im], R[None, :] / rs[im]
)
```

i.e. `mean_sigma(x=s_edges/rs, x_mis=R/rs)` -- the *theta-shell's own
transverse offset* `s` is passed as the "radius" argument, and the
*query point* `R` is passed as the "miscentering offset" argument. Naive
physical intuition says this should be the other way round: we want "how
much of a halo sitting at transverse offset $s$ from the cluster falls
within radius $R$ of the cluster", i.e. radius$=R$, offset$=s$.

Worked through (not empirically tested) whether this is actually a bug:
the *local* (differential) miscentered surface density is symmetric
under swapping radius$\leftrightarrow$offset (the 3D separation
$\sqrt{R^2+s^2-2Rs\cos\varphi}$ is symmetric in $R,s$, and azimuthal
integration doesn't break that), so `sigma_hat(x,x_mis)` alone may not
even care which argument is which. But `shell_masses`'s own docstring
explicitly frames the identity as "the enclosed mass of the halo *offset
by R*" -- i.e. it deliberately treats $R$ as the offset and the
integration variable $s$ (running over the shell edges) as the radius,
which is a real by-parts trick (`$\int_{s_1}^{s_2}2\pi s f(s)ds =
\pi[s^2 F(s)]$` requires $F$ to be a *mean-within* function of its first
argument), not obviously a copy-paste swap. This is the same
enclosed-mass-via-offset construction likely shared with the
already-validated `clenspy.selection.miscentering` module
(`docs/miscentering_math.md`, its own test suite).

**This was left unresolved by reasoning alone -- next step should be an
empirical test, not more inference:** build one NFW halo at a known
transverse offset, compute its aperture-enclosed mass at a grid of $R$
via `shell_masses`, and independently via brute-force 2D numerical
integration (direct $\int\!\int \Sigma_{\rm halo}(|\vec r - \vec
d|)\,d^2r$ over a disk of radius $R$ centred on the cluster, $\vec d$ the
halo's true offset). If they agree, the argument order is fine and the
$2\times$ shift comes from somewhere else (theta-shell resolution/width
relative to $r_s$? the exclusion-radius interaction with the innermost
shells? the `r_trunc`/`tail_masses` subtraction?). If they disagree, the
argument order is the bug and swapping it is the fix -- but verify against
`clenspy.selection.miscentering`'s own tests first, since `shell_masses`
is shared: if the miscentering module calls it with the *same* argument
convention and passes its own validated tests, the convention is
probably a documented feature of the by-parts identity, not a bug, and
the $2\times$ shift has a different cause entirely (worth checking
whether `clenspy.selection.miscentering` even exercises the
`which="sigma"` un-truncated path this bug hunt is in, versus only the
`which="ds"` signed path -- they could be independently correct/incorrect).

## Pending, not yet actioned

The user's last instruction before this summary was: **"Feel to strip
all the MassShell shit"** -- i.e. consider removing `MassShells`
entirely and replacing the theta-shell-to-R mapping with something
simpler/more transparent. This is a **large, invasive change**:
`MassShells` is load-bearing throughout `SigmaPrj.sigma_prj`/
`deltasigma_prj`, and removing it would affect the entire
`projection_lensing` validation suite (`validation/validate_sigma_prj_mock.py`,
`tests/test_projection.py`, `tests/test_sigma_prj_profile.py`, the whole
`papers/projection_lensing` report). **Was not started** -- I asked what
should replace it (a point-mass/delta-function approximation at
$R=\theta\chi_o$? a different, from-scratch profile-convolution
derivation? something else?) before touching anything this size, and the
conversation was interrupted before getting an answer. **Do not start
deleting `MassShells` without first confirming the replacement design**
-- this is exactly the kind of hard-to-reverse, wide-blast-radius change
that needs explicit scope agreement first.

## Recommended order for the next session

1. Confirm with the user what should replace `MassShells` (if the
   "strip it" instruction still stands) -- or do the brute-force
   NFW-offset cross-check above first, to know whether `MassShells` is
   actually wrong before deciding to remove it. The cross-check is
   cheap and non-destructive; strongly prefer it before any deletion.
2. Check whether `clenspy.selection.miscentering`'s own tests exercise
   the same `shell_masses`/`which="sigma"` path this investigation is
   about, and what argument convention they use -- fastest way to learn
   whether the convention is a documented feature or a fresh bug.
3. Only after (1)/(2): either fix `shell_masses`'s argument order (if
   confirmed wrong) or pursue a different explanation for the $2\times$
   shift (theta-shell resolution near $r_s$, exclusion-radius coupling,
   `r_trunc`/`tail_masses` interaction).
4. Once resolved, rerun `validate_fig6_digitized.py` and
   `validate_sigma_prj_mock.py` (numbers will likely change --
   {doc}`plan-bsel-stable-closure`'s median-residual numbers were
   computed *before* this shape issue was found, so they may improve or
   need re-stating), and update `docs/validation.md` /
   `papers/validations` accordingly.
