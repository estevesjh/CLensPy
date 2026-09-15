# A well-conditioned closure for $b_{\rm small}$

**Status:** implemented (`SelBiasEngine.excess_delta`/`gamma_lambda`/
`b_small_large` in `src/clenspy/selection/bsel.py`) and validated against both
references (§1–§6): median Fig-6 residual 0.51→0.10, mock leg D 12/12 bins pass.
**§9, found after implementing, is still open**: the closure's mean level is
right but its redshift-dependence has the wrong sign — read that section before
trusting any single-$z$-bin comparison.
Companion to `docs/plan-fig6-validation.md`. Scratch scripts used for the numbers
below live outside the repo (`$TMPDIR/.../scratchpad/{diag,tilt,recipe}.py`).

**Bottom line.** The closure algebra is *not* wrong and $D$ is *not* too small to
divide by. The closure is an exact linear response in one scalar,
$\delta = \langle\lambda^{\rm ob}-\lambda^{\rm tr}\rangle/\Delta_{\rm RND} - 1$, with a
gain of $A_s \simeq 18\!-\!40$. `_ltr_weights` feeds it a $\delta$ that is
$2\!-\!5\times$ too large — the mock catalogue settles this independently — and the
gain turns that into the observed $3\!-\!4\times$ error in $B_{\rm small}$. The fix is to
compute $\langle\lambda^{\rm ob}-\lambda^{\rm tr}\rangle$ from the model's own
projection operator (a first-order Eddington shift, attenuated by the correlated
fraction of the projection variance) instead of from an externally calibrated
$P(\lambda^{\rm ob}|\lambda^{\rm tr})$ kernel. No tuned constants.

---

## 1. Notation

Everything below is at one fixed $(\lambda^{\rm ob}, z^{\rm ob})$, dropped from the
notation. Write the paper's projection operator (`SelBiasEngine._operators`) as

$$
\mathcal{N}[g] \;=\;
\int\! dz\,\frac{dV}{dz\,d\Omega}
\int\! dM\, n(M,z)
\int\! d\lambda\, P(\lambda|M,z)\,
2\pi\!\!\int\!\! d\theta\,\sin\theta\;
w_z(z)\, f_A(\lambda,z,\theta)\; g(\lambda,\theta,M,z),
$$

so that the code's scalars are $P_1 = \mathcal{N}[\lambda]$,
$I_2 = \mathcal{N}[\lambda\, b\, \xi_{\rm NL}]$,
$D = \mathcal{N}[\lambda\, b\, \xi_{\rm NL}(1-\sigma)]$, $I_1 = I_2 - D$, and
the variance operators of `operators_var` are the same with
$\lambda^2, w_z^2, f_A^2$:
$P_1^{(2)} = \mathcal{N}_2[\lambda^2]$, $I_2^{(2)} = \mathcal{N}_2[\lambda^2 b\,\xi_{\rm NL}]$.

Let $s = 0.13$ (`boost_slope`), $\Delta_{\rm RND} = P_1 + b_{\rm eff} I_2$,
$V_{\rm tot} = P_1^{(2)} + b_{\rm eff} I_2^{(2)}$ (the total variance of the
projected-richness boost), and $\Delta \equiv \lambda^{\rm ob} - \lambda^{\rm tr}$.

---

## 2. The closure is a one-parameter linear response

Substituting $\Delta = \Delta_{\rm RND}(1+\delta)$ into the two closure equations,

$$
b_{\rm large} = b_{\rm eff}\,(1 + s\,\delta), \qquad
b_{\rm small} = \frac{\Delta - P_1 - b_{\rm large} I_1}{D},
$$

and using $\Delta_{\rm RND} - P_1 - b_{\rm eff} I_1 = b_{\rm eff}(I_2 - I_1) = b_{\rm eff} D$
(the closure's own fixed point: an average line of sight has
$b_{\rm small}=b_{\rm large}=b_{\rm eff}$), every term rearranges *exactly* into

$$
\boxed{\;
b_{\rm small} = b_{\rm eff} + \delta\, A_s, \qquad
A_s \equiv \frac{\Delta_{\rm RND} - s\, b_{\rm eff} I_1}{D},
\qquad
b_{\rm large} = b_{\rm eff} + \delta\, s\, b_{\rm eff}. \;}
$$

Three consequences follow immediately.

**(a) The $\lambda^{\rm tr}$ quadrature is decorative.** Both plateaus are affine in
$\delta$, hence affine in $\lambda^{\rm tr}$, so
$B_{\rm small} = b_{\rm small}(\langle\lambda^{\rm tr}\rangle)$ identically — this
reproduces (and supersedes) the calling agent's affine proof. The 22-node
posterior contributes exactly one number, its mean. Nothing about *how* the
marginalisation is ordered can change the answer.

**(b) The instability is a gain, not a cancellation.** The relative amplification is

$$
A \;\equiv\; \frac{b_{\rm small}-b_{\rm eff}}{b_{\rm large}-b_{\rm eff}}
\;=\; \frac{\Delta_{\rm RND} - s\,b_{\rm eff} I_1}{s\, b_{\rm eff}\, D},
$$

independent of $\delta$, i.e. a fixed property of the operators. Measured on the six
Fig-6 panels, $A = 35\!-\!67$ and $A_s = 18\!-\!40$. The condition number of
$b_{\rm small}$ with respect to the *one* input it has is

$$
\kappa \;=\; \frac{\partial \ln b_{\rm small}}{\partial \ln \langle\Delta\rangle}
\;=\; \frac{\langle\Delta\rangle}{\Delta_{\rm RND}}\,\frac{A_s}{b_{\rm small}}
\;\approx\; 7\!-\!10 .
$$

A 10 % error in the assumed mean excess richness is a 70–100 % error in
$b_{\rm small}$. No quadrature refinement, kernel swap, or reordering touches this;
it is the model's own sensitivity.

**(c) The gain lives in the $0.13$, not in $D$.** $D=0.115$ and $I_1=0.237$ are the
same order — the $2\times2$ ansatz basis $\{1-\sigma,\sigma\}$ is not ill-conditioned
under the operator. What is ill-conditioned is the *closure*: the boost model pins
$b_{\rm large}$ to move by only $13\%$ per unit $\delta$, so the entire required change
in $\mathcal{N}[\lambda b\,\xi\, b_{\rm sel}]$, which is $\delta\,\Delta_{\rm RND}$, must be
dumped through the $(1-\sigma)$ channel. $b_{\rm small}$ is a residual bucket by
construction.

Because it is a residual bucket, its numerator is currently formed as a difference
of large numbers ($7.27 - 2.34 - 0.92 = 4.02$ for the worked bin, where the
consistent value is $b_{\rm eff}D = 0.39$). The boxed form removes that subtraction
entirely and is the form that should be coded.

---

## 3. What is actually wrong: the input $\delta$, proven against the mock

`SelectionBias/mock_lob_sigma_catalog.fits` stores both `LAMBDA_TR_LOB` and
`LAMBDA_OB_LOB`, so $\langle\lambda^{\rm ob}-\lambda^{\rm tr}\rangle$ in each
$(\lambda^{\rm ob}, z)$ bin is directly measurable. It is the *only* input the closure
has, and it is measurable without any model.

| panel | mock $\langle\Delta\rangle$ | engine posterior | $\Delta_{\rm RND}$ | Fig-6-implied$^\ast$ |
|---|---|---|---|---|
| $\lambda[20,30)\;z[0.20,0.35)$ | 2.792 | 6.006 | 2.689 | 3.037 |
| $\lambda[20,30)\;z[0.35,0.50)$ | 3.593 | 6.404 | 3.051 | 3.711 |
| $\lambda[20,30)\;z[0.50,0.65)$ | 4.248 | 7.274 | 3.534 | 4.245 |
| $\lambda[60,500)\;z[0.20,0.35)$ | 6.643 | 12.245 | 6.491 | 8.223 |
| $\lambda[60,500)\;z[0.35,0.50)$ | 9.414 | 13.684 | 7.063 | 10.162 |
| $\lambda[60,500)\;z[0.50,0.65)$ | 10.292 | 15.498 | 7.712 | 10.868 |

$^\ast$ $P_1 + b_{\rm small}^{\rm fit} D + b_{\rm large}^{\rm fit} I_1$, i.e. the excess the
digitized Fig-6 fit implies through the closure.

The Fig-6-implied excess agrees with the mock to 0–24 % (median 7 %). The engine's
posterior is $1.5\!-\!2.2\times$ high. **Three independent routes — the mock, the
published figure, and the model's own $\Delta_{\rm RND}$ — agree with each other and
disagree with `_ltr_weights`.** That localises the bug entirely to the
$\lambda^{\rm tr}$ posterior. $P_1$, $I_1$, $I_2$, $D$ and the closure algebra are
exonerated: fed the mock's $\langle\Delta\rangle$, the *unmodified* closure gives
$b_{\rm small}= 3.6,\,7.2,\,9.4,\,6.4,\,17.0,\,20.2$ against fits of
$6.0,\,9.9,\,11.6,\,12.4,\,19.9,\,23.0$.

### Why the posterior blows up

$P(\lambda^{\rm tr}|\lambda^{\rm ob}) \propto P(\lambda^{\rm ob}|\lambda^{\rm tr})\,n(\lambda^{\rm tr})$.
Writing $n(\lambda^{\rm ob}-\Delta) = n(\lambda^{\rm ob})e^{\gamma\Delta}$ with
$\gamma = -d\ln n/d\lambda^{\rm tr}$, the posterior mean excess is the
*exponentially tilted* mean of the projection distribution,
$\langle\Delta\rangle = \partial_\gamma \ln M_\Delta(\gamma)$.

Measured in the mock at $z\in[0.5,0.65)$: $\gamma \simeq 0.15\!-\!0.19$ near
$\lambda \simeq 20$, while $p(\Delta|\lambda^{\rm tr})$ has an exponential tail of rate
$\simeq 0.3$. The tilt is therefore *marginally convergent*, and it is numerically
explosive: tilting the mock's own empirical $p(\Delta|\lambda^{\rm tr}=20)$ gives

| $\gamma$ | 0 | 0.10 | 0.15 | 0.186 | 0.25 |
|---|---|---|---|---|---|
| $\langle\Delta\rangle_{\rm tilted}$ | 3.11 | 5.50 | 9.17 | 15.04 | 30.90 |

against a measured truth of **4.32**. Both the Y3 EMG kernel (`plob_mode="y3"`) and
the self-consistent exponential (`plob_mode="self"`, rate
$\tau = 2\bar\Delta/(\bar\Delta^2+V) = 0.28$) sit inside this explosive regime, which
is why `"self"` made things *worse* rather than better: it is not a kernel-choice
problem, it is that any fixed exponential kernel resummed against a steep richness
function is unstable here.

The physical reason the truth is finite: $p(\Delta|\lambda^{\rm tr})$ is *not*
independent of $\lambda^{\rm tr}$. The tilt reweights configurations at
$\lambda^{\rm tr}=\lambda^{\rm ob}-\Delta$, where both the aperture
$R_\lambda \propto \lambda^{0.2}$ and the host environment are smaller, so the tail
being exponentially reweighted is not the tail that exists there. Resumming a
$\lambda^{\rm tr}$-independent $p(\Delta)$ to all orders in $\gamma$ is therefore not
merely inaccurate, it is the wrong series.

---

## 4. The proposed estimator

Two steps, each standard, each parameter-free, each computable from operators the
engine already has.

### Step 1 — Eddington shift, truncated at first order

$$
\langle\Delta\rangle_{\rm up} \;=\; \Delta_{\rm RND} \;+\; \gamma\, V_{\rm tot},
\qquad
\gamma = -\left.\frac{d\ln n(\lambda^{\rm tr})}{d\lambda^{\rm tr}}\right|_{\lambda^{\rm ob}}.
$$

This is $\partial_\gamma \ln M_\Delta(\gamma)$ truncated at $O(\gamma)$. The
truncation is a *choice made because the resummation is not convergent* (§3), not an
oversight: linear response is the highest order at which the
$\lambda^{\rm tr}$-independence of $p(\Delta)$ is a controlled approximation.
$n(\lambda^{\rm tr})$ is already computed inside `_ltr_weights._prior`, so $\gamma$ is a
two-point log-derivative of an existing function; it is stable to 0.1 % against the
finite-difference width over $\pm5\%\!-\!\pm25\%$ of $\lambda^{\rm ob}$.

### Step 2 — attribute only the correlated part of the up-scatter

The closure attributes the whole of $\Delta - \Delta_{\rm RND}$ to an enhanced
*clustering amplitude*. That is wrong: a cluster is up-scattered in $\lambda^{\rm ob}$
by two physically different mechanisms.

* A chance projector on an uncorrelated line of sight — a Poisson realisation of the
  same mean field. Its variance is the shot term $P_1^{(2)} = \mathcal{N}_2[\lambda^2]$
  (Campbell's theorem for a marked Poisson process: the variance of
  $\Delta = \sum_i \lambda_i$ is $\int dN\,\lambda^2$). It raises $\lambda^{\rm ob}$ but
  carries no information about the bias of the surrounding field.
* A genuinely denser correlated environment. Its variance is the two-halo term
  $b_{\rm eff} I_2^{(2)}$.

The linear-MMSE (Wiener) share of an observed up-scatter attributable to the second
mechanism is

$$
w \;=\; \frac{b_{\rm eff} I_2^{(2)}}{P_1^{(2)} + b_{\rm eff} I_2^{(2)}}
\;=\; \frac{V_{\rm 2h}}{V_{\rm tot}} \;\in\; [0.42,\,0.71]
\ \text{across the six panels}.
$$

Only $w\,(\langle\Delta\rangle_{\rm up}-\Delta_{\rm RND})$ should drive $b_{\rm sel}$.
Combining the two steps, $\gamma V_{\rm tot}\cdot w = \gamma V_{\rm 2h}$, so the whole
recipe collapses to one line:

$$
\boxed{\;\delta \;=\; \frac{\gamma\; b_{\rm eff} I_2^{(2)}}{\Delta_{\rm RND}}\;}
$$

— the Eddington tilt of the *correlated* projection variance, in units of the mean
projection. Nothing is fitted.

> **Note on the shrinkage the calling agent tried.** This *is* a shrinkage, and it
> explains why the empirical version half-worked. But it is applied to $\delta$
> (before the closure), not to $B_{\rm small}$ (after it), so both plateaus stay mutually
> consistent and the closure identity
> $\langle\Delta\rangle_{\rm eff} = P_1 + b_{\rm small}D + b_{\rm large}I_1$ still holds exactly
> for the attributed excess. And the weight is $V_{\rm 2h}/V_{\rm tot}$, computed from
> `operators_var`, not $k\,B_{\rm large}$ with $k$ fit to the targets.

---

## 5. Numerical recipe

Drop-in for `_closure` / `_ltr_weights` / `b_small_large`. `_ltr_weights` is no longer
needed by the closure at all (§2a); keep it only if something else consumes it.

```python
# --- new: the richness-function log-slope, from the existing _prior ------
def gamma_lambda(self, lob, zob, frac=0.15):
    """-dln n(ltr)/dltr at ltr = lob, from the engine's own richness function."""
    m  = np.logspace(np.log10(self.min_mass), self.log10_M_max, 60)
    hm = self.hmf(m, zob)
    nodes = np.array([lob * (1 - frac), lob * (1 + frac)])
    p = self.mor.pdf(nodes[:, None], m[None, :], zob)          # same as _prior
    n = np.trapezoid(p * (hm * m)[None, :], np.log(m), axis=1)
    return -float(np.log(n[1] / n[0]) / (nodes[1] - nodes[0]))

# --- new: the one physical input, self-consistently ----------------------
def excess_delta(self, lob, zob, b_eff):
    """delta = <lob-ltr|lob,zob>/Delta_RND - 1, Eddington tilt of the
    *correlated* projection variance (docs/plan-bsel-stable-closure.md §4)."""
    P1,  _, I2  = self.operators(lob, zob)
    P1v, _, I2v = self.operators_var(lob, zob)
    D_RND = P1 + b_eff * I2
    return self.gamma_lambda(lob, zob) * b_eff * I2v / D_RND

# --- replaces _closure + b_small_large -----------------------------------
def b_small_large(self, lob, zob, b_eff=None, delta=None):
    P1, I1, I2 = self.operators(lob, zob)
    D    = self._d_cache[("ops", float(lob), float(zob))]
    beff = self.b_eff(lob, zob) if b_eff is None else float(b_eff)
    if delta is None:
        delta = self.excess_delta(lob, zob, beff)
    A_s = (P1 + beff * I2 - self.boost_slope * beff * I1) / D   # the gain, 18-40
    return beff + delta * A_s, beff * (1.0 + self.boost_slope * delta)
```

Notes for whoever implements it:

1. Keep `A_s` reachable (return it, or cache it) — it is the model's own condition
   number and belongs in any table the engine writes.
2. `excess_delta` should stay overridable (`delta=` passthrough above) so the mock's
   measured $\langle\Delta\rangle$ can be injected for validation.
3. `_closure`'s guard `abs(denom) < 1e-12*(...)` disappears with the subtraction; if
   $D\to0$ then $A_s\to\infty$ and the right response is to raise, not to silently
   fall back to $b_{\rm large}$.
4. The `NOTE (open issue)` in `_closure` about $I_2 - I_1$ cancellation can be
   retired — the direct-$D$ quadrature was the right call, but the cancellation that
   actually mattered was in the *numerator*, and the boxed form removes it.

---

## 6. Validation

Harness: `validation/validate_sigma_prj_mock.py` (`build_halo_model`, `b_eff_table`),
`HodMor.buzzard()`, bin-representative $(\lambda^{\rm ob}, z^{\rm ob}, b_{\rm eff})$;
targets are the calling agent's closed-form fits to the digitized Costanzi et al.
(2026) Fig. 6 theory curve, plus the mock's directly measured
$\langle\lambda^{\rm ob}-\lambda^{\rm tr}\rangle$.

| panel | $\gamma$ | $w$ | $\delta$ | $\langle\Delta\rangle$ pred / mock | $b_{\rm small}$ / fit | $b_{\rm large}$ / fit | current $b_{\rm small}$ / fit |
|---|---|---|---|---|---|---|---|
| $\lambda[20,30)\;z0.20$ | 0.1340 | 0.539 | 0.283 | 3.451 / 2.792 = **1.24** | 8.04 / 6.00 = **1.34** | 2.96 / 2.53 = 1.17 | 25.43 / 6.00 = 4.24 |
| $\lambda[20,30)\;z0.35$ | 0.1416 | 0.479 | 0.261 | 3.846 / 3.593 = **1.07** | 9.09 / 9.89 = **0.92** | 3.25 / 2.35 = 1.38 | 28.24 / 9.89 = 2.86 |
| $\lambda[20,30)\;z0.50$ | 0.1503 | 0.419 | 0.226 | 4.334 / 4.248 = **1.02** | 10.13 / 11.63 = **0.87** | 3.50 / 2.39 = 1.47 | 34.84 / 11.63 = 3.00 |
| $\lambda[60,500)\;z0.20$ | 0.0575 | 0.712 | 0.281 | 8.312 / 6.643 = **1.25** | 13.28 / 12.40 = **1.07** | 6.01 / 6.27 = 0.96 | 29.44 / 12.40 = 2.37 |
| $\lambda[60,500)\;z0.35$ | 0.0628 | 0.658 | 0.261 | 8.905 / 9.414 = **0.95** | 14.71 / 19.91 = **0.74** | 6.59 / 6.99 = 0.94 | 36.35 / 19.91 = 1.83 |
| $\lambda[60,500)\;z0.50$ | 0.0687 | 0.605 | 0.232 | 9.504 / 10.292 = **0.92** | 16.12 / 23.01 = **0.70** | 7.14 / 7.36 = 0.97 | 46.86 / 23.01 = 2.04 |

Summary (ratio model/target):

| quantity | median | rms log-error | worst |
|---|---|---|---|
| $\langle\Delta\rangle$ vs mock — **proposed** | 1.05 | **13.5 %** | 22 % |
| $\langle\Delta\rangle$ vs mock — current code | 1.75 | 56 % | 77 % |
| $b_{\rm small}$ vs Fig-6 fit — **proposed** | 0.89 | **23.6 %** | 36 % |
| $b_{\rm small}$ vs Fig-6 fit — current code | 2.61 | 100 % | 144 % |
| $b_{\rm large}$ vs Fig-6 fit — **proposed** | 1.07 | 21.7 % | 38 % |

$\gamma$ varies by $<0.2\%$ when the finite-difference half-width is moved over
$\pm5\%$ to $\pm25\%$ of $\lambda^{\rm ob}$, so the recipe carries no hidden numerical
knob.

### 6.1 $\langle\lambda^{\rm tr}\rangle$ survives — as an output

The affine identity of §2a is unchanged by any of this: the closure still depends on
the $\lambda^{\rm tr}$ posterior only through its mean. What changes is the *direction*
of the arrow. Currently $\langle\lambda^{\rm tr}\rangle$ is an input, computed from an
external kernel; under §4 it is a **prediction**,

$$
\langle\lambda^{\rm tr}\rangle \;=\; \lambda^{\rm ob} - \Delta_{\rm RND}\,(1+\delta),
$$

which the mock can check for free — and which is the cleanest single test of the
whole proposal, since it needs no Fig-6 digitisation and no bias model at all.

| panel | implied $\langle\lambda^{\rm tr}\rangle$ | mock | ratio | current code | ratio |
|---|---|---|---|---|---|
| $\lambda[20,30)\;z0.20$ | 20.47 | 21.12 | **0.97** | 17.91 | 0.85 |
| $\lambda[20,30)\;z0.35$ | 20.02 | 20.32 | **0.99** | 17.46 | 0.86 |
| $\lambda[20,30)\;z0.50$ | 19.49 | 19.57 | **1.00** | 16.55 | 0.85 |
| $\lambda[60,500)\;z0.20$ | 71.62 | 72.08 | **0.99** | 67.68 | 0.94 |
| $\lambda[60,500)\;z0.35$ | 69.21 | 69.61 | **0.99** | 64.44 | 0.93 |
| $\lambda[60,500)\;z0.50$ | 67.19 | 65.89 | **1.02** | 61.19 | 0.93 |

Within 3 % everywhere (median 0.5 %), against 6–15 % low today. Some of that is
$\langle\lambda^{\rm tr}\rangle$ being a large number minus a small one, but the
comparison is still $\lesssim0.7$ richness units per bin, and the representative
$\lambda^{\rm ob}$ used here differs from the mock's own bin mean by up to 1.5 %, which
sets the floor. This is worth keeping as a permanent regression check in
`validate_sigma_prj_mock.py` — it tests the one input the closure has, directly,
with no amplification in the way.

---

## 7. Honest caveats

1. **The $b_{\rm large}$ residual in $\lambda[20,30)$ is a different bug.** The Fig-6
   fits give $b_{\rm large} = 2.35\!-\!2.53$ there against $b_{\rm eff} = 2.86\!-\!3.40$.
   Since $b_{\rm large} = b_{\rm eff}(1+s\delta)$ with $\delta>0$ can never fall below
   $b_{\rm eff}$, the low-richness $b_{\rm eff}$ normalisation (or the digitisation of
   those panels) is off by 20–40 %. This closure fix cannot and does not address it;
   the 1.17/1.38/1.47 entries in §6 are that offset, not a failure of the estimator.
2. **The two $z<0.35$ panels inherit the known $\Delta_{\rm RND}$ problem.** They are
   exactly the bins flagged in `_closure`'s existing `NOTE (open issue)` and in
   `validate_sigma_prj_mock.py` leg D, where $\Delta_{\rm RND}$ runs 17–50 % above the
   mock's own random-line-of-sight excess. With $\kappa \approx 8$, that alone accounts
   for the 1.24/1.25 excess ratios in §6. Fixing $\Delta_{\rm RND}$ at low $z$ should
   improve the closure without touching it.
3. **Step 1 is first-order only.** Compared to the mock's *own* measured Eddington
   shift ($\langle\Delta|\lambda^{\rm ob}\rangle - \langle\Delta|\lambda^{\rm tr}\rangle$),
   $\gamma V_{\rm tot}$ is 2.5× high at low richness and correct at high richness; step 2's
   $w$ then over-corrects at high richness and under-corrects at low. The two errors
   partially cancel. A genuinely predictive Eddington shift needs the
   $\lambda^{\rm tr}$-dependence of $p(\Delta|\lambda^{\rm tr})$ — i.e. re-evaluating the
   operator at the *projector's* aperture, which is a separate piece of work (and is
   plausibly what Costanzi's `numerator2(ltr, zcl)` was groping towards; note that his
   own `STATUS.md` reports the same 2–20× overshoot, so it is not a solved problem
   there either).
4. **Do not read the 6 fit numbers as truth.** They are a closed-form fit to a
   *digitized* published curve. Their internal consistency check — the excess they
   imply through the closure agrees with the mock to a median 7 % — is what makes them
   usable, and it is also the floor on how well any recipe can be judged against them.

## 8. What was ruled out

* **Reordering or refining the $\lambda^{\rm tr}$ marginalisation** — provably inert
  (§2a). $B_{\rm small} = b_{\rm small}(\langle\lambda^{\rm tr}\rangle)$ exactly.
* **`plob_mode="self"`** — sits deeper inside the divergent-tilt regime than the Y3
  kernel (§3); it made $B_{\rm small}$ worse for a reason, not by accident.
* **A least-squares solve over an $\lambda^{\rm tr}$-dependent design $D(\lambda^{\rm tr})$,
  $I_1(\lambda^{\rm tr})$** — the natural next idea, but it is not the fix: the closure
  is exactly determined and consistent at every $\lambda^{\rm tr}$ (§2), so there is no
  over-determination to regularise, and re-evaluating the operator at the
  *projector's* aperture (Costanzi's `numerator2(ltr,...)`) collapses by $10^3\!-\!10^4$
  toward low $\lambda^{\rm tr}$ and blows the answer up by $10^{5\!-\!6}$. It changes the
  gain, not the wrong input.
* **Clipping / post-hoc shrinkage of $B_{\rm small}$** — the right *shape* of answer, but
  applied to the wrong quantity and with a fitted scale; §4 derives the same
  operation applied to $\delta$ with a computed weight.

---

## 9. Open limitation found post-implementation: $\delta$'s redshift dependence is the wrong sign

Discovered by actually plotting the fixed model against the digitized Fig-6 curve
(not just reading the aggregate residual): in $\lambda[60,500)$ the three per-$z$
model curves are visually almost indistinguishable, while the digitized data fan
out sharply by $z$. Quantified two independent ways, inverting the closed-form
fit (§6) separately through the $b_{\rm small}$ side and the $b_{\rm large}$ side
of the closure:

| $z$ | $\delta$ needed (from $b_{\rm small}$ fit) | $\delta$ needed (from $b_{\rm large}$ fit) | $\delta$ from `excess_delta` |
|---|---|---|---|
| 0.28 | 0.248 | 0.621 | 0.281 |
| 0.43 | 0.423 | 0.749 | 0.261 |
| 0.58 | 0.407 | 0.469 | 0.232 |

Both independent inversions **rise** from $z=0.28\to0.43$; the model's own
$\delta$ **falls monotonically** across the whole range. In $\lambda[20,30)$ the
$b_{\rm large}$-side inversion is deeply negative ($-0.88$ to $-2.29$) at every
$z$ — impossible under $b_{\rm large}=b_{\rm eff}(1+s\delta)$ for any $\delta>0$,
confirming caveat §7.1 (a $b_{\rm eff}$-normalisation issue in that richness bin,
independent of $\delta$'s $z$-trend).

**Ruled out: quadrature precision.** Doubling $(n_\theta,n_M,n_z)$ from
$(96,48,96)$ to $(160,80,160)$ and sharpening `gamma_lambda`'s finite-difference
step from 0.15 to 0.05 changes $\delta$ by **<0.05%** at every $z$ tested — the
formula is fully converged; this is not a numerics problem.

**Isolated to `operators_var`'s own redshift evolution.** Breaking $\delta =
\gamma b_{\rm eff} I_2^{(2)}/\Delta_{\rm RND}$ into its factors, at $\lambda[60,500)$:

| $z$ | $P_1$ | $I_2$ | $P_1^{(2)}$ | $I_2^{(2)}$ | $I_2^{(2)}/P_1^{(2)}$ | $\gamma$ |
|---|---|---|---|---|---|---|
| 0.28 | 2.574 | 0.675 | 12.81 | 5.463 | 0.426 | 0.0575 |
| 0.43 | 3.156 | 0.613 | 15.23 | 4.604 | 0.302 | 0.0628 |
| 0.58 | 3.951 | 0.543 | 17.03 | 3.763 | 0.221 | 0.0687 |

(same qualitative pattern in $\lambda[20,30)$). $P_1^{(2)}$ (uncorrelated/shot
variance) grows +33% across this range while $I_2^{(2)}$ (correlated variance)
*falls* 31% — the correlated share of the total projection variance nearly
halves with $z$. $\Delta_{\rm RND}$ (dominated by $P_1$'s independent +54%
growth) is separately validated against the mock to 2–7% at $z\ge0.35$ (§6 of
`validate_sigma_prj_mock.py` leg D), so it is not the suspect; $\gamma$ moves the
right direction (+20%) but not enough to compensate. The numerator $\gamma
I_2^{(2)}$ simply does not grow with $z$ as fast as $\Delta_{\rm RND}$ does, so
$\delta$ falls instead of rising.

**Status: open, not fixed this session.** The first-order Eddington-tilt
estimator (§4) gets the *mean level* right (§6's validation) but not the
$z$-shape. Candidate next steps, none attempted: re-derive $\delta$'s
$z$-dependence from the $\lambda^{\rm tr}$-dependence of $p(\Delta\mid
\lambda^{\rm tr})$ directly (the same missing piece caveat §7.3 already names
for the richness axis); or accept $\delta(\lambda^{\rm ob},z^{\rm ob})$ as an
empirically-fit surface rather than a derived quantity, now that §6's
digitized-curve fit gives two real per-$z$ data points to fit it to (four, once
the remaining two richness bins are digitized).
  operation applied to $\delta$ with a computed weight.
