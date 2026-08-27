# CLensPy → cosmology-code: migration plan

> **Amended 2026-08-26.** Sections 1.4 (`survey/`) and 1.4 (`kernels/`) as
> originally written get the Survey integrand and the projection integrand
> wrong. See **Errata: the Survey and projection integrands** below, which
> supersedes them. The authority is the production pipeline in
> `y3_cluster_cpp`, not this plan.

Assessed against the `cosmology-code` skill (SKILL.md rules 1–8). Ordered by risk, not
by effort. P0 items are correctness traps; P1 is structure; P2 is discipline that pays
off once the package grows a survey-facing layer.

---

## Where you already comply

Worth naming, because these are the expensive parts and they are done.

- **The specification exists.** `docs/einasto_proj_density_v3.tex`, `einasto_math.md`,
  `miscentering_math.md`, `fractional_derivative_einasto.tex`. Rule 1 is satisfied at the
  source: there is a written derivation for the hard parts, and the code cites it.
- **Notation already transliterates the notes.** `einasto_lown.py` carries `A_k`, `D_k`,
  `S_j`, `T_j`, `x = R/h`, `nu_k` straight off the Mellin–Barnes kernel. That is exactly
  rule 1, done well.
- **Provenance is cited inline** — Retana-Montenegro et al. (2012), Tinker et al. (2010),
  Johnston et al. (2007), McClintock et al. (2019, eq. 27).
- **Real external validation exists**: `test_nfw.py` against `pyccl`, `test_twohalo.py`
  against a cluster_toolkit-validated reference. Most packages never get here.
- **Approximations carry their domain of validity** — `einasto.py`'s docstring states the
  `n` and `R/h` ranges where each branch holds, and there is an `order_for_tol` method.
  That is rule 5 done better than the exemplar package.
- Modern packaging: `src/` layout, `pyproject.toml`, CI, readthedocs.

---

## P0 — correctness traps

### 0.1 `model="Einasto"` is unreachable and the error message misleads

`LensingProfile.__init__` does `self.model = model.upper()`, so `"Einasto"` becomes
`"EINASTO"`. `_validate_inputs` then tests `self.model not in ["NFW", "Einasto"]`, which
is true, and raises

```
Model 'EINASTO' not supported. Available: NFW, Einasto
```

— an error that lists the model it just rejected. Even if the case matched,
`_setup_halo_profile` raises `NotImplementedError` for anything but NFW, so the validator
and the factory disagree about what is supported.

**Fix.** Normalise once, compare against a single module-level tuple, and let the factory
be the only authority:

```python
SUPPORTED_MODELS = ("nfw", "einasto")
self.model = model.lower()
```

and wire `EinastoProfile` into `_setup_halo_profile`, or drop `"Einasto"` from the
supported list until it is wired. Do not leave a third state.

### 0.2 Two density conventions, one of them exported and dead

`config.RHOCRIT = 2.77533742639e11  # Msun h^2/Mpc^3` is re-exported at package top level
(`from .config import DEFAULT_COSMOLOGY, RHOCRIT`) and **is never used anywhere in the
package**. Every density that actually gets used comes from astropy and is h-free:

```python
nfw.py:      rhoc = cosmo.critical_density(0).to_value("Msun/Mpc^3")
bias.py:     self.rhom = self.cosmo.critical_density(0).to_value(...) * self.omega_m
profile.py:  rhocrit0 = self.cosmo.critical_density0.to_value(...)
```

A public constant in $h^2M_\odot/{\rm Mpc}^3$ sitting next to an all-h-free codebase is a
factor-of-$h^2$ waiting for a user. Skill rule 4: never expose a name whose unit
convention differs from the calculation's.

**Fix.** Delete `RHOCRIT` from `config.py` and `__init__.py`. If you want it for
h-convention interop, move it to `utils/constants.py` under the name
`RHO_CRIT_WITH_H` with the instruction in the comment
(`# h^2 Msun/Mpc^3 -- multiply by h**2 before use`), and do not re-export it.

### 0.3 `h` means two different things

In `einasto*.py`, `h` is the Einasto **scale radius** (`rho_0 exp[-(r/h)^(1/n)]`,
`x = R/h`). Everywhere else in a cosmology package `h` is $H_0/100$. Worse, the
constructor signature is `EinastoProfile(alpha, rho_0, r_s, ...)` while the docstrings and
the internal algebra say `h` — so the public name and the documented symbol already
disagree.

**Fix.** Pick `r_s` and use it in the code; keep `h` only inside the `.tex` notes, and add
one line to the module docstring: `NOTE: the notes write h for the scale radius r_s; this
module uses r_s throughout.` The rule is that the *code* may not overload a symbol that
means something else one directory over.

### 0.4 `m200` is ambiguous between 200c and 200m

`nfw.py` computes `r200` from `rhom = rhoc * Om0`, i.e. 200× the **mean** matter density,
and `test_nfw.py` confirms it against `ccl.halos.massdef.MassDef200m`. But the attribute
is `m200`, the docstring says $M_{200}$, and `BiasModel(..., odelta=200)` gives no hint
either. Anyone arriving from an X-ray or SZ background will read $M_{200c}$.

**Fix.** Rename to `m200m` / `r200m` / `c200m`, or — if the rename is too invasive right
now — state the definition in the first line of the class docstring and in the README
Quick Start, the way the exemplar paper does ("Throughout this work, we use
$M_{\rm 200m}$…"). A one-line `NOTE:` is cheap; a silent 200c/200m mismatch is a
30% mass error.

### 0.5 `LensingProfile.__init__` runs CAMB

The constructor builds a `PkGrid`, evaluates $P(k)$, builds a `TwoHaloTerm`, builds a
`BiasModel`, and calls `bias(m200)` — all before the caller has asked for anything. With
`include_2halo=True` as the default, `LensingProfile(z_cluster=0.3, m200=1e14)` fires up a
Boltzmann solver.

This breaks two rules at once: **constructors store, they do not compute** (leanness
budget), and **easy to start an object** — the property you named as central to the style.

**Fix.** Store the inputs; build the 2-halo machinery lazily on first use, or accept
pre-built collaborators:

```python
def __init__(self, z_cluster, m200, cosmology, concentration=4.0,
             halo_profile=None, two_halo=None, bias=None, z_source=1.0):
    self.halo_profile = halo_profile     # built by the caller, or lazily below
    self._two_halo = two_halo
```

The exemplar packages take `co=`, `su=`, `sr=` as already-constructed objects for exactly
this reason: the driver decides what is expensive, not the library.

### 0.6 `check_structure.py` is stale and machine-specific

Hardcodes `base_dir = "/Users/esteves/Documents/Projetos/CLensPy"`, contains the typo
`"sec/clenspy/utils"`, and checks for files that do not exist (`test_lensing.py`,
`test_utils.py`, `docs/tutorials/tutorial1.ipynb`, `setup.cfg`, `demo_profile_fit.ipynb`).
It cannot pass on any machine including yours.

**Fix.** Delete it. Replace with `tests/test_protocols.py` (below), which checks something
real and runs in CI.

---

## P1 — structure

### 1.1 Target tree

```
src/clenspy/
  protocols.py          NEW  structural contracts, imported by nothing at runtime
  utils/
    constants.py        NEW  from config.py, units in trailing comments
    special.py          NEW  from einasto.py: expn_fast, expint_*, Catalan, asymptotics
    integrate.py             (unchanged)
    interpolate.py           (unchanged)
  cosmology/
    distances.py        RENAME from cosmology/utils.py, minus sigma_critical
    pkgrid.py                (unchanged)
    bias.py             MOVE  from halo/  -- structure formation, not a halo profile
    concentration.py    MOVE  from halo/  -- currently a 6-line stub, fill or delete
  survey/               NEW
    survey.py           NEW   p(z), sigma_gamma, n_src, zs range  -- the Survey protocol
  selection/            NEW
    miscentering.py     MOVE  from lensing/
    boost.py            MOVE  from lensing/
    scaling_relation.py NEW   when you need mass-observable
  kernels/              NEW
    sigma_crit.py       MOVE  sigma_critical out of cosmology/utils.py
  halo/
    nfw.py                   (unchanged)
    einasto.py          SPLIT profile class only, ~350 lines
    einasto_series.py   NEW   the _pk_* branch evaluators from einasto.py
    einasto_lown.py          (unchanged -- this is the production series backend)
    twohalo.py               (unchanged)
  lensing/
    profile.py               (slimmed per 0.5)
validation/             NEW   pyccl / cluster_toolkit / published-value comparisons
tests/                        does-it-run + protocol conformance only
```

### 1.2 `einasto.py` is 1263 lines and is three modules

Thirty-odd module-level private functions before the class even starts: Gauss–Laguerre
nodes, Kummer confluent functions, a Mellin–Barnes contour, Filon quadrature, asymptotic
expansions of $E_\nu$, Catalan numbers, six `_pk_*` branch evaluators. The budget is
~300 lines and one physical concept per module.

**Split three ways.** `utils/special.py` gets the general-purpose special functions
(`expn_fast`, `expint_asymptotic`, `_expint_*`, `_catalan_over_4k`, `_asymptotic_polys`)
— they are not Einasto-specific and other profiles will want them. `halo/einasto_series.py`
gets the `_pk_*` branch machinery. `halo/einasto.py` keeps `EinastoProfile` and the
branch-selection logic. Nothing about the physics changes; the file becomes readable.

### 1.3 Four Einasto implementations, two in limbo

`einasto_v2.py` and `einasto_v3.py` subclass `EinastoProfile`, are self-labelled
*"benchmark / research implementation"*, are **not** exported from `halo/__init__.py`, and
are **not** referenced by any test. They are neither shipped nor archived — the worst of
both, because a reader cannot tell whether they are live.

The skill's rule (`references/review.md`, "Recording negative results") says keep them —
they document roads taken and are the companion code to `einasto_proj_density_v2.tex` and
`_v3.tex`. But move them to `benchmarks/`, add one line at the top of each stating which
note it implements and why the v1 path is production, and add a line to
`docs/einasto_series_investigation.md` pointing at them. `einasto_lown.py` stays in the
package — it is the production backend for general `n`, and `einasto.py` imports it.

### 1.4 The missing layers

You have `cosmology`, `utils`, `halo`, `lensing`. The spine wants `survey` and
`selection` too, and you have the contents for one of them already.

**`selection/`** — `miscentering.py` and `boost.py` are in `lensing/` but neither is a
lensing observable. Both describe how the *observed sample* deviates from the theory
prediction: an offset distribution between the assumed and true centre, and dilution of
the source sample by cluster members. Apply the placement rule from the skill —
*which of these would a referee ask you to vary while holding the others fixed?* Both are
nuisance models that get marginalised over. They belong in `selection/`.

**`survey/`** — currently a scalar `z_source: float = 1.0` buried as a `LensingProfile`
constructor default. That is the single biggest structural gap between "profile calculator"
and "analysis package". Everything about the observation is absent: $p(z)$,
$\sigma_\gamma$, $n_{\rm src}$, footprint. Adding the `Survey` protocol is what unlocks
$\langle\Sigma_{\rm crit}^{-1}\rangle$ over a real source distribution, shape-noise
covariance, and anything survey-facing.

**`kernels/`** — small for now: `sigma_critical` moves out of `cosmology/utils.py`, where
it does not belong (it is a lensing kernel quantity, not a background quantity). This
folder is where a Limber projection and bin-averaged Bessel transforms go if you ever add
a covariance module.

### 1.5 Two "utils" namespaces

`utils/` (package) and `cosmology/utils.py` (module). The package is fine — it holds
named modules. The module is a dumping ground holding `sigma_critical` (a kernel),
`comoving_to_theta`/`theta_to_comoving` (geometry), `critical_density`/`hubble_parameter`
(thin astropy wrappers). Rename to `cosmology/distances.py`, move `sigma_critical` to
`kernels/`, and consider dropping the two one-line astropy passthroughs — a wrapper that
adds nothing is a name a reader has to learn for no reason.

---

## Errata: the Survey and projection integrands

Section 1.4 sketches `survey/` as "p(z), sigma_gamma, n_src, footprint" and
`kernels/` as "a Limber projection". Both are wrong for these observables.
The specification is the production pipeline:

- `y3_cluster_cpp/src/modules/average_sigma_crit_inv/average_sigma_crit_inv.py`
- `y3_cluster_cpp/src/pipelines/des_y3/shear_projection/python/0d/shear_prj_gl.py`
- `y3_cluster_cpp/src/pipelines/des_y3/number_counts/python/0d/numcounts_explicit_gl.py`

**Scope of that authority: the numerics, not the structure.** Those files are
CosmoSIS modules — datablock reads, `setup`/`execute`/`cleanup`, hardcoded
output sections, wall grids. None of that shape comes across. What comes
across is the *integrand*: which factors appear, which cancel, which kernel
is used where, and the quadrature that makes it converge. Transcribe the
equations; leave the plumbing behind. The structure is decided by the skill —
one physical concept per module, one equation per method, everything a method
needs in its signature.

### E.1 The Survey layer is $\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)$, not a source p(z)

The quantity the observable actually needs is the *source-averaged inverse*
critical surface density, tabulated against **lens** redshift:

$$
\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)
= h_0 \int \! dz_s \; p(z_s + \Delta z)\,
  \frac{4\pi G}{c^2}\,
  \frac{D_A(z_l)\, D_A(z_l, z_s)}{D_A(z_s)},
\qquad
\gamma_t = \Delta\Sigma \cdot \langle\Sigma_{\rm crit}^{-1}\rangle .
$$

Four things a naive `Survey` gets wrong:

1. **Average the inverse, never invert the average.**
   $\langle\Sigma_{\rm crit}^{-1}\rangle \neq 1/\langle\Sigma_{\rm crit}\rangle$,
   and the difference *is* the source weighting.
2. **Clamp the integrand at zero**, `np.maximum(0, ...)`. Sources in front of
   the lens contribute nothing; they must not contribute negatively.
3. **The angular diameter distance is the flat subtraction form**
   $D_A(z_l,z_s) = D_A(z_s) - \frac{1+z_l}{1+z_s} D_A(z_l)$,
   *not* $D_A(z_s) - D_A(z_l)$.
4. **$\Delta z$ is a photo-z bias nuisance** shifting the source p(z),
   `p(z_s + delta_z)` — it is marginalised over, so it belongs in the
   signature, not in a config.

The clean protocol seam: setting $\langle\Sigma_{\rm crit}^{-1}\rangle \equiv 1$
makes every downstream consumer emit $\Delta\Sigma$ instead of $\gamma_t$.
The production module exposes exactly this as a `unity` switch. `Survey`
should be one callable `sci(z_l)`, with $\Omega(z)$ a *separate* concern
(see E.2).

### E.2 $\Omega(z)$ belongs to counts, and cancels in shear

$$
N_{ij} = \int\! dz \int\! d\ln M \int\! d\lambda_{\rm tr}\;
  n(M,z)\, \frac{dV}{d\Omega\, dz}\, \boldsymbol{\Omega(z)}\,
  K_j(z)\, S_i(\lambda_{\rm tr}, z)\, P_{\rm HOD}(\lambda_{\rm tr} \mid M, z)
$$

but the shear projection carries **no** $\Omega(z)$ — it cancels in the
surface density, and the exact C++ core hard-excludes it. Folding the survey
footprint into a lensing weight is a silent normalisation error. Any shared
weight builder must therefore take $\Omega(z)$ as an explicit per-observable
argument, never as an ambient survey property applied to both.

### E.3 The projection is an exact angular integral, not Limber

$$
\Delta\Sigma_{\rm prj}(R) = \int\! d\theta\; 2\pi \sin\theta
\Big[ \textstyle\sum_M w_{\rm rnd}(M)\, \Delta\Sigma_{\rm mis}(R, \theta D_A \mid M)
 + b_{\rm sel}(\theta) \sum_M w_{\rm cl}(\theta, M)\, \Delta\Sigma_{\rm mis}(R, \theta D_A \mid M) \Big]
$$

with the exact per-slice redshift weights

$$
\begin{aligned}
w_{\rm rnd}(M) &= \int\! dz\; {\rm common}(z)\, n(M,z) \\
w_{\rm cl}(\theta, M) &= \int\! dz\; {\rm common}(z)\,
   \xi_{\rm NL}\big(|d\chi|(z,\theta), z_{\rm ob}\big)\, n(M,z)\, b(M,z)\,
   \mathbb{1}[\theta > \theta_{\rm excl}(z)] \\
{\rm common}(z) &= \frac{dV}{d\Omega\, dz}(z)\; w_{pz}(z; z_{\rm ob})\; w_z^{\rm GL}
\end{aligned}
$$

Traps:

- **The measure is $2\pi\sin\theta\, d\theta$** (solid angle on the sphere),
  not the flat-sky $2\pi\theta\, d\theta$. There is no Limber approximation
  and no Bessel transform anywhere in this observable.
- **$|d\chi|$ and $\theta_{\rm excl}(z)$ are law-of-cosines**, not
  $|\chi_z - \chi_o|$:
  $d\chi^2 = \chi_z^2 + \chi_o^2 - 2\chi_z\chi_o\cos\theta$.
- **The photo-z weight is parabolic**, $w_{pz} = 1 - u^2$ for $|u| < 1$ else 0,
  with $u = (z - z_{\rm ob})/\sigma_z(z)$ — *not* Gaussian. Number counts use
  a Gaussian $K_j$ instead. Two observables, two different photo-z kernels;
  they must not share one implementation.
- **$b_{\rm sel}(\theta)$ multiplies only the correlated (`cl`) channel**,
  never the random (`rnd`) channel.
- **Keep `rnd` and `cl` stored separately** and sum at the end (rule 6). The
  production module writes `{vals, rnd, cl}` for exactly this reason.

### E.4 Consequence for the plan

`survey/survey.py` (step 11) is not "add p(z) and sigma_gamma". It is:
`kernels/sigma_crit.py` holding $\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)$
per E.1, and a `survey/` that owns $\Omega(z)$ and the source p(z) separately,
so that a counts consumer and a shear consumer cannot accidentally pick up
each other's factors. Write the notation table (step 10) *before* either.

### E.5 What actually moved into `selection/` (step 6, done)

Step 6 as written was "move `lensing/miscentering.py` and `lensing/boost.py`
into `selection/`". Followed literally that inverts the dependency arrow:
`MiscenteringProfile` subclasses `LensingProfile`, so putting it in
`selection/` makes the systematics layer import the probe layer.

The layer boundary that does hold is **a systematic is defined relative to a
profile**, so `selection/` sits above `halo/` and below `lensing/`:

```
cosmology/ → utils/ → halo/ → selection/ → kernels/ → lensing/ → covariance/
```

which gives a different and better split of the same three files:

| was | is | why |
|---|---|---|
| `halo/miscentering_table.py` | `selection/miscentering.py` | the runtime correction — a systematic, not a halo property |
| `halo/miscentering_kernel.py` | `selection/miscentering_kernel.py` | the offline generator for that table |
| `lensing/boost.py` | `selection/boost.py` | $\mathcal B(R)$ corrects the source sample |
| `lensing/miscentering.py` | *unchanged* | `MiscenteringProfile` is a `LensingProfile`; it reads `selection/` |

So the move also fixes an existing misplacement: the miscentering table was
in `halo/`, one layer too low, and `halo/__init__.py` was exporting it.

`clenspy.lensing.boost` keeps a `DeprecationWarning` alias for one release.
The two miscentering modules get **no** alias: they were added on this same
refactor branch and have never been released, so there is nothing to keep
compatible. Adding a shim for them would be plumbing for no reader.

---

## P2 — contracts and discipline

### 2.1 Protocols

Copy `assets/protocols.py` from the skill to `src/clenspy/protocols.py`, trim to the
contracts you actually have (`Cosmology`, `Profile`, and — once written — `Survey`,
`Selection`), and add:

```python
# tests/test_protocols.py
def test_profiles_conform():
    for cls in [NfwProfile(m200=1e14, c200=4.0),
                EinastoProfile(alpha=0.2, rho_0=1e15, r_s=0.3)]:
        assert isinstance(cls, Profile)
```

This is what `check_structure.py` was reaching for, done in ten lines against something
that matters. `NfwProfile` and `EinastoProfile` should expose the same `density`, `sigma`,
`deltasigma`, `fourier` surface — they nearly do; the test will find where they do not.

### 2.2 A notation table

You have the derivations but not the dictionary. Add `docs/notation.md`: symbol, meaning,
code name, units, module — one row per quantity, harvested from the `.tex` notes. It
becomes the review checklist, and it is what a new contributor (or agent) reads first.
Start with the rows that are already ambiguous: $M_{200m}$ vs $M_{200c}$, $h$ (scale
radius) vs $h$ (Hubble), $\rho_m$ comoving vs physical, $k$ physical vs $h$-scaled.

### 2.3 Unit `NOTE:` in every class docstring

`bias.py` already does this well ("All quantities are in absolute (physical) units
throughout — mass in Msun…"). Propagate it. Every class that touches a density, a
distance, or a wavenumber states its convention in the first three lines. `nfw.py`,
`twohalo.py`, `profile.py`, and all the Einasto modules currently do not.

### 2.4 Split `tests/` and `validation/`

`test_nfw.py` contains a `pyccl` comparison; `test_twohalo.py` contains a
cluster_toolkit-validated reference pattern. Those are validations wearing a test costume
— they are skipped when the optional dependency is missing, which means CI green does not
mean "agrees with pyccl". Move them to `validation/`, keep a fast smoke test in `tests/`,
and let `validation/` produce the figures that already live in `docs/_static/img/`
(`einasto_deltasigma_validation.png` et al.) rather than `docs/make_einasto_figures.py`
doing it from the docs directory.

### 2.5 Docstrings are carrying a changelog

`LensingProfile`'s class `Notes` and `fourier_profile`'s `Notes` narrate bugs that were
found and fixed — *"Previously called a nonexistent `self.two_halo_profile.pk(...)` …
Fixed to use `p_kz`"*, *"`deltasigma`'s 2-halo term previously multiplied by a hardcoded
1e12 …"*. That is git history in the API reference; a user reading `help(LensingProfile)`
does not need it, and it will rot.

Keep the *physics* — which normalisation is correct and why `rho_m` rather than `1e12`,
which is a genuine units caveat worth a `NOTE:`. Move the archaeology to `CHANGELOG.md`.

### 2.6 Decorators that hide inputs

`default_rvals_z` substitutes `self.reval`/`self.zvec` when arguments are `None`. That is
the opposite of rule 3 — everything a method needs should be visible in its signature, and
this makes the effective inputs invisible at the call site. `time_method` mutates
`self.timings` as a side effect. `scalar_array_output` is fine and worth keeping.

Prefer explicit defaults on the method, or an explicit `self.default_grid()` call in the
body. In a package whose selling point is followable numerics, a decorator that silently
supplies the radial grid is the wrong kind of clever.

### 2.7 `config.py` as a singleton

`DEFAULT_COSMOLOGY = FlatLambdaCDM(H0=70, Om0=0.3)` is a shared module-level instance used
as a default argument in `NfwProfile`, `LensingProfile`, and others. It works, but it means
the cosmology is ambient rather than passed, and a mutation anywhere is global.
`PI = 3.14159265359` next to `numpy` should go regardless.

Split it: physical constants and conversions to `utils/constants.py` (units in trailing
comments, per the skill); the fiducial cosmology to a function
`fiducial_cosmology() -> FlatLambdaCDM` so each caller gets its own instance and the
default is visible as a call rather than a shared object.

---

## Suggested PR sequence

Each step is independently reviewable and leaves the package working.

1. **P0 fixes** — `model` normalisation, delete `RHOCRIT`, delete `check_structure.py`,
   `NOTE:` lines for $M_{200m}$ and the Einasto `h`/`r_s` clash. Small diff, removes traps.
2. **`utils/constants.py` + `fiducial_cosmology()`**, retire `config.py`.
3. **Split `einasto.py`** into `utils/special.py` + `halo/einasto_series.py` + the class.
   Pure moves; the existing 431-line `test_einasto.py` is your safety net.
4. ~~Move `einasto_v2/v3` to `benchmarks/`~~ — **superseded**: they were
   deleted instead. The `.tex` notes remain the record of those roads, and
   the code is in git history.
5. **`protocols.py` + `tests/test_protocols.py`**; fix whatever divergence it exposes
   between `NfwProfile` and `EinastoProfile`.
6. **Create `selection/`** — done, but not as written: see errata E.5 for
   what moved and why the literal reading inverted the dependency arrow.
7. **Create `kernels/`** — done. `sigma_critical` moved to
   `kernels/sigma_crit.py` and `cosmology/utils.py` became
   `cosmology/distances.py`. `cosmology.sigma_critical` keeps a lazy
   `DeprecationWarning` alias (lazy so `cosmology/` gains no dependency on
   `kernels/`, which sits below it). The unused `critical_density` and
   `hubble_parameter` wrappers were deleted rather than moved -- one-line
   passthroughs to `astropy` with no caller, the same dead-export case as
   `RHOCRIT` in P0 §0.2.
8. **Lazy `LensingProfile`** — done. `Pkvec`, `two_halo_profile`,
   `bias_model`, `bias`, `halo_profile` and `sigma_crit` are
   `functools.cached_property`; construction is ~5 ms and does not even
   import `camb`. `halo_profile=`, `two_halo=`, `bias=` and `k_grid=` are
   constructor arguments, so a driver can supply collaborators or reuse one
   P(k). `_validate_inputs` stays eager. `tests/test_lensing_profile.py`
   asserts the laziness structurally, so it cannot regress silently.
9. **`validation/`** — done. `validate_nfw_pyccl.py` and
   `validate_twohalo_chain.py`, plus `analytic_nfw.py` as an independent
   closed-form reference and `docs/validation.md` as the results page. Two
   things came out of the move: the pyccl tolerances were 5e-3 against a
   measured 1e-10, and the two-halo comparison is now the **per-stage**
   chain bench from `y3_cluster_cpp/validations/second_halo_term/
   10_chain_residuals.py` (CLensPy issue #4), which localises a
   disagreement to one transform instead of reporting one number at the end
   of the chain. `tests/test_twohalo.py` had only that comparison in it, so
   it was rewritten as dependency-free invariants.
10. **`docs/notation.md`**, then the unit `NOTE:` sweep — done. Every
    class in the package now declares its unit convention, in its own
    docstring or its module's, and `tests/test_docstrings.py` asserts it so
    a new class cannot be silent. The sweep found four substantive things,
    not just missing prose: `TwoHaloTerm`'s docstring told callers to
    multiply by $\rho_m(z)$ when the convention is the comoving
    $\Omega_{m,0}\rho_{c,0}$ (the $E^2(z)$ trap of P0 §0.3, +34% at
    z=0.25); its example called a `buildAll` that does not exist; its
    docstring was not raw, so `\rho` was a literal carriage return that
    silently broke sphinx section parsing; and `boost.py` used `l`/`z` for
    bin indices where `RichnessBin` already establishes `i_lam`/`i_z`.
11. **`survey/survey.py`** — the new capability, once the spine is in place.

---

## What not to change

- **The `.tex` notes.** They are the specification and they are the reason this package can
  be brought into line at all. Keep writing them.
- **The Einasto branch-selection logic and `order_for_tol`.** Documented convergence
  control is rule 5 done properly; do not simplify it away during the file split.
- **The `n`-range and `R/h`-range validity statements.** Same reason.
- **`einasto_lown.py`'s comment explaining *why* the legacy Catalan evaluation was
  abandoned** (30–200% relative error for $n=3.3$–10, absolute rather than relative
  truncation error). That is a recorded negative result and it is exactly right.
- **Type hints.** The exemplar package has none; you have them and they are good. The
  skill's "no type hints" observation was a description of one author's habit, not a rule —
  ignore it. Keep the hints, keep `Protocol` structural.

---

## Addendum: the Survey, Selection and Kernel layers, and covariance

> Added 2026-08-27, after reading the three role models:
> `cluster-lensing-cov/` (the exemplar package),
> `y3_cluster_cpp/src/pipelines/systematics/` and
> `.../shared/`. Supersedes step 11 and extends the sequence.

The observables these layers exist to serve are the two redMaPPer
quantities: $N_{ij}$ and $\Delta\Sigma_{ij}$, binned in observed richness
$\lambda^{\rm ob}$ and observed redshift $z^{\rm ob}$. **The covariance of
both is computed here too**, which makes the skill's `Estimator` contract
live rather than hypothetical.

### A.1 The interface is already specified by a consumer

`cluster-lensing-cov` is downstream of CLensPy and already imports from it.
`clens/covariance/limber.py` is a *thin adapter* over
`clenspy.lensing.limber.LimberProjector`, and `clens/covariance/inputs.py`
declares the contract as frozen dataclasses. That fixes the API rather than
leaving it to taste:

| Contract | Fields |
|---|---|
| `CosmologyInputs` | `chi(z)`, `pk_lin(k,z)`, `rho_mean0`, `growth(z)`, `sigma_R0(R)` |
| `SourceInputs` | `sigma_gamma`, `n_src_arcmin2`, `q_sigma(z_l,z_h)`, `mean_sigma_crit(z_h)`, `f_src_behind(z_h)`, `zs_max` |
| `LensSample` | bin edges, `counts`, `bias`, `bN`, `volume`, `sigma_w`, optional `pk_hh` / `pk_hm` / `intrinsic_cov` |
| `SurveyGeometry` | `f_sky`, `area_sr` |

So the **Kernel layer's public surface is three callables** —
`q_sigma`, `mean_sigma_crit`, `f_src_behind` — plus $\Sigma_{\rm crit}$
itself. Nothing else is required of it.

Modules `cluster-lensing-cov` imports that this branch does **not** have:

```
clenspy.lensing.limber.LimberProjector
clenspy.utils.fftlog_cov          (j2_bin_averaged)
clenspy.halo.mass_function        (SigmaGrid: .rho_m0, __call__(M, z))
clenspy.clusters.BinHaloModelSpectra
clenspy.IntrinsicProfileVariance
docs/covariance_fftlog_math.md
```

All of these exist on the `codex/clusters` branch. **They are to be
rebuilt, not merged.** That branch does not follow the skill — modules
carry several physical concepts at once, integrands are hard to follow
against the papers, and `clusters/` bundles survey, selection, kernels,
photo-z, MOR and observables into one flat package. Use it only to locate
which equations were already transcribed; take the structure from the skill
and the numerics from the papers and the C++ core.

`src/clenspy/clusters/` has been removed from this branch. (It survived as a
bare `__pycache__`, which made `import clenspy.clusters` succeed against
stale bytecode — worth knowing, since that silently shadows a real module.)

### A.2 `survey/` — the footprint and the source population

Per-survey presets, one per data release: **DES Y1, DES Y3, SDSS**.

Two separable things, and they must not be conflated (errata E.2):

**Footprint $\Omega(z)$** [rad²], the effective solid angle, which falls at
high $z$ as the red-sequence contrast degrades. Polynomial fits transcribed
from `y3_cluster_cpp/src/models/omega_z_{des,sdss}.hh`:

- SDSS: one degree-11 fit in $(z - 0.2)$, ~3.13 sr plateau over $z\in[0.1,0.4]$.
- DES: three-piece fit, breaks at $z = 0.504$ and $0.7$, ~0.45 sr plateau.
  (The DES header names its coefficient arrays `SDSS_fit` — a copy-paste
  artifact in the C++. Do not replicate the name.)

$\Omega(z)$ enters $N_{ij}$ and **cancels** in $\Delta\Sigma$. A shared
weight builder must therefore take it as an explicit per-observable
argument, never as an ambient survey property.

**Source population** — the `Survey` contract proper: `pz_src(z_s)`
normalised, `sigma_gamma`, `n_src_arcmin`, `zs_min`, `zs_max`. Three
$p(z_s)$ forms, following `clens/util/survey.py`:

$$p(z_s) \propto z_s^{m}\exp\!\big[-(z_s/z_\star)^{\beta}\big]$$

(Rozo et al. 2011 eq. 14, the default), a top-hat, or an interpolated
tabulated $dn/dz$. Two departures from the exemplar: it subclasses `dict`
and prints on construction — do neither.

### A.3 `kernels/` — $\Sigma_{\rm crit}$, its inverse, and Limber

- `sigma_crit(z_l, z_s)` — already in `cosmology/utils.py`, moves here (step 7).
- $\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)$ per errata E.1: average the
  **inverse**, clamp the integrand at zero so foreground sources
  contribute nothing, use the flat subtraction form for $D_A(z_l,z_s)$, and
  carry the photo-z bias shift $\Delta z$ in the signature.
- `mean_sigma_crit(z_h)`, `f_src_behind(z_h)`, `q_sigma(z_l, z_h)` — the
  three the covariance consumes. The exemplar's `LensingKernel` computes
  all three by trapezoid over $p(z_s)$; that is the specification.
- **Two photo-z kernels, deliberately distinct** (errata E.3): a Gaussian
  CDF difference $\mathcal S_j$ for counts, and the parabolic weight
  $w_{pz} = 1-u^2$, $u = (z-z^{\rm ob})/\sigma_z(z)$, for the projection.
  $\sigma_z(z)$ is a 120-node table (`shared/z_kernel.py`).
- Limber projection, written **once**, with windows passed in — not one
  `calc_C_ell_*` per spectrum.

### A.4 `selection/` — $\mathcal S_{ij}(\ln M, z)$

The closed-form richness selection of
`RichnessSelection/docs/richness_selection_function.tex`:

$$\mathcal S_i = (1-f^{\rm prj})\,\Phi\big|_{\Delta\lambda_i} + f^{\rm prj} F_{\rm EMG}\big|_{\Delta\lambda_i}$$

with the EMG CDF in closed form, so only the $\lambda^{\rm tr}$ integral is
numerical (Gauss–Legendre on the per-$(M,z)$ bracket). Two mass–observable
relations: log-normal (Costanzi et al. 2021) and the HOD shifted-Poisson.
`miscentering.py` (already table-backed) and `boost.py` move here from
`lensing/` per step 6, joined by the $\bar b_{\rm sel}$ sigmoid.

### A.5 `covariance/` — the `Estimator` layer

Keep the decomposition and sum at the end (rule 6); the exemplar's shape:

- **Counts**: Poisson (diagonal) + sample variance (same-$z$ blocks fully
  correlated across richness, window r.m.s. $\sigma_W = \sigma_R(R_{\rm eff})D(z)$).
- **$\Delta\Sigma$**, Wu et al. 2019 eq. 22: `cosmic_shear`, `shape_noise`,
  `cross`, each stored separately with switches to isolate them.
- FFTLog engine for the $\ell$ integral, plus the trapz-over-$\ln\ell$
  version kept as a **test-only reference** so equivalence tests isolate the
  integration method.

### A.6 Revised sequence

Steps 1–5 are done. Steps 6–10 stand. Then:

11. `survey/` — **done**, as one module plus a config directory.
    `survey/survey.py` holds $\Omega(z)$; the analysis choices live in
    `clenspy/configs/<survey>.json`. `Survey` is now in `protocols.py`, and
    it is the source population — *not*
    $\langle\Sigma_{\rm crit}^{-1}\rangle$, *not* $\Omega(z)$.

    **The split is code vs. configuration.** A footprint fit is a
    transcribed polynomial, so it is code — one mistyped digit is a silent
    normalisation error. Bin edges, $\sigma_z$, $\sigma_\gamma$,
    $n_{\rm src}$ and the $p(z_s)$ parameters are analysis choices, so
    they are JSON with a `_provenance` string on every group. Changing an
    analysis edits a config, never this module.

    Five things settled by reading the sources rather than assuming:

    - **`y3_cluster::OMEGA_Z_DES` is DES Y1, not Y3.** 1494 deg² at
      $z=0.2$ against Y1's published 1437; Y3 is 4143. The y3 repo's own
      python transcription names it `omega_z_des_y1`.
    - **No $z$-dependent DES Y3 fit exists anywhere**, so `omega_des_y3`
      is flat at the 4143 deg² **gold** footprint — the area of the data.
      The downstream `configs/des_y3.json` carries 5000, but that file is a
      forecast ("DES Y1 counts scaled by 5000/1437"), so the two describe
      different things and must not be reconciled. Precedent for a flat
      placeholder is the C++'s own `OMEGA_Z_Y3XSPT`.
    - **$\sigma_z = 0.01$, not 0.03.** The `SIGMA_Z = 0.03` in the y3
      production config is the **3σ window**, not the scatter. Using it as
      $\sigma_z$ widens every photo-z kernel threefold.
    - **The DES Y1 fit has three pathologies**, all in the C++ and now
      pinned by tests: discontinuous by $-0.37\%$ at $z=0.504$ and
      $-30.6\%$ at $z=0.700$, and it crosses zero at $z=0.9378$. Clamped
      at zero here — a deliberate divergence above $z=0.94$. Domain of
      validity is the analysis range $[0.20, 0.65]$.
    - **The DES header names its arrays `SDSS_fit`** — a copy-paste
      artifact the C++ flags itself. Renamed on transcription.

    Coverage is uneven on purpose. SDSS has an $\Omega(z)$ fit, because
    one exists upstream, but no config — so `load_config("sdss")` raises.
    A bin edge is an integration limit, and a guessed one is a wrong number
    that looks right. Same principle as the missing Einasto miscentering
    table.

12. `kernels/` — **the lensing kernel is done.** `LensingKernel` holds
    $\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)$ and the three callables the
    covariance consumes, and reproduces the frozen Stage-A reference
    (`validation/frozen_inputs/kernels.npz`) to **2.8e-7** on all four.
    **The two photo-z kernels are done too** (`kernels/photoz.py`), and the
    key point is that they are *different functions*: a Gaussian CDF
    difference keyed on the bin **edges** for the counts, and a
    compactly-supported parabola about one $z^{\rm ob}$ for the projection.
    The counts kernel is a probability with support everywhere and
    $\int \mathcal S_j\,dz^{\rm tr} = $ the bin width; the projection
    weight is unnormalised with $\int w_{pz}\,dz = \frac43 n_\sigma\sigma_z$
    and is exactly zero beyond $n_\sigma\sigma_z$. Substituting one for the
    other puts weight along the whole line of sight.

    **And this is where the 0.03 finally resolved.** $\sigma_z = 0.01$ is
    the scatter; the 0.03 in the y3 configs is $3\sigma_z$, the **window
    half-width of the parabolic $b_{\rm sel}$ weight**. So the projection
    kernel takes `n_sigma=3` and the counts kernel takes $\sigma_z$ itself.
    Confusing them widens one kernel threefold or narrows the other
    threefold, in opposite directions and silently — which is why the two
    signatures take their width differently and a test asserts the factor.

    **Limber is done** (`kernels/limber.py`), transcribed from **Wu et al.
    (2019), MNRAS 490, 2606** — reading the paper's `.tex` rather than the
    old docstring's paraphrase, which changed the design:

    - The paper writes all three spectra as **one** formula,
      $C_\ell^{AB} = \int d\chi (F_A/\chi)(F_B/\chi) P_{AB}$, so that is
      how it is written: `limber` once, with `F_h` / `F_Sigma` passed in.
      The three near-duplicate slab loops it replaces differed only in
      which windows they carried — the skill's rule and the paper's own
      structure turned out to be the same instruction.
    - Names follow the paper: `C_ell_hh`, `C_ell_SS`, `C_ell_hS`, `F_h`,
      `F_Sigma`, `shot_noise_h`, `shape_noise_Sigma`. The old
      `c_ell_sigma` / `c_ell_h` / `c_ell_h_sigma` remain as aliases for one
      release, and `clenspy.lensing.limber` is a deprecation shim, because
      `cluster-lensing-cov` imports that path.
    - **The paper vindicates the $q_\Sigma$ range.** Eq. `F_Sigma`
      integrates sources from $\chi_{\rm lss}$ to $\infty$ — keyed on the
      line-of-sight structure, not the halo. So
      $F_\Sigma = \bar\rho\,q_\Sigma(z_{\rm lss}, z_h)$ *exactly*, and the
      range choice that looked arbitrary in the code (and that I first got
      wrong) is Wu et al.'s definition.
    - One deviation from the paper is recorded rather than hidden: the
      shape-noise term divides by $n_{\rm s} f_{\rm src}$ where the paper
      has $n_{\rm s}$. That is what the frozen reference was built with;
      passing `f_src_behind=lambda z: 1.0` recovers the paper exactly.

    The test that matters needs no reference data: with linear bias
    $P_{\rm hm}^2 = P_{\rm hh}P_{\rm mm}$, so on a common $\chi$ range
    $(C_\ell^{\rm h\Sigma})^2 = C_\ell^{\rm hh}C_\ell^{\Sigma\Sigma}$
    **exactly** — verified to 1e-12. The windows cancel, so it checks that
    the three spectra carry the *right* windows, and it fails on any stray
    power of $\bar\rho$, $V$ or $\chi$.

    Step 12 is complete.

    Five things this settled, none of which were guessable:

    - **The exemplar's $\Sigma_{\rm crit}$ is comoving**, `clenspy`'s
      existing `sigma_critical` is physical, and the two differ by exactly
      $(1+z_l)^2$ — verified, not assumed. The y3 module
      `average_sigma_crit_inv.py` uses a *third* convention (physical,
      times $h_0$). Comoving is right here, because `clenspy`'s
      $\Delta\Sigma$ is comoving and $\gamma_t$ must be dimensionless.
    - **The residual 0.138% against the reference is their rounded $c$.**
      They use $3\times10^5$ km/s; $(299792.458/3\times10^5)^2 =
      0.99861687$ accounts for all of it. Applying the correction to the
      wrong *power* doubles the residual instead of removing it, which is
      how the direction got caught.
    - **$\langle\Sigma_{\rm crit}\rangle$ and $q_\Sigma$ are
      logarithmically divergent.** They exist only relative to the minimum
      lens-source separation, 0.01, which is a *definition* and a floor —
      and, through the endpoint's trapezoid weight, to the node count:
      100 → 800 nodes moves the answer 5%. Both are arguments now, and a
      test asserts the non-convergence so nobody "fixes" it.
      $\langle\Sigma_{\rm crit}^{-1}\rangle$ is convergent, which is the
      deeper reason E.1 says to average the inverse.
    - **$q_\Sigma$ is signed and must not be clamped.** Its source range
      is keyed on $z_l$, not $\max(z_l, z_h)$, so it includes sources in
      front of the halo where $\Sigma_{\rm crit}(z_h, z_s) < 0$; the frozen
      reference runs $-2.29$ to $+3.91$. It also puts the $z_s = z_h$ pole
      inside the integral, which is where its $\pm 4$ spikes come from.
      Faithfully reproduced; documented as an artifact of the definition.
    - **No Cauchy–Schwarz bound relates the two averages**, tempting as it
      is: they are not taken against the same measure. The product crosses
      1 across the bin range (1.09, 1.10, 0.89, 0.55). My first test
      asserted the bound and was wrong.
13. `cosmology/{growth,sigma,mass_function}.py` — **done**, as three
    modules rather than one, and **ported** from `y3_cluster_cpp`'s in-repo
    replacement for CosmoSIS's `MfTinker`
    (`mf_tinker_cpp/python/tinker_core.py`) rather than reimplemented: that
    evaluator already agrees with arbitrary-precision mpmath to 4.4e-16.

    Six conventions came across with it, each stated where it is used:
    the $k \le 20/R$ upper limit **depends on $R$** and is
    algorithm-defining; **FFTLog cannot express that**, so the fast path
    computes the untruncated quantity and its docstring says to validate it
    against `truncate=False` only (the gap it cannot capture is 7.0e-3 in
    $dn/d\ln M$ over $0\le z\le2$); $d\sigma^2/d\ln R$ is taken under the
    integral sign and the moving boundary contributes a Leibniz term;
    $\delta_c = 1.6865$ in the mass function but **1.686** in the bias and
    in $M_\star$, and both are kept separately; the mass axis carries no
    $\Omega_m$, so it is in $\Omega_m h^{-1}M_\odot$; and these three
    modules are h-scaled, which every identifier says.

    **13b.** `BiasModel` now shares that one `SigmaGrid` instead of
    computing $\sigma(M)$ by a second FFTLog. Four defects went with the
    old inline version: it splined the *variance* linearly in $\log_{10}R$
    then square-rooted it, it let `np.interp` silently **clamp** outside
    the FFTLog range, it rebuilt the FFTLog every call, and `bias()` cached
    $\nu$ from the first mass and returned it for every later one.

14. `selection/` — **done**. `richness_kernel.py` (the Costanzi EMG kernel
    and its closed-form bin integral), `scaling_relation.py` (log-normal
    and HOD shifted-Poisson), `selection_function.py`
    ($\mathcal S_{ij} = S_i\,\mathcal S_j$).

    The EMG CDF is **not** evaluated in the form the derivation produces.
    $\Phi(z) - e^{A}\Phi(z-\tau\sigma)$ is a product of a factor that
    overflows and one that underflows: for $\tau\sigma \gtrsim 40$ it is
    `inf * 0 = nan` where the true value is an ordinary number in $[0,1]$.
    The `erfcx` rewrite absorbs the exponent exactly, via
    $A - u^2 = -z^2/2$.

    Two things found and recorded rather than smoothed over: the continuous
    shifted-Poisson density's first moment sits **exactly 1.0** above
    $\lambda^{\rm cen}+\langle\lambda^{\rm sat}\rangle$ at
    $\sigma_{\rm intr}=0$ (an artifact of interpolating a discrete law, so
    `HodMor.mean` is documented as being for bracket placement only), and
    the log-normal Poisson floor is
    $(\langle\lambda\rangle-1)/\langle\lambda\rangle^2$, **not**
    $1/\langle\lambda\rangle$ — the central galaxy carries no shot noise,
    and the term goes negative below $\langle\lambda\rangle = 1$.

15. `observables/` — **done**. One weight, two contractions: contracting
    $W_{ij}$ against 1 gives $\langle N_{ij}\rangle$ and against
    $\Delta\Sigma(R\mid M,z)$ gives the stack. `StackedDeltaSigma` owns no
    weight of its own, so it *cannot* disagree with the counts about the
    sample — checked by identities that an implementation with two weights
    fails: unity stacks to exactly 1, $M$ stacks to exactly
    $\langle M\rangle_{ij}$, $\Delta\Sigma = M/R$ stacks to
    $\langle M\rangle_{ij}/R$.

    E.2 is now executable: $\Omega(z)$ scales the counts by exactly 2 when
    doubled and cancels to 1e-14 in every average.

16. `covariance/` — **done**. Counts (Poisson + sample variance, the latter
    **rank one** per redshift slice and exactly zero between slices) and
    $\Delta\Sigma$ (Wu et al. `eq:cov_DS`, bracket expanded into its five
    terms with a `terms` selector).

    Two results worth carrying forward. First, `j2_bin`'s published closed
    form is **unusable** for $\ell\theta \lesssim 1$ — its bracket cancels
    to nine orders, leaving about four correct digits, measured at 4.8e-4
    against quadrature — so a Taylor series branch covers $x < 1$ and
    neither form alone suffices. Second, **the FFTLog engine this step
    asked for buys nothing**: the integral is a *bilinear* form,
    $\hat J_2(kr_p)\hat J_2(kr_p')$ under one $k$ integral, not a Hankel
    transform of a single function, so it does not factorise. Written as
    $A^{\rm T}{\rm diag}(wP)A$ it costs $O(n_k n_r^2)$ and is already
    negligible; the log-$k$ quadrature *is* the engine, with a
    convergence-tested grid.

Validate each against `cluster-lensing-cov`'s frozen Stage-A snapshots,
which exist precisely to pin refactor equivalence.
