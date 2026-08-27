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
8. **Lazy `LensingProfile`** — stop running CAMB in `__init__`.
9. **`validation/`** — move the pyccl and cluster_toolkit comparisons out of `tests/`.
10. **`docs/notation.md`**, then the unit `NOTE:` sweep.
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

11. `survey/` — $\Omega(z)$ for DES Y1 / DES Y3 / SDSS, and the `Survey`
    source-population contract. Add `Survey` to `protocols.py` only once it
    exists.
12. `kernels/` — $\Sigma_{\rm crit}$, $\langle\Sigma_{\rm crit}^{-1}\rangle$,
    the three covariance callables, the two photo-z kernels, Limber.
13. `halo/mass_function.py` — `SigmaGrid`, the halo mass function, the
    growth factor. Needed by both counts and covariance.
14. `selection/` — the EMG richness kernel, the two MORs, $\mathcal S_{ij}$.
15. `observables/` — $N_{ij}$ and $\Delta\Sigma_{ij}$ as contractions of the
    weights against an integrand.
16. `covariance/` — counts and $\Delta\Sigma$ blocks, FFTLog engine,
    reference implementation, assembly.

Validate each against `cluster-lensing-cov`'s frozen Stage-A snapshots,
which exist precisely to pin refactor equivalence.
