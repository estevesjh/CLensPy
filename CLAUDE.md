# CLensPy

Cluster weak-lensing toolkit: halo profiles, halo bias, two-halo term,
projection/miscentering lensing, selection effects. Dependency chain drives
everything: `cosmology → P(k) → ξ/σ² → HMF/Bias → TwoHaloTerm/SigmaPrj`.
Read code physics-first, like reading the paper it implements — see
`docs/index.md`'s Theory toctree (one page per physical effect: prose,
governing equation, runnable snippet) and `docs/development.md` for
running tests/lint/docs build.

Consuming CLensPy from another project? Read `docs/llm_quickstart.md`
(task recipes + the three anti-patterns: never `import camb`, never
`np.trapz` the FFTLog/Abel transforms, never loop over array inputs) and
the README's "Using CLensPy from your own project" section.

`docs/_archive/` holds past session planning notes (refactor plans,
cleanup plans). They're historical, not maintained — don't treat them as
current state; check the code and the Theory pages instead.

## House rules

1. **No doc essays in modules.** No NOTEs/warnings/derivations at the top
   of `.py` files — that content belongs in `docs/*.md`. A short equation
   reference inline on a method is fine.
2. **Reuse the house decorators** in `utils/decorators.py`
   (`@default_rvals_z`, `@default_mvals_z`, `@scalar_array_output`,
   `@time_method`) for grid reshaping and call signatures — don't
   reimplement that plumbing per class.
3. **Big kwargs → a config dataclass is correct**, not something to trim
   (e.g. `SigmaPrjConfig`-style bags). Don't flatten these back to loose
   kwargs.
4. **Integration numerics live outside the physics classes** — module-level
   functions (e.g. `utils/integrate.py`) called by `TwoHaloTerm`/`SigmaPrj`,
   not methods on those classes.
5. **Keep the `build()` pattern**: `SigmaPrj`/`HMF`/`Bias` construct their
   own missing upstream dependencies when not injected.
6. **Share one m/z/radial grid** across objects wherever possible, rather
   than each object owning its own.
7. **No pairwise evaluation, ever — everything is grids.** Interpolators
   and evaluators take vector `x` + vector `z` and return the outer grid
   `(nx, nz)`; scalar+scalar → float; vector+scalar → `(nx,)`. Never add a
   `pairs()`/pointwise method. If a caller seems to need paired `(r, z)`
   points, restructure to loop the small axis (z) with grid calls
   `f(r_vec, z_scalar)` instead.
8. **Never derive ΔΣ(R) from a tabulated Σ(R)** — no `cumulative_trapezoid`,
   no reconstructing ΔΣ from a Σ̄(<R) quadrature. ΔΣ is its own integral;
   for population integrals swap the kernel inside the same operator
   (e.g. the signed `ds_hat` table), don't post-process Σ.

## Docs math style

Every symbol is defined in the document itself. Big formulas in display
(`$$...$$`) blocks, never inline. A section is always prose → equation →
prose: a lead-in sentence before the display block, follow-up explaining
the symbols/consequences after — never a heading followed immediately by
a `$$` block with no lead-in.
