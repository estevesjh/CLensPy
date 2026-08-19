# Einasto Profile: Pitfalls & Cases Needing Special Solutions

**Design decision:** the current code enforces **n > 3/2** (α < 2/3). This
sidesteps the impractical-convergence regime and covers all physically relevant
dark-matter haloes (n ~ 4–6 for spirals, n ~ 2–3 for clusters).

---

## 1. Catalan Series — Algebraic Convergence Only

The projected quantities (Σ, M_2D, ΔΣ) converge as `K^{-1/2}` (terms decay as
`k^{-3/2}`). This means:

| n (shape) | K for 1% rel err | K for 0.1% | Notes |
|-----------|------------------|------------|-------|
| 0.5       | impractical      | —          | REJECTED (n ≤ 3/2) |
| 1.0       | ~5000            | >50000     | REJECTED (n ≤ 3/2) |
| 2.0       | ~1500            | ~15000     | Clusters; slow convergence |
| 4.0       | ~160             | ~15000     | Spiral haloes; workable |
| 5.0       | ~40              | ~4000      | Spiral haloes; easy |
| 6.0       | ~10              | ~900       | Fat cores; easy |

**Takeaway:** for n ~ 4–6 (spirals), order=15 gives ~1% at scales near r_s.
Use `order_for_tol(tol)` for automatic selection.

---

## 2. E_ν(x) Dispatch — Four Branches (No External Dependency for Integer n)

The generalized exponential integral is evaluated by:

| Branch | Condition | Method | Speed |
|--------|-----------|--------|-------|
| `scipy.special.expn` | integer ν ≥ 1 | Exact | vectorized, ~μs |
| `_expint_gamma` | ν < 1 (a = 1−ν > 0) | `z^{p-1} gammaincc(a,z) Γ(a)` | vectorized, ~μs |
| DLMF 8.20 asymptotic | ν ≥ threshold(rtol) | Polynomial recurrence | vectorized, ~μs |
| `mpmath.expint` | non-integer ν ∈ (1, threshold) | Arbitrary precision | **scalar, ~0.25 ms/pt** |

For **integer n** (4, 5, 6): all ν_k are integers → branches 1+2 cover
everything. **mpmath never called. ~3 μs/point.**

For **non-integer n** (4.5, 5.5): ν₁ = n+1 (e.g. 5.5) is non-integer and
falls in (1, threshold) → mpmath. **~0.25 ms/point.** Acceptable for typical
use (100 radii = 25 ms). The `rtol` parameter controls the asymptotic
threshold: lower rtol → higher threshold → more mpmath calls.

### Threshold formula

```
nu_min = rtol^{-1/nterms}
```

| rtol | nu_min (5 terms) | mpmath range for n=4.5 |
|------|------------------|------------------------|
| 1e-4 | 6.3 | ν₁=5.5 → mpmath |
| 1e-6 | 15.8 (default) | ν₁=5.5 → mpmath |
| 1e-8 | 39.8 | ν₁=5.5 → mpmath |

### Pitfall: Dunster (1996) OCR corruption

The mathpix transcription of Dunster's recursion (2.11)/(2.13) for the
asymptotic coefficients U_s(q) is **wrong at s ≥ 2** — gives stuck ~1e-3 error
vs the verified DLMF. Do NOT implement from that source; use DLMF 8.20.4
directly (validated to machine precision against mpmath).

### Why not scipy `gammaincc` for all ν?

`E_p(z) = z^{p-1} Γ(1-p, z)` requires `a = 1-p > 0`. For ν > 1 (all k ≥ 1),
`a = 1-ν < 0` → `scipy.special.gammaincc` returns **nan**. Scipy does not
implement the upper incomplete gamma for negative first argument. Only mpmath
(or the DLMF asymptotic) handles this.

A recurrence-based approach (`s E_{s+1} = e^{-x} - x E_s`, starting from ν<1)
achieves machine precision without mpmath but is not yet implemented. It would
eliminate the mpmath dependency entirely.

---

## 3. Power Spectrum P(k) — Two-Branch Structure

### 3a. n > 1 (spiral haloes): large-k series in `(kh)^{-1/n}`

- Converges for ALL k (mathematically), but partial sums blow up at kt ≪ 1.
- **Numerically usable only for kt = kh ≳ 1** (small scales).
- Leading term gives the power-law tail `P ∝ k^{-(3+1/n)}`.
- Code emits `RuntimeWarning` for kt < 1.

### 3b. n < 1: small-k series in `(kh)^2`

- Converges for all k; fastest at small k.
- Useless for n > 1 (diverges).
- Not reachable with n > 3/2 constraint.

### 3c. n = 1: elementary closed form `P = ρ₀h³/[2π(1+k²h²)²]`

- Not reachable with n > 3/2 constraint.

### 3d. Pitfall: No single series covers P(k) at kt < 1 for n > 1

For n > 1 at kt < 1, the only options are:
1. Direct numerical quadrature of the Hankel integral (robust, slow).
2. mcfit FFTLog from ρ(r) → P(k) numerically.
3. Retana-Montenegro Fox-H representation.

**Current code warns** and returns unreliable values below kt ~ 1.

---

## 4. Retana-Montenegro et al. (2012) Fox H / Meijer G

Express Σ(R) as a Fox H-function or Meijer-G function via the Mellin integral
transform. **Advantages:**
- Numerically stable for **all** n via standard Meijer-G evaluation (mpmath).
- Single formula, no truncation order to choose.
- Exact representation (no truncation error).

**Disadvantages:**
- mpmath Meijer-G is slow (per-element, not vectorized).
- Not in scipy (no array-level speed).

**Status:** NOT implemented. Useful as:
- Ground-truth validator for any (n, R).
- Backup for P(k) at kt < 1 via Mellin-space evaluation.

See `2012A&A...540A..70R` in library.

---

## 5. Recommendations for Future Work

| Problem | Solution | Priority |
|---------|----------|----------|
| Eliminate mpmath for non-integer n | Implement E_ν recurrence (shown to work at machine precision) | Medium |
| P(k) at kt < 1, n > 1 | mcfit FFTLog (ρ → P numerically) | High |
| Ground truth / validator | Retana-Montenegro Meijer G | Low |
| n ≤ 3/2 support (if ever needed) | Retana-Montenegro or closed forms (Gaussian, K₁) | Not planned |
