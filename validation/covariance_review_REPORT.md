# Covariance review: CLensPy against cluster-lensing-cov `cddbb2a`

This report covers the Gaussian covariance of the excess surface density
$\Delta\Sigma$ in `clenspy`, compared with Hao-Yi Wu's `cluster-lensing-cov`
pinned at `cddbb2a` (Wu et al. 2019, arXiv:1907.06611). The comparison is a
validation ladder (V0-V8). Each rung tests one layer and takes its inputs
from her snapshot `validation/data/clc_cddbb2a_reference.npz`. The snapshot
was not regenerated.

## What was fixed

| commit | change |
|---|---|
| `30f883f` | (earlier) the $\ell\,d\ell = \chi_h^2 k\,dk$ measure and $\Omega_{\rm ann} = A_{\rm ann}/\chi_h^2$ in the closed-form `shot_shape` |
| `3ac9756` | **shape noise**: the conditional mean $\langle\Sigma_{\rm crit}\rangle/f_{\rm src}$ with the cut $z_h + 0.1$ on both integrals (`MIN_LENS_SOURCE_SEPARATION_NOISE`), as in her `cddbb2a`. Before the fix it was 12-40% high. `LensingKernel.f_src_behind` gains `min_separation`/`n_nodes`, with defaults unchanged |
| `04c5ecd` | **optional Wu et al. $\ell$ range**: `DeltaSigmaGaussianCovariance(..., ell_range="wu2019")` reproduces her per-pair truncated range. The default stays `"converged"`. A 12-bin covariance takes 0.1 s (hers takes about 6.6 s). Under `"wu2019"`, `shot_shape` defaults to her quadrature, so her 0.9% truncation is reproduced. Pass `exact_shot_shape=True` to override |
| `2802d2e` | **$q_\Sigma$ foreground pole**: diagnosed, default unchanged (see the open decision below). Adds `diagnose_qsigma_pole.py`, slab-convergence INFO rows in V2, and docstring notes |
| `7d7516b` | hygiene: the V1 0/0 row is masked, the V0 growth threshold is now $10^{-8}$ (with the reasoning), and the README and `docs/validation.md` have a ladder section |

## Final validation results (real output)

```
V1  -> V1 kernels: 0 unexpected FAIL, 0 known-bug FAIL*, 11 PASS        (exit 0)
      worst PASS: q_Sigma(z_h=0.425) 8.5e-12; <Sc> shape-noise convention 1.0e-14
V2  C_hh   thin0/1/2/wide max(band) 1.05e-10 / 5.09e-11 / 3.17e-11 / 1.05e-10  PASS (tol 1e-8)
    C_SS   5.74e-01 / 9.40e-01 / 4.00e-01 / 5.74e-01                           INFO
    C_hS   1.75e-05 / 9.66e-03 / 6.78e-02 / 1.75e-05                           INFO
    slab convergence (signed q_Sigma), max|C(dz)/C(0.01)-1|, dz=0.05 / 0.02:
      thin0 C_SS 4.43e-02 / 2.66e+00   C_hS 3.86e-03 / 2.08e-02             INFO
      thin1 C_SS 7.58e+05 / 1.53e-01   C_hS 4.96e-03 / 4.19e-02             INFO
      thin2 C_SS 7.16e-01 / 1.47e+00   C_hS 1.12e-02 / 6.44e-02             INFO   (exit 0)
V3  shot noise ours/hers - 1 = 0.00e+00 (4 cases)                          PASS
    shipped shape_noise_Sigma vs hers, raw  -2.76e-03 (4 cases, tol 1.2e-2) PASS
    shipped shape_noise_Sigma vs hers, exact (c^4, p-norm removed)
      -3.3e-16 / 2.9e-15 / 1.7e-14 / -3.3e-16 (tol 1e-6)                    PASS
    V3 PASS                                                                 (exit 0)
V4  20/20 terms PASS, tol 1e-3 of sqrt(C_ii C_jj); worst 1.21e-05 (lss_lss thin2);
    shot_shape 1.4e-12 .. 3.8e-12; "all terms within tolerance"             (exit 0)
V8  A closed form 4.4e-16 (tol 1e-9) PASS; quadrature 2.0e-4 (tol 5e-3) PASS
    B MC/code 0.969..1.028, max 1.4 sigma PASS;  "V8 PASS"                  (exit 0)
old validate_lensing_kernel.py: "all comparisons pass" (corrected 2.8e-07)  (exit 0)
V0  make_clc_reference.py: not re-run (by instruction)
```

`pytest tests/test_covariance.py tests/test_fftlog_cov.py tests/test_limber.py
tests/test_lensing_kernel.py --no-cov`: **147 passed** in 6.1 s.

## Remaining known differences from hers

1. **Constants.** Her $c = 3\times10^5$ km/s gives $-1.38\times10^{-3}$ on
   $\Sigma_{\rm crit}$ and $-2.76\times10^{-3}$ on the shape noise. Her
   `arange` normalisation of $p(z)$ gives $-2.8\times10^{-7}$. Both are
   exact and removed analytically in V1 and V3.
2. **Her $\ell$ range** (only with `ell_range="converged"`, which is our
   default). At bin 0, her `lss_lss` and `cross` are 47-50% below the
   converged values: converged/hers $- 1$ is +89% to +98% for `lss_lss` and
   +91% for `cross`. At bin 11 the difference is under 1.2%. Her
   `shot_shape` is 0.9% low in every bin, and `lss_shape`/`shot_lss` are
   2.3-2.4% low at bin 0. 47% of the `lss_lss` integral at the smallest
   $r_p$ lies below her $1/\theta_{\max}$, and none of it is at $\ell < 10$.
   With `ell_range="wu2019"` we reproduce her to 1.3e-5.
3. **$C^{\Sigma\Sigma}$, 40-94% off (V2), and $C^{h\Sigma}$, up to 6.8% off.**
   This is the $q_\Sigma$ pole below and is not an implementation
   disagreement. On her nodes, $q_\Sigma$ agrees with hers to 1e-11 (V1).
   She stores it on a 100-point $z$ grid and interpolates linearly. For
   foreground slabs, that interpolation lands on values dominated by the
   pole: at $z_h = 0.425$, $z_{\rm lss} = 0.153$ ours is $-0.03$ and hers is
   $+3.39$.

## Open decision: the definition of $q_\Sigma$ for foreground slabs

**Evidence (step i).** For $z_{\rm lss} < z_h$, the source integral in
$q_\Sigma$ runs over $z_s$ from $z_{\rm lss} + 0.01$, so it crosses
$z_s = z_h$. There, $\Sigma_{\rm crit}(z_s, z_h) \propto 1/(\chi_s - \chi_h)$
has a simple pole. At $z_h = 0.425$, $z_{\rm lss} = 0.153$ the integrand is
$-926$ at $z_s = 0.4249$ and $+927$ at $0.4251$. The trapezoid value then
depends on where the nodes fall relative to the pole: it is $-0.03$, 0.24,
0.44, 1.24, 1.07, 0.88 and 0.68 for 100, 101, 200, 400, 800, 1600 and 3200
nodes. It does not converge, so the pole hypothesis is confirmed.

**What the paper and her code say (step ii).** Wu et al. (2019), sec. 6 and
app. C, write $F_\Sigma = \bar\rho\int_{\chi_{\rm lss}}^\infty d\chi_s\,
p\,\Sigma_{\rm crit}(z_s,z_h)/\Sigma_{\rm crit}(z_s,z_{\rm lss})$. Read
literally, the lower limit includes foreground sources. The paper derives it
by "integrating $\int d\chi_s\, p\,\Sigma_{\rm crit}(z_s, z_h)$ twice" over
the $\gamma_t$ covariance. That only makes sense for $z_s > z_h$, because a
source in front of the halo has no $\Sigma_{\rm crit}(z_s, z_h)$. The paper
also writes $\langle\Sigma_{\rm crit}\rangle$ as $\int_0^\infty$, yet her
code restricts it to $z_s > z_h + 0.1$. Her `calc_kernel_Sigma` keeps the
foreground sources and the sign, with `max(zl+0.01, zs_min)`. Neither source
states how foreground sources should be treated. The literal formula is
ill-defined, and the code gives one particular grid-dependent value.

**Candidates (step iii)**, from `validation/diagnose_qsigma_pole.py`.
"Total" is the covariance diagonal at the largest radial bin (11), relative
to the shipped definition A at `DZ_SLAB = 0.1`. It uses our spectra, her
shot noise and our shipped shape noise, with the converged $\ell$ range.

| definition | $C^{\Sigma\Sigma}$ slab change 0.05→0.01 | total, thin0 / thin1 / thin2 (dz 0.01) |
|---|---|---|
| A: signed, sources from $z_{\rm lss}+0.01$ (hers, shipped) | 4.4% / $7.6\times10^5$ / 72% | 0.764 / 1.420 / 2.281 (vs 1 at dz 0.1) |
| B: sources behind the halo, from $\max(z_{\rm lss}, z_h)+0.01$ | 0.26% / 0.26% / 0.23% | 0.791 / 0.908 / 1.133 |
| C: shape-noise sample, from $\max(z_{\rm lss}+0.01, z_h+0.1)$, $/f_{\rm src}(0.1)$ | 0.25% / 0.23% / 0.20% | 0.849 / 0.934 / 1.200 |

$C^{h\Sigma}$ moves by 0.2-0.6% between widths under B and C, compared with
0.4-6.4% under A. In her matrices, the $C^{\Sigma\Sigma}$-carrying terms
(`lss_lss` + `shot_lss`) are 28-90% of the bin-11 diagonal. That is why the
choice matters at the 10-130% level.

**Decision not made.** B and C both converge (under 1% between slab widths
of 0.05 and 0.01, as required), but the paper and the code do not choose
between them, and they differ by 3-7% on the bin-11 diagonal. C is the
version consistent with the cddbb2a source sample: the ΔΣ estimator
averages over sources behind $z_h+0.1$, and the shape noise uses the same
conditional normalisation. B is the minimal "no foreground sources" fix.
Either one departs from her numbers by design. **The default is unchanged
(A).** I recommend asking Hao-Yi Wu which source sample $F_\Sigma$ is meant
to carry, and then implementing that choice as an argument to
`LensingKernel.q_sigma` (for example `sources="behind_halo"`), keeping A
reachable.

## Notes

- `validation/validate_cov_anchors.py` (V8) is still an uncommitted user
  file. I ran it but did not edit it, and the README and docs refer to it.
- `LimberProjector.shape_noise_Sigma` passes `min_separation=0.1` to the
  `mean_sigma_crit` / `f_src_behind` callables when their signature accepts
  it. A plain callable of $z$ alone is used as given, so a frozen table must
  already carry the 0.1 convention.

## Update after the PR #8 review: Gauss–Legendre for the source integrals

Review comment: the source integrals used a trapezoid; use Gauss–Legendre.
Measured against a 2048-node rule, 128 GL nodes reach 2e-12 on
$\langle\Sigma_{\rm crit}\rangle$ (0.01 cut) and about 1e-13 on the rest;
the 100-node trapezoid was off by 2e-4 ($f_{\rm src}$), 5e-4 to 2e-3
($\langle\Sigma_{\rm crit}\rangle$, 0.1 cut) and 4–8%
($\langle\Sigma_{\rm crit}\rangle$, 0.01 cut). `f_src_behind`,
`mean_sigma_crit` and `mean_inverse_sigma_crit` now use `gl_nodes`
(`N_ZS_GL = 128`). `q_sigma` keeps the trapezoid because of the pole.

Consequences for the numbers above: the earlier statement that
$\langle\Sigma_{\rm crit}\rangle$ "does not converge when refined" was the
trapezoid, not the physics. The raw $-2.76\times10^{-3}$ on the shape noise
was her $c$; with Gauss–Legendre the shipped shape noise differs from hers
(constants removed) by $-1.6\times10^{-3}$ to $-4.2\times10^{-3}$, which is
her trapezoid error. V1 and V3 now compare our formulas on her rule
(floating point) and report the converged-vs-hers difference separately.

