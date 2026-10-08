# validation/

Comparisons against published results, other libraries, and analytic limits.

`tests/` asks *does it run*; this directory asks *does it reproduce a number
someone else got*. Nothing here runs in CI: each script pulls in a heavy
optional dependency, and the assertions are about physics agreement rather
than about the code executing.

Every script prints its error norms, exits nonzero on failure, and takes
`--plot` to write the figure that shows the agreement.

| script | reference | needs |
|---|---|---|
| `analytic_nfw.py` | direct quadrature (self-check) | scipy |
| `validate_nfw_pyccl.py` | `pyccl.halos.HaloProfileNFW` | pyccl |
| `validate_twohalo_chain.py` | closed-form NFW, per chain stage | cluster_toolkit, clmm, pyccl |
| `validate_lensing_kernel.py` | `cluster-lensing-cov` frozen Stage-A kernels | `$CLUSTER_LENSING_COV_DIR` |
| `validate_miscentering_table.py` | `cluster_toolkit.miscentering`, y3 tables | cluster_toolkit, `$Y3_CLUSTER_CPP_DIR` |
| `validate_sigma_prj_mock.py` | Costanzi mock halo catalogue (`mock_lob_sigma_catalog.fits`) | astropy, camb, `$SELECTION_BIAS_DIR` |

The **covariance ladder** compares the Gaussian $\Delta\Sigma$ covariance
with `cluster-lensing-cov` pinned at `cddbb2a` (Wu et al. 2019). Each rung
isolates one layer by feeding it the layer below from the reference. Only
V0 imports her code; the others read its snapshot,
`validation/data/clc_cddbb2a_reference.npz`. None of them take `--plot`
except V1.

| rung | script | isolates |
|---|---|---|
| V0 | `make_clc_reference.py` | writes the snapshot from her code at the pin (needs `$CLUSTER_LENSING_COV_DIR`); do not re-run casually |
| V1 | `validate_cov_kernels.py` | lensing kernels $\langle\Sigma_{\rm crit}^{-1}\rangle$, $q_\Sigma$, $\langle\Sigma_{\rm crit}\rangle$, $f_{\rm src}$ on her nodes |
| V2 | `validate_cov_spectra.py` | Limber spectra $C^{hh}$ (hard), $C^{\Sigma\Sigma}$, $C^{h\Sigma}$ (INFO) with her $P(k,z)$, plus slab-width convergence |
| V3 | `validate_cov_noise.py` | halo shot noise and the shape noise on $\Sigma$ (cddbb2a convention) |
| V4 | `validate_cov_gaussian.py` | the five covariance terms with *her* spectra and noises, on her per-pair $\ell$ range (`ell_range="wu2019"`); `--converged` shows her truncation |
| V8 | `validate_cov_anchors.py` | anchors independent of her code: closed-form shot\_shape vs the physical sample variance, a Monte-Carlo stack, where the $\ell$ integral lives |
| — | `diagnose_qsigma_pole.py` | not a rung: the foreground pole in $q_\Sigma$ and three candidate fixes |

Shared plumbing (pin, matched configuration, snapshot loader, verdict
table) is `_clc_reference.py`. Findings are in
`covariance_review_REPORT.md` and `docs/validation.md`.

```bash
python validation/analytic_nfw.py                    # check the reference first
python validation/validate_nfw_pyccl.py     --plot
python validation/validate_twohalo_chain.py  --plot
python validation/validate_lensing_kernel.py --plot
python validation/validate_miscentering_table.py
python validation/validate_cov_kernels.py            # V1 .. V8, after V0
python validation/validate_cov_spectra.py
python validation/validate_cov_noise.py
python validation/validate_cov_gaussian.py --converged
python validation/validate_cov_anchors.py
SELECTION_BIAS_DIR=../SelectionBias python validation/validate_sigma_prj_mock.py --plot
```

`analytic_nfw.py` is the reference the chain bench compares against, and is
a **deliberate second copy** of formulae `clenspy.halo.nfw` also carries — a
reference that imports the code under test validates nothing. Run it first;
its own quadrature self-check is what makes it usable as a truth.

Figures are written to `docs/_static/validation/` so `docs/validation.md`
can show them. That page carries the results, the residual tables, and what
each comparison does and does not test.
