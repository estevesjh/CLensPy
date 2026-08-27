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

```bash
python validation/analytic_nfw.py                    # check the reference first
python validation/validate_nfw_pyccl.py     --plot
python validation/validate_twohalo_chain.py  --plot
python validation/validate_lensing_kernel.py --plot
python validation/validate_miscentering_table.py
```

`analytic_nfw.py` is the reference the chain bench compares against, and is
a **deliberate second copy** of formulae `clenspy.halo.nfw` also carries — a
reference that imports the code under test validates nothing. Run it first;
its own quadrature self-check is what makes it usable as a truth.

Figures are written to `docs/_static/validation/` so `docs/validation.md`
can show them. That page carries the results, the residual tables, and what
each comparison does and does not test.
