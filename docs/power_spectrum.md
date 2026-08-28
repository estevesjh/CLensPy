# Power Spectrum

Every structure-formation calculation downstream — the mass function, the
halo bias, the two-halo term — needs a linear matter power spectrum
$P(k,z)$. CLensPy does not compute one from scratch: it wraps a Boltzmann
solver (CAMB, or PyCCL) and caches the result, so a calculation that needs
$P(k)$ at many $(k,z)$ pays the solver once.

## Getting $P(k)$ from CAMB

`PkGrid` takes the cosmology you already built ({doc}`cosmology`) and
returns an interpolator over a cached $(k,z)$ grid:

```python
from clenspy.cosmology import PkGrid

pk_grid = PkGrid(cosmo=cosmo, nonlinear=False)  # linear P(k,z), CAMB-backed
k = pk_grid.k                                   # the grid's own k array
Pk = pk_grid(k, z=0.0)
```

The $(k,z)$ grid itself is log-spaced in $k$, linear in $z$, and sized by
four keywords with package defaults — override any of them explicitly if
your calculation needs a wider or finer grid:

| Keyword | Default | Meaning |
|---|---|---|
| `k_range` | `(1e-4, 10.0)` ${\rm Mpc}^{-1}$ | grid endpoints, log-spaced |
| `nk` | `512` | number of $k$ points |
| `z_range` | `(0.0, 1.0)` | grid endpoints, linear-spaced |
| `nz` | `100` | number of $z$ points |

```python
pk_grid = PkGrid(
    cosmo=cosmo, nonlinear=False,
    k_range=(1e-5, 50.0), nk=1024,   # e.g. extend to smaller scales
    z_range=(0.0, 3.0), nz=61,       # and a higher-redshift tracer
)
```

The result is cached to disk (`$CLENSPY_DATA/pk_cache`, or `clenspy/data/`
by default) keyed on the cosmology and grid parameters, so re-running the
same setup skips the Boltzmann solve entirely.

```{figure} _static/img/power_spectrum.png
:alt: Linear matter power spectrum P(k) from CAMB at z = 0 and z = 1
:width: 75%
:align: center

$P(k)$ falls off at $z=1$ relative to $z=0$ by the growth factor squared,
$D^2(z)$; the turnover near $k\sim10^{-2}\,{\rm Mpc}^{-1}$ marks
matter-radiation equality.
```

```{admonition} Units are h-free absolute here
:class: note
`PkGrid` follows the rest of the package: wavenumbers in ${\rm Mpc}^{-1}$,
$P(k)$ in ${\rm Mpc}^3$ — **not** the h/Mpc, $(\rm Mpc/h)^3$ convention
common in the structure-formation literature. Both backends (CAMB, PyCCL)
work internally in h-scaled units and are converted at the boundary.
```

## The mass variance $\sigma^2(R)$

The matter density contrast $\delta(\mathbf{x}) = [\rho(\mathbf{x}) -
\bar\rho]/\bar\rho$ has power spectrum $P(k)$,

$$
\langle\delta(\mathbf{k})\delta^*(\mathbf{k}')\rangle
  = (2\pi)^3 P(k)\,\delta_D(\mathbf{k}-\mathbf{k}').
$$

Its variance smoothed with a top-hat window $W$ of radius $R$ is

$$
\sigma^2(R) = \left\langle \delta_R^2(\mathbf{x}) \right\rangle
            = \frac{1}{2\pi^2}\int dk\,k^{2}\,P(k)\,W^{2}(kR).
$$

Collapse occurs where $\delta_R$ exceeds $\delta_c$: $\nu =
\delta_c/\sigma(R)$ is the peak height {doc}`mass_function` and
{doc}`halo_bias` are fit to. `clenspy.cosmology.sigma` evaluates the
integral above in the h-scaled convention of the Fortran reference it
reproduces: $k$ in h/Mpc, $P(k)$ in $(\rm Mpc/h)^3$. Crossing from
`PkGrid`'s h-free output into that convention is one visible conversion,
done once:

$$
k_h = \frac{k}{h}, \qquad P_{h^3}(k_h) = h^3\,P(k).
$$

```{figure} _static/img/sigma_r.png
:alt: Top-hat variance sigma(R) vs R at z = 0 and z = 1, with sigma_8 marked
:width: 75%
:align: center

$\sigma(R)$ falls with $R$ — larger spheres average over more scales and
fluctuate less — and with $z$ by the growth factor $D(z)$ (not $D^2$: this
is $\sigma$, not $\sigma^2$). The marked point is $\sigma_8 \equiv
\sigma(R{=}8\,h^{-1}{\rm Mpc})$ at $z=0$.
```

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"power-spectrum\"]"
:end-before: "%% [markdown]"
:language: python
```

```
P(k) from CAMB: k in [1.0e-04, 1.0e+01] 1/Mpc, P in [2.37e-01, 7.32e+04] Mpc^3
sigma(R=  1.0 Mpc/h) = 2.3870
sigma(R=  8.0 Mpc/h) = 0.8001
sigma(R= 20.0 Mpc/h) = 0.3917
```

$\sigma(R{=}8\,h^{-1}{\rm Mpc}) \approx 0.80$ is $\sigma_8$: this cosmology's
amplitude was set to it (`PkGrid` reads `sigma8` off the cosmology, defaulting
to 0.8), so recovering it here from the integral is a check that the CAMB
call and the conversion above are correct.

See also: {doc}`mass_function` and {doc}`halo_bias`, the two consumers that
build their own $\sigma^2(R)$ evaluator internally, either straight from
`cosmo` (repeating the `PkGrid` step above for you) or from a custom
`(k_h, pk_h3)`; {doc}`api/index` for the full `clenspy.cosmology` reference.
