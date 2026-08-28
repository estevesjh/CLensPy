# The Lensing Kernel

$\Sigma_{\rm crit}$ is not a property of the universe or of the halo — it
is a property of the geometry of one lens and one source. A real survey
never has a single source plane, so what a lensing weight actually needs
is $\Sigma_{\rm crit}$ averaged over the survey's source population,
$p(z_s)$ ({doc}`survey`).

```{figure} _static/img/lensing_kernel.png
:alt: Critical surface density vs source redshift, and DES Y1 source-averaged weights
:width: 95%
:align: center

Left: $\Sigma_{\rm crit}(z_l,z_s)$ diverges as $z_s\to z_l$ and flattens
for distant sources. Right: DES Y1's two source-averaged weights —
$\langle\Sigma_{\rm crit}^{-1}\rangle$ and $1/\langle\Sigma_{\rm
crit}\rangle$ agree only near the peak of the lens-redshift range and
diverge from each other elsewhere, which *is* the source weighting.
```

## One pair, then a population

For a single lens-source pair, using the flat-subtraction angular-diameter
distance (never $D_A(z_s)-D_A(z_l)$, which is wrong by 34% at
$z_l=0.35,z_s=1$):

$$
\Sigma_{\rm crit}(z_l,z_s) = \frac{c^2}{4\pi G}\,
\frac{D_A(z_s)}{D_A(z_l)\,D_A(z_l,z_s)}.
$$

`sigma_critical` takes scalar redshifts only — vectorize over `z_source`
by looping, or use the comoving `sigma_crit_comoving` (`LensingKernel`'s
internal convention), which is array-valued. The population average that
a real lensing weight needs is

$$
\left\langle\Sigma_{\rm crit}^{-1}\right\rangle(z_l) = \int dz_s\,
p(z_s+\Delta z)\,\max\!\big[0,\,\Sigma_{\rm crit}^{-1}(z_l,z_s)\big],
\qquad
\gamma_t = \Delta\Sigma\cdot\left\langle\Sigma_{\rm crit}^{-1}
\right\rangle(z_l),
$$

with $\Delta z$ a photo-z bias nuisance parameter shifting $p(z_s)$, kept
as an explicit argument rather than a stored constant since it is
marginalized over.

```{note}
**Average the inverse, never invert the average.** $\langle\Sigma_{\rm
crit}^{-1}\rangle \ne 1/\langle\Sigma_{\rm crit}\rangle$ — the difference
*is* the source weighting, visible in the figure above. Sources in front
of the lens must contribute exactly zero, not a negative number, to
either average — `LensingKernel` clamps the integrand rather than the
result. `mean_inverse_sigma_crit` is the convergent one of the two:
refining the source-redshift grid moves it by $<10^{-4}$, while
`mean_sigma_crit` is only cutoff-defined — sources just behind the lens
have unbounded $\Sigma_{\rm crit}$, so its value depends on
`MIN_LENS_SOURCE_SEPARATION` (0.01 in redshift) by definition, not by
numerical error.
```

`LensingKernel` also builds the three quantities a covariance consumes —
`q_sigma`, `mean_sigma_crit`, `f_src_behind` — and none of them are
lazily-cached beyond the $\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)$
interpolant `kernel_z` builds on first use.

## The photo-z kernel

The other half of {doc}`selection_function`'s exact factorization is a
Gaussian CDF difference — the probability that a cluster at true redshift
$z^{\rm tr}$ lands in the observed bin $\Delta z_j$:

$$
\mathcal S_j(z^{\rm tr}) = \Phi\!\left(\frac{z_j^{\max}-z^{\rm tr}}
{\sigma_z}\right) - \Phi\!\left(\frac{z_j^{\min}-z^{\rm tr}}{\sigma_z}
\right),
$$

`photoz_counts`, with $\Phi$ the standard normal CDF (written out via
`scipy.special.erf` rather than `scipy.stats.norm.cdf`, ~40x faster per
call in this module's inner loop).

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"lensing-kernel\"]"
:end-before: "%% [markdown]"
:language: python
```

```
z_s=0.60  Sigma_crit=4.333e+15 Msun/Mpc^2
z_s=1.00  Sigma_crit=2.796e+15 Msun/Mpc^2
z_s=1.50  Sigma_crit=2.383e+15 Msun/Mpc^2
z_s=2.00  Sigma_crit=2.222e+15 Msun/Mpc^2

<Sigma_crit^-1>   = [3.61149037e-16 4.32085305e-16 3.81417233e-16 2.75419611e-16]
1/<Sigma_crit>    = [3.31695530e-16 3.92057624e-16 4.27008960e-16 5.01833643e-16]
ratio (!= 1)      = [1.08879682 1.10209642 0.89323005 0.54882652]
f_src_behind(z_l) = [0.96883223 0.87738012 0.72350911 0.53406002]
```

See also: {doc}`api/index` for the full `clenspy.kernels` reference,
{doc}`notation` for the symbol table, {doc}`lensing_profile` for
$\Sigma_{\rm crit}$'s role in turning $\Delta\Sigma$ into a shear.
