# Boost Factor

Cluster members and correlated structure scattered into the background
source catalogue are not lensed by the cluster — they dilute the measured
shear. The boost factor $B(R)$ is the multiplicative correction that
removes that dilution, applied to $\Delta\Sigma$ before it is compared to
a model.

```{figure} _static/img/boost_factor.png
:alt: NFW boost factor B(R) for three amplitudes
:width: 75%
:align: center

$B(R)$ at three amplitudes $B_0$, fixed NFW scale radius. Dilution is
largest in the cluster core, where correlated members dominate the source
catalogue, and $B(R)\to1$ at large $R$.
```

## The NFW dilution model

$B(R) \equiv \Sigma_{\rm crit}/\Sigma_{\rm crit}^{\rm eff}$ compares the
true critical surface density to the effective one diluted by
non-background galaxies. McClintock et al. (2019, Eq. 27) model it with
the same radial shape as an NFW projection:

$$
B(R) = 1 + B_0\,\frac{1-f(x)}{x^2-1}, \qquad x = \frac{R}{r_s}, \qquad
f(x) = \begin{cases}
\dfrac{\operatorname{arctanh}\sqrt{1-x^2}}{\sqrt{1-x^2}}, & x<1\\[2mm]
1, & x=1\\[2mm]
\dfrac{\arctan\sqrt{x^2-1}}{\sqrt{x^2-1}}, & x>1
\end{cases}
$$

with $B_0$ a free amplitude fit per richness/redshift bin and $r_s$ the
same NFW scale radius as {doc}`density_profiles`. $B(R)$ is dimensionless
and $\ge1$ by construction.

```{note}
`boost_factor_nfw` is the *model* — the loaders in the same module,
`load_boost_factor_data`/`load_boost_factor_collection`, read a
*measurement*: the DES Y1 unblinded boost factor files, not distributed
with `clenspy`. The two are independent uses of this module, not a
model-then-fit pipeline living in one function.
```

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"boost-factor\"]"
:end-before: "%% [markdown]"
:language: python
```

```
B0=0.05  B(R) = [1.05491131 1.01995534 1.003816   1.00057211 1.00005803]
B0=0.10  B(R) = [1.10982262 1.03991067 1.00763199 1.00114423 1.00011605]
B0=0.20  B(R) = [1.21964523 1.07982134 1.01526398 1.00228846 1.00023211]
```

See also: {doc}`api/index` for the full `clenspy.selection` reference,
{doc}`notation` for the symbol table, {doc}`miscentering` for the other
selection-driven correction to a stacked profile.
