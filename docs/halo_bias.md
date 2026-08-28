# Halo Bias

The linear bias $b(M,z)$ relating halo clustering to the matter clustering
it sits in, from Tinker et al. (2010).

## Peak height and the bias fit

`BiasModel` is fit to the same peak height as {doc}`mass_function`,

$$
\nu(M,z) = \frac{\delta_c}{\sigma(M,z)}, \qquad \delta_c = 1.686,
$$

with $\sigma(M,z) = D(z)\,\sigma(M,0)$ ({doc}`cosmology`) and $\sigma(M,0)$
the same top-hat variance defined in {doc}`power_spectrum`. The bias itself
is (Tinker et al. 2010, Eq. 6)

$$
b(\nu) = 1 - A\,\frac{\nu^{a}}{\nu^{a}+\delta_c^{a}} + B\,\nu^{b} + C\,\nu^{c},
$$

with $(A,a,B,b,C,c)$ fit as functions of the spherical overdensity $\Delta$
(`odelta`) — unlike the Tinker (2008) mass function, these coefficients
carry no separate $z$-dependence: the entire redshift evolution of $b(M,z)$
runs through $\sigma(M,z)$.

Like {doc}`mass_function`, `BiasModel`'s constructor only stores its
collaborators — pass `cosmo=` for the chain (`cosmo` → `PkGrid` → `SigmaGrid`,
lazily, on first use), or `k=`/`P=` to override the `PkGrid` step with a
custom spectrum. No h-scaling boundary this time: `BiasModel` already uses
the package's h-free convention, matching `PkGrid` directly.

```{figure} _static/img/halo_bias.png
:alt: Tinker (2010) linear halo bias b(M) at z = 0, 0.5, 1
:width: 85%
:align: center

$b(M,z)$ at $z=0,0.5,1$ — massive haloes are always more strongly biased,
and a fixed mass is *more* biased at higher $z$, since it is a rarer peak
against a smaller, less-grown $\sigma(M,z)$.
```

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"halo-bias\"]"
:end-before: "%% [markdown]"
:language: python
```

```
nu(M)   = [1.11168598 1.71766385 2.46837069 2.93659406]
b(M)    = [1.07022151 1.88289029 3.51661308 4.92175586]
b(M=1e14, z=0.0) = 1.8829
b(M=1e14, z=0.5) = 2.8981
b(M=1e14, z=1.0) = 4.5033
```

See also: {doc}`api/index` for the full `clenspy.cosmology` reference,
{doc}`notation` for the symbol table.
