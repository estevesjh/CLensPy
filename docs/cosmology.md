# Cosmology

Cosmology parameters are set by the user upfront, once, as a single object —
every other module in the package (profiles, kernels, the mass function)
takes this object as an argument and computes from it, rather than reading
global parameters of its own. If no cosmology is set up, `clenspy` falls
back to its own default. The object itself is a plain
`astropy.cosmology.FlatLambdaCDM` — not a package-specific type — so it
composes with the rest of the Astropy/SciPy ecosystem. This page also
covers the two background quantities built directly on top of it: distances
and the linear growth factor.

## Defining a cosmology

`fiducial_cosmology()` returns an `astropy.cosmology.FlatLambdaCDM` set to
the package's default parameters — a flat universe with $H_0=70\,{\rm
km\,s^{-1}\,Mpc^{-1}}$, $\Omega_m=0.3$. Call it with no arguments to get
that default, or pass `H0`/`Om0` to set your own:

```python
from clenspy.cosmology import fiducial_cosmology

cosmo = fiducial_cosmology(H0=70.0, Om0=0.3)  # flat LambdaCDM
```

`cosmo` is then the one object every other calculation in the package —
profiles, kernels, the mass function — takes as an argument, so a single
choice of $H_0$, $\Omega_m$ propagates consistently through the whole
chain. Each call to `fiducial_cosmology()` returns an independent object,
so building `cosmo` once and passing it everywhere never risks a later
call silently changing the cosmology out from under an earlier one.

### Using a custom cosmology

`fiducial_cosmology()` only exposes `H0` and `Om0` — the two parameters
`clenspy` itself varies. For anything else (baryon density, neutrino mass,
$T_{\rm CMB}$, or a non-flat/non-$\Lambda$CDM model), build the
`astropy.cosmology` object yourself and pass it in instead of
`fiducial_cosmology()`'s result — every downstream function only requires
the same interface (`comoving_distance`, `H`, etc.), not that it came from
`fiducial_cosmology()` specifically:

```python
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315, Ob0=0.049, Tcmb0=2.7255, m_nu=[0, 0, 0.06])
```

or reuse one of Astropy's own published cosmologies directly:

```python
from astropy.cosmology import Planck18 as cosmo
```

## Distances

The comoving distance $\chi(z)$ and angular diameter distance $D_A(z) =
\chi(z)/(1+z)$ come directly from `cosmo`. CLensPy adds the angle–distance
conversion at fixed lens redshift $z_l$,

$$
\theta = \frac{D_c}{D_A(z_l)},
$$

used throughout the package to move between a projected physical separation
$R$ and an angular separation on the sky.

## Linear growth factor $D(z)$

For a flat cosmology with any dark-energy equation of state, the growing
mode has the closed-form quadrature (Child et al. 2018, Sec. 4)

$$
D^{+}(a) = \frac{5\Omega_{m,0}}{2}\,\frac{H(a)}{H_0}
           \int_0^{a} \frac{da'}{\left[a'H(a')/H_0\right]^{3}},
\qquad
D(a) = \frac{D^{+}(a)}{D^{+}(a=1)},
$$

normalised so $D(z{=}0)=1$ — the convention in which $P_{\rm lin}(k,z) =
D^2(z)\,P_{\rm lin}(k,0)$, and the one {doc}`mass_function` uses to evolve
$\sigma(M,z) = D(z)\,\sigma(M,0)$ instead of recomputing a power spectrum at
every redshift.

```{figure} _static/img/cosmology_distances_growth.png
:alt: Comoving and angular diameter distance vs redshift, and the linear growth factor D(z) vs redshift, for flat LambdaCDM
:width: 100%
:align: center

$\chi(z)$ and $D_A(z)$ (left) turn over past $z\sim1.5$ because a fixed
comoving separation subtends a smaller angle as $D_A$ itself starts to
fall; $D(z)$ (right) falls monotonically from 1.
```

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"cosmology\"]"
:end-before: "%% [markdown]"
:language: python
```

```
FlatLambdaCDM(H0=70.0 km / (Mpc s), Om0=0.3, Tcmb0=0.0 K, Neff=3.04, m_nu=None, Ob0=0.0)
theta [arcmin] = [ 0.3373975   3.37397499 33.73974985]
round trip     = [ 0.1  1.  10. ]
D(z) =          [1.         0.834242   0.61180575 0.4214457 ]
```

See also: {doc}`api/index` for the full `clenspy.cosmology` reference,
{doc}`notation` for the symbol table.
