# Survey

A lensing weight needs the source population: how sources are distributed
in redshift, how noisy their shapes are, how many there are per unit sky
area. A cluster-counts prediction additionally needs the footprint and the
observed bin grid. `clenspy` keeps these three kinds of thing apart on
purpose — `Survey` is only the first.

```{figure} _static/img/survey.png
:alt: Source redshift distributions for three configs, and two survey footprints
:width: 95%
:align: center

Left: $p(z_s)$ for every shipped config — Buzzard, DES Y1, and DES Y3 sit
on top of each other, since all three use the same Smail shape parameters
$(z_\star,m,\beta)=(0.74,1.68,2.33)$; only $\sigma_\gamma$ and
$n_{\rm src}$ differ between them. Right: the DES Y1 and SDSS effective
footprints $\Omega(z)$, each valid only over the redshift range its
analysis actually bins.
```

## Three kinds of thing, kept apart

**$\Omega(z)$ is code.** Each footprint is a polynomial fit transcribed
coefficient-by-coefficient from `y3_cluster_cpp` — a fit is not a number
to retype, so it lives in `clenspy.survey` and only the *choice* of which
fit to use is configurable. `omega_des_y1` clamps at zero outside
$z\in[0.20,0.65]$ (`DES_Y1_Z_RANGE`), since the fit is discontinuous at
its two internal breaks and crosses zero entirely at $z=0.94$; `omega_sdss`
does the same outside $[0.10,0.33]$, since a degree-11 polynomial diverges
fast beyond the range it was fit on. `omega_des_y3` is flat at the
published 4143 deg² gold footprint — no redshift-dependent Y3 fit exists
yet, so this is a stated approximation, not a transcription.

**Bins and source properties are configuration.** Richness edges, redshift
edges, $\sigma_z$, $\sigma_\gamma$, $n_{\rm src}$, and the $p(z_s)$
parameters are analysis choices, not physics — they live in
`clenspy/configs/<survey>.yaml` and `Survey.from_config`/`survey_bins`
read them. Changing an analysis means editing a config, not this module.

**`Survey` is the source population, and nothing else.** $p(z_s)$,
$\sigma_\gamma$, $n_{\rm src}$, and the redshift support — exactly what a
lensing weight asks for. Build one from a config, or directly from a shape
(`smail`, `top_hat`, `tabulated`):

$$
p(z_s) \propto z_s^{m}\,\exp\!\left[-(z_s/z_\star)^{\beta}\right]
\qquad \text{(Rozo et al. 2011, Eq. 14 — the Smail form)}.
$$

```{note}
`Survey` carries **no** $\Omega(z)$ and **no** bin grid. The footprint
appears in the cluster **counts** and *cancels* in the shear projection —
folding it into a lensing weight would be a silent normalization error —
so it is reached through `survey_area`, a separate function, rather than
read off the survey object.
```

## Example

```{literalinclude} ../examples/getting_started.py
:start-after: "tags=[\"survey\"]"
:end-before: "%% [markdown]"
:language: python
```

```
Survey('DES Y1', sigma_gamma=0.3, n_src_arcmin=6.28, zs=[0, 3])
p(z_s) = [0.65542688 1.15967077 0.06176611]
12 bins = 4 richness x 3 redshift
Omega(z) [deg^2] = [1494.02461986 1502.86763277 1511.32308133  649.07197582]
```

See also: {doc}`api/index` for the full `clenspy.survey` reference,
{doc}`notation` for the symbol table, {doc}`lensing_kernel` for how
$p(z_s)$ and $\Sigma_{\rm crit}$ combine into a lensing efficiency.
