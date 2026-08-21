# Miscentering: exact single-offset kernel

This note derives the miscentered lensing observables implemented in
`clenspy.lensing.miscentering`, for a single (delta-function) offset. The
key result is a closed-form reduction of the miscentered aperture mean to
a **single smooth integral over the known centered mean profile** — no
nested cumulative integral, no integrable cusp, no endpoint singularity.

## 1. Definitions

Symbols used throughout:

- $\Sigma(R)$ — centered (azimuthally symmetric) surface density of the
  halo, at projected halo-centric radius $R$.
- $\bar\Sigma({<}R)$ — mean of $\Sigma$ inside the disk of radius $R$
  about the halo center,

$$
\bar\Sigma({<}R) \;=\; \frac{2}{R^{2}} \int_0^{R} \Sigma(R')\, R'\, dR' .
$$

- $\Delta\Sigma(R) = \bar\Sigma({<}R) - \Sigma(R)$ — centered excess
  surface density. For NFW, $\Sigma$ and $\Delta\Sigma$ (hence
  $\bar\Sigma = \Delta\Sigma + \Sigma$) have the Wright & Brainerd (2000)
  closed forms.
- $R_{\rm mis}$ — distance between the true halo center and the assumed
  (observed) center.
- $R$ — projected radius measured from the **assumed** center.
- $u(t)$ — halo-centric distance of a point at radius $R$ from the
  assumed center and azimuth $t$ (law of cosines),

$$
u(t) \;=\; \sqrt{R^{2} + R_{\rm mis}^{2} - 2\,R\,R_{\rm mis}\cos t }.
$$

Subscript "mis" denotes the corresponding quantity measured about the
assumed center for a halo offset by $R_{\rm mis}$.

## 2. Miscentered surface density

$\Sigma_{\rm mis}$ is the azimuthal average of the centered profile over
the circle of radius $R$ about the assumed center (Yang et al. 2006;
Johnston et al. 2007):

$$
\Sigma_{\rm mis}(R \mid R_{\rm mis})
\;=\; \frac{1}{\pi} \int_0^{\pi} \Sigma\!\big(u(t)\big)\, dt .
$$

The integrand is smooth except when $R = R_{\rm mis}$, where
$u(t) \to 0$ as $t \to 0$ and NFW's $\Sigma \sim -\ln u$ gives an
integrable logarithmic endpoint singularity (handled in §5).

## 3. Miscentered aperture mean: aperture-mass identity

The naive route to $\bar\Sigma_{\rm mis}({<}R)$ is the cumulative
integral of §2's result,

$$
\bar\Sigma_{\rm mis}({<}R \mid R_{\rm mis})
\;=\; \frac{2}{R^{2}} \int_0^{R} \Sigma_{\rm mis}(R' \mid R_{\rm mis})\,
R'\, dR' ,
$$

a nested double integral whose inner integrand has a cusp at
$R' = R_{\rm mis}$. Both problems disappear by computing the aperture
mass directly in halo-centric coordinates. The mass inside the aperture
(disk of radius $R$ about the assumed center) is a weighted integral of
the **known** $\Sigma$ over halo-centric annuli:

$$
\bar\Sigma_{\rm mis}({<}R \mid R_{\rm mis})
\;=\; \frac{1}{\pi R^{2}} \int_0^{\infty} \Sigma(u)\, u\, \Lambda(u)\, du ,
$$

where $\Lambda(u)$ is the overlap angle — the angular extent of the
halo-centric circle of radius $u$ that lies inside the aperture. With
$A(u)$ the cosine of the half-opening angle,

$$
A(u) \;=\; \frac{u^{2} + R_{\rm mis}^{2} - R^{2}}{2\,u\,R_{\rm mis}},
\qquad
\Lambda(u) \;=\;
\begin{cases}
2\pi, & u \le R - R_{\rm mis} \quad (\text{only if } R_{\rm mis} < R),\\[1mm]
2\arccos A(u), & |R - R_{\rm mis}| \le u \le R + R_{\rm mis},\\[1mm]
0, & \text{otherwise.}
\end{cases}
$$

This is a single integral, but $\Sigma(u)$ still carries its central
$\ln u$ divergence when $R_{\rm mis} < R$ (the region $u \to 0$
contributes with weight $2\pi$).

## 4. Integration by parts onto the known mean

Since

$$
\frac{d}{du}\!\left[\frac{u^{2}}{2}\,\bar\Sigma({<}u)\right]
\;=\; \Sigma(u)\, u ,
$$

split §3 at $u_- = |R - R_{\rm mis}|$ and $u_+ = R + R_{\rm mis}$ and
integrate the $[u_-, u_+]$ piece by parts. The endpoint values of the
overlap angle are $\Lambda(u_+) = 0$ always, and
$\Lambda(u_-) = 2\pi$ if $R_{\rm mis} < R$ (the circle $u = u_-$ lies
entirely inside the aperture) or $\Lambda(u_-) = 0$ if
$R_{\rm mis} > R$. When $R_{\rm mis} < R$, the by-parts boundary term at
$u_-$,

$$
-\,\frac{u_-^{2}}{2}\,\bar\Sigma({<}u_-)\,\Lambda(u_-)
\;=\; -\,\pi\, u_-^{2}\,\bar\Sigma({<}u_-),
$$

cancels the inner-disk contribution
$\int_0^{u_-} \Sigma(u)\, u \cdot 2\pi\, du
= \pi\, u_-^{2}\, \bar\Sigma({<}u_-)$ **exactly**; when
$R_{\rm mis} > R$ both pieces are individually zero. In either case no
boundary term survives:

$$
\bar\Sigma_{\rm mis}({<}R \mid R_{\rm mis})
\;=\; -\frac{1}{\pi R^{2}} \int_{u_-}^{u_+}
\frac{u^{2}}{2}\, \bar\Sigma({<}u)\, \Lambda'(u)\, du ,
$$

with

$$
\Lambda'(u) \;=\; -\,\frac{2\,A'(u)}{\sqrt{1 - A^{2}(u)}}
\;=\; -\,\frac{u^{2} + R^{2} - R_{\rm mis}^{2}}
{u^{2}\, R_{\rm mis}\, \sqrt{1 - A^{2}(u)}} .
$$

The central divergence of $\Sigma$ is gone — only the smooth
$\bar\Sigma$ appears (near $u=0$ the combination
$u^2 \bar\Sigma({<}u) \to 0$ for any profile with finite central mass) —
but $\Lambda'(u)$ still has inverse-square-root singularities at both
endpoints, since

$$
1 - A^{2}(u) \;=\;
\frac{\big(u_+^{2} - u^{2}\big)\big(u^{2} - u_-^{2}\big)}{4\,u^{2}\,R_{\rm mis}^{2}} .
$$

## 5. The substitution that removes everything

Factor the endpoint singularity using the identity above:

$$
\frac{u^{2}}{2}\,\Lambda'(u)\, du
\;=\; -\,\frac{u\,\big(u^{2} + R^{2} - R_{\rm mis}^{2}\big)}
{\sqrt{\big(u_+^{2} - u^{2}\big)\big(u^{2} - u_-^{2}\big)}}\; du ,
$$

and substitute the law-of-cosines parametrization

$$
u(t)^{2} \;=\; R^{2} + R_{\rm mis}^{2} - 2\,R\,R_{\rm mis}\cos t
\;=\; \tfrac{1}{2}\big(u_+^{2} + u_-^{2}\big)
- \tfrac{1}{2}\big(u_+^{2} - u_-^{2}\big)\cos t ,
\qquad t \in [0, \pi].
$$

Then

$$
u_+^{2} - u^{2} = \tfrac{1}{2}\big(u_+^{2} - u_-^{2}\big)(1 + \cos t),
\qquad
u^{2} - u_-^{2} = \tfrac{1}{2}\big(u_+^{2} - u_-^{2}\big)(1 - \cos t),
$$

so the square root becomes
$\tfrac{1}{2}(u_+^{2} - u_-^{2})\sin t$, while
$2\,u\,du = \tfrac{1}{2}(u_+^{2} - u_-^{2})\sin t\, dt$ — the Jacobian
cancels the singular factor **exactly**, leaving

$$
\boxed{\;
\bar\Sigma_{\rm mis}({<}R \mid R_{\rm mis})
\;=\; \frac{1}{2\pi R^{2}} \int_0^{\pi}
\Big[\, u(t)^{2} + R^{2} - R_{\rm mis}^{2} \,\Big]\,
\bar\Sigma\!\big({<}u(t)\big)\, dt
\;}
$$

with $u(t)$ exactly the nodes already used for $\Sigma_{\rm mis}$ in §2.
The integrand is smooth for all $R \ne R_{\rm mis}$; at
$R = R_{\rm mis}$ the factor $u^{2}\,\bar\Sigma({<}u) \to 0$ suppresses
the (integrable) $\ln u$ tail of $\bar\Sigma$.

The miscentered excess surface density follows as

$$
\Delta\Sigma_{\rm mis}(R \mid R_{\rm mis})
\;=\; \bar\Sigma_{\rm mis}({<}R \mid R_{\rm mis})
\;-\; \Sigma_{\rm mis}(R \mid R_{\rm mis}) .
$$

## 6. Analytic checks

**Centered limit.** $R_{\rm mis} \to 0$: $u(t) \to R$, the bracket is
$2R^{2}$, and the boxed formula returns $\bar\Sigma({<}R)$.

**Uniform sheet.** $\bar\Sigma \equiv \Sigma_0$: using
$\int_0^\pi \cos t\, dt = 0$,

$$
\frac{\Sigma_0}{2\pi R^{2}} \int_0^{\pi}
\big(2R^{2} - 2RR_{\rm mis}\cos t\big)\, dt \;=\; \Sigma_0
\qquad \text{for every } R_{\rm mis},
$$

so a constant surface density is invariant under miscentering, as it
must be — on both sides of $R_{\rm mis} = R$, confirming the boundary
cancellation of §4.

**Point mass.** $\bar\Sigma({<}u) = M/(\pi u^{2})$ for a point mass $M$
at the true center. With the standard integral

$$
\int_0^{\pi} \frac{dt}{a - b\cos t} \;=\; \frac{\pi}{\sqrt{a^{2}-b^{2}}},
\qquad a = R^{2} + R_{\rm mis}^{2},\; b = 2RR_{\rm mis},\;
\sqrt{a^{2}-b^{2}} = \big|R^{2} - R_{\rm mis}^{2}\big| ,
$$

the boxed formula gives

$$
\bar\Sigma_{\rm mis}({<}R \mid R_{\rm mis}) \;=\;
\frac{M}{2\pi^{2} R^{2}}
\left[\pi + \big(R^{2} - R_{\rm mis}^{2}\big)
\frac{\pi}{\big|R^{2} - R_{\rm mis}^{2}\big|}\right]
=
\begin{cases}
\dfrac{M}{\pi R^{2}}, & R_{\rm mis} < R,\\[2mm]
0, & R_{\rm mis} > R,
\end{cases}
$$

i.e. exactly the enclosed point mass — zero when the true center lies
outside the aperture.

## 7. The sign of $\Delta\Sigma_{\rm mis}$ — do not clamp

$\Delta\Sigma_{\rm mis}$ is genuinely signed: positive for
$R_{\rm mis} \ll R$ and **negative** for $R_{\rm mis} \gtrsim R$
(dimensionless NFW, $r_s = 1$, $\Sigma = f(x)$: at $R = 1$ it crosses
from $+0.0760$ at $R_{\rm mis}=0.8$ to $-0.0628$ at $R_{\rm mis}=1.0$).
Two facts show the negative lobe is physical:

1. The point-mass result of §6 gives
   $\Delta\Sigma_{\rm mis} = 0$ exactly for $R_{\rm mis} > R$; the
   negative values for an extended profile are a finite-profile gradient
   effect — the halo's mass leaks into the aperture and rises toward its
   edge, so $\Sigma_{\rm mis}(R) > \bar\Sigma_{\rm mis}({<}R)$.

2. A population of halos with offsets uniform over the plane is a
   uniform mass sheet, which has zero lensing contrast. Hence the
   mean-field cancellation

$$
\int_0^{\infty} \Delta\Sigma_{\rm mis}(R \mid R_{\rm mis})\;
2\pi R_{\rm mis}\, dR_{\rm mis} \;=\; 0 ,
$$

and it is the negative lobe that delivers it (numerically the truncated
integral falls to $0.3\%$ of its $L_1$ norm by
$R_{\rm mis}^{\max} = 60\,r_s$). Clamping
$\Delta\Sigma_{\rm mis} \ge 0$ pins the integral at roughly $+50\%$ of
the $L_1$ norm — it breaks the cancellation rather than protecting it.

## 8. Numerical scheme and accuracy

Both integrals of §2 and §5 are evaluated by fixed-order Gauss–Legendre
quadrature after the node-clustering map

$$
t \;=\; \pi s^{2}, \qquad s \in [0, 1], \qquad dt = 2\pi s\, ds ,
$$

which concentrates nodes at $t = 0$, where $u \to |R - R_{\rm mis}|$ and
the integrand of §2 develops its logarithmic behavior when
$R \approx R_{\rm mis}$. The radii are evaluated in the
cancellation-free form

$$
u^{2} \;=\; \big(R - R_{\rm mis}\big)^{2}
+ 4\,R\,R_{\rm mis}\sin^{2}(t/2) .
$$

One further ingredient is needed: the Wright & Brainerd closed forms for
the NFW kernels $f(x) = \Sigma/(2 r_s \rho_s)$ and
$g(x) = \Delta\Sigma/(r_s \rho_s)$ are $0/0$ at $x = 1$ and lose
$\sim 10^{-16}/|x-1|$ to cancellation nearby — and the clustered nodes
land precisely there whenever $u$ crosses the scale radius. `NfwProfile`
therefore switches to the exact Taylor expansions for $|x - 1| \le 10^{-2}$
(both one-sided forms continue to the same analytic series):

$$
f(1+d) = \tfrac{1}{3} - \tfrac{2}{5}d + \tfrac{13}{35}d^{2}
- \tfrac{20}{63}d^{3} + \cdots,
\qquad
g(1+d) = \sum_{n\ge 0} \Big[\, p_n + (-1)^{n+1}\, 4(n{+}1)\ln 2 \,\Big] d^{n},
$$

with $p_0 = \tfrac{10}{3},\; p_1 = -\tfrac{88}{15},\;
p_2 = \tfrac{296}{35},\; p_3 = -\tfrac{3508}{315}, \ldots$ (nine terms
kept; truncation $\sim 10^{-19}$, direct forms outside the window
accurate to $\lesssim 2\times 10^{-14}$).

Validation against mpmath (30-digit) adaptive quadrature of the same
integrals, dimensionless NFW, $n_{\rm nodes} = 128$–$256$:

| quantity | typical relative error | worst case tested |
|---|---|---|
| $\bar\Sigma_{\rm mis}$ | $\lesssim 10^{-13}$ | $10^{-10}$ at $R = R_{\rm mis}$ |
| $\Sigma_{\rm mis}$ | $\lesssim 10^{-13}$ | $2\times 10^{-10}$ at $R = R_{\rm mis}$ |

The scheme is a fixed matrix–vector product per radius vector (one
profile evaluation on an $(n_R \times n_{\rm nodes})$ grid), with no
adaptive quadrature anywhere.

## 9. References

- Wright, C. O. & Brainerd, T. G. 2000, ApJ, 534, 34 — closed-form NFW
  $\Sigma$, $\Delta\Sigma$.
- Yang, X. et al. 2006, MNRAS, 373, 1159 — azimuthal-average form of
  $\Sigma_{\rm mis}$.
- Johnston, D. E. et al. 2007, arXiv:0709.1159 — miscentering formalism
  for stacked cluster lensing.
- Simet, M. et al. 2017, MNRAS, 466, 3103 — offset distributions for
  redMaPPer clusters.
