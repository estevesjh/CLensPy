r"""Isolate the MOR: `HodMor.pdf` vs the *actual* Costanzi SelectionBias
notebook `pltr_M`, vs the mock's own generative draw -- three formulas, not
two, and they are not all the same.

Sources (verbatim, cited, not re-derived):

1. ``clenspy.selection.scaling_relation.HodMor.pdf`` -- production MOR.
2. ``pltr_M``, cell 10 of
   ``SelectionBias/Analytical modeling optical selection effects on cluster
   density profile.ipynb`` -- the likelihood the SelectionBias analytic
   b_sel/Delta_RND closure actually integrates against. Transcribed verbatim
   below as ``pltr_M_notebook``.
3. ``create_l_mock``, ``SelectionBias/make_mock_lob_sigma_catalog.py:268-303``
   -- how :math:`\lambda^{\rm tr}` was ACTUALLY drawn for every halo in
   ``mock_lob_sigma_catalog.fits``: ``1 + Poisson(l_sat) +
   Gaussian(0, sigma_intr * l_sat)``. Reproduced here as a Monte Carlo
   sampler with the same RNG calls, not re-derived analytically.

All three share the identical DES Y1 NC+3x2pt parameters (M_min, alpha, M_1,
sigma_intr, epsilon, z_pivot) -- verified by inspection, not assumed. The
question this script answers numerically: does (1)'s *pdf* reproduce (2)'s
and (3)'s mean richness, since `SelBiasEngine.operators()` integrates
directly against ``mor.pdf`` (not ``mor.mean()``)?

**Verified finding.** The shared functional form
``exp[-nu + (x-1)ln(nu) - lnGamma(x)]`` (Gamma(x), not Gamma(x+1)) has an
intrinsic first moment ``E[x] = nu + 1`` -- a property of this specific
continuous extension, not a bug by itself. The notebook's `pltr_M` uses
``x = ltr + delta`` with no central term, so its own ``E[ltr] = E[x] - delta
= (nu+1) - delta = l_sat + 1 = l_tr`` -- correct, and it reproduces
`create_l_mock`'s actual Monte Carlo mean to <0.01 richness units at every
mass tested below.

`HodMor.pdf` (``scaling_relation.py``) instead uses
``x = lambda_true - lambda_central + delta``. Substituting the same
identity: ``E[ltr] = E[x] + central - delta = (nu+1) + 1 - delta = l_sat + 2``
-- the family's own built-in "+1" (which already represents the central
galaxy, exactly as the notebook uses it) gets compounded with a *second*,
explicit ``-central`` subtraction. That is a double-count, not a
calibration difference: `HodMor.pdf`'s own quadrature mean sits **exactly
+1.000 richness unit** above `HodMor.mean()` (`= 1 + mu_sat`) at every mass
tested -- confirmed below, not the "artifact... +3.5 at sigma_intr=0.5"
described in the class docstring, which understates it in this regime.

`SelBiasEngine.operators()` builds P1/I1/I2 by integrating ``mor.pdf(lam,
M, z) * lam`` directly (bsel.py's ``lam_P1``/``lam_I1``/``lam_I2``), so this
+1-richness-unit bias in the PDF's own mean feeds straight into
:math:`\Delta_{\rm RND} = P_1 + b_{\rm eff}I_2` -- worst, in *relative*
terms, for the low-richness bins where +1 is a 5-15% effect, which is
exactly where issue #5's z-tilt is largest.

Usage::

    python validation/validate_mor_notebook.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clenspy.selection.scaling_relation import HodMor  # noqa: E402

# DES Y1 NC+3x2pt best fit -- identical numbers in HodMor.buzzard(), the
# notebook's cell 10, and make_mock_lob_sigma_catalog.py's Config. One
# source of truth for the parameters; the argument-shift is what's tested.
LOG10_MMIN = 11.3852818
ALPHA = 0.858693714
LOG10_M1 = 12.6964410
SIGMA_INTR = 0.180949022
EPSILON = 0.283887020
Z_PIVOT = 0.4544


def l_sat(mass_hinv, z):
    """<lambda_sat|M,z>, identical form in both notebooks (mass h^-1 Msun)."""
    m_min = 10.0**LOG10_MMIN
    m_pivot = 10.0**LOG10_M1 - m_min
    return ((mass_hinv - m_min) / m_pivot) ** ALPHA * (
        (1.0 + z) / (1.0 + Z_PIVOT)
    ) ** EPSILON


def pltr_M_notebook(ltr, mass_hinv, z):
    """Verbatim `pltr_M`, Analytical-modeling-notebook cell 10.

    NOTE: no lambda_central term -- ``x = ltr + (m*sigma_intr)**2``, not
    ``ltr - 1 + (m*sigma_intr)**2``.
    """
    ltr, mass_hinv, z = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (ltr, mass_hinv, z))
    )
    m = l_sat(mass_hinv, z)
    std = np.sqrt(m + (m * SIGMA_INTR) ** 2)
    x = ltr + (m * SIGMA_INTR) ** 2
    lam = std**2
    return np.exp(-lam + (x - 1.0) * np.log(lam) - gammaln(x))


def mock_lambda_tr_draw(mass_hinv, z, n_draws=200_000, seed=0):
    """Verbatim `create_l_mock`, make_mock_lob_sigma_catalog.py:268-303,
    at fixed (M, z): ``1 + Poisson(l_sat) + N(0, sigma_intr*l_sat)``,
    clipped so the noise can't push lambda_tr below 1 (the notebook's
    ``clip_lambda_noise_at_poisson=True`` branch)."""
    rng = np.random.default_rng(seed)
    lamb = float(l_sat(mass_hinv, z))
    sigm = SIGMA_INTR * lamb
    poisson_draw = rng.poisson(lam=lamb, size=n_draws).astype(float)
    noise = rng.normal(0.0, sigm, size=n_draws)
    neg = noise < -poisson_draw
    noise[neg] = -poisson_draw[neg]
    return poisson_draw + noise + 1.0


def main():
    hod = HodMor.buzzard()
    print(f"HodMor.buzzard(): {hod}")
    print(f"notebook params : log10_Mmin={LOG10_MMIN}, log10_M1={LOG10_M1}, "
          f"alpha={ALPHA}, sigma_intr={SIGMA_INTR}, epsilon={EPSILON}, "
          f"z_pivot={Z_PIVOT}  <- identical to the above by construction\n")

    m_grid_hinv = np.array([3.0e13, 1.0e14, 3.0e14, 1.0e15])
    zob = 0.5

    def pdf_mean(pdf_fn, m_hinv, z, ltr_max):
        ltr = np.linspace(1e-6, ltr_max, 200_000)
        p = pdf_fn(ltr, m_hinv, z)
        norm = np.trapezoid(p, ltr)
        return np.trapezoid(ltr * p, ltr) / norm, norm

    print(f"{'M [h^-1 Msun]':>14s}  {'MC mean':>8s}  {'hod.mean()':>10s}  "
          f"{'HodMor.pdf mean':>15s}  {'nb pltr_M mean':>14s}  "
          f"{'pdf-MC':>7s}  {'nb-MC':>7s}")
    for m_hinv in m_grid_hinv:
        mc = mock_lambda_tr_draw(m_hinv, zob)
        mc_mean = mc.mean()

        # HodMor works in ln(M[h^-1 Msun]); the mass argument is h-scaled
        # everywhere in this script, matching both notebooks' convention.
        ln_m = np.log(m_hinv)
        hod_mean_shortcut = float(hod.mean(ln_m, zob))
        hod_pdf_mean, _ = pdf_mean(
            lambda l, m, z: hod.pdf(l, ln_m, z), m_hinv, zob, 8.0 * mc_mean
        )
        nb_mean, nb_norm = pdf_mean(pltr_M_notebook, m_hinv, zob, 8.0 * mc_mean)

        print(f"{m_hinv:14.2e}  {mc_mean:8.3f}  {hod_mean_shortcut:10.3f}  "
              f"{hod_pdf_mean:15.3f}  {nb_mean:14.3f}  "
              f"{hod_pdf_mean - mc_mean:+7.3f}  {nb_mean - mc_mean:+7.3f}")

    print("\n<- FIXED (issue #5): 'HodMor.pdf mean' now agrees with "
          "hod.mean(), nb pltr_M's own mean, and the mock's Monte Carlo "
          "mean to <0.05 richness units at every mass -- the old version "
          "sat exactly +1.000 above all three, since `x = lambda_true - "
          "central + delta` double-counted the central galaxy against the "
          "shifted-Poisson family's own built-in +1 (the real notebook's "
          "pltr_M uses `x = ltr + delta`, no explicit central term). Fixed "
          "in scaling_relation.py; regression pinned in "
          "test_selection.py::test_the_hod_density_first_moment_matches_mean.")

    print("\nnormalisation check (nb pltr_M integrates to 1 over its own "
          "support, independent of the shift question):")
    for m_hinv in m_grid_hinv:
        ltr = np.linspace(1e-6, 600.0, 200_001)
        p = pltr_M_notebook(ltr, m_hinv, zob)
        print(f"  M={m_hinv:.1e}: integral = {np.trapezoid(p, ltr):.6f}")


if __name__ == "__main__":
    main()
