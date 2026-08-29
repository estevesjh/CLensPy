r"""Projection lensing :math:`\Sigma_{\rm prj}` / :math:`\Delta\Sigma_{\rm prj}`.

Tests of the E.3 machinery with the package's own halo model — Buzzard
cosmology, CAMB halofit :math:`\xi_{\rm NL}` (`PkGrid` is disk-cached, so
CAMB runs once ever), Tinker (2008) mass function and Tinker (2010) bias.
No toy stand-ins: the channel split, the exclusion semantics, the kernel
limits, the annihilation of constants by :math:`\Delta\Sigma`, and the
large-radius two-halo limit are all statements about the real integrand.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from clenspy.cosmology import TinkerMassFunction
from clenspy.cosmology.bias import BiasModel
from clenspy.cosmology.pkgrid import PkGrid
from clenspy.lensing import SigmaPrj
from clenspy.selection import SigmoidBias, XiNL


class BuzzardCosmology(FlatLambdaCDM):
    """Buzzard v1.1; class attrs are what PkGrid reads via getattr."""

    sigma8 = 0.82
    n_s = 0.96


COSMO = BuzzardCosmology(H0=70.0, Om0=0.286, Ob0=0.046)
H = 0.7
OMEGA_M = 0.286
LOB, ZOB = 20.0, 0.5
R = np.array([0.5, 2.0, 8.0, 25.0])  # comoving Mpc


@pytest.fixture(scope="module")
def model():
    """The real halo-model trio in SelBiasEngine's physical-Msun,
    broadcast-callable convention (`clenspy.utils.decorators` classes are
    grid/pairwise; the ravel makes every broadcast query pairwise)."""
    pk = PkGrid(cosmo=COSMO, nonlinear=True)
    xi_nl = XiNL(pk, clip=False)
    tmf = TinkerMassFunction(cosmo=COSMO, zvec=np.linspace(0.0, 1.0, 21))
    bm = BiasModel(cosmo=COSMO)

    def hmf(mass, z):
        """dn/dM [Msun^-1 Mpc^-3] at physical Msun. The one unit boundary:
        physical Msun -> the Tinker grid's Omega_m h^-1 Msun."""
        m, zz = np.broadcast_arrays(np.asarray(mass, float),
                                    np.asarray(z, float))
        m_tilde = m.ravel() * H / OMEGA_M
        vals = tmf.dndlnm(m_tilde, zz.ravel())        # pairwise, h^3 Mpc^-3
        return vals.reshape(m.shape) * H**3 / m       # -> per dM, h-free

    def bias(mass, z):
        m, zz = np.broadcast_arrays(np.asarray(mass, float),
                                    np.asarray(z, float))
        return np.asarray(
            bm.bias(m.ravel(), zz.ravel())
        ).reshape(m.shape)

    return SimpleNamespace(xi_nl=xi_nl, hmf=hmf, bias=bias)


def _prj(model, **kw):
    kwargs = dict(cosmology=COSMO, xi_nl=model.xi_nl, hmf=model.hmf,
                  bias=model.bias, n_theta=48, n_z_side=24, n_M=16,
                  los_window="hard", los_depth=50.0 / H, exclusion="cl")
    kwargs.update(kw)
    return SigmaPrj(**kwargs)


def _bsel(prj):
    return SigmoidBias(lob=LOB, zob=ZOB,
                       theta_lambda=prj.r_excl(LOB, ZOB) / prj.chi(ZOB),
                       b_small=4.0, b_large=3.0)


# -- channel split ----------------------------------------------------------

def test_bsel_zero_kills_cl_channel(model):
    prj = _prj(model)
    total = prj.sigma_prj(R, LOB, ZOB, lambda th: 0.0, channel="sum")
    assert np.allclose(prj.cl, 0.0)
    assert np.allclose(total, prj.rnd)


def test_xi_zero_kills_cl_channel(model):
    prj = _prj(model,
               xi_nl=lambda r, zob: np.zeros_like(np.asarray(r, float)))
    prj.sigma_prj(R, LOB, ZOB, _bsel(_prj(model)))
    assert np.allclose(prj.cl, 0.0)


def test_sum_is_rnd_plus_cl_and_components(model):
    prj = _prj(model)
    total = prj.sigma_prj(R, LOB, ZOB, _bsel(prj), channel="sum")
    parts = prj.components()
    assert np.allclose(parts["rnd"] + parts["cl"], total)
    assert np.allclose(parts["sum"], total)
    assert np.all(parts["rnd"] > 0)
    assert np.all(parts["cl"] > 0)


def test_cl_scales_linearly_with_constant_bsel(model):
    prj = _prj(model)
    prj.sigma_prj(R, LOB, ZOB, lambda th: 1.0)
    cl_1 = prj.cl.copy()
    prj.sigma_prj(R, LOB, ZOB, lambda th: 2.5)
    assert np.allclose(prj.cl, 2.5 * cl_1, rtol=1e-12)


# -- exclusion semantics ------------------------------------------------------

def test_exclusion_ball_reduces_rnd_none_does_not(model):
    prj_none = _prj(model, exclusion="none")
    prj_ball = _prj(model, exclusion="ball")
    prj_none.sigma_prj(R, LOB, ZOB, lambda th: 0.0)
    prj_ball.sigma_prj(R, LOB, ZOB, lambda th: 0.0)
    # the ball removes neighbours at the smallest radii only
    assert prj_ball.rnd[0] < prj_none.rnd[0]
    # far outside the ball the two agree to the excluded neighbours'
    # distant-tail contribution
    assert prj_ball.rnd[-1] == pytest.approx(prj_none.rnd[-1], rel=1e-4)


def test_exclusion_cl_keeps_rnd(model):
    prj_cl = _prj(model, exclusion="cl")
    prj_none = _prj(model, exclusion="none")
    b = _bsel(prj_cl)
    prj_cl.sigma_prj(R, LOB, ZOB, b)
    rnd_cl, cl_cl = prj_cl.rnd.copy(), prj_cl.cl.copy()
    prj_none.sigma_prj(R, LOB, ZOB, b)
    assert np.allclose(rnd_cl, prj_none.rnd)
    assert cl_cl[0] < prj_none.cl[0]


def test_floor_matches_unfloored_when_bracket_positive(model):
    # scale xi down so 1 + b_sel b xi > 0 everywhere: floor is a no-op
    tiny = lambda r, zob: 1e-6 * model.xi_nl(r, zob)
    prj_a = _prj(model, xi_nl=tiny, floor_one_plus_bxi=False)
    prj_b = _prj(model, xi_nl=tiny, floor_one_plus_bxi=True)
    b = _bsel(prj_a)
    sa = prj_a.sigma_prj(R, LOB, ZOB, b)
    sb = prj_b.sigma_prj(R, LOB, ZOB, b)
    assert np.allclose(sa, sb, rtol=1e-9)


# -- geometry -----------------------------------------------------------------

def test_dchi_is_law_of_cosines_not_delta_chi(model):
    prj = _prj(model)
    thetas, _ = prj.theta_grid(ZOB)
    zs, _ = prj._z_grid(ZOB)
    d = prj.dchi(thetas, zs, ZOB)
    naive = np.abs(prj.chi(zs) - prj.chi(ZOB))[None, :]
    iz = int(np.argmin(naive[0]))
    # at the largest theta the transverse leg dominates near the ring
    assert d[-1, iz] > 10.0 * max(naive[0, iz], 1e-3)
    # the chord is never smaller than the line-of-sight separation
    assert np.all(d >= naive - 1e-9)


def test_kernel_cell_mass_matches_brute_force(model):
    r"""The exact-cell trick: by the symmetry
    :math:`\Sigma_{\rm mis}(R, s) = \Sigma_{\rm mis}(s, R)`, the cell
    integral is an annulus-mass difference. Check one ring-crossing cell
    against a dense pointwise quadrature of :math:`\Sigma_{\rm mis}` —
    the thing the trick replaces."""
    prj = _prj(model)
    chi_o = float(prj.chi(ZOB))
    s_edges = prj.theta_edges(ZOB) * chi_o
    rs, sigma0 = prj._profiles(ZOB)
    im = rs.size // 2
    R0 = 5.0  # comoving Mpc
    it = int(np.searchsorted(s_edges, R0)) - 1  # the ring-crossing cell
    K = prj.kernel(np.array([R0]), LOB, ZOB, "sigma")
    # brute force on the same cell: 3000 nodes across the ring
    s = np.linspace(s_edges[it], s_edges[it + 1], 3000)
    sig_mis = sigma0[im] * prj.mis_table.sigma_hat(R0 / rs[im], 0.0) * 0.0
    sig_mis = np.array([
        sigma0[im] * prj.mis_table.sigma_hat(
            np.array([R0 / rs[im]]), si / rs[im])[0]
        for si in s
    ])
    brute = np.trapezoid(2.0 * np.pi * s * sig_mis, s) / chi_o**2
    brute *= np.sin(np.sqrt(s_edges[it] * s_edges[it + 1]) / chi_o) / (
        np.sqrt(s_edges[it] * s_edges[it + 1]) / chi_o)
    assert K[it, im, 0] == pytest.approx(brute, rel=0.02)


# -- DeltaSigma: kernel swap, signedness, annihilation of constants -----------

def test_deltasigma_annihilates_a_uniform_sheet(model):
    r"""DeltaSigma_prj is its own operator, :math:`\Delta\Sigma_{\rm prj}
    \propto \int dz \int dM \int d\theta \ldots \Delta\Sigma_{\rm mis}` —
    never a reconstruction from :math:`\Sigma_{\rm prj}`. Its defining
    property: the excess functional annihilates constants. With a stub
    table describing a uniform sheet (:math:`\hat\Sigma = {\rm const}`,
    :math:`\widehat{\Delta\Sigma} = 0`), sigma_prj is the sheet times the
    channel weights and deltasigma_prj is exactly zero."""
    class _SheetTable:
        @staticmethod
        def sigma_hat(x, x_mis):
            return np.full_like(np.atleast_1d(np.asarray(x, float)), 0.7)
        @staticmethod
        def ds_hat(x, x_mis):
            return np.zeros_like(np.atleast_1d(np.asarray(x, float)))

    prj = _prj(model, mis_table=_SheetTable())
    b = _bsel(prj)
    R_test = np.array([0.5, 5.0, 20.0])
    sig = prj.sigma_prj(R_test, LOB, ZOB, b)
    ds = prj.deltasigma_prj(R_test, LOB, ZOB, b)
    assert np.all(sig > 0)
    assert np.allclose(ds, 0.0, atol=1e-10 * sig.max())


def test_deltasigma_kernel_is_signed(model):
    # the ds_hat lobe at R_theta > R is negative and must never be
    # clamped -- it is what makes DeltaSigma annihilate a uniform sheet
    prj = _prj(model)
    K = prj.kernel(np.array([0.3, 1.0]), LOB, ZOB, "ds")
    assert K.min() < 0.0


def test_deltasigma_rnd_channel_vanishes(model):
    r"""For :math:`\Delta\Sigma_{\rm prj}` the rnd channel must be zero.

    The rnd channel is a uniform sheet, and the excess functional
    annihilates constants: mass conservation of the azimuthal average
    gives :math:`\int d^2s\,\Sigma_{\rm mis}(R, s \mid M) = M_{2D}`
    independent of :math:`R`, hence the same for
    :math:`\bar\Sigma_{\rm mis}(<R)`, hence
    :math:`\int d^2s\,\Delta\Sigma_{\rm mis}(R, s \mid M) = 0` — the
    negative lobe cancels the core exactly. Only the finite
    :math:`\theta_{\max}` truncation survives, so the rnd channel is a
    boundary term, small against the cl channel."""
    prj = _prj(model, exclusion="none", n_theta=96,
               theta_perp_range=(1e-3, 200.0))
    b = _bsel(prj)
    R_in = np.array([0.5, 2.0, 8.0])  # << theta_max span
    prj.deltasigma_prj(R_in, LOB, ZOB, b)
    ds_rnd, ds_cl = prj.rnd.copy(), prj.cl.copy()
    # and against the uniform sheet it came from
    prj.sigma_prj(R_in, LOB, ZOB, b)
    sheet = prj.rnd.copy()
    # measured residual: 0.05-0.2% of the sheet (theta-truncation
    # boundary term plus the smooth-part trapezoid), <=1.2% of cl
    assert np.all(np.abs(ds_rnd) < 0.02 * np.abs(ds_cl))
    assert np.all(np.abs(ds_rnd) < 3e-3 * sheet)


# -- the two-halo limit -----------------------------------------------------------

def test_two_halo_limit_wiring_with_flat_xi(model):
    r"""The two-halo limit of the cl channel, in the regime where it is
    exact. In the continuum

    .. math::
        \Sigma_{\rm cl}(R) = b_{\rm sel} \int dz\,{\rm common}\,\xi\,
        \frac{1}{\chi_o^2}\int dM\,n\,b\,M_{2D}

    holds exactly when :math:`\xi` is flat across the kernel's support
    (mass conservation of the azimuthal average) — with a real
    :math:`\xi(r)` the NFW wings spread :math:`M_{2D}` logarithmically in
    offset, so :math:`\Sigma_{\rm cl}(R)` sits *below* the point-mass
    value by the wing-weighted :math:`\xi` deficit; that physical
    comparison against :math:`b\,\rho_m\,\Sigma_{2h}` lives in
    ``validation/validate_sigma_prj_mock.py``. Here :math:`\xi = c`
    checks the whole wiring — spherical measure, z weights, mass budget —
    against the independent closed form :math:`M_{2D} = \pi s_{\max}^2
    \bar\Sigma_{\rm NFW}(<s_{\max})`."""
    xi_c = 0.37
    prj = _prj(model, exclusion="none", n_theta=96, n_z_side=32,
               theta_perp_range=(1e-3, 200.0),
               xi_nl=lambda r, zob: np.full_like(
                   np.asarray(r, float), xi_c))
    b_const = 2.0
    R_any = np.array([2.0, 15.0, 25.0])  # exact at every R for flat xi
    prj.sigma_prj(R_any, LOB, ZOB, lambda th: b_const)
    cl = prj.cl.copy()

    chi_o = float(prj.chi(ZOB))
    zs, wzs = prj._z_grid(ZOB)
    cmn = prj.common(zs, ZOB) * wzs
    Ms, Mw = prj._mass_nodes()
    from clenspy.halo.nfw import NfwProfile
    prof = NfwProfile(m200=Ms, c200=prj.concentration(Ms, ZOB),
                      rho_ref=prj.rho_m)
    s_max = prj.theta_perp_range[1]
    # closed form, independent of the miscentering table: for s_max >> R
    # the offset aperture mass is the centred one to O((R/s_max)^2)
    M2D = np.pi * s_max**2 * np.asarray(prof.mean_sigma(s_max)).ravel()
    n_mz = prj.hmf(Ms[:, None], zs[None, :])
    b_mz = prj.bias(Ms[:, None], zs[None, :])
    budget_z = np.einsum("m,mz->z", Mw, n_mz * b_mz * M2D[:, None])
    ref = b_const * xi_c * np.sum(cmn * budget_z) / chi_o**2
    assert np.allclose(cl, ref, rtol=0.02)


# -- input validation ------------------------------------------------------------

def test_bad_switches_raise(model):
    with pytest.raises(ValueError):
        _prj(model, exclusion="slab")
    with pytest.raises(ValueError):
        _prj(model, los_window="gaussian")
    with pytest.raises(ValueError):
        _prj(model, los_window="hard", los_depth=None)


def test_counter_term_exclusion(model):
    r"""The Costanzi counter term: inside the chord ball the correlated
    integrand is -1, cancelling the background's +1. Same TOTAL as
    deleting the neighbours (``"ball"``), but the background stays
    strictly uniform and the exclusion hole is booked in the cl channel."""
    b = _bsel(_prj(model))
    R_in = np.array([0.1, 0.5, 5.0])
    prj_c = _prj(model, exclusion="counter")
    prj_b = _prj(model, exclusion="ball")
    prj_slab = _prj(model, exclusion="cl")   # same z-grid split, uniform rnd
    tot_c = prj_c.sigma_prj(R_in, LOB, ZOB, b, channel="sum")
    rnd_c, cl_c = prj_c.rnd.copy(), prj_c.cl.copy()
    tot_b = prj_b.sigma_prj(R_in, LOB, ZOB, b, channel="sum")
    prj_slab.sigma_prj(R_in, LOB, ZOB, b, channel="sum")
    # identical totals, to floating point
    assert np.allclose(tot_c, tot_b, rtol=1e-12)
    # background strictly uniform: identical to the unmasked-rnd slab run
    # (same quadrature grid, so the match is exact)
    assert np.allclose(rnd_c, prj_slab.rnd, rtol=1e-12)
    # the hole lives in cl: the counter term pulls cl below the
    # merely-zeroed slab cl at small R
    assert cl_c[0] < prj_slab.cl[0]
    assert cl_c[0] < prj_b.cl[0]
