"""Profiling fixture for the grid-reuse path of ``SigmaPrj``.

Run explicitly with ``pytest -s tests/test_sigma_prj_profile.py`` to print
the cumulative profile.  All cosmology products are materialized before the
profile starts, so the recorded work is strictly SigmaPrj construction,
grid reuse, and one Abel channel evaluation.
"""

from __future__ import annotations

import cProfile
import io
import pstats

import numpy as np

from clenspy.cosmology.bias import BiasModel
from clenspy.cosmology.fiducial import fiducial_cosmology
from clenspy.cosmology.halo_mass_function import TinkerMassFunction
from clenspy.cosmology.pkgrid import PkGrid
from clenspy.halo.twohalo import TwoHaloTerm
from clenspy.lensing import SigmaPrj, SigmaPrjConfig


def test_sigma_prj_prebuilt_grid_profile(capsys):
    cosmo = fiducial_cosmology()
    config = SigmaPrjConfig(
        n_theta=16,
        n_M=8,
        n_u_inside=6,
        n_u_outside=16,
        los_window="hard",
        los_depth=50.0,
    )
    pk = PkGrid(cosmo=cosmo, nonlinear=True)
    pk0 = pk(pk.k, z=0.0)
    mass = np.geomspace(1.0e13 / cosmo.h, 10.0**15.5 / cosmo.h, 256)
    hmf = TinkerMassFunction(cosmo=cosmo, k=pk.k, pk=pk0,
                             mvec=mass, zvec=pk.z)
    bias = BiasModel(cosmo=cosmo, k=pk.k, P=pk0, mvec=mass, zvec=pk.z)
    two_halo = TwoHaloTerm(pk.k, pk.pk, zvec=pk.z)
    two_halo.xi()

    profiler = cProfile.Profile()
    profiler.enable()
    prj = SigmaPrj(
        cosmology=cosmo,
        pk=pk,
        hmf=hmf,
        two_halo=two_halo,
        bias=bias,
        config=config,
    )
    n_rnd_in, n_rnd_out, n_lss = prj.n_los_integral(
        10.0, 0.3, lambda theta: np.ones_like(theta))
    rnd, exc = prj.k_exc.channels(n_rnd_in, n_rnd_out, n_lss)
    profiler.disable()

    assert prj.hmf_model is hmf
    assert prj.two_halo is two_halo
    assert prj.bias_model is bias
    assert np.isfinite(rnd).all() and np.isfinite(exc).all()

    output = io.StringIO()
    pstats.Stats(profiler, stream=output).sort_stats("cumulative").print_stats(12)
    with capsys.disabled():
        print(output.getvalue())
