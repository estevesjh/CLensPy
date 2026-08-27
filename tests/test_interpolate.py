"""`LogGridInterpolator`: the auto-transpose of ``values``, the ``z=None``
default, the garbage-shape rejection, and the scalar-in/float-out branch.
Also `make_log_interpolation`'s scalar-in/float-out branch (its array-input
path is already exercised indirectly via `clenspy.halo.einasto`).
"""

import numpy as np
import pytest

from clenspy.utils.interpolate import LogGridInterpolator, make_log_interpolation


def _grid():
    x = np.logspace(-2, 1, 20)
    z = np.array([0.0, 0.5, 1.0])
    # separable power law, so the log-log interpolant is exact
    values = np.outer(x**-1.5, 1.0 / (1.0 + z))  # shape (nk, nz)
    return x, z, values


def test_values_already_nk_nz_and_transposed_nz_nk_agree():
    x, z, values = _grid()
    direct = LogGridInterpolator(xvec=x, zvec=z, values=values)         # (nk, nz)
    transposed = LogGridInterpolator(xvec=x, zvec=z, values=values.T)   # (nz, nk)

    xq = np.array([0.05, 0.5, 5.0])
    for zq in (0.0, 0.5, 1.0):
        np.testing.assert_allclose(direct(xq, zq), transposed(xq, zq))


def test_garbage_shape_raises_value_error():
    x, z, values = _grid()
    bad = np.zeros((values.shape[0] + 1, values.shape[1] + 1))
    with pytest.raises(ValueError, match="incompatible"):
        LogGridInterpolator(xvec=x, zvec=z, values=bad)


def test_z_none_matches_the_stored_zvec():
    x, z, values = _grid()
    interp = LogGridInterpolator(xvec=x, zvec=z, values=values)
    xq = np.array([0.05, 0.5, 5.0])
    np.testing.assert_allclose(interp(xq, None), interp(xq, interp.zvec))


def test_scalar_r_returns_a_python_float():
    x, z, values = _grid()
    interp = LogGridInterpolator(xvec=x, zvec=z, values=values)
    out = interp(0.5, 0.5)
    assert isinstance(out, float)


def test_make_log_interpolation_scalar_r_returns_a_python_float():
    x = np.logspace(-2, 1, 20)
    y = x ** -1.5  # exact under log-log linear interpolation
    f = make_log_interpolation(x, y)
    out = f(0.5)
    assert isinstance(out, float)
    assert np.isclose(out, 0.5 ** -1.5, rtol=1e-10)


def test_make_log_interpolation_array_r_returns_an_array():
    x = np.logspace(-2, 1, 20)
    y = x ** -1.5
    f = make_log_interpolation(x, y)
    out = f(np.array([0.05, 0.5, 5.0]))
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, np.array([0.05, 0.5, 5.0]) ** -1.5, rtol=1e-10)
