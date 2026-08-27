"""`scalar_array_output`'s scalar-cast branch, and `time_method`'s
verbose-print branch.
"""

import numpy as np

from clenspy.utils.decorators import scalar_array_output, time_method


class _Demo:
    @scalar_array_output
    def square(self, x):
        return np.atleast_1d(np.asarray(x, dtype=float)) ** 2

    @scalar_array_output
    def already_scalar(self, x):
        # returns a plain Python float, not an ndarray -- hits the
        # `return float(result)` branch directly, distinct from the
        # size-1-ndarray-collapse branch `square` exercises above.
        return float(x) ** 2

    @time_method
    def slow(self, n):
        return float(np.sum(np.arange(n, dtype=float)))


def test_scalar_array_output_casts_a_size_one_result_to_float():
    d = _Demo()
    out = d.square(3.0)
    assert isinstance(out, float)
    assert out == 9.0


def test_scalar_array_output_casts_a_non_array_scalar_result_to_float():
    d = _Demo()
    out = d.already_scalar(3.0)
    assert isinstance(out, float)
    assert out == 9.0


def test_scalar_array_output_leaves_array_results_alone():
    d = _Demo()
    out = d.square(np.array([1.0, 2.0, 3.0]))
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [1.0, 4.0, 9.0])


def test_time_method_prints_when_verbose(capsys):
    d = _Demo()
    d.verbose = True
    d.slow(100)
    captured = capsys.readouterr()
    assert "slow" in captured.out
    assert "took" in captured.out
    assert "slow" in d.timings
    assert len(d.timings["slow"]) == 1


def test_time_method_is_silent_when_not_verbose(capsys):
    d = _Demo()
    d.verbose = False
    d.slow(100)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_time_method_is_silent_by_default():
    """No ``verbose`` attribute at all: ``getattr(..., False)`` covers it."""
    d = _Demo()
    d.slow(100)
    assert "slow" in d.timings
