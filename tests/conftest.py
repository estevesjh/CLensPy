"""Test-suite configuration.

Force matplotlib's non-interactive Agg backend. The comparison tests carry
`is_plot` switches for interactive debugging; with Agg selected here,
flipping one to True renders to a buffer instead of blocking the run on
`plt.show()`. Import is guarded because matplotlib is an optional
dependency.
"""

import pytest

try:
    import matplotlib
except ImportError:  # pragma: no cover - matplotlib is optional
    pass
else:
    matplotlib.use("Agg")


@pytest.fixture
def timing_report(twohalo, request):
    """Capture timing profile and report after test."""
    yield
    if hasattr(twohalo, "timings") and twohalo.timings:
        print(f"\n  Timings for {request.node.name}:")
        for method, times in sorted(twohalo.timings.items()):
            total = sum(times)
            count = len(times)
            avg = total / count if count else 0
            print(f"    {method:20s}: {total:8.3f}s ({count:2d} calls, {avg:7.4f}s avg)")
