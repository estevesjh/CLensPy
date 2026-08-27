"""Test-suite configuration.

Force matplotlib's non-interactive Agg backend. The comparison tests carry
`is_plot` switches for interactive debugging; with Agg selected here,
flipping one to True renders to a buffer instead of blocking the run on
`plt.show()`. Import is guarded because matplotlib is an optional
dependency.
"""

try:
    import matplotlib
except ImportError:  # pragma: no cover - matplotlib is optional
    pass
else:
    matplotlib.use("Agg")
