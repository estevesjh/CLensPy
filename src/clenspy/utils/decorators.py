import time
from functools import wraps

import numpy as np


def scalar_array_output(method):
    """
    Decorator: return a Python scalar if the method's first positional
    argument was scalar-like **and** the result really is a single number;
    otherwise return the array unchanged.

    NOTE: the size check is not belt-and-braces, it is required. A scalar
    first argument does not imply a scalar result: `NfwProfile.fourier`
    with an array ``m200`` and a scalar ``k`` legitimately returns one
    value per halo. The earlier version keyed only on the argument and
    called ``.item()`` unconditionally, which raised
    ``ValueError: can only convert an array of size 1`` for exactly that
    call -- a real combination, just one no test happened to make.

    Parameters
    ----------
    method : callable
        Method with signature ``(self, x, ...)`` returning an array-like.

    Returns
    -------
    callable
        Wrapped method with the same signature.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)

        # Was the first positional argument scalar-like?
        scalar_in = np.ndim(args[0]) == 0

        if scalar_in:
            if isinstance(result, np.ndarray):
                # only collapse when there is genuinely one number to give
                if result.size == 1:
                    return result.reshape(()).item()
                return result
            return float(result)                 # already a scalar
        return result
    return wrapper


def default_rvals_z(method):
    """
    Decorator for methods with signature ``(self, R_vals=None, z=None)``:
    substitutes ``self.reval``/``self.zvec`` when ``R_vals``/``z`` are None.

    Parameters
    ----------
    method : callable
        Method with signature ``(self, R_vals, z, ...)``.

    Returns
    -------
    callable
        Wrapped method accepting ``R_vals=None``/``z=None``.
    """
    from functools import wraps

    @wraps(method)
    def wrapper(self, R_vals=None, z=None, *args, **kwargs):
        if R_vals is None:
            R_vals = self.reval
        if z is None:
            z = self.zvec
        return method(self, R_vals, z, *args, **kwargs)

    return wrapper


def time_method(func):
    """
    Decorator: record each call's wall-clock time in ``self.timings``
    (a dict mapping method name to a list of elapsed seconds), and print
    it if ``self.verbose`` is truthy.

    Parameters
    ----------
    func : callable
        Method to time.

    Returns
    -------
    callable
        Wrapped method with the same signature and return value.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "timings"):
            self.timings = {}
        t0 = time.time()
        result = func(self, *args, **kwargs)
        t1 = time.time()
        elapsed = t1 - t0
        fname = func.__name__
        self.timings.setdefault(fname, []).append(elapsed)
        # Only print if verbose is set and true
        if getattr(self, "verbose", False):
            print(f"{fname} took {elapsed:.3f} s")
        return result

    return wrapper


if __name__ == "__main__":
    import numpy as np

    class Demo:
        """A stand-in showing what each decorator changes."""

        @scalar_array_output
        def square(self, x):
            return np.atleast_1d(np.asarray(x, dtype=float)) ** 2

        @time_method
        def slow(self, n):
            return float(np.sum(np.arange(n, dtype=float)))

    d = Demo()
    print("scalar_array_output: a scalar in gives a scalar out")
    print(f"  square(3.0)          = {d.square(3.0)!r}  "
          f"(ndim {np.ndim(d.square(3.0))})")
    print(f"  square([1, 2, 3])    = {d.square([1.0, 2.0, 3.0])!r}")

    print("\ntime_method: prints its own wall time")
    d.slow(1_000_00)
