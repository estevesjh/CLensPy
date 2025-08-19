import time
from functools import wraps

import numpy as np


def scalar_array_output(method):
    """Return a scalar for scalar input, ndarray otherwise."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)

        # Was the first positional argument scalar-like?
        scalar_in = np.ndim(args[0]) == 0

        if scalar_in:
            # Accept numpy scalars, 0-D arrays or size-1 arrays
            if isinstance(result, np.ndarray):
                return result.squeeze().item()   # safe, future-proof
            return float(result)                 # already a scalar
        return result
    return wrapper


def default_rvals_z(method):
    """
    Decorator for methods with signature (self, R_vals=None, z=None):
    - If R_vals is None, use self.reval.
    - If z is None, use self.zvec.
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
