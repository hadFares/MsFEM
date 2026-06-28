"""
Solver timing.

Robust measurement of a solver run time (warm-up + median of several calls).
Used to compare the cost of coarse P1 / fine P1 / MsFEM.

Note: the solvers use np.linalg.solve (dense, O(m^3)). The measured times
therefore reflect this implementation, not the optimal tridiagonal complexity.
"""

import time
import numpy as np


def time_solver(solve_fn, *args, repeats: int = 5, **kwargs):
    """
    Time a solver and return its solution and its median run time.

    Parameters
    ----------
    solve_fn : callable
        Solver function, e.g. fem_p1_solve or msfem.solve.
    *args, **kwargs
        Arguments passed to solve_fn.
    repeats : int, optional
        Number of timed runs (median returned), default 5.

    Returns
    -------
    solution : object
        Result of the last call to solve_fn.
    t_median : float
        Median run time, in seconds.
    """
    solve_fn(*args, **kwargs)   # warm-up (caches, lazy imports)

    times = np.empty(repeats)
    solution = None
    for i in range(repeats):
        t0 = time.perf_counter()
        solution = solve_fn(*args, **kwargs)
        times[i] = time.perf_counter() - t0

    return solution, float(np.median(times))
