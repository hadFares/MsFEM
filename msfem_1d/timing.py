"""
Chronométrage des solveurs.

Mesure robuste du temps d'exécution d'un solveur (warm-up + médiane de plusieurs
appels). Utilisé pour comparer le coût P1 grossier / P1 fin / MsFEM.

Note : les solveurs utilisent np.linalg.solve (dense, O(m^3)). Les temps mesurés
reflètent donc cette implémentation, pas la complexité optimale tridiagonale.
"""

import time
import numpy as np


def time_solver(solve_fn, *args, repeats: int = 5, **kwargs):
    """
    Chronomètre un solveur et renvoie sa solution et son temps médian.

    Parameters
    ----------
    solve_fn : callable
        Fonction de résolution, ex. fem_p1_solve ou msfem.solve.
    *args, **kwargs
        Arguments transmis à solve_fn.
    repeats : int, optional
        Nombre d'exécutions chronométrées (médiane retournée), défaut 5.

    Returns
    -------
    solution : object
        Résultat du dernier appel à solve_fn.
    t_median : float
        Temps d'exécution médian, en secondes.
    """
    solve_fn(*args, **kwargs)   # warm-up (caches, imports paresseux)

    times = np.empty(repeats)
    solution = None
    for i in range(repeats):
        t0 = time.perf_counter()
        solution = solve_fn(*args, **kwargs)
        times[i] = time.perf_counter() - t0

    return solution, float(np.median(times))
