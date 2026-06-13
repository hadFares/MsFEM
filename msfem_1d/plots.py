"""
Tracés : solutions, erreurs, courbes de convergence.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_solutions(solutions_dict: dict, n_pts=500, title="Solutions", ax=None):
    """
    solutions_dict : {"label": solution_obj, ...}
    Chaque solution_obj doit avoir .value(x).
    """
    if ax is None:
        fig, ax = plt.subplots()
    x = np.linspace(0.0, 1.0, n_pts)
    for label, sol in solutions_dict.items():
        ax.plot(x, sol.value(x), label=label)
    ax.set_xlabel("x")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_convergence(H_list, errors_dict: dict, title="Convergence", ax=None, refs=None):
    """
    H_list       : liste de pas grossiers
    errors_dict  : {"label": [err(H) pour chaque H], ...}
    refs         : {slope: label}  ex. {2: "O(H^2)", 1: "O(H)"}
    """
    if ax is None:
        fig, ax = plt.subplots()

    H_arr = np.array(H_list)
    for label, errs in errors_dict.items():
        ax.loglog(H_arr, errs, marker="o", label=label)

    if refs is not None:
        x0, y0 = H_arr[-1], max(e[-1] for e in errors_dict.values())
        for slope, lbl in refs.items():
            ref_y = y0 * (H_arr / H_arr[-1])**slope
            ax.loglog(H_arr, ref_y, "k--", alpha=0.5, label=lbl)

    ax.set_xlabel("H")
    ax.set_title(title)
    ax.legend()
    return ax
