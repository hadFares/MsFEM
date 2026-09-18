"""
plot_exact.py — solution exacte du problème modèle pour plusieurs ε.

Problème : -d/dx(A(x) u') = 1 sur (0,1), u(0)=u(1)=0,
avec A(x) = 1 / (2 + cos(2πx/ε)).

La solution est tracée à partir de sa FORME FERMÉE analytique (aucune
résolution EF, aucune quadrature) : classe ExactSolution de msfem_1d.problem.

Rappel de la forme close (voir problem.py) : avec k = 2π/ε,
    u(x) = 2Cx - x² + (C - x) sin(kx)/k + (1 - cos kx)/k²,
    C = (1 + sin k / k + (cos k - 1)/k²) / (2 + sin k / k).

Lancement : python plot_exact.py
"""

import numpy as np
import matplotlib.pyplot as plt

from msfem_1d.problem import exact_solution, HomogSolution

# Valeurs de ε à comparer (de la plus grande à la plus petite oscillation).
EPS_LIST = [0.25]


def plot_exact(eps_list=EPS_LIST):
    x = np.linspace(0.0, 1.0, 4000)

    # Solution homogénéisée u_0(x) = x(1-x), limite quand ε -> 0.
    u_hom = HomogSolution()
    y_hom = u_hom.value(x)

    # Grille de sous-graphes : un ε par sous-graphe.
    n = len(eps_list)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, eps in zip(axes, eps_list):
        u_ex = exact_solution(eps=eps)          # forme fermée analytique
        ax.plot(x, y_hom, "k--", lw=1.5, label="Homogénéisée  u₀ = x(1-x)")
        ax.plot(x, u_ex.value(x), "C0", lw=1.2, label="Exacte")
        ax.set_title(f"ε = {eps:.4g}")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Masque les sous-graphes inutilisés si len(eps_list) est impair.
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Solution exacte (forme fermée) pour plusieurs ε")
    plt.tight_layout()
    plt.savefig("exact_solutions.png", dpi=150)
    plt.show()
    print("Figure sauvegardée : exact_solutions.png")


if __name__ == "__main__":
    plot_exact()
