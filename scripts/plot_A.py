"""
plot_A.py — visualisation du coefficient oscillant du problème modèle.

    A(x) = 1 / (2 + cos(2 pi x / eps))

C'est ce coefficient à petite échelle eps qui rend la méthode P1 classique
inefficace (il faut H << eps) et que la base MsFEM capture par des problèmes
locaux. Ce script trace A pour plusieurs valeurs de eps.

Lancement : python scripts/plot_A.py
"""

import matplotlib.pyplot as plt
import numpy as np


def A(x, eps):
    """Coefficient du problème modèle : A(x) = 1 / (2 + cos(2*pi*x/eps))."""
    return 1.0 / (2.0 + np.cos(2.0 * np.pi * x / eps))


def plot_A(eps_list):
    x = np.linspace(0.0, 1.0, 2000)
    n = len(eps_list)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, eps in zip(axes, eps_list):
        ax.plot(x, A(x, eps))
        ax.set_title(f"A(x), ε = {eps:.3g}")
        ax.set_xlabel("x")
        ax.grid(True)
    axes[0].set_ylabel("A(x)")

    fig.tight_layout()
    fig.savefig("coefficient_A.png", dpi=150)
    plt.show()
    print("Figure sauvegardée : coefficient_A.png")


if __name__ == "__main__":
    plot_A([0.1, 0.2, 0.3])
