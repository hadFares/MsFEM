"""
main.py — études paramétriques et vérifications.

Lance : python main.py
"""

import numpy as np
import matplotlib.pyplot as plt

from msfem_1d import Mesh1D, exact_solution, HomogSolution, fem_p1_solve
from msfem_1d import errors, plots
from msfem_1d.problem import A, f

EPS = 1 / 8


# ===========================================================================
# 1. Comparaison visuelle : P1 fin vs P1 grossier vs solution exacte
# ===========================================================================

def demo_solutions():
    u_ex = exact_solution(eps=EPS)
    u_hom = HomogSolution()

    mesh_fine   = Mesh1D(N=256, n=1)   # P1 très fin (référence numérique)
    mesh_coarse = Mesh1D(N=8,   n=1)   # P1 grossier, H = 1/8 ~ eps

    u_p1_fine   = fem_p1_solve(mesh_fine,   A, f)
    u_p1_coarse = fem_p1_solve(mesh_coarse, A, f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plots.plot_solutions(
        {"Exacte": u_ex, "P1 fin (H=1/256)": u_p1_fine,
         "P1 grossier (H=1/8)": u_p1_coarse, "Homogénéisée": u_hom},
        title=f"Solutions — ε={EPS:.3g}", ax=axes[0]
    )

    # Erreur pointwise
    x = np.linspace(0, 1, 500)
    axes[1].plot(x, np.abs(u_p1_coarse.value(x) - u_ex.value(x)), label="|P1 grossier - exact|")
    axes[1].plot(x, np.abs(u_p1_fine.value(x)   - u_ex.value(x)), label="|P1 fin - exact|")
    axes[1].set_xlabel("x")
    axes[1].set_title("Erreur pointwise")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("demo_solutions.png", dpi=150)
    plt.show()
    print("Figure sauvegardée : demo_solutions.png")


# ===========================================================================
# 2. Courbe de convergence P1 en H (norme L2 et H1)
# ===========================================================================

def convergence_p1():
    u_ex = exact_solution(eps=EPS)

    N_list = [4, 8, 16, 32, 64, 128, 256]
    H_list, err_L2, err_H1 = [], [], []

    for N in N_list:
        mesh = Mesh1D(N=N, n=1)
        u_h  = fem_p1_solve(mesh, A, f)
        # Grille fine de référence pour la quadrature
        ref_nodes = np.linspace(0, 1, 2000)
        H_list.append(mesh.H)
        err_L2.append(errors.norm_L2(u_h, u_ex, ref_nodes))
        err_H1.append(errors.norm_H1_semi(u_h, u_ex, ref_nodes))
        print(f"N={N:4d}  H={mesh.H:.4f}  L2={err_L2[-1]:.3e}  H1={err_H1[-1]:.3e}")

    fig, ax = plt.subplots(figsize=(7, 5))
    plots.plot_convergence(
        H_list,
        {"L2": err_L2, "H1 semi": err_H1},
        title=f"Convergence P1 — ε={EPS:.3g}",
        ax=ax,
        refs={2: "O(H²)", 1: "O(H)"}
    )
    # Marque la zone H ~ eps
    ax.axvline(EPS, color="r", linestyle=":", alpha=0.7, label=f"H=ε={EPS:.3g}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("convergence_p1.png", dpi=150)
    plt.show()
    print("Figure sauvegardée : convergence_p1.png")


# ===========================================================================
# 3. Test résonant (H/eps entier)
# ===========================================================================

def test_resonant():
    """
    Pour H = eps (N = 1/eps = 8), le P1 grossier doit sous-estimer la solution exacte
    d'un facteur sqrt(3)/2 ≈ 0.866 au pic.
    """
    N_res = int(1 / EPS)   # H = eps
    mesh  = Mesh1D(N=N_res, n=1)
    u_p1  = fem_p1_solve(mesh, A, f)
    u_ex  = exact_solution(eps=EPS)

    x_mid = 0.5
    ratio = u_p1.value(x_mid) / u_ex.value(x_mid)
    expected = np.sqrt(3) / 2
    print(f"\n=== Test résonant (H=ε={EPS:.3g}) ===")
    print(f"  u_P1(0.5)    = {u_p1.value(x_mid):.6f}")
    print(f"  u_exact(0.5) = {u_ex.value(x_mid):.6f}")
    print(f"  ratio        = {ratio:.6f}  (attendu ≈ {expected:.6f})")
    print(f"  écart relatif = {abs(ratio - expected)/expected * 100:.2f}%")


# ===========================================================================

if __name__ == "__main__":
    print("=== Démonstration solutions ===")
    demo_solutions()

    print("\n=== Convergence P1 ===")
    convergence_p1()

    test_resonant()
