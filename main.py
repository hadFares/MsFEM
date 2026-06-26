"""
main.py — études paramétriques, comparaisons et vérifications.

Lance : python main.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# Console Windows (cp1252) : autorise les caractères type ε dans les prints.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from msfem_1d import (Mesh1D, exact_solution, HomogSolution,
                      fem_p1_solve, msfem_solve, msfem_basis)
from msfem_1d import errors, plots, timing
from msfem_1d.problem import A, f

EPS = 1 / 8

# Grille de référence fine commune pour la quadrature des normes d'erreur.
REF_NODES = np.linspace(0.0, 1.0, 4096 + 1)


# ===========================================================================
# 1. Comparaison visuelle : P1 fin vs P1 grossier vs MsFEM vs exacte
# ===========================================================================

def demo_solutions_msfem():
    u_ex = exact_solution(eps=EPS)
    u_hom = HomogSolution()

    u_p1_fine   = fem_p1_solve(Mesh1D(N=256, n=1), A, f)   # référence numérique
    u_p1_coarse = fem_p1_solve(Mesh1D(N=8,   n=1), A, f)   # P1 grossier, H ~ eps
    u_msfem     = msfem_solve(Mesh1D(N=8,   n=32), A, f)   # MsFEM, mêmes H

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plots.plot_solutions(
        {"Exacte": u_ex, "P1 fin (H=1/256)": u_p1_fine,
         "P1 grossier (H=1/8)": u_p1_coarse, "MsFEM (H=1/8, n=32)": u_msfem,
         "Homogénéisée": u_hom},
        title=f"Solutions — ε={EPS:.3g}", ax=axes[0]
    )
    plots.plot_pointwise_error(
        {"|P1 grossier - exact|": u_p1_coarse, "|MsFEM - exact|": u_msfem},
        u_ex, title="Erreur pointwise (H ~ ε)", ax=axes[1]
    )

    plt.tight_layout()
    plt.savefig("solutions_msfem.png", dpi=150)
    plt.show()
    print("Figure sauvegardée : solutions_msfem.png")


# ===========================================================================
# 2. Fonctions de base multi-échelles vs chapeaux P1
# ===========================================================================

def demo_basis():
    mesh = Mesh1D(N=4, n=200)
    basis = msfem_basis(mesh, A)

    x = np.linspace(0.0, 1.0, 1000)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for i, phi in basis.items():
        xi = mesh.nodes_coarse[i]
        line, = axes[0].plot(x, phi.value(x), label=f"Φ_{i} (x={xi:.2g})")
        # chapeau P1 classique de même support, en pointillé pour contraster
        hat = np.clip(1.0 - np.abs(x - xi) / mesh.H, 0.0, None)
        axes[0].plot(x, hat, "--", color=line.get_color(), alpha=0.5)
        axes[1].plot(x, phi.grad(x), label=f"Φ_{i}'")

    axes[0].set_title(f"Base MsFEM (traits pleins) vs chapeaux P1 (pointillés) — ε={EPS:.3g}")
    axes[0].set_xlabel("x"); axes[0].legend()
    axes[1].set_title("Dérivées Φ_i' (oscillations du flux)")
    axes[1].set_xlabel("x"); axes[1].legend()

    plt.tight_layout()
    plt.savefig("basis_msfem.png", dpi=150)
    plt.show()
    print("Figure sauvegardée : basis_msfem.png")


# ===========================================================================
# 3. Convergence en H : MsFEM vs P1 grossier
# ===========================================================================

def convergence_compare():
    u_ex = exact_solution(eps=EPS)
    N_list = [4, 8, 16, 32, 64, 128]
    n_fine = 16   # sous-maillage fin du MsFEM

    H_list = []
    err = {"MsFEM L2": [], "MsFEM H1": [], "P1 grossier L2": [], "P1 grossier H1": []}

    for N in N_list:
        u_ms = msfem_solve(Mesh1D(N=N, n=n_fine), A, f)
        u_p1 = fem_p1_solve(Mesh1D(N=N, n=1), A, f)
        H_list.append(1.0 / N)
        err["MsFEM L2"].append(errors.norm_L2(u_ms, u_ex, REF_NODES))
        err["MsFEM H1"].append(errors.norm_H1_semi(u_ms, u_ex, REF_NODES))
        err["P1 grossier L2"].append(errors.norm_L2(u_p1, u_ex, REF_NODES))
        err["P1 grossier H1"].append(errors.norm_H1_semi(u_p1, u_ex, REF_NODES))
        print(f"N={N:4d}  H={1/N:.4f}  "
              f"MsFEM L2={err['MsFEM L2'][-1]:.3e}  P1 L2={err['P1 grossier L2'][-1]:.3e}")

    fig, ax = plt.subplots(figsize=(7, 5))
    plots.plot_convergence(
        H_list, err,
        title=f"Convergence MsFEM vs P1 grossier — ε={EPS:.3g}",
        ax=ax, refs={2: "O(H²)", 1: "O(H)"}
    )
    ax.axvline(EPS, color="r", linestyle=":", alpha=0.7)
    ax.text(EPS, ax.get_ylim()[1], " H=ε", color="r", va="top")
    plt.tight_layout()
    plt.savefig("convergence_compare.png", dpi=150)
    plt.show()
    print("Figure sauvegardée : convergence_compare.png")


# ===========================================================================
# 4. Benchmark : temps de chaque solveur + diagramme travail-précision
# ===========================================================================

def benchmark():
    u_ex = exact_solution(eps=EPS)
    N_list = [8, 16, 32, 64, 128]
    n_fine = 16

    res = {"P1 grossier": {"t": [], "e": [], "dof": []},
           "P1 fin":      {"t": [], "e": [], "dof": []},
           "MsFEM":       {"t": [], "e": [], "dof": []}}

    print("\n=== Benchmark (solveur dense O(m³) — temps indicatifs) ===")
    print(f"{'N':>5} {'méthode':>12} {'DDL':>7} {'temps (ms)':>12} {'L2':>11}")
    for N in N_list:
        runs = {
            "P1 grossier": (fem_p1_solve, Mesh1D(N=N, n=1),         N - 1),
            "P1 fin":      (fem_p1_solve, Mesh1D(N=N * n_fine, n=1), N * n_fine - 1),
            "MsFEM":       (msfem_solve,  Mesh1D(N=N, n=n_fine),     N - 1),
        }
        for name, (fn, mesh, dof) in runs.items():
            sol, t = timing.time_solver(fn, mesh, A, f)
            e = errors.norm_L2(sol, u_ex, REF_NODES)
            res[name]["t"].append(t); res[name]["e"].append(e); res[name]["dof"].append(dof)
            print(f"{N:>5} {name:>12} {dof:>7} {t * 1e3:>12.3f} {e:>11.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, d in res.items():
        axes[0].loglog(N_list, np.array(d["t"]) * 1e3, marker="o", label=name)
        axes[1].loglog(d["t"], d["e"], marker="o", label=name)
    axes[0].set_xlabel("N (mailles grossières)"); axes[0].set_ylabel("temps (ms)")
    axes[0].set_title("Temps de résolution vs N"); axes[0].legend()
    axes[1].set_xlabel("temps (s)"); axes[1].set_ylabel("erreur L2")
    axes[1].set_title("Travail-précision (en bas à gauche = mieux)"); axes[1].legend()

    plt.tight_layout()
    plt.savefig("benchmark.png", dpi=150)
    plt.show()
    print("Figure sauvegardée : benchmark.png")


# ===========================================================================
# 5. Test résonant (H/eps entier) — vérification du facteur sqrt(3)/2
# ===========================================================================

def test_resonant():
    """À H = eps, le P1 grossier sous-estime l'exacte d'un facteur sqrt(3)/2."""
    N_res = int(1 / EPS)
    u_p1 = fem_p1_solve(Mesh1D(N=N_res, n=1), A, f)
    u_ms = msfem_solve(Mesh1D(N=N_res, n=32), A, f)
    u_ex = exact_solution(eps=EPS)

    x_mid = 0.5
    ratio = u_p1.value(x_mid) / u_ex.value(x_mid)
    print(f"\n=== Test résonant (H=ε={EPS:.3g}) ===")
    print(f"  u_P1(0.5)    = {u_p1.value(x_mid):.6f}")
    print(f"  u_MsFEM(0.5) = {u_ms.value(x_mid):.6f}")
    print(f"  u_exact(0.5) = {u_ex.value(x_mid):.6f}")
    print(f"  ratio P1/exact = {ratio:.6f}  (attendu ≈ {np.sqrt(3)/2:.6f})")


# ===========================================================================

if __name__ == "__main__":
    print("=== Solutions MsFEM vs P1 ===")
    demo_solutions_msfem()

    print("\n=== Fonctions de base multi-échelles ===")
    demo_basis()

    print("\n=== Convergence MsFEM vs P1 grossier ===")
    convergence_compare()

    benchmark()

    test_resonant()
