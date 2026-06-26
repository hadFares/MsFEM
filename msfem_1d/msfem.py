"""
Assemblage et résolution MsFEM 1D.

Deux étages :
  1. problème local par maille grossière -> fonctions de base multi-échelles
     (résolues par P1 sur le maillage fin) ;
  2. assemblage du système global de Galerkin sur ces fonctions de base.

Les intégrales du coefficient A sur chaque sous-intervalle fin (alpha_m) sont
approchées par la formule de Simpson, puis réutilisées aux deux étages.

solve(mesh, A_func, f_func) -> P1Interpolant
"""

import numpy as np
from .mesh import Mesh1D
from .solution import P1Interpolant


def _simpson_alphas(y, A_func):
    """
    Intègre A sur chaque sous-intervalle fin par la formule de Simpson.

    Parameters
    ----------
    y : ndarray, shape (n+1,)
        Nœuds fins d'une maille grossière.
    A_func : callable
        Coefficient A(x), vectorisé.

    Returns
    -------
    alpha : ndarray, shape (n,)
        alpha[m] = \\int_{y[m]}^{y[m+1]} A(x) dx, m = 0..n-1.
    """
    yl, yr = y[:-1], y[1:]
    h = yr - yl
    return (h / 6.0) * (A_func(yl) + 4.0 * A_func(0.5 * (yl + yr)) + A_func(yr))


def solve_local_cell(mesh: Mesh1D, i: int, A_func):
    """
    Résout les deux problèmes locaux homogènes sur la i-ème maille grossière.

    Sur K_i = [x_i, x_{i+1}], chaque fonction de base résout -(A phi')' = 0 par
    P1 sur le maillage fin, avec relèvement de la condition de Dirichlet.

    Parameters
    ----------
    mesh : Mesh1D
        Maillage à deux niveaux (grossier H, fin h = H/n).
    i : int
        Indice de la maille grossière (0-indexé).
    A_func : callable
        Coefficient A(x), vectorisé.

    Returns
    -------
    alpha : ndarray, shape (n,)
        Intégrales de A sur les sous-intervalles fins (réutilisées au global).
    U : ndarray, shape (n+1,)
        Base ancrée à gauche : valeurs nodales fines, U[0]=1, U[n]=0.
    V : ndarray, shape (n+1,)
        Base ancrée à droite : valeurs nodales fines, V[0]=0, V[n]=1.
    """
    y = mesh.fine_nodes_in_element(i)
    n = mesh.n
    alpha = _simpson_alphas(y, A_func)

    U = np.zeros(n + 1)
    V = np.zeros(n + 1)
    U[0] = 1.0
    V[n] = 1.0

    # Pas de nœud intérieur (n == 1) : la base se réduit au P1 classique.
    if n > 1:
        # Système tridiagonal (n-1) x (n-1) ; le facteur 1/h^2 se simplifie.
        diag = alpha[:-1] + alpha[1:]
        off = -alpha[1:-1]
        K = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)

        # Seconds membres issus du relèvement (gauche : phi(x_i)=1 ; droite : =1 à droite).
        b = np.zeros((n - 1, 2))
        b[0, 0] = alpha[0]
        b[-1, 1] = alpha[-1]

        sol = np.linalg.solve(K, b)
        U[1:-1] = sol[:, 0]
        V[1:-1] = sol[:, 1]

    return alpha, U, V


def _stiffness(alpha, W1, W2, h):
    """
    Contribution \\int A W1' W2' dx sur une maille (dérivées P1 par morceaux).

    Parameters
    ----------
    alpha : ndarray, shape (n,)
        Intégrales de A sur les sous-intervalles fins.
    W1, W2 : ndarray, shape (n+1,)
        Valeurs nodales fines des deux fonctions.
    h : float
        Pas fin.

    Returns
    -------
    float
        Somme des contributions de rigidité sur la maille.
    """
    return np.sum(alpha * np.diff(W1) * np.diff(W2)) / h**2


def solve(mesh: Mesh1D, A_func, f_func) -> P1Interpolant:
    """
    Résout -d/dx(A u') = f par MsFEM, Dirichlet homogène u(0)=u(1)=0.

    Les fonctions de base multi-échelles sont calculées maille par maille
    (solve_local_cell), puis le système global tridiagonal (N-1) x (N-1) est
    assemblé et résolu. La solution est reconstruite sur les nœuds fins.

    Parameters
    ----------
    mesh : Mesh1D
        Maillage à deux niveaux.
    A_func : callable
        Coefficient A(x), vectorisé.
    f_func : callable
        Second membre f(x), vectorisé.

    Returns
    -------
    P1Interpolant
        Solution MsFEM échantillonnée exactement sur les nœuds fins.
    """
    N, n, h = mesh.N, mesh.n, mesh.h

    # Étage 1 : résolution des problèmes locaux sur chaque maille grossière.
    alpha = [None] * N
    U = [None] * N   # base ancrée à gauche (1 -> 0)
    V = [None] * N   # base ancrée à droite (0 -> 1)
    for i in range(N):
        alpha[i], U[i], V[i] = solve_local_cell(mesh, i, A_func)

    # Étage 2 : assemblage du système global tridiagonal sur les nœuds intérieurs.
    K = np.zeros((N - 1, N - 1))
    F = np.zeros(N - 1)
    for j in range(1, N):
        p = j - 1   # indice matriciel du nœud intérieur j
        # Phi_j vit sur K_{j-1} (ancrée à droite) et K_j (ancrée à gauche).
        K[p, p] = (_stiffness(alpha[j - 1], V[j - 1], V[j - 1], h)
                   + _stiffness(alpha[j], U[j], U[j], h))
        if j < N - 1:
            # Recouvrement de Phi_j et Phi_{j+1} sur la maille commune K_j.
            kij = _stiffness(alpha[j], U[j], V[j], h)
            K[p, p + 1] = kij
            K[p + 1, p] = kij

        # Second membre : \\int f Phi_j sur les deux mailles (trapèzes sur maillage fin).
        yl = mesh.fine_nodes_in_element(j - 1)
        yr = mesh.fine_nodes_in_element(j)
        F[p] = (np.trapezoid(f_func(yl) * V[j - 1], dx=h)
                + np.trapezoid(f_func(yr) * U[j], dx=h))

    U_coarse = np.zeros(N + 1)   # valeurs aux nœuds grossiers, bords à 0
    U_coarse[1:N] = np.linalg.solve(K, F)

    # Reconstruction : u_H = sum_i U_coarse[i] Phi_i, échantillonnée sur les nœuds fins.
    u_fine = np.zeros(N * n + 1)
    for i in range(N):
        seg = slice(i * n, i * n + n + 1)
        u_fine[seg] = U_coarse[i] * U[i] + U_coarse[i + 1] * V[i]

    return P1Interpolant(mesh.nodes_fine, u_fine)


def msfem_basis(mesh: Mesh1D, A_func) -> dict:
    """
    Construit les fonctions de base multi-échelles Phi_i (nœuds intérieurs).

    Phi_i a pour support [x_{i-1}, x_{i+1}] : ancrée à droite (0->1) sur la maille
    i-1, ancrée à gauche (1->0) sur la maille i. Utile pour la visualisation.

    Parameters
    ----------
    mesh : Mesh1D
        Maillage à deux niveaux.
    A_func : callable
        Coefficient A(x), vectorisé.

    Returns
    -------
    dict[int, P1Interpolant]
        Pour chaque nœud intérieur i (1..N-1), Phi_i échantillonnée sur tous les
        nœuds fins (nulle hors de son support [x_{i-1}, x_{i+1}]).
    """
    N, n = mesh.N, mesh.n

    # Résolution locale par maille (réutilisée par les Phi_i adjacentes).
    U = [None] * N   # base ancrée à gauche (1 -> 0)
    V = [None] * N   # base ancrée à droite (0 -> 1)
    for c in range(N):
        _, U[c], V[c] = solve_local_cell(mesh, c, A_func)

    basis = {}
    for i in range(1, N):
        # Support sur tous les nœuds fins : nul ailleurs (évite l'extrapolation).
        values = np.zeros(N * n + 1)
        values[(i - 1) * n : i * n + 1] = V[i - 1]   # maille i-1 (0 -> 1)
        values[i * n : (i + 1) * n + 1] = U[i]       # maille i   (1 -> 0)
        basis[i] = P1Interpolant(mesh.nodes_fine, values)

    return basis
