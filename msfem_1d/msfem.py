"""
1D MsFEM assembly and solve.

Two stages:
  1. a local problem per coarse cell -> multiscale basis functions
     (solved with P1 on the fine mesh);
  2. assembly of the global Galerkin system on these basis functions.

The integrals of the coefficient A over each fine sub-interval (alpha_m) are
approximated with Simpson's rule, then reused at both stages.

solve(mesh, A_func, f_func) -> P1Interpolant
"""

import numpy as np
from .mesh import Mesh1D
from .solution import P1Interpolant


def _simpson_alphas(y, A_func):
    """
    Integrate A over each fine sub-interval with Simpson's rule.

    Parameters
    ----------
    y : ndarray, shape (n+1,)
        Fine nodes of one coarse cell.
    A_func : callable
        Coefficient A(x), vectorized.

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
    Solve the two homogeneous local problems on the i-th coarse cell.

    On K_i = [x_i, x_{i+1}], each basis function solves -(A phi')' = 0 with P1
    on the fine mesh, using a lifting of the Dirichlet condition.

    Parameters
    ----------
    mesh : Mesh1D
        Two-level mesh (coarse H, fine h = H/n).
    i : int
        Index of the coarse cell (0-indexed).
    A_func : callable
        Coefficient A(x), vectorized.

    Returns
    -------
    alpha : ndarray, shape (n,)
        Integrals of A over the fine sub-intervals (reused in the global stage).
    U : ndarray, shape (n+1,)
        Left-anchored basis: fine nodal values, U[0]=1, U[n]=0.
    V : ndarray, shape (n+1,)
        Right-anchored basis: fine nodal values, V[0]=0, V[n]=1.
    """
    y = mesh.fine_nodes_in_element(i)
    n = mesh.n
    alpha = _simpson_alphas(y, A_func)

    U = np.zeros(n + 1)
    V = np.zeros(n + 1)
    U[0] = 1.0
    V[n] = 1.0

    # No interior node (n == 1): the basis reduces to standard P1.
    if n > 1:
        # Tridiagonal system (n-1) x (n-1); the 1/h^2 factor cancels out.
        diag = alpha[:-1] + alpha[1:]
        off = -alpha[1:-1]
        K = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)

        # Right-hand sides from the lifting (left: phi(x_i)=1; right: =1 on the right).
        b = np.zeros((n - 1, 2))
        b[0, 0] = alpha[0]
        b[-1, 1] = alpha[-1]

        sol = np.linalg.solve(K, b)
        U[1:-1] = sol[:, 0]
        V[1:-1] = sol[:, 1]

    return alpha, U, V


def _stiffness(alpha, W1, W2, h):
    """
    Contribution \\int A W1' W2' dx over a cell (piecewise P1 derivatives).

    Parameters
    ----------
    alpha : ndarray, shape (n,)
        Integrals of A over the fine sub-intervals.
    W1, W2 : ndarray, shape (n+1,)
        Fine nodal values of the two functions.
    h : float
        Fine step.

    Returns
    -------
    float
        Sum of the stiffness contributions over the cell.
    """
    return np.sum(alpha * np.diff(W1) * np.diff(W2)) / h**2


def solve(mesh: Mesh1D, A_func, f_func) -> P1Interpolant:
    """
    Solve -d/dx(A u') = f with MsFEM, homogeneous Dirichlet u(0)=u(1)=0.

    The multiscale basis functions are computed cell by cell
    (solve_local_cell), then the global tridiagonal system (N-1) x (N-1) is
    assembled and solved. The solution is rebuilt on the fine nodes.

    Parameters
    ----------
    mesh : Mesh1D
        Two-level mesh.
    A_func : callable
        Coefficient A(x), vectorized.
    f_func : callable
        Right-hand side f(x), vectorized.

    Returns
    -------
    P1Interpolant
        MsFEM solution sampled exactly on the fine nodes.
    """
    N, n, h = mesh.N, mesh.n, mesh.h

    # Stage 1: solve the local problems on each coarse cell.
    alpha = [None] * N
    U = [None] * N   # left-anchored basis (1 -> 0)
    V = [None] * N   # right-anchored basis (0 -> 1)
    for i in range(N):
        alpha[i], U[i], V[i] = solve_local_cell(mesh, i, A_func)

    # Stage 2: assemble the global tridiagonal system on the interior nodes.
    K = np.zeros((N - 1, N - 1))
    F = np.zeros(N - 1)
    for j in range(1, N):
        p = j - 1   # matrix index of interior node j
        # Phi_j lives on K_{j-1} (right-anchored) and K_j (left-anchored).
        K[p, p] = (_stiffness(alpha[j - 1], V[j - 1], V[j - 1], h)
                   + _stiffness(alpha[j], U[j], U[j], h))
        if j < N - 1:
            # Overlap of Phi_j and Phi_{j+1} on the shared cell K_j.
            kij = _stiffness(alpha[j], U[j], V[j], h)
            K[p, p + 1] = kij
            K[p + 1, p] = kij

        # Right-hand side: \\int f Phi_j over the two cells (trapezoid on fine mesh).
        yl = mesh.fine_nodes_in_element(j - 1)
        yr = mesh.fine_nodes_in_element(j)
        F[p] = (np.trapezoid(f_func(yl) * V[j - 1], dx=h)
                + np.trapezoid(f_func(yr) * U[j], dx=h))

    U_coarse = np.zeros(N + 1)   # values at coarse nodes, boundaries at 0
    U_coarse[1:N] = np.linalg.solve(K, F)

    # Reconstruction: u_H = sum_i U_coarse[i] Phi_i, sampled on the fine nodes.
    u_fine = np.zeros(N * n + 1)
    for i in range(N):
        seg = slice(i * n, i * n + n + 1)
        u_fine[seg] = U_coarse[i] * U[i] + U_coarse[i + 1] * V[i]

    return P1Interpolant(mesh.nodes_fine, u_fine)


def msfem_basis(mesh: Mesh1D, A_func) -> dict:
    """
    Build the multiscale basis functions Phi_i (interior nodes).

    Phi_i has support [x_{i-1}, x_{i+1}]: right-anchored (0->1) on cell i-1,
    left-anchored (1->0) on cell i. Useful for visualization.

    Parameters
    ----------
    mesh : Mesh1D
        Two-level mesh.
    A_func : callable
        Coefficient A(x), vectorized.

    Returns
    -------
    dict[int, P1Interpolant]
        For each interior node i (1..N-1), Phi_i sampled on all fine nodes
        (zero outside its support [x_{i-1}, x_{i+1}]).
    """
    N, n = mesh.N, mesh.n

    # Local solve per cell (reused by the adjacent Phi_i).
    U = [None] * N   # left-anchored basis (1 -> 0)
    V = [None] * N   # right-anchored basis (0 -> 1)
    for c in range(N):
        _, U[c], V[c] = solve_local_cell(mesh, c, A_func)

    basis = {}
    for i in range(1, N):
        # Support over all fine nodes: zero elsewhere (avoids extrapolation).
        values = np.zeros(N * n + 1)
        values[(i - 1) * n : i * n + 1] = V[i - 1]   # cell i-1 (0 -> 1)
        values[i * n : (i + 1) * n + 1] = U[i]       # cell i   (1 -> 0)
        basis[i] = P1Interpolant(mesh.nodes_fine, values)

    return basis
