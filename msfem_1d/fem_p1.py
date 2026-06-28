"""
Standard P1 assembly and solve on a uniform mesh.

solve(mesh, problem) -> P1Interpolant

Element assembly on each cell [x_i, x_{i+1}]:
  K_loc = (1/h) * A_mean * [[1, -1], [-1, 1]]
  F_loc = (h/2) * f_mean * [1, 1]

A_mean and f_mean are approximated by Gauss quadrature with n_gauss points
on each fine cell (if mesh.n > 1) or coarse cell (if mesh.n == 1).
"""

import numpy as np
from .mesh import Mesh1D
from .solution import P1Interpolant


def solve(mesh: Mesh1D, A_func, f_func, n_gauss: int = 5) -> P1Interpolant:
  """
  Solve -d/dx(A u') = f with P1 elements on the fine mesh of mesh.

  If mesh.n == 1, the P1 elements are the coarse cells.
  If mesh.n > 1, the P1 elements are the fine cells (reference fine P1).

  Return a P1Interpolant on the fine nodes.
  """
  nodes = mesh.nodes_fine        # N*n + 1 nodes
  M = len(nodes) - 1             # number of elements

  # Stiffness matrix K (symmetric) and right-hand side vector F, size = node count
  K = np.zeros((len(nodes), len(nodes)))
  F = np.zeros(len(nodes))

  # Gauss-Legendre points and weights on [-1,1]
  xi_g, w_g = np.polynomial.legendre.leggauss(n_gauss)

  for e in range(M):
    xa, xb = nodes[e], nodes[e + 1]
    h = xb - xa
    # Change of variable: t in [xa,xb] <-> s in [-1,1]
    t_pts = 0.5 * (xa + xb) + 0.5 * h * xi_g   # Gauss points in [xa,xb]
    w_pts = 0.5 * h * w_g

    A_vals = A_func(t_pts)
    f_vals = f_func(t_pts)

    # P1 basis functions: phi_0 = (xb-t)/h,  phi_1 = (t-xa)/h
    # dphi_i' = ±1/h
    # K_loc[i,j] = integral_xa^xb A * (±1/h)^2 dt  =  (1/h^2) * dot(w,A)
    # dot(w_pts, A_vals) ~ integral_xa^xb A dt  (the Jacobian h/2 is in w_pts)
    k00 = np.dot(w_pts, A_vals) / h**2
    k11 = k00            # phi_1'^2 = phi_0'^2 = 1/h^2
    k01 = -k00           # phi_0' phi_1' = -1/h^2

    # Scatter the 2x2 local matrix into the global matrix
    K[e,   e  ] += k00
    K[e,   e+1] += k01
    K[e+1, e  ] += k01
    K[e+1, e+1] += k11

    # F_loc[i] = integral f * phi_i dt
    phi0 = (xb - t_pts) / h
    phi1 = (t_pts - xa) / h
    F[e  ] += np.dot(w_pts, f_vals * phi0)
    F[e+1] += np.dot(w_pts, f_vals * phi1)

  # Homogeneous Dirichlet boundary conditions u(0)=u(1)=0 by elimination:
  # we only solve on the interior nodes (the boundary stays at 0).
  interior = slice(1, len(nodes) - 1)
  K_int = K[interior, interior]
  F_int = F[interior]

  U_int = np.linalg.solve(K_int, F_int)

  U = np.zeros(len(nodes))   # boundaries already at 0
  U[interior] = U_int

  return P1Interpolant(nodes, U)
