"""
Assemblage et résolution P1 classique sur maillage uniforme.

solve(mesh, problem) -> P1Interpolant

Assemblage élémentaire sur chaque maille [x_i, x_{i+1}] :
  K_loc = (1/h) * A_moy * [[1, -1], [-1, 1]]
  F_loc = (h/2) * f_moy * [1, 1]

A_moy et f_moy sont approchés par quadrature de Gauss à n_gauss points
sur chaque maille fine (si mesh.n > 1) ou grossière (si mesh.n == 1).
"""

import numpy as np
from .mesh import Mesh1D
from .solution import P1Interpolant


def solve(mesh: Mesh1D, A_func, f_func, n_gauss: int = 5) -> P1Interpolant:
  """
  Résout -d/dx(A u') = f par éléments P1 sur le maillage fin de mesh.

  Si mesh.n == 1, les éléments P1 sont les mailles grossières.
  Si mesh.n > 1, les éléments P1 sont les mailles fines (P1 fin de référence).

  Retourne un P1Interpolant sur les nœuds fins.
  """
  nodes = mesh.nodes_fine        # N*n + 1 nœuds
  M = len(nodes) - 1             # nombre d'éléments

  # Matrice de rigidité K (symétrique) et vecteur second membre F, taille = nb nœuds
  K = np.zeros((len(nodes), len(nodes)))
  F = np.zeros(len(nodes))

  # Points et poids Gauss-Legendre sur [-1,1]
  xi_g, w_g = np.polynomial.legendre.leggauss(n_gauss)

  for e in range(M):
    xa, xb = nodes[e], nodes[e + 1]
    h = xb - xa
    # Changement de variable : t in [xa,xb] <-> s in [-1,1]
    t_pts = 0.5 * (xa + xb) + 0.5 * h * xi_g   # points de Gauss dans [xa,xb]
    w_pts = 0.5 * h * w_g

    A_vals = A_func(t_pts)
    f_vals = f_func(t_pts)

    # Fonctions de base P1 : phi_0 = (xb-t)/h,  phi_1 = (t-xa)/h
    # dphi_i' = ±1/h
    # K_loc[i,j] = integral_xa^xb A * (±1/h)^2 dt  =  (1/h^2) * dot(w,A)
    # dot(w_pts, A_vals) ≈ integral_xa^xb A dt  (le Jacobien h/2 est dans w_pts)
    k00 = np.dot(w_pts, A_vals) / h**2
    k11 = k00            # phi_1'^2 = phi_0'^2 = 1/h^2
    k01 = -k00           # phi_0' phi_1' = -1/h^2

    # Dispersion de la matrice locale 2x2 vers la matrice globale
    K[e,   e  ] += k00
    K[e,   e+1] += k01
    K[e+1, e  ] += k01
    K[e+1, e+1] += k11

    # F_loc[i] = integral f * phi_i dt
    phi0 = (xb - t_pts) / h
    phi1 = (t_pts - xa) / h
    F[e  ] += np.dot(w_pts, f_vals * phi0)
    F[e+1] += np.dot(w_pts, f_vals * phi1)

  # Conditions aux limites Dirichlet homogènes u(0)=u(1)=0 par élimination :
  # on ne résout que sur les nœuds intérieurs (le bord reste à 0).
  interior = slice(1, len(nodes) - 1)
  K_int = K[interior, interior]
  F_int = F[interior]

  U_int = np.linalg.solve(K_int, F_int)

  U = np.zeros(len(nodes))   # bords déjà à 0
  U[interior] = U_int

  return P1Interpolant(nodes, U)
