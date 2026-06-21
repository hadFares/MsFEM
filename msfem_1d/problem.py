"""
Problème modèle : -d/dx(A(x) u') = f sur (0,1), Dirichlet homogène.
A(x) = 1 / (2 + cos(2 pi x / eps)),  f = 1.

Fournit :
  - les coefficients A, f
  - la constante homogénéisée A_hom = 1/2
  - la solution homogénéisée u_hom(x) = x(1-x)  (forme close)
  - la solution exacte sous forme close analytique (ExactSolution, interface
    DiscreteSolution)
"""

import numpy as np
from .solution import DiscreteSolution


EPS_DEFAULT = 1 / 8


def A(x, eps=EPS_DEFAULT):
    return 1.0 / (2.0 + np.cos(2.0 * np.pi * x / eps))


def f(x):
    return np.ones_like(np.asarray(x, dtype=float))


# Moyenne harmonique : <1/A> = <2 + cos(...)> = 2  =>  A_hom = 1/2
A_HOM = 0.5


def u_hom(x):
    """Solution homogénéisée : -A_hom u'' = 1  =>  u_0(x) = x(1-x)."""
    x = np.asarray(x, dtype=float)
    return x * (1.0 - x)


def du_hom(x):
    x = np.asarray(x, dtype=float)
    return 1.0 - 2.0 * x


# ---------------------------------------------------------------------------
# Solution exacte sous forme close
# ---------------------------------------------------------------------------
# IMPORTANT : la forme close ci-dessous suppose f ≡ 1 (second membre constant).
#
# En 1D, -(A u')' = 1 s'intègre une fois en  A(x) u'(x) = C - x, d'où
#     u'(x) = (C - x)(2 + cos kx),     k = 2 pi / eps.
# Une seconde intégration donne (valable pour tout eps, pas seulement le cas
# résonant) :
#     u(x)  = 2 C x - x^2 + (C - x) sin(kx)/k + (1 - cos kx)/k^2.
# La constante de flux C est fixée par u(1) = 0 :
#     C = (1 + sin(k)/k + (cos k - 1)/k^2) / (2 + sin(k)/k).

class ExactSolution:
    """Solution exacte du problème modèle, conforme à l'interface DiscreteSolution."""

    def __init__(self, eps=EPS_DEFAULT):
        self.eps = eps
        k = 2.0 * np.pi / eps
        self.k = k
        # Forme close de la constante de flux C (valable pour tout eps ; en
        # régime résonant la formule redonne 0.5 à la précision machine).
        self.C = (1.0 + np.sin(k) / k + (np.cos(k) - 1.0) / k**2) \
            / (2.0 + np.sin(k) / k)

    def value(self, x):
        x = np.asarray(x, dtype=float)
        k, C = self.k, self.C
        return (2.0 * C * x - x**2
                + (C - x) * np.sin(k * x) / k
                + (1.0 - np.cos(k * x)) / k**2)

    def grad(self, x):
        x = np.asarray(x, dtype=float)
        return (self.C - x) * (2.0 + np.cos(self.k * x))


def exact_solution(eps=EPS_DEFAULT) -> ExactSolution:
    return ExactSolution(eps=eps)


class HomogSolution:
    """Solution homogénéisée u_0(x) = x(1-x), conforme à l'interface."""

    def value(self, x):
        return u_hom(x)

    def grad(self, x):
        return du_hom(x)
