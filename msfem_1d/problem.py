"""
Problème modèle : -d/dx(A(x) u') = f sur (0,1), Dirichlet homogène.
A(x) = 1 / (2 + cos(2 pi x / eps)),  f = 1.

Fournit :
  - les coefficients A, f
  - la constante homogénéisée A_hom = 1/2
  - la solution homogénéisée u_hom(x) = x(1-x)  (forme close)
  - la solution exacte par quadrature (ExactSolution wrappée en DiscreteSolution)
"""

import numpy as np
from scipy import integrate
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
# Solution exacte par quadrature
# ---------------------------------------------------------------------------
# En 1D : A(x) u'(x) = C - x  (primitives de -f = -1)
# => u'(x) = (C - x) * (2 + cos(2 pi x / eps))
# => u(x)  = integral_0^x u'(t) dt
# C est fixé par u(1) = 0.

def _build_exact(eps=EPS_DEFAULT, n_quad=4000):
    """
    Retourne (u_func, du_func) : fonctions évaluables sur array 1D.
    n_quad : nombre de points Gauss-Legendre sur [0,1] pour les intégrales.
    """
    # Quadrature fine sur [0,1]
    t_nodes, t_weights = np.polynomial.legendre.leggauss(n_quad)
    # changement de variable [0,1] -> [-1,1] : t in [0,1] => s = 2t-1
    t01 = 0.5 * (t_nodes + 1.0)      # points dans [0,1]
    w01 = 0.5 * t_weights             # poids

    def _integrand_C(t, C):
        return (C - t) * (2.0 + np.cos(2.0 * np.pi * t / eps))

    # Cherche C tel que u(1) = integral_0^1 u'(t) dt = 0
    # u(1) = integral_0^1 (C-t)(2+cos...) dt = C*I1 - I2
    # I1 = integral_0^1 (2+cos(2pi t/eps)) dt = 2  (cos s'annule)
    # I2 = integral_0^1 t (2+cos(2pi t/eps)) dt
    I1 = np.dot(w01, 2.0 + np.cos(2.0 * np.pi * t01 / eps))
    I2 = np.dot(w01, t01 * (2.0 + np.cos(2.0 * np.pi * t01 / eps)))
    C = I2 / I1

    # u'(x) = (C - x) A_inv(x)  avec A_inv = 2 + cos(...)
    def du_func(x):
        x = np.asarray(x, dtype=float)
        return (C - x) * (2.0 + np.cos(2.0 * np.pi * x / eps))

    # u(x) = integral_0^x u'(t) dt  par quadrature sur chaque sous-intervalle
    def u_func(x):
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        result = np.empty_like(x)
        for i, xi in enumerate(x):
            if xi == 0.0:
                result[i] = 0.0
            else:
                # quadrature sur [0, xi]
                ti = 0.5 * xi * (t_nodes + 1.0)
                wi = 0.5 * xi * t_weights
                result[i] = np.dot(wi, (C - ti) * (2.0 + np.cos(2.0 * np.pi * ti / eps)))
        return result[0] if scalar else result

    return u_func, du_func, C


class ExactSolution:
    """Solution exacte du problème modèle, conforme à l'interface DiscreteSolution."""

    def __init__(self, eps=EPS_DEFAULT, n_quad=4000):
        self.eps = eps
        self._u, self._du, self.C = _build_exact(eps, n_quad)

    def value(self, x):
        return self._u(x)

    def grad(self, x):
        return self._du(x)


def exact_solution(eps=EPS_DEFAULT, n_quad=4000) -> ExactSolution:
    return ExactSolution(eps=eps, n_quad=n_quad)


class HomogSolution:
    """Solution homogénéisée u_0(x) = x(1-x), conforme à l'interface."""

    def value(self, x):
        return u_hom(x)

    def grad(self, x):
        return du_hom(x)
