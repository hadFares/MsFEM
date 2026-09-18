"""
Model problem: -d/dx(A(x) u') = f on (0,1), homogeneous Dirichlet.
A(x) = 1 / (2 + cos(2 pi x / eps)),  f = 1.

Provides:
  - the coefficients A, f
  - the homogenized constant A_hom = 1/2
  - the homogenized solution u_hom(x) = x(1-x)  (closed form)
  - the exact solution in analytic closed form (ExactSolution, with the
    DiscreteSolution interface)
"""

import numpy as np
from .solution import DiscreteSolution


EPS_DEFAULT = 1 / 16


def A(x, eps=EPS_DEFAULT):
    return 1.0 / (2.0 + np.cos(2.0 * np.pi * x / eps))


def f(x):
    return np.ones_like(np.asarray(x, dtype=float))


# Harmonic mean: <1/A> = <2 + cos(...)> = 2  =>  A_hom = 1/2
A_HOM = 0.5


def u_hom(x):
    """Homogenized solution: -A_hom u'' = 1  =>  u_0(x) = x(1-x)."""
    x = np.asarray(x, dtype=float)
    return x * (1.0 - x)


def du_hom(x):
    x = np.asarray(x, dtype=float)
    return 1.0 - 2.0 * x


# ---------------------------------------------------------------------------
# Exact solution in closed form
# ---------------------------------------------------------------------------
# IMPORTANT: the closed form below assumes f == 1 (constant right-hand side).
#
# In 1D, -(A u')' = 1 integrates once into  A(x) u'(x) = C - x, hence
#     u'(x) = (C - x)(2 + cos kx),     k = 2 pi / eps.
# A second integration gives (valid for any eps, not only the resonant case):
#     u(x)  = 2 C x - x^2 + (C - x) sin(kx)/k + (1 - cos kx)/k^2.
# The flux constant C is set by u(1) = 0:
#     C = (1 + sin(k)/k + (cos k - 1)/k^2) / (2 + sin(k)/k).

class ExactSolution:
    """Exact solution of the model problem, matching the DiscreteSolution interface."""

    def __init__(self, eps=EPS_DEFAULT):
        self.eps = eps
        k = 2.0 * np.pi / eps
        self.k = k
        # Closed form of the flux constant C (valid for any eps; in the
        # resonant case the formula returns 0.5 up to machine precision).
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
    """Homogenized solution u_0(x) = x(1-x), matching the interface."""

    def value(self, x):
        return u_hom(x)

    def grad(self, x):
        return du_hom(x)
