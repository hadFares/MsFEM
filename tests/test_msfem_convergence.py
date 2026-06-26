"""
Tests end-to-end du pipeline MsFEM complet (problèmes locaux + assemblage
global + résolution + reconstruction).

On vérifie : la réduction exacte au P1 grossier en coefficient constant,
l'exactitude nodale en régime résonant, la supériorité sur le P1 grossier là
où celui-ci s'effondre (H ~ eps), et la convergence en H.
"""

import numpy as np

from msfem_1d.mesh import Mesh1D
from msfem_1d import problem, fem_p1, msfem
from msfem_1d import errors


# Grille de référence fine commune (multiple de tous les maillages testés).
REF_NODES = np.linspace(0.0, 1.0, 4096 + 1)


class _ConstSolution:
    """Solution exacte de -c u'' = 1, Dirichlet homogène : u = x(1-x)/(2c)."""

    def __init__(self, c):
        self.c = c

    def value(self, x):
        x = np.asarray(x, float)
        return x * (1.0 - x) / (2.0 * self.c)

    def grad(self, x):
        x = np.asarray(x, float)
        return (1.0 - 2.0 * x) / (2.0 * self.c)


def test_constant_coeff_reduces_to_coarse_p1():
    """En coefficient constant, MsFEM(N,n) coïncide avec le P1 grossier."""
    c = 0.7
    A = lambda x: np.full_like(np.asarray(x, float), c)
    f = problem.f
    ms = msfem.solve(Mesh1D(N=10, n=5), A, f)
    p1 = fem_p1.solve(Mesh1D(N=10, n=1), A, f)         # P1 sur le maillage grossier
    xs = np.linspace(0.0, 1.0, 257)
    assert np.max(np.abs(ms.value(xs) - p1.value(xs))) < 1e-12
    # nodalement exact aux nœuds grossiers
    xc = Mesh1D(N=10).nodes_coarse
    assert np.max(np.abs(ms.value(xc) - _ConstSolution(c).value(xc))) < 1e-12


def test_resonant_nodally_exact():
    """En régime résonant (H/eps entier), MsFEM est exact aux nœuds grossiers."""
    eps = problem.EPS_DEFAULT                          # 1/8
    A, f = lambda x: problem.A(x, eps), problem.f
    exact = problem.exact_solution(eps)
    mesh = Mesh1D(N=8, n=64)                            # H = eps
    ms = msfem.solve(mesh, A, f)
    err = np.max(np.abs(ms.value(mesh.nodes_coarse) - exact.value(mesh.nodes_coarse)))
    assert err < 1e-4


def test_resonant_node_error_decreases_with_fine_mesh():
    """L'erreur nodale résonante chute quand le maillage fin se raffine (Simpson)."""
    eps = problem.EPS_DEFAULT
    A, f = lambda x: problem.A(x, eps), problem.f
    exact = problem.exact_solution(eps)
    errs = []
    for n in (8, 16, 32, 64):
        mesh = Mesh1D(N=8, n=n)
        ms = msfem.solve(mesh, A, f)
        errs.append(np.max(np.abs(ms.value(mesh.nodes_coarse)
                                  - exact.value(mesh.nodes_coarse))))
    errs = np.array(errs)
    assert np.all(np.diff(errs) < 0.0)


def test_msfem_beats_coarse_p1_in_resonant_regime():
    """À H ~ eps, MsFEM est nettement plus précis que le P1 grossier."""
    eps = problem.EPS_DEFAULT
    A, f = lambda x: problem.A(x, eps), problem.f
    exact = problem.exact_solution(eps)
    ms = msfem.solve(Mesh1D(N=8, n=64), A, f)
    p1 = fem_p1.solve(Mesh1D(N=8, n=1), A, f)
    e_ms = errors.norm_L2(ms, exact, REF_NODES)
    e_p1 = errors.norm_L2(p1, exact, REF_NODES)
    assert e_ms < e_p1 / 3.0


def test_msfem_H_convergence():
    """À maillage fin fixe (h constant), l'erreur L2 décroît quand H diminue."""
    eps = 1.0 / 16.0
    A, f = lambda x: problem.A(x, eps), problem.f
    exact = problem.exact_solution(eps)
    errs = []
    for N in (4, 8, 16, 32):
        mesh = Mesh1D(N=N, n=1024 // N)                 # h = 1/1024 constant
        ms = msfem.solve(mesh, A, f)
        errs.append(errors.norm_L2(ms, exact, REF_NODES))
    errs = np.array(errs)
    assert np.all(np.diff(errs) < 0.0)
    assert errs[0] / errs[-1] > 4.0
