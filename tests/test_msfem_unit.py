"""
Tests unitaires des briques du solveur MsFEM (msfem.py).

On valide isolément : la quadrature de Simpson des alpha_m, la résolution des
problèmes locaux (conditions aux limites, conservation du flux discret, cas
constant), et la contribution de rigidité élémentaire.
"""

import numpy as np
import pytest

from msfem_1d.mesh import Mesh1D
from msfem_1d.msfem import _simpson_alphas, solve_local_cell, _stiffness


# ---------------------------------------------------------------------------
# _simpson_alphas
# ---------------------------------------------------------------------------
def test_simpson_constant_is_exact():
    """Pour A constant, alpha_m = A * h sur chaque sous-intervalle."""
    y = np.linspace(0.3, 0.7, 9)        # 8 sous-intervalles
    A = lambda x: np.full_like(np.asarray(x, float), 1.3)
    alpha = _simpson_alphas(y, A)
    assert np.allclose(alpha, 1.3 * np.diff(y))


def test_simpson_exact_on_cubic():
    """Simpson intègre exactement les polynômes jusqu'au degré 3."""
    y = np.linspace(0.0, 1.0, 6)
    A = lambda x: 1.0 + 2.0 * x - 0.5 * x**2 + 4.0 * x**3
    prim = lambda x: x + x**2 - x**3 / 6.0 + x**4
    expected = prim(y[1:]) - prim(y[:-1])
    assert np.allclose(_simpson_alphas(y, A), expected, atol=1e-14)


# ---------------------------------------------------------------------------
# solve_local_cell
# ---------------------------------------------------------------------------
def test_local_boundary_conditions():
    """Les deux bases respectent les conditions de Dirichlet aux bords."""
    mesh = Mesh1D(N=4, n=10)
    A = lambda x: 1.0 / (2.0 + np.cos(2.0 * np.pi * x / 0.25))
    _, U, V = solve_local_cell(mesh, 1, A)
    assert U[0] == 1.0 and U[-1] == 0.0
    assert V[0] == 0.0 and V[-1] == 1.0


def test_local_discrete_flux_is_constant():
    """Sans source, le flux discret A phi' est constant le long de la maille."""
    mesh = Mesh1D(N=4, n=20)
    A = lambda x: 1.0 / (2.0 + np.cos(2.0 * np.pi * x / 0.2))
    alpha, U, V = solve_local_cell(mesh, 2, A)
    qU = alpha * np.diff(U)              # flux par sous-intervalle (à h près)
    qV = alpha * np.diff(V)
    assert np.ptp(qU) < 1e-12
    assert np.ptp(qV) < 1e-12


def test_local_constant_coeff_is_linear():
    """Pour A constant, les bases sont affines et V = 1 - U."""
    mesh = Mesh1D(N=4, n=8)
    A = lambda x: np.full_like(np.asarray(x, float), 0.9)
    _, U, V = solve_local_cell(mesh, 0, A)
    assert np.allclose(U, np.linspace(1.0, 0.0, mesh.n + 1))
    assert np.allclose(V, 1.0 - U)


def test_local_n1_reduces_to_hat():
    """Sans nœud intérieur (n=1), les bases sont les chapeaux P1."""
    mesh = Mesh1D(N=5, n=1)
    A = lambda x: 1.0 / (2.0 + np.cos(2.0 * np.pi * x / 0.1))
    alpha, U, V = solve_local_cell(mesh, 0, A)
    assert alpha.shape == (1,)
    assert np.array_equal(U, [1.0, 0.0])
    assert np.array_equal(V, [0.0, 1.0])


# ---------------------------------------------------------------------------
# _stiffness
# ---------------------------------------------------------------------------
def test_stiffness_constant_diagonal():
    """Sur une maille à A constant et base affine, la rigidité vaut A/H."""
    mesh = Mesh1D(N=4, n=6)            # H = 0.25
    alpha = np.full(mesh.n, 0.8 * mesh.h)
    U = np.linspace(1.0, 0.0, mesh.n + 1)
    assert _stiffness(alpha, U, U, mesh.h) == pytest.approx(0.8 / mesh.H)
