"""
Unit tests for the building blocks of the MsFEM solver (msfem.py).

We validate in isolation: the Simpson quadrature of the alpha_m, the solve of
the local problems (boundary conditions, discrete flux conservation, constant
case), and the element stiffness contribution.
"""

import numpy as np
import pytest

from msfem_1d.mesh import Mesh1D
from msfem_1d.msfem import _simpson_alphas, solve_local_cell, _stiffness


# ---------------------------------------------------------------------------
# _simpson_alphas
# ---------------------------------------------------------------------------
def test_simpson_constant_is_exact():
    """For constant A, alpha_m = A * h on each sub-interval."""
    y = np.linspace(0.3, 0.7, 9)        # 8 sub-intervals
    A = lambda x: np.full_like(np.asarray(x, float), 1.3)
    alpha = _simpson_alphas(y, A)
    assert np.allclose(alpha, 1.3 * np.diff(y))


def test_simpson_exact_on_cubic():
    """Simpson integrates polynomials up to degree 3 exactly."""
    y = np.linspace(0.0, 1.0, 6)
    A = lambda x: 1.0 + 2.0 * x - 0.5 * x**2 + 4.0 * x**3
    prim = lambda x: x + x**2 - x**3 / 6.0 + x**4
    expected = prim(y[1:]) - prim(y[:-1])
    assert np.allclose(_simpson_alphas(y, A), expected, atol=1e-14)


# ---------------------------------------------------------------------------
# solve_local_cell
# ---------------------------------------------------------------------------
def test_local_boundary_conditions():
    """Both bases satisfy the Dirichlet conditions at the boundaries."""
    mesh = Mesh1D(N=4, n=10)
    A = lambda x: 1.0 / (2.0 + np.cos(2.0 * np.pi * x / 0.25))
    _, U, V = solve_local_cell(mesh, 1, A)
    assert U[0] == 1.0 and U[-1] == 0.0
    assert V[0] == 0.0 and V[-1] == 1.0


def test_local_discrete_flux_is_constant():
    """Without a source, the discrete flux A phi' is constant along the cell."""
    mesh = Mesh1D(N=4, n=20)
    A = lambda x: 1.0 / (2.0 + np.cos(2.0 * np.pi * x / 0.2))
    alpha, U, V = solve_local_cell(mesh, 2, A)
    qU = alpha * np.diff(U)              # flux per sub-interval (up to h)
    qV = alpha * np.diff(V)
    assert np.ptp(qU) < 1e-12
    assert np.ptp(qV) < 1e-12


def test_local_constant_coeff_is_linear():
    """For constant A, the bases are affine and V = 1 - U."""
    mesh = Mesh1D(N=4, n=8)
    A = lambda x: np.full_like(np.asarray(x, float), 0.9)
    _, U, V = solve_local_cell(mesh, 0, A)
    assert np.allclose(U, np.linspace(1.0, 0.0, mesh.n + 1))
    assert np.allclose(V, 1.0 - U)


def test_local_n1_reduces_to_hat():
    """Without an interior node (n=1), the bases are the P1 hats."""
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
    """On a cell with constant A and affine basis, the stiffness equals A/H."""
    mesh = Mesh1D(N=4, n=6)            # H = 0.25
    alpha = np.full(mesh.n, 0.8 * mesh.h)
    U = np.linspace(1.0, 0.0, mesh.n + 1)
    assert _stiffness(alpha, U, U, mesh.h) == pytest.approx(0.8 / mesh.H)
