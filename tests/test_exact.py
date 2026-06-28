"""
Tests for the closed-form exact solution (ExactSolution).

We check that the closed form matches a reference computation by quadrature
(the old engine, reimplemented here as a test helper), and that ExactSolution
is consistent with homogenization when eps -> 0.
"""

import numpy as np
import pytest

from msfem_1d import ExactSolution


# ---------------------------------------------------------------------------
# Test helper: old Gauss-Legendre quadrature engine
# ---------------------------------------------------------------------------
def _exact_by_quadrature(eps, n_quad=4000):
    """
    Rebuild (u_func, C) by quadrature, independently of the closed form.
    Serves as a reference to validate ExactSolution.

    u'(x) = (C - x)(2 + cos(2 pi x / eps)),  C set by u(1) = 0.
    """
    t_nodes, t_weights = np.polynomial.legendre.leggauss(n_quad)
    t01 = 0.5 * (t_nodes + 1.0)   # points in [0,1]
    w01 = 0.5 * t_weights         # matching weights

    # u(1) = C * I1 - I2 = 0  with
    #   I1 = ∫_0^1 (2 + cos(2π t/eps)) dt
    #   I2 = ∫_0^1 t (2 + cos(2π t/eps)) dt
    I1 = np.dot(w01, 2.0 + np.cos(2.0 * np.pi * t01 / eps))
    I2 = np.dot(w01, t01 * (2.0 + np.cos(2.0 * np.pi * t01 / eps)))
    C = I2 / I1

    def u_func(x):
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        result = np.empty_like(x)
        for i, xi in enumerate(x):
            if xi == 0.0:
                result[i] = 0.0
            else:
                # quadrature of u'(t) on [0, xi]
                ti = 0.5 * xi * (t_nodes + 1.0)
                wi = 0.5 * xi * t_weights
                result[i] = np.dot(
                    wi, (C - ti) * (2.0 + np.cos(2.0 * np.pi * ti / eps))
                )
        return result[0] if scalar else result

    return u_func, C


@pytest.mark.parametrize("eps", [1.0 / 8.0, 1.0 / 7.3])
def test_closed_form_matches_quadrature(eps):
    """The closed form matches the reference quadrature to 1e-10."""
    exact = ExactSolution(eps)
    u_ref, C_ref = _exact_by_quadrature(eps)

    # flux constant
    assert abs(exact.C - C_ref) < 1e-10

    # values on a fine grid
    grid = np.linspace(0.0, 1.0, 257)
    assert np.max(np.abs(exact.value(grid) - u_ref(grid))) < 1e-10

    # homogeneous Dirichlet conditions
    assert abs(exact.value(0.0)) < 1e-12
    assert abs(exact.value(1.0)) < 1e-10


def test_homogenization_consistency():
    """
    When eps -> 0: C -> 1/2 and u -> u_0(x) = x(1-x), with the gap decreasing
    as O(eps).
    """
    epss = np.array([1.0 / 8.0, 1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0])
    grid = np.linspace(0.0, 1.0, 513)
    u0 = grid * (1.0 - grid)

    errs = []
    for eps in epss:
        exact = ExactSolution(eps)
        assert abs(exact.C - 0.5) < eps  # C -> 1/2
        errs.append(np.max(np.abs(exact.value(grid) - u0)))
    errs = np.array(errs)

    # the error decreases
    assert np.all(np.diff(errs) < 0.0)

    # rate ~ O(eps): halving eps roughly halves the error.
    ratios = errs[:-1] / errs[1:]
    assert np.all(ratios > 1.7)   # close to 2, margin for higher-order terms.


def test_resonant_eps_gives_half():
    """In the resonant case (eps = 1/n integer), C = 1/2 up to machine precision."""
    for n in (4, 8, 16):
        assert abs(ExactSolution(1.0 / n).C - 0.5) < 1e-12
