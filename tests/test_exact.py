"""
Tests de la solution exacte sous forme close (ExactSolution).

On vérifie que la forme close coïncide avec un calcul de référence par
quadrature (l'ancien moteur, réimplémenté ici comme helper de test), et que
ExactSolution est cohérente avec l'homogénéisation quand eps -> 0.
"""

import numpy as np
import pytest

from msfem_1d import ExactSolution


# ---------------------------------------------------------------------------
# Helper de test : ancien moteur de quadrature Gauss-Legendre
# ---------------------------------------------------------------------------
def _exact_by_quadrature(eps, n_quad=4000):
    """
    Reconstruit (u_func, C) par quadrature, indépendamment de la forme close.
    Sert de référence pour valider ExactSolution.

    u'(x) = (C - x)(2 + cos(2 pi x / eps)),  C fixé par u(1) = 0.
    """
    t_nodes, t_weights = np.polynomial.legendre.leggauss(n_quad)
    t01 = 0.5 * (t_nodes + 1.0)   # points dans [0,1]
    w01 = 0.5 * t_weights         # poids associés

    # u(1) = C * I1 - I2 = 0  avec
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
                # quadrature de u'(t) sur [0, xi]
                ti = 0.5 * xi * (t_nodes + 1.0)
                wi = 0.5 * xi * t_weights
                result[i] = np.dot(
                    wi, (C - ti) * (2.0 + np.cos(2.0 * np.pi * ti / eps))
                )
        return result[0] if scalar else result

    return u_func, C


@pytest.mark.parametrize("eps", [1.0 / 8.0, 1.0 / 7.3])
def test_closed_form_matches_quadrature(eps):
    """La forme close coïncide avec la quadrature de référence à 1e-10."""
    exact = ExactSolution(eps)
    u_ref, C_ref = _exact_by_quadrature(eps)

    # constante de flux
    assert abs(exact.C - C_ref) < 1e-10

    # valeurs sur une grille fine
    grid = np.linspace(0.0, 1.0, 257)
    assert np.max(np.abs(exact.value(grid) - u_ref(grid))) < 1e-10

    # conditions de Dirichlet homogènes
    assert abs(exact.value(0.0)) < 1e-12
    assert abs(exact.value(1.0)) < 1e-10


def test_homogenization_consistency():
    """
    Quand eps -> 0 : C -> 1/2 et u -> u_0(x) = x(1-x), l'écart décroissant en
    O(eps).
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

    # l'erreur décroît
    assert np.all(np.diff(errs) < 0.0)

    # taux ~ O(eps) : en divisant eps par 2, l'erreur est ~ divisée par 2.
    ratios = errs[:-1] / errs[1:]
    assert np.all(ratios > 1.7)   # proche de 2, marge pour les termes d'ordre sup.


def test_resonant_eps_gives_half():
    """En régime résonant (eps = 1/n entier), C = 1/2 à la précision machine."""
    for n in (4, 8, 16):
        assert abs(ExactSolution(1.0 / n).C - 0.5) < 1e-12
