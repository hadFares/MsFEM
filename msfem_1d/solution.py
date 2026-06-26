"""
Interface commune pour toutes les solutions discrètes (P1, MsFEM, exacte).

Toute méthode renvoie un objet avec :
  .value(x)  -> u(x)   sur scalaire ou array numpy
  .grad(x)   -> u'(x)  idem

P1Interpolant est la réalisation concrète pour les méthodes FEM nodales.
Les solutions exactes/homogénéisées implémentent la même interface directement
dans problem.py (ExactSolution, HomogSolution).
"""

import numpy as np


class DiscreteSolution:
    """Classe de base (interface). Sous-classer ou instancier P1Interpolant."""

    def value(self, x):
        raise NotImplementedError

    def grad(self, x):
        raise NotImplementedError


class P1Interpolant(DiscreteSolution):
    """
    Interpolant P1 par morceaux construit à partir de valeurs nodales.

    Paramètres
    ----------
    nodes  : array (M+1,) nœuds triés x_0 < x_1 < ... < x_M
    values : array (M+1,) valeurs u_i = u(x_i)
    """

    def __init__(self, nodes: np.ndarray, values: np.ndarray):
        self.nodes = np.asarray(nodes, dtype=float)
        self.values = np.asarray(values, dtype=float)

    def _locate(self, x):
        """Indice i de l'élément contenant x : nodes[i] <= x < nodes[i+1]."""
        idx = np.searchsorted(self.nodes, x, side="right") - 1
        # clip pour rester un indice d'élément valide (x au bord ou hors [0,1])
        return np.clip(idx, 0, len(self.nodes) - 2)

    def value(self, x):
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        i = self._locate(x)
        x0, x1 = self.nodes[i], self.nodes[i + 1]
        u0, u1 = self.values[i], self.values[i + 1]
        # interpolation linéaire en coordonnée locale t in [0,1]
        h = x1 - x0
        t = (x - x0) / h
        result = (1.0 - t) * u0 + t * u1
        return result[0] if scalar else result

    def grad(self, x):
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        i = self._locate(x)
        x0, x1 = self.nodes[i], self.nodes[i + 1]
        u0, u1 = self.values[i], self.values[i + 1]
        # dérivée constante par élément (P1)
        result = (u1 - u0) / (x1 - x0)
        return result[0] if scalar else result
