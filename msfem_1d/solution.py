"""
Common interface for all discrete solutions (P1, MsFEM, exact).

Every method returns an object with:
  .value(x)  -> u(x)   on a scalar or a numpy array
  .grad(x)   -> u'(x)  same

P1Interpolant is the concrete implementation for nodal FEM methods.
The exact/homogenized solutions implement the same interface directly in
problem.py (ExactSolution, HomogSolution).
"""

import numpy as np


class DiscreteSolution:
    """Base class (interface). Subclass it or use P1Interpolant."""

    def value(self, x):
        raise NotImplementedError

    def grad(self, x):
        raise NotImplementedError


class P1Interpolant(DiscreteSolution):
    """
    Piecewise P1 interpolant built from nodal values.

    Parameters
    ----------
    nodes  : array (M+1,) sorted nodes x_0 < x_1 < ... < x_M
    values : array (M+1,) values u_i = u(x_i)
    """

    def __init__(self, nodes: np.ndarray, values: np.ndarray):
        self.nodes = np.asarray(nodes, dtype=float)
        self.values = np.asarray(values, dtype=float)

    def _locate(self, x):
        """Index i of the element containing x: nodes[i] <= x < nodes[i+1]."""
        idx = np.searchsorted(self.nodes, x, side="right") - 1
        # clip to keep a valid element index (x on the boundary or outside [0,1])
        return np.clip(idx, 0, len(self.nodes) - 2)

    def value(self, x):
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        i = self._locate(x)
        x0, x1 = self.nodes[i], self.nodes[i + 1]
        u0, u1 = self.values[i], self.values[i + 1]
        # linear interpolation in local coordinate t in [0,1]
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
        # derivative is constant per element (P1)
        result = (u1 - u0) / (x1 - x0)
        return result[0] if scalar else result
