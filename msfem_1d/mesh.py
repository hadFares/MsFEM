"""
Uniform 1D mesh.

Coarse mesh: N+1 nodes, step H = 1/N.
Fine sub-mesh: each coarse cell is split into n sub-intervals, step h = H/n.
"""

import numpy as np


class Mesh1D:
    """
    Parameters
    ----------
    N   : number of coarse cells (N+1 nodes, including boundaries)
    n   : number of fine sub-cells per coarse cell
    """

    def __init__(self, N: int, n: int = 1):
        self.N = N
        self.n = n
        self.H = 1.0 / N
        self.h = self.H / n

        # Coarse nodes (N+1 points, indices 0..N)
        self.nodes_coarse = np.linspace(0.0, 1.0, N + 1)

        # Fine nodes (N*n + 1 points)
        self.nodes_fine = np.linspace(0.0, 1.0, N * n + 1)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def coarse_element(self, i: int):
        """Return (x_left, x_right) of the i-th coarse cell (0-indexed)."""
        return self.nodes_coarse[i], self.nodes_coarse[i + 1]

    def fine_nodes_in_element(self, i: int):
        """Fine nodes inside the i-th coarse cell (both boundaries included)."""
        start = i * self.n
        return self.nodes_fine[start : start + self.n + 1]

    def element_of(self, x):
        """
        Index of the coarse cell that contains x (or the last one if x==1).
        Works on a scalar or an array.
        """
        x = np.asarray(x, dtype=float)
        idx = np.floor(x / self.H).astype(int)   # cell = integer part of x/H
        np.clip(idx, 0, self.N - 1, out=idx)     # bound the x == 1 case
        return idx

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return (f"Mesh1D(N={self.N}, n={self.n}, "
                f"H={self.H:.4g}, h={self.h:.4g})")
