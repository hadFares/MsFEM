"""
Maillage 1D uniforme.

Maillage grossier : N+1 nœuds, pas H = 1/N.
Sous-maillage fin : chaque maille grossière est subdivisée en n sous-intervalles,
  pas h = H/n.
"""

import numpy as np


class Mesh1D:
    """
    Paramètres
    ----------
    N   : nombre de mailles grossières (N+1 nœuds, y.c. bords)
    n   : nombre de sous-mailles fines par maille grossière
    """

    def __init__(self, N: int, n: int = 1):
        self.N = N
        self.n = n
        self.H = 1.0 / N
        self.h = self.H / n

        # Nœuds grossiers (N+1 points, indices 0..N)
        self.nodes_coarse = np.linspace(0.0, 1.0, N + 1)

        # Nœuds fins (N*n + 1 points)
        self.nodes_fine = np.linspace(0.0, 1.0, N * n + 1)

    # ------------------------------------------------------------------
    # Utilitaires géométriques
    # ------------------------------------------------------------------

    def coarse_element(self, i: int):
        """Retourne (x_left, x_right) de la i-ème maille grossière (0-indexé)."""
        return self.nodes_coarse[i], self.nodes_coarse[i + 1]

    def fine_nodes_in_element(self, i: int):
        """Nœuds fins dans la i-ème maille grossière (inclus les deux bords)."""
        start = i * self.n
        return self.nodes_fine[start : start + self.n + 1]

    def element_of(self, x):
        """
        Indice de la maille grossière contenant x (ou la dernière si x==1).
        Fonctionne sur scalaire ou array.
        """
        x = np.asarray(x, dtype=float)
        idx = np.floor(x / self.H).astype(int)
        np.clip(idx, 0, self.N - 1, out=idx)
        return idx

    # ------------------------------------------------------------------
    # Représentation
    # ------------------------------------------------------------------

    def __repr__(self):
        return (f"Mesh1D(N={self.N}, n={self.n}, "
                f"H={self.H:.4g}, h={self.h:.4g})")
