"""
Error norms between any two solutions (DiscreteSolution or exact).

All functions take two objects with .value(x) and .grad(x), plus a fine grid
on which the quadrature is done.
No "if method == ..." here.
"""

import numpy as np


def _gauss_on_intervals(nodes, n_gauss=5):
    """
    Return Gauss (points, weights) on the union of intervals [nodes[i], nodes[i+1]].
    """
    xi_g, w_g = np.polynomial.legendre.leggauss(n_gauss)
    all_pts, all_w = [], []
    for i in range(len(nodes) - 1):
        xa, xb = nodes[i], nodes[i + 1]
        h = xb - xa
        all_pts.append(0.5 * (xa + xb) + 0.5 * h * xi_g)
        all_w.append(0.5 * h * w_g)
    return np.concatenate(all_pts), np.concatenate(all_w)


def norm_L2(u, v, ref_nodes, n_gauss=5) -> float:
    """||u - v||_{L^2(0,1)}"""
    pts, w = _gauss_on_intervals(ref_nodes, n_gauss)
    diff = u.value(pts) - v.value(pts)
    return float(np.sqrt(np.dot(w, diff**2)))


def norm_H1_semi(u, v, ref_nodes, n_gauss=5) -> float:
    """|(u - v)|_{H^1(0,1)} = ||u' - v'||_{L^2}"""
    pts, w = _gauss_on_intervals(ref_nodes, n_gauss)
    diff_g = u.grad(pts) - v.grad(pts)
    return float(np.sqrt(np.dot(w, diff_g**2)))


def norm_H1(u, v, ref_nodes, n_gauss=5) -> float:
    """||u - v||_{H^1} = sqrt(L2^2 + H1semi^2)"""
    e_l2 = norm_L2(u, v, ref_nodes, n_gauss)
    e_h1s = norm_H1_semi(u, v, ref_nodes, n_gauss)
    return float(np.sqrt(e_l2**2 + e_h1s**2))


def norm_energy(u, v, A_func, ref_nodes, n_gauss=5) -> float:
    """||u - v||_{energy} = sqrt( integral A |u'-v'|^2 dx )"""
    pts, w = _gauss_on_intervals(ref_nodes, n_gauss)
    diff_g = u.grad(pts) - v.grad(pts)
    A_vals = A_func(pts)
    return float(np.sqrt(np.dot(w, A_vals * diff_g**2)))
