"""Figure 3: three Laplacian layouts of the Delaunay example.

(a) Hall's layout,      eigenvectors q_2, q_3 of L;
(b) degree-normalized,  generalized problem L q = mu D q  (Koren);
(c) anchored embedding, L_FF p_F = -L_FA p_A  (Tutte / leader-follower).

The graph is planar and 3-connected and the anchor set is its convex hull, so
(c) satisfies the hypotheses of Tutte's theorem and must come out crossing-free;
the script checks that it does.
"""
import itertools

import numpy as np
from scipy.linalg import eigh

from common import (graph_drawing, incidence, normalize_layout, spectra,
                    update_results, write_csv, write_edge_polyline)


def anchored_layout(L, anchors, anchor_pos):
    n = L.shape[0]
    free = np.array([i for i in range(n) if i not in anchors])
    P = np.zeros((n, 2))
    P[anchors] = anchor_pos
    P[free] = np.linalg.solve(L[np.ix_(free, free)],
                              -L[np.ix_(free, anchors)] @ anchor_pos)
    return P, float(np.linalg.eigvalsh(L[np.ix_(free, free)]).min())


def crossings(edges, P):
    def side(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    c = 0
    for (a, b), (x, y) in itertools.combinations(edges, 2):
        if len({a, b, x, y}) < 4:
            continue
        d1, d2 = side(P[a], P[b], P[x]), side(P[a], P[b], P[y])
        d3, d4 = side(P[x], P[y], P[a]), side(P[x], P[y], P[b])
        if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
            c += 1
    return c


def dump(tag, P, edges, per_axis=True, extra=None):
    P = normalize_layout(P, per_axis=per_axis)
    hdr = ["x", "y", "id"] + (["anchor"] if extra is not None else [])
    rows = []
    for i in range(len(P)):
        row = [float(P[i, 0]), float(P[i, 1]), i + 1]
        if extra is not None:
            row.append(int(extra[i]))
        rows.append(row)
    write_csv(f"layout_{tag}_nodes.csv", hdr, rows)
    write_edge_polyline(f"layout_{tag}_edges.csv", P, edges)


def main():
    G, A, d, L, P_true, hull = graph_drawing()
    B, edges = incidence(G)
    lam, Q, *_ = spectra(L, d)
    n = G.number_of_nodes()

    dump("hall", np.column_stack([Q[:, 1], Q[:, 2]]), edges)

    mu, V = eigh(L, np.diag(d))                     # L q = mu D q
    dump("koren", np.column_stack([V[:, 1], V[:, 2]]), edges)

    # anchors: the convex hull, pinned at its own positions -- a convex polygon,
    # exactly the configuration of Tutte's theorem
    P_t, lmin = anchored_layout(L, hull, P_true[hull])
    is_anchor = np.zeros(n, dtype=int)
    is_anchor[hull] = 1
    dump("tutte", P_t, edges, per_axis=False, extra=is_anchor)
    dump("true", P_true, edges, per_axis=False, extra=is_anchor)

    update_results({
        "hall_energy": float(Q[:, 1] @ L @ Q[:, 1]),
        "koren_mu_2": float(mu[1]),
        "anchor_count": len(hull),
        "grounded_lambda_min": lmin,
        "tutte_crossings": crossings(edges, P_t),
        "hall_crossings": crossings(edges, np.column_stack([Q[:, 1], Q[:, 2]])),
        "tutte_max_displacement": float(
            np.abs(normalize_layout(P_t, False)
                   - normalize_layout(P_true, False)).max()),
    })


if __name__ == "__main__":
    main()
