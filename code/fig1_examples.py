"""Figure 1: the four examples, and their Laplacian spectra."""
import networkx as nx
import numpy as np

from common import (graph_ahp, graph_clustering, graph_consensus, graph_drawing,
                    incidence, normalize_layout, spectra, update_results,
                    write_csv, write_edge_polyline, write_table)


def pca2(X):
    X = X - X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T


def dump_graph(tag, P, G, per_axis=False, edges=True, extra=None):
    P = normalize_layout(P, per_axis=per_axis)
    hdr = ["x", "y"] + ([extra[0]] if extra else [])
    rows = [[float(P[i, 0]), float(P[i, 1])] + ([extra[1][i]] if extra else [])
            for i in range(len(P))]
    write_csv(f"ex_{tag}_nodes.csv", hdr, rows)
    if edges:
        write_edge_polyline(f"ex_{tag}_edges.csv", P, sorted(G.edges()))


def main():
    res = {}

    Gc, Ac, dc, Lc, Pc = graph_consensus()
    Gd, Ad, dd, Ld, Pd, hull = graph_drawing()
    Gk, Ak, dk, Lk, chain, xyz = graph_clustering()
    Ga, Aa, da, La = graph_ahp()

    dump_graph("consensus", Pc, Gc)
    dump_graph("drawing", Pd, Gd)
    # 574 residues: draw the nodes only, in the plane of the two principal axes
    dump_graph("clustering", pca2(xyz), Gk, edges=False,
               extra=("chain", [ord(c) - ord("A") for c in chain]))
    pos = nx.circular_layout(Ga)
    dump_graph("ahp", np.array([pos[i] for i in Ga.nodes()]), Ga)

    for tag, (G, A, d, L) in [("consensus", (Gc, Ac, dc, Lc)),
                              ("drawing", (Gd, Ad, dd, Ld)),
                              ("clustering", (Gk, Ak, dk, Lk)),
                              ("ahp", (Ga, Aa, da, La))]:
        lam, Q, Lnorm, lam_n, Q_n = spectra(L, d)
        B, _ = incidence(G)
        n = G.number_of_nodes()
        write_table(f"spectrum_{tag}.csv", ["i", "lambda", "scaled"],
                    np.column_stack([np.arange(1, n + 1), lam, lam / lam[-1]]))
        res.update({
            f"{tag}_n": n, f"{tag}_m": G.number_of_edges(),
            f"{tag}_d_min": int(d.min()), f"{tag}_d_max": int(d.max()),
            f"{tag}_lambda_2": float(lam[1]),
            f"{tag}_lambda_n": float(lam[-1]),
            f"{tag}_lambda_2_norm": float(lam_n[1]),
            f"{tag}_spectral_ratio": float(lam[-1] / lam[1]),
            f"{tag}_incidence_residual": float(np.abs(B @ B.T - L).max()),
        })

    res["drawing_hull_size"] = len(hull)
    res["drawing_planar"] = bool(nx.check_planarity(Gd)[0])
    res["drawing_connectivity"] = int(nx.node_connectivity(Gd))
    res["clustering_chains"] = sorted(set(chain.tolist()))
    update_results(res)


if __name__ == "__main__":
    main()
