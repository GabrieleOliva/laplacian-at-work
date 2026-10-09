"""Figure 4: spectral clustering of the haemoglobin contact network.

Human deoxyhaemoglobin is a tetramer, so the ground truth is known: the four
chains A, C (alpha) and B, D (beta). Neither the number of clusters nor the
assignment is given to the algorithm.
"""
import itertools

import numpy as np
from scipy.linalg import eigh

from common import (SEED, graph_clustering, normalize_layout, spectra,
                    update_results, write_csv, write_table)


def kmeans(X, k, rng, iters=300, restarts=60):
    best_lab, best_J = None, np.inf
    for _ in range(restarts):
        c = X[rng.choice(len(X), k, replace=False)].copy()
        lab = -np.ones(len(X), dtype=int)
        for _ in range(iters):
            new = ((X[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
            if np.array_equal(new, lab):
                break
            lab = new
            for q in range(k):
                if np.any(lab == q):
                    c[q] = X[lab == q].mean(0)
        J = ((X - c[lab]) ** 2).sum()
        if J < best_J:
            best_J, best_lab = J, lab
    return best_lab, float(best_J)


def ncut(labels, A, d):
    return float(sum(A[np.ix_(labels == q, labels != q)].sum() / d[labels == q].sum()
                     for q in np.unique(labels)))


def best_match(lab, truth, k):
    """Agreement with the ground truth, maximized over cluster relabelings."""
    names = sorted(set(truth.tolist()))
    best, perm_best = 0.0, None
    for perm in itertools.permutations(range(k)):
        m = {names[i]: perm[i] for i in range(len(names))}
        acc = float(np.mean([lab[i] == m[truth[i]] for i in range(len(lab))]))
        if acc > best:
            best, perm_best = acc, m
    return best, perm_best


def pca2(X):
    X = X - X.mean(0)
    return X @ np.linalg.svd(X, full_matrices=False)[2][:2].T


def main():
    G, A, d, L, chain, xyz = graph_clustering()
    lam, Q, Lnorm, lam_n, Q_n = spectra(L, d)
    n = G.number_of_nodes()
    rng = np.random.default_rng(SEED)

    gaps = np.diff(lam_n)
    k_star = int(np.argmax(gaps[1:9]) + 2)          # search k in {2,...,9}
    write_table("spectrum_normalized.csv", ["i", "lambda"],
                np.column_stack([np.arange(1, 11), lam_n[:10]]))

    U = Q_n[:, 1:k_star]
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    lab, _ = kmeans(U, k_star, rng)
    acc, mapping = best_match(lab, chain, k_star)
    # relabel clusters so that cluster q corresponds to the q-th chain
    inv = {v: k for k, v in mapping.items()}
    order = {q: sorted(set(chain.tolist())).index(inv[q]) for q in range(k_star)}
    lab = np.array([order[q] for q in lab])

    P_emb = normalize_layout(np.column_stack([Q_n[:, 1] / np.sqrt(d),
                                              Q_n[:, 2] / np.sqrt(d)]),
                             per_axis=True)
    P_xyz = normalize_layout(pca2(xyz), per_axis=False)
    write_csv("cluster_embedding.csv", ["x", "y", "class"],
              [[float(P_emb[i, 0]), float(P_emb[i, 1]), int(lab[i])]
               for i in range(n)])
    write_csv("cluster_structure.csv", ["x", "y", "class", "chain"],
              [[float(P_xyz[i, 0]), float(P_xyz[i, 1]), int(lab[i]),
                ord(chain[i]) - ord("A")] for i in range(n)])

    conf = [[int(((chain == c) & (lab == q)).sum()) for q in range(k_star)]
            for c in sorted(set(chain.tolist()))]
    update_results({
        "eigengap_k_star": k_star,
        "eigengap_largest": float(gaps[k_star - 1]),
        "chain_agreement": acc,
        "chain_misassigned": int(round((1 - acc) * n)),
        "ncut_k_star": ncut(lab, A, d),
        "cluster_sizes": [int((lab == q).sum()) for q in range(k_star)],
        "chain_sizes": [int((chain == c).sum())
                        for c in sorted(set(chain.tolist()))],
        "confusion_chain_cluster": conf,
    })


if __name__ == "__main__":
    main()
