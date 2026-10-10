"""Figure 4: spectral clustering of the haemoglobin contact network.

Human deoxyhaemoglobin is a tetramer, so an external reference partition is
available: the four chains A, C (alpha) and B, D (beta). The assignment is not
given to the algorithm. The number of clusters is not supplied directly: the
eigengap rule selects it within the prespecified range 2, ..., 9.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform

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
    """Agreement with the ground truth, maximized over one-to-one matchings of
    the true classes with the k clusters.

    The optimal matching is found by the Hungarian algorithm on the confusion
    matrix (O(k^3), valid for any k) instead of enumerating the k! permutations.
    Returns (accuracy, {class: cluster}); when k is smaller than the number of
    classes some classes stay unmatched and are absent from the mapping, when it
    is larger some clusters stay unmatched.
    """
    names = sorted(set(truth.tolist()))
    conf = np.array([[np.sum((truth == c) & (lab == q)) for q in range(k)]
                     for c in names])
    rows, cols = linear_sum_assignment(conf, maximize=True)
    mapping = {names[r]: int(c) for r, c in zip(rows, cols)}
    return float(conf[rows, cols].sum() / len(lab)), mapping


def eigengap(lam_n, kmax=9):
    """Eigengap rule (24) on k=2,...,kmax.

    Returns (k*, largest gap, runner-up gap / largest gap).  The upper limit is
    a prespecified scale choice; the paper uses kmax=9 for coarse partitions.
    """
    g = np.diff(lam_n)[1:kmax]                      # gaps after k = 2, ..., kmax
    top = np.sort(g)[::-1]
    return int(np.argmax(g) + 2), float(top[0]), float(top[1] / top[0])


def cutoff_scan(xyz, cutoffs, lower=4.0):
    """k* and margin of the eigengap rule when the contact cut-off changes
    (the robustness statement of Section 5.4)."""
    D = squareform(pdist(xyz))
    out = {}
    for c in cutoffs:
        Ac = ((D >= lower) & (D <= c)).astype(float)
        np.fill_diagonal(Ac, 0.0)
        dc = Ac.sum(1)
        if dc.min() == 0:
            out[f"{c:g}"] = {"isolated_residues": int((dc == 0).sum())}
            continue
        Ln = Ac * 0.0 - Ac / np.sqrt(np.outer(dc, dc)) + np.eye(len(dc))
        k, _, ratio = eigengap(np.linalg.eigvalsh(Ln))
        out[f"{c:g}"] = {"contacts": int(Ac.sum() / 2), "k_star": k,
                         "margin": float(1 - ratio)}
    return out


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

    # Ng--Jordan--Weiss symmetric normalized spectral clustering uses the
    # first k eigenvectors of L_norm, including the trivial first one.
    U = Q_n[:, :k_star]
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    lab, _ = kmeans(U, k_star, rng)
    acc, mapping = best_match(lab, chain, k_star)
    # Stable display labels: matched clusters come first in chain order and any
    # unmatched clusters follow.  Enumerating that order keeps the labels
    # contiguous also when k* is smaller than the number of reference chains.
    names = sorted(set(chain.tolist()))
    cluster_order = [q for c, q in sorted(mapping.items(),
                                          key=lambda item: names.index(item[0]))]
    cluster_order += [q for q in range(k_star) if q not in cluster_order]
    order = {q: new for new, q in enumerate(cluster_order)}
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
    # the 18 of the 28 disagreements of Section 5.4: beta residues absorbed by the
    # alpha cluster of the same dimer, and how many touch that alpha chain
    partner = {"B": "A", "D": "C"}
    names = sorted(set(chain.tolist()))
    absorbed = [i for i in range(n) if chain[i] in partner
                and lab[i] == names.index(partner[chain[i]])]
    touching = [i for i in absorbed
                if any(chain[j] == partner[chain[i]] for j in np.flatnonzero(A[i]))]
    _, _, runner_up = eigengap(lam_n)
    update_results({
        "eigengap_runner_up_ratio": runner_up,
        "beta_absorbed_by_alpha": len(absorbed),
        "beta_absorbed_touching_alpha": len(touching),
        "cutoff_scan": cutoff_scan(xyz, [7.0, 7.5, 8.0, 8.5, 9.0]),
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
