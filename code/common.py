"""
Shared utilities for the numerical examples of

    G. Oliva, "The Graph Laplacian at Work: One Least-Squares Problem Behind
    Consensus, Graph Drawing, Clustering and Decision Making",
    SICE JCMSI (tutorial paper).

The four applications discussed in the paper (average consensus, graph
drawing, spectral clustering and sparse AHP) are instances of one
least-squares problem, min ||B^T x - b||_W^2; each is illustrated on the
graph that suits it (see the table below and Section 2 of the paper).
"""

from __future__ import annotations

import json
import os

import networkx as nx
import numpy as np
from scipy.linalg import eigh
from scipy.spatial import ConvexHull

SEED = 20261202

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(DATA, exist_ok=True)


# ----------------------------------------------------------------------
# the four examples
#
# The paper makes one point -- four problems are one least-squares problem --
# and illustrates each of them on the graph that suits it:
#
#   Section 3  consensus   random geometric graph      (a sensor network)
#   Section 4  drawing     Delaunay triangulation      (planar and 3-connected,
#                                                       i.e. Tutte's hypotheses)
#   Section 5  clustering  haemoglobin contact network (PDB 2HHB)
#   Section 6  sparse AHP  Watts-Strogatz small world  (a sparse elicitation)
# ----------------------------------------------------------------------
def _wrap(G):
    """Return (G, A, d, L) for a simple unweighted graph on 0..n-1."""
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    A = nx.to_numpy_array(G, dtype=float, weight=None)
    d = A.sum(axis=1)
    return G, A, d, np.diag(d) - A


def graph_consensus(n=50, radius=0.24, seed=7):
    """Random geometric graph: n nodes thrown in the unit square, linked when
    closer than `radius`. The standard model of a wireless sensor network, and
    the one used in the experiments of Oliva, Setola and Scala (Automatica,
    2017)."""
    rng = np.random.default_rng(seed)
    pos = {i: p for i, p in enumerate(rng.random((n, 2)))}
    G = nx.random_geometric_graph(n, radius, pos=pos)
    if not nx.is_connected(G):
        raise RuntimeError("random geometric graph is disconnected")
    P = np.array([pos[i] for i in range(n)])
    return _wrap(G) + (P,)


def graph_drawing(n=30, seed=11):
    """Delaunay triangulation of n random points in the unit square.

    Planar, with the convex hull as the outer face, and -- for this seed --
    3-connected (not every Delaunay triangulation is; fig1_examples.py checks
    it): exactly the hypotheses of Tutte's theorem. Irregular enough
    (degrees 3 to 8) that Hall's layout and its degree-normalized variant
    differ visibly.
    """
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(seed)
    P = rng.random((n, 2))
    tri = Delaunay(P)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for s in tri.simplices:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            G.add_edge(int(s[a]), int(s[b]))
    # convex hull in cyclic order: the outer face, to be used as the anchor set
    hull = list(ConvexHull(P).vertices)
    return _wrap(G) + (P, hull)


def graph_clustering():
    """Haemoglobin protein contact network, read from assets/.

    Human deoxyhaemoglobin (PDB 2HHB) is a tetramer of four chains, A and C
    (alpha) and B and D (beta). Two residues are in contact when their C-alpha
    atoms are 4 to 8 angstrom apart, the convention of Di Paola et al. Rebuild
    the files with `python3 code/fetch_hemoglobin.py`.
    """
    nodes = os.path.join(ASSETS, "hemoglobin_nodes.csv")
    edges = os.path.join(ASSETS, "hemoglobin_edges.csv")
    if not (os.path.exists(nodes) and os.path.exists(edges)):
        raise FileNotFoundError("run code/fetch_hemoglobin.py first")
    chain, xyz = [], []
    with open(nodes) as f:
        next(f)
        for line in f:
            _, ch, _, x, y, z = line.strip().split(",")
            chain.append(ch)
            xyz.append([float(x), float(y), float(z)])
    G = nx.Graph()
    G.add_nodes_from(range(len(chain)))
    with open(edges) as f:
        next(f)
        for line in f:
            i, j = line.strip().split(",")
            G.add_edge(int(i), int(j))
    if not nx.is_connected(G):
        raise RuntimeError("contact network is disconnected")
    return _wrap(G) + (np.array(chain), np.array(xyz))


def graph_ahp(n=40, k=4, p=0.3, seed=23):
    """Watts-Strogatz small world: a sparse elicitation pattern in which each
    alternative is compared with a few neighbours plus a few distant ones."""
    return _wrap(nx.connected_watts_strogatz_graph(n, k, p, seed=seed))


def incidence(G):
    """Oriented incidence matrix B (n x m), with the identity L = B B^T.

    Edge h = {v_i, v_j}, i < j, contributes +1 in row i and -1 in row j; the
    orientation is arbitrary, and L does not depend on it.
    """
    edges = sorted(tuple(sorted(e)) for e in G.edges())
    B = np.zeros((G.number_of_nodes(), len(edges)))
    for h, (i, j) in enumerate(edges):
        B[i, h], B[j, h] = 1.0, -1.0
    return B, edges


def spectra(L, d):
    """Eigendecompositions of L and of the normalized Laplacian."""
    lam, Q = eigh(L)
    lam[0] = max(lam[0], 0.0)
    Dm12 = np.diag(1.0 / np.sqrt(d))
    Lnorm = Dm12 @ L @ Dm12
    lam_n, Q_n = eigh(Lnorm)
    lam_n[0] = max(lam_n[0], 0.0)
    return lam, Q, Lnorm, lam_n, Q_n


# ----------------------------------------------------------------------
# CSV / JSON output helpers  (all figures are drawn in TikZ from these files)
# ----------------------------------------------------------------------
def write_csv(name, header, rows, fmt="%.6f"):
    path = os.path.join(DATA, name)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(("nan" if (isinstance(v, float) and np.isnan(v))
                              else (fmt % v if isinstance(v, float) else str(v)))
                             for v in r) + "\n")
    print(f"  wrote {os.path.relpath(path, ROOT)}  ({len(rows)} rows)")


def write_table(name, header, array, fmt="%.6f"):
    write_csv(name, header, [list(row) for row in np.asarray(array)], fmt)


def write_edge_polyline(name, P, edges):
    """Edges as a single polyline broken by NaN rows.

    pgfplots joins consecutive rows of a table and breaks the path at a NaN,
    so one \\addplot command draws the whole edge set.
    """
    rows = []
    for (i, j) in edges:
        rows.append([float(P[i, 0]), float(P[i, 1])])
        rows.append([float(P[j, 0]), float(P[j, 1])])
        rows.append([float("nan"), float("nan")])
    write_csv(name, ["x", "y"], rows)


def normalize_layout(P, per_axis=False):
    """Centre and scale a layout to fit in [-1, 1]^2.

    With per_axis=False the aspect ratio is preserved (used for the true
    positions and for the anchored layout, whose anchors keep their true
    positions). With per_axis=True each coordinate
    is scaled independently, which is standard practice for spectral layouts:
    the eigenvectors have very different norms once weighted by 1/sqrt(lambda),
    and an unscaled plot degenerates into a thin sliver.
    """
    P = np.asarray(P, dtype=float)
    P = P - P.mean(0)
    s = np.abs(P).max(axis=0) if per_axis else np.abs(P).max()
    s = np.where(np.asarray(s) > 0, s, 1.0)
    return P / s


def update_results(new):
    """Merge numbers into data/results.json (quoted verbatim in the paper)."""
    path = os.path.join(DATA, "results.json")
    cur = {}
    if os.path.exists(path):
        with open(path) as f:
            cur = json.load(f)
    cur.update(new)
    with open(path, "w") as f:
        json.dump(cur, f, indent=2, sort_keys=True)
    for k, v in new.items():
        print(f"  {k} = {v}")
