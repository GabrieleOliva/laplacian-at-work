"""Figure 5: sparse AHP on the small-world elicitation graph, solved by the
Sparse Logarithmic Least Squares method.

The agents observe perturbed log-ratios on the EDGES of the comparison graph,
    b_h = ln S_ij = ln(w_i / w_j) + eps_ij,
and recover the utilities by solving the normal equations L r = B b -- i.e.
by running an average consensus iteration driven by a constant bias.
"""
import numpy as np
from scipy.stats import kendalltau

from common import (SEED, graph_ahp, incidence, spectra, update_results,
                    write_csv)

SIGMA_SHOW = 0.30
N_RUNS = 300


def log_ratios(w, edges, sigma, rng):
    lw = np.log(w)
    ell = np.array([lw[i] - lw[j] for (i, j) in edges])
    return ell + rng.normal(0.0, sigma, len(edges))


def slls(L_pinv, B, ell):
    """Centralized solution of L r = s, normalized so that 1^T r = 0."""
    r = L_pinv @ (B @ ell)
    return r - r.mean()


def main():
    G, A, d, L = graph_ahp()
    B, edges = incidence(G)
    lam, *_ = spectra(L, d)
    n = G.number_of_nodes()
    L_pinv = np.linalg.pinv(L)
    rng = np.random.default_rng(SEED)

    # ---- one instance, shown in panels (a) and (b)
    w_true = np.exp(rng.uniform(np.log(1.0), np.log(9.0), n))
    w_true /= w_true.sum()
    ell = log_ratios(w_true, edges, SIGMA_SHOW, rng)
    r_star = slls(L_pinv, B, ell)
    w_hat = np.exp(r_star)
    w_hat /= w_hat.sum()

    # ---- distributed iteration r(k+1) = (I - theta L) r(k) + theta s
    theta = 1.0 / d.max()
    s = B @ ell
    M = np.eye(n) - theta * L
    rho = max(abs(1 - theta * lam[1]), abs(1 - theta * lam[-1]))
    K = 1200
    r = rng.normal(0.0, 1.0, n)
    # The iteration preserves 1^T r, hence it converges to the representative
    # of the affine solution set having the same mean as r(0).
    r_inf = r_star + r.mean()
    err0 = float(np.linalg.norm(r - r_inf))
    rows = []
    for k in range(K):
        rows.append([k, float(np.linalg.norm(r - r_inf)),
                     float(err0 * rho ** k)])
        r = M @ r + theta * s

    write_csv("slls_convergence.csv", ["k", "err", "bound"], rows[::4],
              fmt="%.12e")
    order = np.argsort(-w_true)
    write_csv("slls_utilities.csv", ["rank", "w_true", "w_hat"],
              [[i + 1, float(w_true[order][i]), float(w_hat[order][i])]
               for i in range(n)])

    # ---- ranking accuracy against the perturbation level
    sigmas = np.linspace(0.0, 1.0, 11)
    curve = []
    for sg in sigmas:
        taus = []
        for _ in range(N_RUNS):
            w = np.exp(rng.uniform(np.log(1.0), np.log(9.0), n))
            taus.append(kendalltau(
                w, np.exp(slls(L_pinv, B, log_ratios(w, edges, sg, rng)))
            ).statistic)
        curve.append([float(sg), float(np.mean(taus)), float(np.std(taus))])
    write_csv("slls_tau.csv", ["sigma", "tau", "std"], curve)

    update_results({
        "slls_sigma_shown": SIGMA_SHOW,
        "slls_tau_shown": float(kendalltau(w_true, w_hat).statistic),
        "slls_theta": float(theta),
        "slls_rate": float(rho),
        "slls_runs_per_sigma": N_RUNS,
        "slls_tau_sigma0": curve[0][1],
        "slls_tau_sigma05": curve[5][1],
        "slls_tau_sigma1": curve[-1][1],
        "slls_iters_to_1e6": int(np.ceil(np.log(1e-6) / np.log(rho))),
        "ahp_settling_time_1e3": float(np.log(1e3) / lam[1]),
        "slls_one_minus_theta_lambda_n": float(abs(1 - theta * lam[-1])),
    })


if __name__ == "__main__":
    main()
