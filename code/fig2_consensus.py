"""Figure 2: average consensus on the sensor-network example."""
import numpy as np

from common import SEED, graph_consensus, spectra, update_results, write_csv


def main():
    G, A, d, L, P = graph_consensus()
    lam, Q, *_ = spectra(L, d)
    n = G.number_of_nodes()
    rng = np.random.default_rng(SEED)

    x0 = rng.uniform(-1.0, 1.0, n)
    x_ave = float(x0.mean())
    t = np.linspace(0.0, 60.0, 3001)
    X = Q @ (np.exp(-np.outer(lam, t)) * (Q.T @ x0)[:, None])
    err = np.linalg.norm(X - x_ave, axis=0)

    sub = slice(None, None, 6)
    write_csv("consensus_trajectories.csv",
              ["t"] + [f"x{i+1}" for i in range(n)],
              [[float(t[k])] + [float(v) for v in X[:, k]]
               for k in range(len(t))[sub]])
    write_csv("consensus_error.csv", ["t", "err", "bound"],
              [[float(t[k]), float(err[k]),
                float(err[0] * np.exp(-lam[1] * t[k]))]
               for k in range(len(t))[sub]], fmt="%.10e")

    theta_deg, theta_max = 1.0 / d.max(), 2.0 / lam[-1]
    theta_star = 2.0 / (lam[1] + lam[-1])          # best constant step
    rho = lambda th: max(abs(1 - th * lam[1]), abs(1 - th * lam[-1]))
    iters = lambda th: int(np.ceil(np.log(1e3) / -np.log(rho(th))))
    update_results({
        "theta_star": float(theta_star),
        "rho_theta_star": float(rho(theta_star)),
        "iters_1e3_theta_deg": iters(theta_deg),
        "iters_1e3_theta_star": iters(theta_star),
        "x_ave": x_ave,
        "theta_degree_bound": float(theta_deg),
        "theta_spectral_bound": float(theta_max),
        "discrete_rate_theta_deg": float(max(abs(1 - theta_deg * lam[1]),
                                             abs(1 - theta_deg * lam[-1]))),
        "settling_time_1e3": float(np.log(1e3) / lam[1]),
    })


if __name__ == "__main__":
    main()
