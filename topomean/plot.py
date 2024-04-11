import argparse
import time
import numpy as np
import scipy as sp
import sklearn.cluster as skc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Topomean synthetic testbed program ablating the algorithm."
    )
    parser.add_argument('-s', '--seed', type=int, default=14258, help="Initial seed for the experiments.")
    parser.add_argument('-a', '--attack-target', type=float, default=6.0)
    parser.add_argument("--padversaries", type=float, default=0.4, help="Proportion of adversaries.")
    args = parser.parse_args()
    print(f"Experiment args: {vars(args)}")
    npoints = 10000
    dimensions = 10

    start_time = time.time()
    rng = np.random.default_rng(args.seed)
    nadversaries = round(npoints * args.padversaries)

    honest_x = rng.normal(1, 3, size=(npoints - nadversaries, dimensions))
    attack_x = rng.normal(args.attack_target, np.std(honest_x, 0), (nadversaries, dimensions))
    x = np.concatenate((honest_x, attack_x))
    mu = np.mean(x, axis=0)
    sigma = np.std(x)
    dists = sp.spatial.distance.cdist(x, x)
    mu_densities = np.linalg.norm(x - mu, axis=1) < sigma
    mu_dists = np.linalg.norm(x - mu, axis=1)
    sonar = np.array([
        np.sum((mu_dists <= i * 0.1 * sigma) & (mu_dists > (i - 1) * 0.1 * sigma))
        for i in range(1, round(3 / 0.1))
    ])
    peak_indices = np.argwhere((sonar[1:-1] >= sonar[2:]) & (sonar[1:-1] > sonar[:-2])).reshape(-1) + 1
    centres = []
    for pi in peak_indices:
        model = skc.KMeans(2)
        model.fit(x[(mu_dists <= pi * 0.1 * sigma) & (mu_dists > (pi - 1) * 0.1 * sigma)])
        centres.append(model.cluster_centers_)
    centres = np.concatenate(centres)
    smaller = centres[np.mean(centres - mu, axis=1) < np.mean(centres - mu)].mean(0).reshape(-1)
    bigger = centres[np.mean(centres - mu, axis=1) >= np.mean(centres - mu)].mean(0).reshape(-1)
    mu_density = mu_densities.sum()
    smaller_density = np.sum(np.linalg.norm(x - smaller, axis=1) < sigma)
    bigger_density = np.sum(np.linalg.norm(x - bigger, axis=1) < sigma)
    print(f"{mu=}, {mu_density=}")
    print(f"{smaller=}, {smaller_density=}, {bigger=}, {bigger_density=}")
    max_density = max(mu_density, smaller_density, bigger_density)
    agg = []
    e1 = 0.05
    print(f"{abs(smaller_density - bigger_density)=}, {e1 * max_density=}")
    # if mu_density != max_density:
    #     agg.append(smaller if smaller_density > bigger_density else bigger)
    # elif abs(smaller_density - bigger_density) < e1 * max_density:
    #     density_sum = smaller_density + mu_density + max_density
    #     agg.extend([
    #         smaller * (smaller_density / density_sum) * 3,
    #         mu * (mu_density / density_sum) * 3,
    #         bigger * (bigger_density / density_sum) * 3
    #     ])
    # else:
    #     if smaller_density >= (1 - e1) * max_density:
    #         agg.append(smaller)
    #     elif bigger_density >= (1 - e1) * max_density:
    #         agg.append(bigger)
    #     else:
    #         agg.append(mu)
    #         agg.append(smaller if smaller_density > bigger_density else bigger)

    density_sum = smaller_density + mu_density + max_density
    agg.extend([
        smaller * (smaller_density / density_sum),
        mu * (mu_density / density_sum),
        bigger * (bigger_density / density_sum),
    ])
    print(f"{np.sum(agg, axis=0)=}")
