import argparse
import time
import numpy as np
import scipy as sp
import sklearn.metrics as skm


def angle_cluster(samples: np.typing.NDArray, mean: np.typing.NDArray) -> np.typing.NDArray:
    sims = skm.pairwise.cosine_similarity(samples - mean)
    leaders = np.unravel_index(sims.argmin(), sims.shape)
    groups = np.array([samples[sims[leader] >= 0].mean(0) for leader in leaders])
    return groups


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
    dimensions = 2
    e1 = 0.1
    e2 = 0.5

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
        np.sum((mu_dists <= i * e1 * sigma) & (mu_dists > (i - 1) * e1 * sigma))
        for i in range(1, round(3 / 0.1))
    ])
    peak_indices = np.argwhere((sonar[1:-1] >= sonar[2:]) & (sonar[1:-1] > sonar[:-2])).reshape(-1) + 1
    centres = []
    for pi in peak_indices:
        considered_samples = x[(mu_dists <= pi * e1 * sigma) & (mu_dists > (pi - 1) * e1 * sigma)]
        centres.append(angle_cluster(considered_samples, mu))
    centres = np.concatenate(centres)
    print(f"{centres=}")
    all_centres = angle_cluster(centres, mu)
    all_centres = np.concatenate([all_centres, np.array([mu])])
    print(f"{all_centres=}")
    densities = np.array([np.sum(np.linalg.norm(x - c, axis=1) < sigma) for c in all_centres])
    max_density = np.max(densities)
    # densities[-1] *= e2
    weights = sp.stats.norm.pdf(np.abs(densities - max_density), scale=e1*max_density)
    p = weights / weights.sum()
    print(f"{densities=}, {weights=}, {p=}")
    agg_x = np.sum((all_centres.T * p).T, axis=0)
    print(f"{agg_x=}")
    print(f"{honest_x.mean(0)=}")
    print(f"Error: {np.linalg.norm(agg_x - honest_x.mean(0))}")

    print(f"Experiment took {time.time() - start_time} seconds")
