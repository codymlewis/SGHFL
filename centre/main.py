import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scipy as sp


def find_topomean(samples: npt.NDArray, threshold: float = 0.1, tol: float = 0.2) -> npt.NDArray:
    for i, sample in enumerate(samples):
        samples = samples[(np.linalg.norm(samples - sample, axis=1) > 0.01) | (np.arange(len(samples)) == i)]
    dists = sp.spatial.distance.cdist(samples, samples)
    radius = np.std(samples) * threshold
    scores = np.sum(dists <= radius, axis=1)
    sorted_score_idx = np.argsort(-scores)
    sphere_scores = []
    sphere_centres = []
    while len(sorted_score_idx):
        i = sorted_score_idx[0]
        neighbourhood = np.argwhere(dists[i] <= radius).reshape(-1)
        sphere_scores.append(scores[i])
        sphere_centres.append(samples[neighbourhood].mean(0))
        sorted_score_idx = np.setdiff1d(sorted_score_idx, neighbourhood)
    sphere_scores = np.array(sphere_scores)
    sphere_centres = np.array(sphere_centres)
    sorted_ssi = np.argpartition(-sphere_scores, len(sphere_scores) // 3)[:len(sphere_scores) // 3]
    sphere_scores = sphere_scores[sorted_ssi]
    sphere_centres = sphere_centres[sorted_ssi]
    centre_dists = sp.spatial.distance.cdist(sphere_centres, sphere_centres)
    ts = centre_dists / np.std(samples)
    overlap = 1 - (sp.stats.norm.cdf(ts) - sp.stats.norm.cdf(-ts))
    p = overlap[np.argmax(overlap.sum(1))]
    p = p / p.sum()
    p = p * sphere_scores
    # return sp.optimize.minimize(
    #     lambda x: np.sum((p * np.linalg.norm(x - sphere_centres, axis=1).T).T / p.sum()),
    #     x0=sphere_centres.mean(0)
    # ).x
    return np.sum((p * sphere_centres.T).T / p.sum(), axis=0)


if __name__ == "__main__":
    rng = np.random.default_rng(72)
    npoints = 10000
    nadversaries = 3000
    attack = "shifted_random"

    honest_x = rng.normal(1, 3, size=(npoints - nadversaries, 2))
    match attack:
        case "lie":
            s = npoints // 2 + 1 - nadversaries
            zmax = sp.stats.norm.ppf((npoints - s) / npoints)
            attack_x = np.tile(np.mean(honest_x, 0) + zmax * np.std(honest_x, 0), (nadversaries, 1))
        case "shifted_random":
            attack_x = rng.normal(9, np.std(honest_x, 0), (nadversaries, 2))
    x = np.concatenate((honest_x, attack_x))
    print(f"{x.mean(0)=}, {x.std(0)=}")
    print(f"{honest_x.mean(0)=}, {honest_x.std(0)=}")
    print(f"{attack_x.mean(0)=}, {attack_x.std(0)=}")

    plt.scatter(honest_x[:, 0], honest_x[:, 1], label="Honest points")
    plt.scatter(attack_x[:, 0], attack_x[:, 1], label="Attack points")
    honest_mean = np.mean(honest_x, 0)
    plt.scatter(honest_mean[0], honest_mean[1], marker="x", label="Honest mean")
    full_mean = np.mean(x, 0)
    plt.scatter(full_mean[0], full_mean[1], marker="x", label="Full mean")
    topomean = find_topomean(x)
    print(f"Topographic mean: {topomean}")
    plt.scatter(topomean[0], topomean[1], marker="x", label="Topomean")

    plt.legend()
    plt.show()
