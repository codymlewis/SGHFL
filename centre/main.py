from typing import Dict, Tuple
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import sklearn.cluster as skc
import scipy as sp


def plusplus_init(samples: npt.NDArray, k: int = 8, seed: int = 0) -> Dict[str, npt.NDArray]:
    "K-Means++ initialisation algorithm from https://dl.acm.org/doi/10.5555/1283383.1283494"
    rng = np.random.default_rng(seed)
    num_samples = samples.shape[0]
    centroid_idx = rng.choice(num_samples)
    centroids = [samples[centroid_idx]]

    for i in range(k):
        dists = sp.spatial.distance.cdist(centroids, samples)
        weights = np.min(dists, axis=0)**2
        centroid_idx = rng.choice(num_samples, p=weights / weights.sum())
        centroid = samples[centroid_idx]
        centroids.append(centroid)

    return {"centroids": np.array(centroids)}


def lloyds(
    params: Dict[str, npt.NDArray], samples: npt.NDArray, num_iterations: int = 300, tol=1e-4
) -> Tuple[npt.NDArray, Dict[str, npt.NDArray]]:
    "Lloyd's algorithm for cluster finding from https://ieeexplore.ieee.org/document/1056489"
    centroids = params['centroids']

    for _ in range(num_iterations):
        dists = sp.spatial.distance.cdist(centroids, samples)
        clusters = np.argmin(dists, axis=0)
        new_centroids = centroids.copy()
        for i, cluster in enumerate(np.unique(clusters)):
            new_centroids[i] = np.mean(samples[clusters == cluster], 0)
        loss = np.linalg.norm(centroids - new_centroids)
        centroids = new_centroids
        if loss < tol:
            break

    return {"centroids": centroids}


def find_centre(samples: npt.NDArray) -> npt.NDArray:
    nsamples = len(samples)
    dists = sp.spatial.distance.cdist(samples, samples)
    threshold = np.max(dists) / np.sqrt(nsamples)
    dists += np.eye(nsamples) * threshold
    # Space sampling
    space_samples = samples[np.all(dists >= threshold, axis=1)]
    samples_from, samples_to = np.where(dists < threshold)
    overlapping_samples = np.unique(samples_from)
    for s in np.unique(samples_from):
        if s in overlapping_samples:
            overlapping_samples = np.setdiff1d(overlapping_samples, samples_from[samples_to == s])
    space_samples = np.concatenate((space_samples, samples[overlapping_samples]))
    # return space_samples.mean(0)

    # Try sphere sliding
    max_dist_idx = np.unravel_index(np.argmax(dists), dists.shape)
    radius = 3 * np.std(space_samples) / 2

    # Try largest cluster
    # clusters = skc.HDBSCAN().fit_predict(space_samples)
    # honest_cluster = np.argmax(np.bincount(clusters[clusters != -1]))
    # centre = space_samples[clusters == honest_cluster].mean(0)
    # return centre


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    npoints = 1000
    nadversaries = 200
    attack = "shifted_random"

    honest_x = rng.normal(1, 3, size=(npoints - nadversaries, 2))
    match attack:
        case "lie":
            s = npoints // 2 + 1 - nadversaries
            zmax = sp.stats.norm.ppf((npoints - s) / npoints)
            attack_x = np.tile(np.mean(honest_x, 0) + zmax * np.std(honest_x, 0), (nadversaries, 1))
        case "model_replacement":
            attack_x = rng.normal(3.5, 0.25, (nadversaries, 2)) * (npoints // 2 + 1)
        case "shifted_random":
            attack_x = rng.normal(5, np.std(honest_x, 0), (nadversaries, 2))
        case "closest_points":
            target = np.array([3.5, 3.5])
            dists = np.sqrt(np.sum(abs(honest_x - target)**2, axis=1))
            attack_x = honest_x[np.argsort(dists)[:nadversaries]] + rng.normal(0, 0.01, (nadversaries, 2))
        case "multi_shifted_random":
            attack_x = np.array([
                rng.normal([[3.5, 3.5], [3.0, 4.0], [4.0, 3.0]][i % 3], np.std(honest_x, 0), 2) for i in range(nadversaries)]
            )
    x = np.concatenate((honest_x, attack_x))
    print(f"{x.mean(0)=}, {x.std(0)=}")
    print(f"{honest_x.mean(0)=}, {honest_x.std(0)=}")
    print(f"{attack_x.mean(0)=}, {attack_x.std(0)=}")
    print(find_centre(x))

    plt.scatter(honest_x[:, 0], honest_x[:, 1], label="Honest points")
    plt.scatter(attack_x[:, 0], attack_x[:, 1], label="Attack points")
    honest_mean = np.mean(honest_x, 0)
    plt.scatter(honest_mean[0], honest_mean[1], marker="x", label="Honest mean")
    full_mean = np.mean(x, 0)
    plt.scatter(full_mean[0], full_mean[1], marker="x", label="Full mean")
    centre = find_centre(x)
    plt.scatter(centre[0], centre[1], marker="x", label="Centre")

    samples = x
    nsamples = len(samples)
    dists = sp.spatial.distance.cdist(samples, samples)
    threshold = np.max(dists) / np.sqrt(nsamples)
    print(f"{threshold=}")
    dists += np.eye(nsamples) * threshold
    # Space sampling
    space_samples = samples[np.all(dists >= threshold, axis=1)]
    samples_from, samples_to = np.where(dists < threshold)
    overlapping_samples = np.unique(samples_from)
    for s in np.unique(samples_from):
        if s in overlapping_samples:
            overlapping_samples = np.setdiff1d(overlapping_samples, samples_from[samples_to == s])
    space_samples = np.concatenate((space_samples, samples[overlapping_samples]))
    plt.scatter(space_samples[:, 0], space_samples[:, 1], marker="+", label="Space samples")

    plt.legend()
    plt.show()
