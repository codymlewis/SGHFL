from typing import Dict, Tuple
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    npoints = 1000
    nadversaries = 500
    attack = "multi_shifted_random"

    honest_x = rng.normal(1, 3, size=(npoints - nadversaries, 2))
    match attack:
        case "lie":
            s = npoints // 2 + 1 - nadversaries
            zmax = sp.stats.norm.ppf((npoints - s) / npoints)
            attack_x = np.tile(np.mean(honest_x, 0) + zmax * np.std(honest_x, 0), (nadversaries, 1))
        case "model_replacement":
            attack_x = rng.normal(3.5, 0.25, (nadversaries, 2)) * (npoints // 2 + 1)
        case "shifted_random":
            attack_x = rng.normal(3.5, np.std(honest_x, 0), (nadversaries, 2))
        case "closest_points":
            target = np.array([3.5, 3.5])
            dists = np.sqrt(np.sum(abs(honest_x - target)**2, axis=1))
            attack_x = honest_x[np.argsort(dists)[:nadversaries]] + rng.normal(0, 0.01, (nadversaries, 2))
        case "multi_shifted_random":
            attack_x = np.array([
                rng.normal([[3.5, 3.5], [3.0, 4.0], [4.0, 3.0]][i % 3], np.std(honest_x, 0), 2) for i in range(nadversaries)]
            )
    x = np.concatenate((honest_x, attack_x))

    plt.scatter(honest_x[:, 0], honest_x[:, 1], label="Honest points")
    plt.scatter(attack_x[:, 0], attack_x[:, 1], label="Attack points")
    honest_mean = np.mean(honest_x, 0)
    plt.scatter(honest_mean[0], honest_mean[1], marker="x", label="Honest mean")
    full_mean = np.mean(x, 0)
    plt.scatter(full_mean[0], full_mean[1], marker="x", label="Full mean")
    params = plusplus_init(x, npoints // 2 + 1)
    params = lloyds(params, x)
    centroids = params['centroids']
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="+", label="Centroids")
    centre_mean = np.mean(centroids, 0)
    plt.scatter(centre_mean[0], centre_mean[1], marker="x", label="Centre")
    # plt.legend()
    # plt.show()
    print(f"{x.mean(0)=}, {x.std(0)=}")
    print(f"{honest_x.mean(0)=}, {honest_x.std(0)=}")
    print(f"{attack_x.mean(0)=}, {attack_x.std(0)=}")
    print(f"{np.mean(centroids, 0)=}, {centroids.std(0)=}")
