import numpy as np
import numpy.typing as npt
import scipy as sp


def mean(samples: npt.NDArray) -> npt.NDArray:
    return np.mean(samples, axis=0)


def median(samples: npt.NDArray) -> npt.NDArray:
    return np.median(samples, axis=0)


def geomedian(samples: npt.NDArray) -> npt.NDArray:
    return sp.optimize.minimize(
        lambda x: np.linalg.norm(samples - x),
        x0=np.median(samples)
    ).x


def krum(samples: npt.NDArray, c: float = 0.5) -> npt.NDArray:
    n = len(samples)
    clip = round(c * n)
    scores = np.zeros(n)
    distances = sp.spatial.distance.cdist(samples, samples)
    for i in range(n):
        scores[i] = np.sum(np.sort(distances[i])[1:((n - clip) - 1)])
    idx = np.argpartition(scores, n - clip)[:(n - clip)]
    return np.mean(samples[idx], axis=0)


def trmean(samples: npt.NDArray, c: float = 0.5) -> npt.NDArray:
    reject_i = round((c / 2) * len(samples))
    sorted_samples = np.sort(samples, axis=0)
    return np.mean(sorted_samples[reject_i:-reject_i], axis=0)


def phocas(samples: npt.NDArray, c: float = 0.5) -> npt.NDArray:
    trimmed_mean = trmean(samples, c)
    tm_closest_idx = np.argsort(np.linalg.norm(samples - trimmed_mean, axis=0))[:round((1 - c) * len(samples))]
    return np.mean(samples[tm_closest_idx], axis=0)


def ssfgm(
    samples: npt.NDArray,
    r: float = 0.01,
    # e1: float = 0.1,
    c: float = 0.8,
    space_sampling: bool = True,
    fractional_geomedian: bool = True,
) -> npt.NDArray:
    """
    Assumptions:
    - Attacking clients are in the minority
    - Updates are i.i.d.
    """
    # Eliminate samples that are too close to eachother, leaving only one representative
    dists = sp.spatial.distance.cdist(samples, samples)
    if space_sampling:
        sigma = np.std(samples)
        far_enough_idx = np.all((dists + (np.eye(len(samples)) * r * sigma)) >= (r * sigma), axis=0)
        samples = samples[far_enough_idx]
    # Perform the fractional geometric median
    if fractional_geomedian:
        # mu = np.mean(samples, axis=0)
        # sigma = np.std(samples)
        # mu_dists = np.linalg.norm(samples - mu, axis=1)
        # sonar = np.array([
        #     np.sum((mu_dists <= i * e1 * sigma) & (mu_dists > (i - 1) * e1 * sigma))
        #     for i in range(1, round(3 / 0.1))
        # ])
        # c = 1 - np.sum((sonar[1:-1] >= sonar[2:]) & (sonar[1:-1] > sonar[:-2])) / len(sonar)
        # print(f"{c=}")
        k = round(len(samples) * c) - 1
        return sp.optimize.minimize(
            lambda x: np.sum(np.partition(np.linalg.norm(samples - x, axis=1), k)[:k]),
            np.mean(samples, axis=0)
        ).x
    else:
        return np.mean(samples, axis=0)
