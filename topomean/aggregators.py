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


def topomean(
    samples: npt.NDArray,
    e1: float = 0.01,
    e2: float = 1.0,
    K: int = 3,
    eliminate_close: bool = True,
    take_dense_spheres: bool = True,
    scale_by_overlap: bool = True,
) -> npt.NDArray:
    """
    Assumptions:
    - Attacking clients are in the minority
    - Updates are i.i.d.
    - Updates follow a normal distribution
    """
    sigma = np.std(samples)
    # Eliminate samples that are too close to eachother, leaving only one representative
    dists = sp.spatial.distance.cdist(samples, samples)
    if eliminate_close:
        far_enough_idx = np.all((dists + (np.eye(len(samples)) * e1 * sigma)) >= (e1 * sigma), axis=0)
        samples = samples[far_enough_idx]
        dists = dists[np.ix_(far_enough_idx, far_enough_idx)]
    # Find and take only the highest scoring neighbourhoods
    sigma = np.std(samples)
    radius = sigma * e2
    if take_dense_spheres:
        neighbourhoods = dists <= radius
        scores = np.sum(neighbourhoods, axis=1)
        sphere_idx = np.argpartition(-scores, len(scores) // K)[:len(scores) // K]
        sphere_scores = scores[sphere_idx]
        sphere_centres = np.einsum('bx,bd -> dx', samples, neighbourhoods / neighbourhoods.sum(1))
        sphere_centres = sphere_centres[sphere_idx]
    else:
        scores = np.sum(dists, axis=1)
        sphere_centres = samples
        sphere_scores = scores
    # Scale scores according to expected proportion of unique points the sphere would contain
    if scale_by_overlap:
        centre_dists = sp.spatial.distance.cdist(sphere_centres, sphere_centres)
        ts = centre_dists / sigma
        non_overlap = 1 - sp.stats.norm.cdf(ts).sum(1)
        # Use scaled density score to weight the average of the sphere centres
        p = non_overlap * sphere_scores
    else:
        p = np.ones(len(sphere_centres))
    return np.average(sphere_centres, weights=p / p.sum(), axis=0)
