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
    e2: float = 0.1,
    c: float = 0.5,
    eliminate_close: bool = True,
    take_topomap: bool = True,
    scale_by_overlap: bool = True,
    overlap_scaling_fn_name: str = "normal"
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
    if take_topomap:
        mu_dists = np.linalg.norm(samples - samples.mean(0), axis=1)
        topomap = np.array([np.sum(mu_dists <= i * e2 * sigma) for i in range(1, round(3 / e2))])
        topomap[1:] -= topomap[:-1]
        peak_indices = np.argwhere((topomap[1:-1] >= topomap[2:]) & (topomap[1:-1] > topomap[:-2])).reshape(-1) + 1
        sphere_centres = [np.array([samples.mean(0)])]
        sphere_scores = [np.array([(mu_dists < sigma).sum()])]
        # Find the densest points in each peak ring and add to sphere centres, use in proceeding part
        for pi in peak_indices:
            idx = np.where((mu_dists >= pi * e2 * sigma) & (mu_dists > (pi + 1) * e2 * sigma))
            spike_scores = (dists[idx] < sigma).sum(1)
            if spike_scores.shape[0] > 0:
                keep_idx = np.where(spike_scores > c * np.max(spike_scores))
                sphere_scores.append(spike_scores[keep_idx])
                sphere_centres.append(samples[keep_idx])
        sphere_centres = np.concatenate(sphere_centres)
        sphere_scores = np.concatenate(sphere_scores)
        sphere_centres, sci = np.unique(sphere_centres, return_index=True, axis=0)
        sphere_scores = sphere_scores[sci]
    else:
        scores = np.sum(dists, axis=1)
        sphere_centres = samples
        sphere_scores = scores
    # Scale scores according to expected proportion of unique points the sphere would contain
    if scale_by_overlap:
        centre_dists = sp.spatial.distance.cdist(sphere_centres, sphere_centres)
        ts = centre_dists / sigma
        match overlap_scaling_fn_name:
            case "non-overlap":
                dist_scaler = (1 - sp.stats.norm.cdf(ts)).sum(1)
            case "overlap":
                dist_scaler = sp.stats.norm.cdf(ts).sum(1)
            case "chi-overlap":
                dist_scaler = sp.stats.chi.cdf(np.sqrt((ts**2).sum(1)), ts.shape[-1])
            case "chi-non-overlap":
                dist_scaler = 1 - sp.stats.chi.cdf(np.sqrt((ts**2).sum(1)), ts.shape[-1])
                if dist_scaler.sum() == 0:
                    dist_scaler = np.ones_like(sphere_scores)
            case "distances":
                dist_scaler = ts.sum(1)
            case "similarities":
                dist_scaler = (1 - (ts - ts.min()) / (ts.max() - ts.min())).sum(1)
            case "density":
                dist_scaler = np.ones_like(sphere_scores)
            case "none":
                dist_scaler = 1 / sphere_scores
        # Use scaled density score to weight the average of the sphere centres
        p = dist_scaler * sphere_scores
    else:
        p = np.ones(len(sphere_centres))
    return np.average(sphere_centres, weights=p / p.sum(), axis=0)
