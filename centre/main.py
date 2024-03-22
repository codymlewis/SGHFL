import argparse
import time
import os
import numpy as np
import numpy.typing as npt
import scipy as sp
from tqdm import trange


def find_topomean(samples: npt.NDArray, e1: float = 0.01, e2: float = 0.1) -> npt.NDArray:
    """
    Assumptions:
    - Attacking clients are in the minority
    - Updates are i.i.d.
    - Updates follow a normal distribution
    """
    # Eliminate samples that are too close to eachother, leaving only one representative
    dists = sp.spatial.distance.cdist(samples, samples)
    far_enough_idx = np.all((dists + (np.eye(len(samples)) * e1)) >= e1, axis=0)
    samples = samples[far_enough_idx]
    dists = dists[np.ix_(far_enough_idx, far_enough_idx)]
    radius = np.std(samples) * e2
    # Find and take only the highest scoring neighbourhoods
    neighbourhoods = dists <= radius
    scores = np.sum(neighbourhoods, axis=1)
    sphere_idx = np.argpartition(-scores, len(scores) // 3)[:len(scores) // 3]
    sphere_scores = scores[sphere_idx]
    sphere_centres = np.einsum('bx,ab->bx', samples, neighbourhoods / neighbourhoods.sum(1))
    sphere_centres = sphere_centres[sphere_idx]
    # Scale scores according to expected proportion of unique points the sphere would contain
    centre_dists = sp.spatial.distance.cdist(sphere_centres, sphere_centres)
    ts = centre_dists / np.std(samples)
    non_overlap = 1 - sp.stats.norm.cdf(ts)
    # Use scaled density score to weight the average of the sphere centres
    p = non_overlap[np.argmax(non_overlap.sum(1))]
    p = (p / p.sum()) * sphere_scores
    return np.average(sphere_centres, weights=p / p.sum(), axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Topomean synthetic testbed program.")
    parser.add_argument('-s', '--seed', type=int, default=14258, help="Initial seed for the experiments.")
    parser.add_argument('-r', '--repetitions', type=int, default=100,
                        help="Number of times to repeat the experiment")
    parser.add_argument('-n', '--npoints', type=int, default=1000, help="Number of points to evaluate for.")
    parser.add_argument('-a', '--attack', type=str, default="shifted_random", help="Type of attack to perform.")
    parser.add_argument('-p', '--padversaries', type=float, default=0.4,
                        help="Proportion of points to assign as adversarial.")
    parser.add_argument("-g", "--aggregator", type=str, default="topomean", help="Aggregation algorithm to use.")
    args = parser.parse_args()
    print(f"Experiment args: {vars(args)}")

    start_time = time.time()
    rng = np.random.default_rng(args.seed)
    nadversaries = round(args.npoints * args.padversaries)

    errors = np.zeros(args.repetitions)
    improvements = np.zeros(args.repetitions)
    for r in (pbar := trange(args.repetitions)):
        honest_x = rng.normal(1, 3, size=(args.npoints - nadversaries, 2))
        match args.attack:
            case "lie":
                s = args.npoints // 2 + 1 - nadversaries
                zmax = sp.stats.norm.ppf((args.npoints - s) / args.npoints)
                attack_x = np.tile(np.mean(honest_x, 0) + zmax * np.std(honest_x, 0), (nadversaries, 1))
            case "shifted_random":
                attack_x = rng.normal(6, np.std(honest_x, 0), (nadversaries, 2))
        x = np.concatenate((honest_x, attack_x))
        agg_mean = find_topomean(x)
        honest_mean = honest_x.mean(0)
        # attack_mean = attack_x.mean(0)
        full_mean = x.mean(0)
        errors[r] = np.linalg.norm(honest_mean - agg_mean)
        improvements[r] = 1 - errors[r] / np.linalg.norm(honest_mean - full_mean)
        pbar.set_postfix_str(f"ERR: {errors[r]:.3f}, IMP: {improvements[r]:.3%}")

    print(("=" * 20) + " Results " + ("=" * 20))
    print(f"Average error: {errors.mean()}")
    print(f"Average improvement: {improvements.mean():%}")
    print("=" * (40 + 9))

    # Save results to csv
    results_fn = "results.csv"
    experiment_results = vars(args)  # Include the information
    experiment_results["error"] = errors.mean()
    experiment_results["improvement"] = improvements.mean()
    if not os.path.exists(results_fn):
        with open(results_fn, "w") as f:
            f.write(",".join(experiment_results.keys()) + "\n")
    with open(results_fn, 'a') as f:
        f.write(",".join([str(v) for v in experiment_results.values()]) + "\n")
    print(f"Results written to {results_fn}")

    print(f"Experiment took {time.time() - start_time} seconds")
