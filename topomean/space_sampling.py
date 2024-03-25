
# # Space sampling: To be used elsewhere
# nsamples = len(samples)
# dists = sp.spatial.distance.cdist(samples, samples)
# threshold = np.max(dists) / np.sqrt(nsamples)
# dists += np.eye(nsamples) * threshold
# space_samples = samples[np.all(dists >= threshold, axis=1)]
# samples_from, samples_to = np.where(dists < threshold)
# overlapping_samples = np.unique(samples_from)
# for s in np.unique(samples_from):
#     if s in overlapping_samples:
#         overlapping_samples = np.setdiff1d(overlapping_samples, samples_from[samples_to == s])
# space_samples = np.concatenate((space_samples, samples[overlapping_samples]))
