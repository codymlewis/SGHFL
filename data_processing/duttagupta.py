"""
The region finding functionality from https://ieeexplore.ieee.org/document/10333889
"""
import numpy as np
from sklearn.cluster import KMeans


def find_regions(client_data):
    processed_dataset = np.array([
        np.concatenate((
            np.mean(cd, axis=0),
            np.median(cd, axis=0),
            np.sum(cd, axis=0),
            np.max(cd, axis=0),
            np.min(cd, axis=0),
        )) for cd in client_data.values()
    ])
    region_ids = KMeans(18).fit_predict(processed_dataset)
    return {cid: rid.tolist() for cid, rid in zip(client_data.keys(), region_ids)}
