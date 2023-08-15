import datasets
import numpy as np
import datasets
import json
import pickle
import sklearn.preprocessing as skp
import einops

import ntmg


def mnist() -> ntmg.Dataset:
    """
    Load the Fashion MNIST dataset http://arxiv.org/abs/1708.07747

    Arguments:
    - seed: seed value for the rng used in the dataset
    """
    ds = datasets.load_dataset("fashion_mnist")
    ds = ds.map(
        lambda e: {
            'X': einops.rearrange(np.array(e['image'], dtype=np.float32) / 255, "h (w c) -> h w c", c=1),
            'Y': e['label']
        },
        remove_columns=['image', 'label']
    )
    features = ds['train'].features
    features['X'] = datasets.Array3D(shape=(28, 28, 1), dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    data = {t: {'X': ds[t]['X'], 'Y': ds[t]['Y']} for t in ['train', 'test']}
    dataset = ntmg.Dataset(data)
    return dataset


def solar_home():
    with open("data/solar_home_data.pkl", 'rb') as f:
        data = pickle.load(f)

    concat_data = np.concatenate(list(data.values()))
    X_processor = skp.StandardScaler().fit(concat_data)
    Y_processor = skp.MinMaxScaler().fit(concat_data[:, 0].reshape(-1, 1))

    def get_customer_data(customer=1):
        idx = np.arange(24, len(data[customer]))
        expanded_idx = np.array([np.arange(i - 24, i - 1) for i in idx])
        return (
            X_processor.transform(data[customer][expanded_idx].reshape(-1, 4)).reshape(-1, 23, 4),
            Y_processor.transform(data[customer][idx, 0].reshape(-1, 1)).reshape(-1)
        )
    return get_customer_data


def solar_home_customer_regions():
    with open("data/customer_regions.json", 'r') as f:
        customer_regions = json.load(f)

    data_collector_counts = {}
    client_ids = {}
    for customer, region in customer_regions.items():
        data_collector = region
        if not data_collector_counts.get(data_collector):
            data_collector_counts[data_collector] = 0
        client_ids[f"{data_collector}-{data_collector_counts[data_collector]}"] = int(customer)
        data_collector_counts[data_collector] += 1
    return data_collector_counts, client_ids