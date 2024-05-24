from __future__ import annotations
import json
import numpy as np
import chex
import einops
from safetensors.numpy import load_file


class Dataset:
    def __init__(self, X: chex.Array, Y: chex.Array, index: int = 0):
        self.X = X
        self.Y = Y
        self.i = index

    def online(dataset_size: int, forecast_window: int) -> Dataset:
        return Dataset(
            np.zeros((dataset_size, 2 * forecast_window + 2)),
            np.zeros((dataset_size, 2)),
            0
        )

    def offline(X: chex.Array, Y: chex.Array) -> Dataset:
        return Dataset(X, Y, Y.shape[0])

    def add(self, x, y):
        self.i = (self.i + 1) % self.Y.shape[0]
        self.X[self.i] = x
        self.Y[self.i] = y

    def __len__(self) -> int:
        return min(self.i, self.Y.shape[0])


def process_regions_json(regions_json_fn):
    with open(regions_json_fn, 'r') as f:
        client_regions = json.load(f)
    regions = [[] for _ in np.unique(list(client_regions.values()))]
    for client, region_i in client_regions.items():
        regions[region_i].append(int(client) - 1)
    return regions


def solar_home():
    train_data = load_file("../data/solar_home_2010-2011.safetensors")
    test_data = load_file("../data/solar_home_2011-2012.safetensors")

    client_data = []
    X_test, Y_test = [], []
    for c in train_data.keys():
        idx = np.arange(24, len(train_data[c]))
        expanded_idx = np.array([np.arange(i - 24, i - 1) for i in idx])
        client_train_X, client_train_Y = train_data[c][expanded_idx], train_data[c][idx, :2]
        client_train_X = einops.rearrange(client_train_X, 'b h s -> b (h s)')
        idx = np.arange(24, len(test_data[c]))
        expanded_idx = np.array([np.arange(i - 24, i - 1) for i in idx])
        client_test_X, client_test_Y = test_data[c][expanded_idx], test_data[c][idx, :2]
        client_test_X = einops.rearrange(client_test_X, 'b h s -> b (h s)')
        client_data.append(Dataset.offline(client_train_X, client_train_Y))
        X_test.append(client_test_X)
        Y_test.append(client_test_Y)
    X_test = np.concatenate(X_test)
    Y_test = np.concatenate(Y_test)
    test_idx = np.random.default_rng(4568).choice(len(Y_test), len(Y_test) // 20, replace=False)

    return client_data, X_test[test_idx], Y_test[test_idx]


def apartment():
    train_data = load_file("../data/apartment_2015.safetensors")
    test_data = load_file("../data/apartment_2016.safetensors")

    client_data = []
    X_test, Y_test = [], []
    for c in train_data.keys():
        idx = np.arange(24, len(train_data[c]))
        expanded_idx = np.array([np.arange(i - 24, i - 1) for i in idx])
        client_train_X, client_train_Y = train_data[c][expanded_idx], train_data[c][idx, 0].reshape(-1, 1)
        client_train_X = einops.rearrange(client_train_X, 'b h s -> b (h s)')
        idx = np.arange(24, len(test_data[c]))
        expanded_idx = np.array([np.arange(i - 24, i - 1) for i in idx])
        client_test_X, client_test_Y = test_data[c][expanded_idx], test_data[c][idx, 0].reshape(-1, 1)
        client_test_X = einops.rearrange(client_test_X, 'b h s -> b (h s)')
        client_data.append(Dataset.offline(client_train_X, client_train_Y))
        X_test.append(client_test_X)
        Y_test.append(client_test_Y)
    X_test = np.concatenate(X_test)
    Y_test = np.concatenate(Y_test)
    test_idx = np.random.default_rng(3784).choice(len(Y_test), len(Y_test) // 10, replace=False)

    return client_data, X_test[test_idx], Y_test[test_idx]


def load_data(dataset):
    match dataset:
        case "solar_home":
            return solar_home()
        case "apartment":
            return apartment()
        case _:
            raise NotImplementedError(f"Dataset {dataset} is not implemented")


def load_regions(dataset, duttagupta=False):
    if dataset not in ["solar_home", "apartment", "l2rpn"]:
        raise NotImplementedError(f"Dataset region for {dataset} is not implemented")
    return process_regions_json(f"../data/{dataset}_{'duttagupta_' if duttagupta else ''}regions.json")
