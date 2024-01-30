import json
import numpy as np
import einops
from safetensors.numpy import load_file

import data_manager


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
        client_data.append(data_manager.Dataset({
            "train": {"X": client_train_X, "Y": client_train_Y},
            "test": {"X": client_test_X, "Y": client_test_Y}
        }))
        X_test.append(client_data[-1]['test']['X'])
        Y_test.append(client_data[-1]['test']['Y'])
    X_test = np.concatenate(X_test)
    Y_test = np.concatenate(Y_test)

    return client_data, X_test, Y_test


def solar_home_regions():
    with open("../data/customer_regions.json", 'r') as f:
        customer_regions = json.load(f)
    regions = [[] for _ in np.unique(list(customer_regions.values()))]
    for customer, region_i in customer_regions.items():
        regions[region_i].append(int(customer) - 1)
    return regions


def apartment():
    train_data = load_file("../data/apartment_2015.safetensors")
    test_data = load_file("../data/apartment_2016.safetensors")

    client_data = []
    X_test, Y_test = [], []
    for c in train_data.keys():
        idx = np.arange(24, len(train_data[c]))
        expanded_idx = np.array([np.arange(i - 24, i - 1) for i in idx])
        client_train_X, client_train_Y = train_data[c][expanded_idx], train_data[c][idx, 0]
        client_train_X = einops.rearrange(client_train_X, 'b h s -> b (h s)')
        idx = np.arange(24, len(test_data[c]))
        expanded_idx = np.array([np.arange(i - 24, i - 1) for i in idx])
        client_test_X, client_test_Y = test_data[c][expanded_idx], test_data[c][idx, 0]
        client_test_X = einops.rearrange(client_test_X, 'b h s -> b (h s)')
        client_data.append(data_manager.Dataset({
            "train": {"X": client_train_X, "Y": client_train_Y},
            "test": {"X": client_test_X, "Y": client_test_Y}
        }))
        X_test.append(client_test_X)
        Y_test.append(client_test_Y)
    X_test = np.concatenate(X_test)
    Y_test = np.concatenate(Y_test)

    return client_data, X_test, Y_test


def apartment_regions():
    return [list(range(i, min(i + 10, 114))) for i in range(0, 114, 10)]


def load_data(dataset):
    return globals()[dataset]()


def load_regions(dataset):
    return globals()[f"{dataset}_regions"]()
