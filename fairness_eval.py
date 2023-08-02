from typing import List
import time
import json
import itertools
import datasets
import numpy as np
import scipy as sp
import sklearn.metrics as skm
import sklearn.cluster as skc
import sklearn.decomposition as skd
import einops
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm.auto import trange

import flagon
from flagon.strategy import FedAVG
from flagon.common import Config, Parameters, Metrics, count_clients, to_attribute_array
from flagon.strategy import FedAVG
import ntmg

import os
os.makedirs("results", exist_ok=True)


def load_mnist() -> ntmg.Dataset:
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


def create_model() -> tf.keras.Model:
    inputs = tf.keras.Input((28, 28, 1))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(100, activation="relu")(x)
    x = tf.keras.layers.Dense(50, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


class Client(flagon.Client):
    def __init__(self, data, create_model_fn):
        self.data = data
        self.model = create_model_fn()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(self.data['train']['X'], self.data['train']['Y'], epochs=config['num_epochs'], steps_per_epoch=config.get("num_steps"), verbose=0)
        return self.model.get_weights(), len(self.data['train']), {k: v[-1] for k, v in history.history.items()}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.data['test']['X'], self.data['test']['Y'], verbose=0)
        return len(self.data['test']), {'loss': loss, 'accuracy': accuracy}


def regional_distribution(labels, network_arch, rng, alpha=0.5):
    nmiddleservers = len(network_arch['clients'])
    nclients = [count_clients(subnet) for subnet in network_arch['clients']]
    distribution = [[] for _ in range(sum(nclients))]
    nclasses = len(np.unique(labels))
    proportions = rng.dirichlet(np.repeat(alpha, sum(nclients)), size=nclasses)
    client_i = 0
    for i in range(nmiddleservers):
        rdist = rng.dirichlet(np.repeat(alpha, nclients[i]))
        proportions[-(i + 1)] = np.zeros_like(proportions[-(i + 1)])
        proportions[-(i + 1)][client_i:client_i + nclients[i]] = rdist
        client_i += nclients[i]

    for c in range(nclasses):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        dists_c = np.split(idx_c, np.round(np.cumsum(proportions[c]) * len(idx_c)).astype(int)[:-1])
        distribution = [distribution[i] + d.tolist() for i, d in enumerate(dists_c)]
    return distribution


def regional_test_distribution(labels, network_arch):
    nmiddleservers = len(network_arch['clients'])
    nclients = [count_clients(subnet) for subnet in network_arch['clients']]
    distribution = [[] for _ in range(sum(nclients))]
    nclasses = len(np.unique(labels))
    client_i = 0
    for i, middle_server_nclients in enumerate(nclients):
        c = nclasses - i - 1
        for j in range(middle_server_nclients):
            distribution[client_i] = distribution[client_i] + np.where(labels == c)[0].tolist()
            client_i += 1

    for i in range(len(distribution)):
        distribution[i] = distribution[i] + np.where(~np.isin(labels, list(range(nclasses - 1, nclasses - nmiddleservers - 1, -1))))[0].tolist()
    return distribution


def create_clients(data, create_model_fn, network_arch, seed=None):
    rng = np.random.default_rng(seed)
    idx = iter(regional_distribution(data['train']['Y'], network_arch, rng))
    test_idx = iter(regional_test_distribution(data['test']['Y'], network_arch))
    nclients = count_clients(network_arch)
    data = data.normalise()

    def create_client(client_id: str):
        return Client(data.select({"train": next(idx), "test": next(test_idx)}), create_model_fn)
    return create_client


def experiment(config):
    aggregate_results = []
    test_results = []
    data = load_mnist()
    data = data.normalise()
    for i in (pbar := trange(config['repeat'])):
        seed = round(np.pi**i + np.exp(i)) % 2**32
        tf.random.set_seed(seed)
        server = flagon.Server(create_model().get_weights(), config, client_manager=DroppingClientManager(config['drop_round'], seed=seed))
        network_arch = {"clients": [{"clients": 3} for _ in range(5)]}
        history = flagon.start_simulation(
            server,
            create_clients(data, create_model, network_arch, seed=seed),
            network_arch
        )
        aggregate_results.append(history.aggregate_history[config['num_rounds']])
        test_results.append(history.test_history[config['num_rounds']])
        pbar.set_postfix(aggregate_results[-1])
    return {"train": aggregate_results, "test": test_results}


def fairness_analytics(client_metrics, client_samples, config):
    distributed_metrics = {k: [v] for k, v in client_metrics[0].items()}
    for cm in client_metrics[1:]:
        for k, v in cm.items():
            distributed_metrics[k].append(v)
    return {f"{k} std": np.std(v) for k, v in distributed_metrics.items()}


class DroppingClientManager(flagon.client_manager.ClientManager):
    def __init__(self, drop_round, seed=None):
        super().__init__()
        self.round = 0
        self.drop_round = drop_round
        self.test_clients = []
        self.rng = np.random.default_rng(seed)

    def register(self, client):
        super().register(client)
        self.test_clients.append(client)

    def sample(self):
        self.round += 1
        if self.round == self.drop_round:
            for _ in range(2):
                self.clients.pop()
                # del self.clients[self.rng.integers(len(self.clients))]
        return super().sample()

    def test_sample(self):
        return self.test_clients


if __name__ == "__main__":
    experiment_config = {"num_rounds": 5, "num_episodes": 1, "num_epochs": 1, "repeat": 10, "analytics": [fairness_analytics], "drop_round": 6}
    results = experiment(experiment_config)

    with open(f"results/fairness_r{experiment_config['num_rounds']}_e{experiment_config['num_episodes']}_s{experiment_config['num_epochs']}_dr{experiment_config['drop_round']}.json", "w") as f:
        json.dump(results, f)