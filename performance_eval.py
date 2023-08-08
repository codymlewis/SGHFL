import argparse
from typing import List
import pickle
import time
import json
import itertools
import datasets
import numpy as np
import einops
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from tqdm.auto import trange

import flagon
from flagon.strategy import FedAVG
from flagon.common import Config, Parameters, Metrics, count_clients, to_attribute_array
from flagon.strategy import FedAVG
import ntmg

import flax_lightning

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


def load_solar_home():
    with open("data/solar_home_data.pkl", 'rb') as f:
        data = pickle.load(f)

    def get_customer_data(customer=1):
        idx = np.arange(24, len(data[customer]))
        expanded_idx = np.array([np.arange(i - 24, i - 1) for i in idx])
        return data[customer][expanded_idx], data[customer][idx, 0]
    return get_customer_data


def load_sh_customer_regions():
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

def create_sh_clients(create_model_fn, nclients, client_ids, seed=None):
    get_customer_data = load_solar_home()

    def create_client(client_id: str):
        client_X, client_Y = get_customer_data(client_ids[client_id])
        client_data = {"train": {"X": client_X[:300 * 24], "Y": client_Y[:300 * 24]}, "test": {"X": client_X[300 * 24:], "Y": client_Y[300 * 24:]}}
        return Client(client_data, create_model_fn)
    return create_client

class Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x = nn.Conv(64, (5, 5))(x)
        # x = nn.relu(x)
        # x = nn.Conv(64, (3, 3))(x)
        # x = nn.relu(x)
        # x = nn.Conv(32, (2, 2))(x)
        # x = nn.relu(x)
        x = einops.rearrange(x, "b h s -> b (h s)")
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(50)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

def create_sh_model(seed=None):
    model = Net()
    params = model.init(jax.random.PRNGKey(seed if seed else 42), jnp.zeros((1, 23, 4)))
    return flax_lightning.Model(
        model,
        params,
        optax.sgd(0.01, momentum=0.9),
        "mean_absolute_error",
        metrics=["mean_absolute_error"],
        seed=seed
    )


class LeNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = einops.rearrange(x, "b w h c -> b (w h c)")
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(50)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return nn.softmax(x)


def create_model(seed=None):
    model = LeNet()
    params = model.init(jax.random.PRNGKey(seed if seed else 42), jnp.zeros((1, 28, 28, 1)))
    return flax_lightning.Model(
        model,
        params,
        optax.sgd(0.01, momentum=0.9),
        "crossentropy_loss",
        metrics=["accuracy", "crossentropy_loss"],
        seed=seed
    )


class Client(flagon.Client):
    def __init__(self, data, create_model_fn, seed=None):
        self.data = data
        self.model = create_model_fn(seed)

    def fit(self, parameters, config):
        self.model.set_parameters(parameters)
        metrics = self.model.step(
            self.data['train']['X'],
            self.data['train']['Y'],
            epochs=config['num_epochs'],
            steps_per_epoch=config.get("num_steps"),
            verbose=0
        )
        return self.model.get_parameters(), len(self.data['train']), metrics

    def evaluate(self, parameters, config):
        self.model.set_parameters(parameters)
        metrics = self.model.evaluate(self.data['test']['X'], self.data['test']['Y'], verbose=0)
        return len(self.data['test']), metrics


class KickbackMomentum(FedAVG):
    def __init__(self):
        self.momentum = None
        self.prev_parameters = None
        self.episode = 0

    def aggregate(
        self, client_parameters: List[Parameters], client_samples: List[int], parameters: Parameters, config: Config
    ) -> Parameters:
        grads = [np.average(clayer, weights=client_samples, axis=0) - slayer for clayer, slayer in zip(to_attribute_array(client_parameters), parameters)]
        if self.episode % config['num_episodes'] == 0:
            if self.momentum is None:
                self.momentum = [np.zeros_like(p) for p in parameters]
                self.prev_parameters = parameters
            else:
                self.momentum = [config["mu1"] * m + (p - pp) for m, pp, p in zip(self.momentum, self.prev_parameters, parameters)]
                self.prev_parameters = parameters
        self.episode += 1
        return [p + config["mu2"] * m + g for p, m, g in zip(self.prev_parameters, self.momentum, grads)]


class IntermediateFineTuner(flagon.MiddleServer):
    def evaluate(self, parameters, config):
        flagon.common.logger.info("Starting finetuning on middle server")
        strategy = FedAVG()  # Use standard FedAVG for finetuning since it does not need to conform with the upper tier
        start_time = time.time()
        tuned_parameters = parameters
        for e in range(1, config['num_finetune_episodes'] + 1):
            client_parameters = []
            client_samples = []
            client_metrics = []
            clients = self.client_manager.sample()
            for c in clients:
                parameters, samples, metrics = c.fit(tuned_parameters, config)
                client_parameters.append(parameters)
                client_samples.append(samples)
                client_metrics.append(metrics)
            tuned_parameters = strategy.aggregate(
                client_parameters, client_samples, tuned_parameters, config
            )
        flagon.common.logger.info(f"Completed middle server finetuning in {time.time() - start_time}s")

        flagon.common.logger.info("Performing analytics on middle server")
        start_time = time.time()
        client_samples = []
        client_metrics = []
        clients = self.client_manager.sample()
        for c in clients:
            samples, metrics = c.evaluate(tuned_parameters, config)
            client_samples.append(samples)
            client_metrics.append(metrics)
        flagon.common.logger.info(f"Completed middle server analytics in {time.time() - start_time}s")
        aggregated_metrics = self.strategy.analytics(client_metrics, client_samples, config)
        flagon.common.logger.info(f"Aggregated final metrics {aggregated_metrics}")

        return sum(client_samples), aggregated_metrics


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


class CosineSimilarity(flagon.common.Metric):
    def __call__(self, global_parameters, client_parameters, client_samples):
        client_parameters = [np.concatenate([cl.reshape(-1) for cl in cp]) for cp in client_parameters]
        similarity_matrix = skm.pairwise.cosine_similarity(client_parameters) - np.eye(len(client_parameters))
        return similarity_matrix.sum() / (len(client_parameters) * (len(client_parameters) - 1))


def experiment(config, strategy_class, middle_server_class=flagon.MiddleServer):
    aggregate_results = []
    test_results = []
    if config['dataset'] == "fmnist":
        data = load_mnist()
        data = data.normalise()
    else:
        data_collector_counts, client_ids = load_sh_customer_regions()
    for i in (pbar := trange(config['repeat'])):
        seed = round(np.pi**i + np.exp(i)) % 2**32
        if config['dataset'] == "fmnist":
            server = flagon.Server(create_model().get_parameters(), config)
            network_arch = {
                "clients": [
                    {"clients": 3, "strategy": strategy_class(), "middle_server_class": middle_server_class} for _ in range(5)
                ]
            }
            clients = create_clients(data, create_model, network_arch, seed=seed)
        else:
            server = flagon.Server(create_sh_model().get_parameters(), config)
            network_arch = {
                "clients": [{"clients": 0} for _ in data_collector_counts.keys()],
            }
            for k, v in data_collector_counts.items():
                network_arch['clients'][k]['clients'] = v
            clients = create_sh_clients(
                create_sh_model, flagon.common.count_clients(network_arch), client_ids, seed=seed
            )
        history = flagon.start_simulation(
            server,
            clients,
            network_arch
        )
        aggregate_results.append(history.aggregate_history[config['num_rounds']])
        test_results.append(history.test_history[config['num_rounds']])
        pbar.set_postfix(aggregate_results[-1])
    return {"train": aggregate_results, "test": test_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform experiments evaluating the fairness when clients drop out from colloboration.")
    parser.add_argument("-i", "--id", type=int, default=1, help="Which of the experiments in the config to perform (counts from 1).")
    parser.add_argument("-d", "--dataset", type=str, default="fmnist", help="Which of the datasets to perform the experiment with.")
    args = parser.parse_args()

    with open("configs/performance.json", 'r') as f:
        experiment_config = json.load(f)[args.id - 1]
    experiment_config["metrics"] = [CosineSimilarity()]
    experiment_config["dataset"] = args.dataset

    results = experiment(
        experiment_config,
        KickbackMomentum if experiment_config.get("mu1") else FedAVG,
        IntermediateFineTuner if experiment_config.get("num_finetune_episodes") else flagon.MiddleServer
    )

    filename = "results/performance_{}{}.json".format(
        '_'.join([f'{k}={v}' for k, v in experiment_config.items() if k not in ['metrics', 'round', 'mu1', 'mu2']]),
        "_momentum" if experiment_config.get("mu1") else ""
    )
    with open(filename, "w") as f:
        json.dump(results, f)
    print(f"Saved results to {filename}")