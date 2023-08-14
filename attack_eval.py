import argparse
from functools import partial
import gc
import json
import numpy as np
import scipy as sp
import einops
from tqdm.auto import trange

import flagon
from flagon.strategy import FedAVG
from flagon.common import Config, Parameters, Metrics, count_clients, to_attribute_array
from flagon.strategy import FedAVG

import load_data
import strategy
import common
import client

import os
os.makedirs("results", exist_ok=True)


class EmptyUpdater(client.Client):
    def __init__(self, data, create_model_fn, corroborator, seed=None):
        super().__init__(data, create_model_fn, seed)

    def fit(self, parameters, config):
        _, samples, metrics = super().fit(parameters, config)
        return parameters, samples, metrics


class LIE(client.Client):
    def __init__(self, data, create_model_fn, corroborator, seed=None):
        super().__init__(data, create_model_fn, seed)
        self.corroborator = corroborator
        self.corroborator.register(self)
    
    def fit(self, parameters, config):
        z_max = self.corroborator.z_max
        history, mu, sigma = self.corroborator.calc_grad_stats(parameters, config)
        return [m + z_max * s for m, s in zip(mu, sigma)], len(self.data['train']), history
    
    def honest_fit(self, parameters, config):
        return super().fit(parameters, config)


class IPM(client.Client):
    def __init__(self, data, create_model_fn, corroborator, seed=None):
        super().__init__(data, create_model_fn, seed)
        self.corroborator = corroborator
        self.corroborator.register(self)
    
    def fit(self, parameters, config):
        history, mu, sigma = self.corroborator.calc_grad_stats(parameters, config)
        grads = [p - m for p, m in zip(parameters, mu)]
        return [p + (1.0 / self.corroborator.nadversaries) * g for p, g in zip(parameters, grads)], len(self.data['train']), history
    
    def honest_fit(self, parameters, config):
        return super().fit(parameters, config)


def backdoor_mapping(data, from_y, to_y): 
    trigger = np.zeros((28, 28, 1))
    trigger[:5, :5, :] = 1
    def _apply(example):
        backdoor_idx = example['Y'] == from_y
        return {
            "X": np.array([bx if backdoor else tx for backdoor, bx, tx in zip(backdoor_idx, np.minimum(example['X'] + trigger, 1), example['X'])]),
            "true X": example['X'],
            "Y": np.where(backdoor_idx, to_y, example['Y']),
            "true Y": example["Y"]
        }
    return _apply


class BackdoorClient(client.Client):
    def __init__(self, data, create_model_fn, from_y, to_y, seed=None):
        super().__init__(data, create_model_fn)
    
    def fit(self, parameters, config):
        self.model.set_parameters(parameters)
        metrics = self.model.step(
            self.data['train']['true X'],
            self.data['train']['true Y'],
            epochs=config['num_epochs'],
            steps_per_epoch=config.get("num_steps"),
            verbose=0
        )
        return self.model.get_parameters(), len(self.data['train']), metrics

    def evaluate(self, parameters, config):
        self.model.set_parameters(parameters)
        metrics = self.model.evaluate(
            self.data['test']['true X'],
            self.data['test']['true Y'],
            verbose=0
        )
        backdoor_idx = self.data['test']['true Y'] == config['from_y']
        attacked_metrics = self.model.evaluate(
            self.data['test']['X'][backdoor_idx],
            self.data['test']['Y'][backdoor_idx],
            verbose=0
        )
        metrics['asr'] = attacked_metrics['accuracy']
        return len(self.data['test']), metrics


class BackdoorLIE(BackdoorClient):
    def __init__(self, data, create_model_fn, corroborator, from_y, to_y, seed=None):
        super().__init__(data, create_model_fn, from_y, to_y, seed)
        self.corroborator = corroborator
        self.corroborator.register(self)
    
    def fit(self, parameters, config):  # Note: the loss function is slightly wrong
        z_max = self.corroborator.z_max
        history, mu, sigma = self.corroborator.calc_grad_stats(parameters, config)
        self.model.set_parameters(parameters)
        self.model.step(
            self.data['train']['X'],
            self.data['train']['Y'],
            epochs=config['num_epochs'],
            steps_per_epoch=config.get("num_steps"),
            verbose=0
        )
        update = [
            np.clip(p, m - z_max * s, m + z_max * s)
            for p, m, s in zip(self.model.get_parameters(), mu, sigma)
        ]
        return update, len(self.data['train']), history
    
    def honest_fit(self, parameters, config):  # Note: only works correctly when there is no moment computation in the optimizer
        return super().fit(parameters, config)


class Corroborator(FedAVG):
    def __init__(self, nclients):
        self.nclients = nclients
        self.adversaries = []
        self.round = 0
        self.mu = None
        self.sigma = None
        self.history = None

    @property
    def nadversaries(self):
        return len(self.adversaries)

    def register(self, adversary):
        self.adversaries.append(adversary)

    @property
    def z_max(self):
        s = self.nclients // 2 + 1 - self.nadversaries
        return sp.stats.norm.ppf((self.nclients - s) / self.nclients)

    def calc_grad_stats(self, parameters, config):
        if self.round == config['round']:
            return self.history, self.mu, self.sigma
        self.round = config['round']

        honest_parameters = []
        honest_samples = []
        honest_metrics = []
        for a in self.adversaries:
            parameters, samples, metrics = a.honest_fit(parameters, config)
            honest_parameters.append(parameters)
            honest_samples.append(samples)
            honest_metrics.append(metrics)

        # Does some aggregation
        attr_honest_parameters = to_attribute_array(honest_parameters)
        self.mu = [np.average(layer, weights=honest_samples, axis=0) for layer in attr_honest_parameters]
        self.sigma = [np.sqrt(np.average((layer - m)**2, weights=honest_samples, axis=0)) for layer, m in zip(attr_honest_parameters, self.mu)]
        self.history = super().analytics(honest_metrics, honest_samples, config)
        return self.history, self.mu, self.sigma


def create_clients(data, create_model_fn, network_arch, nadversaries, adversary_type, seed=None):
    Y = data['train']['Y']
    rng = np.random.default_rng(seed)
    idx = iter(common.lda(Y, count_clients(network_arch), rng, alpha=1000))
    nclients = count_clients(network_arch)
    corroborator = Corroborator(nclients)

    def create_client(client_id: str) -> client.Client:
        if int(client_id) > (nclients - nadversaries - 1):
            return adversary_type(data.select({"train": next(idx), "test": np.arange(len(data['test']))}), create_model_fn, corroborator)
        return client.Client(data.select({"train": next(idx), "test": np.arange(len(data['test']))}), create_model_fn, seed)
    return create_client


def bd_create_clients(data, create_model_fn, network_arch, nadversaries, adversary_type, from_y, to_y, seed=None):
    Y = data['train']['true Y']
    rng = np.random.default_rng(seed)
    idx = iter(common.lda(Y, count_clients(network_arch), rng, alpha=1000))
    nclients = count_clients(network_arch)
    corroborator = Corroborator(nclients)

    def create_client(client_id: str) -> client.Client:
        client_idx = next(idx)
        if int(client_id) > (nclients - nadversaries - 1):
            return adversary_type(data.select({"train": np.arange(len(data['train'])), "test": np.arange(len(data['test']))}), create_model_fn, corroborator, from_y, to_y, seed)
        return BackdoorClient(data.select({"train": client_idx, "test": np.arange(len(data['test']))}), create_model_fn, from_y, to_y, seed)
    return create_client


def experiment(config):
    results = {}
    if config['dataset'] == "fmnist":
        data = load_data.mnist()
    if config.get("from_y"):
        data.map(backdoor_mapping(data, config['from_y'], config['to_y']))
    data = data.normalise()

    strategy_type = {"fedavg": FedAVG, "median": strategy.Median, "centre": strategy.Centre}[config['aggregator']]
    if config.get("from_y"):
        adversary_type = BackdoorLIE
    else:
        adversary_type = {"empty": EmptyUpdater, "ipm": IPM, "lie": LIE}[config['attack']]
    train_results = []
    test_results = []
    for i in (pbar := trange(config['repeat'])):
        seed = round(np.pi**i + np.exp(i)) % 2**32
        server = flagon.Server(
            {"fmnist": common.create_fmnist_model}[config['dataset']]().get_parameters(),
            config,
            strategy=strategy_type()
        )
        network_arch = {"clients": config['num_clients']}
        history = flagon.start_simulation(
            server,
            (partial(bd_create_clients, from_y=config['from_y'], to_y=config['to_y']) if config.get("from_y") else create_clients)(
                data,
                common.create_fmnist_model,
                network_arch,
                nadversaries=config['num_adversaries'],
                adversary_type=adversary_type,
                seed=seed
            ),
            network_arch
        )
        train_results.append(history.aggregate_history)
        test_results.append(history.test_history)
        del server
        del network_arch
        gc.collect()
    return {"train": train_results, "test": test_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform experiments evaluating attacks upon FL.")
    parser.add_argument("-i", "--id", type=int, default=1, help="Which of the experiments in the config to perform (counts from 1).")
    parser.add_argument("-d", "--dataset", type=str, default="fmnist", help="Which of the datasets to perform the experiment with.")
    args = parser.parse_args()

    with open("configs/attack.json", 'r') as f:
        experiment_config = common.get_experiment_config(json.load(f), args.id)
    experiment_config["dataset"] = args.dataset

    results = experiment(experiment_config)

    filename = "results/attack_{}.json".format(
        '_'.join([f'{k}={v}' for k, v in experiment_config.items()])
    )
    with open(filename, "w") as f:
        json.dump(results, f)
    print(f"Saved results to {filename}")