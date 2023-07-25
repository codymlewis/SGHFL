from typing import List
import time
import gc
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


class EmptyUpdater(Client):
    def __init__(self, data, create_model_fn, corroborator):
        super().__init__(data, create_model_fn)

    def fit(self, parameters, config):
        _, samples, metrics = super().fit(parameters, config)
        return parameters, samples, metrics


class LIE(Client):
    def __init__(self, data, create_model_fn, corroborator):
        super().__init__(data, create_model_fn)
        self.corroborator = corroborator
        self.corroborator.register(self)
    
    def fit(self, parameters, config):
        z_max = self.corroborator.z_max
        history, mu, sigma = self.corroborator.calc_grad_stats(parameters, config)
        return [m + z_max * s for m, s in zip(mu, sigma)], len(self.data['train']), history
    
    def honest_fit(self, parameters, config):
        return super().fit(parameters, config)


class IPM(Client):
    def __init__(self, data, create_model_fn, corroborator):
        super().__init__(data, create_model_fn)
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


class BackdoorClient(Client):
    def __init__(self, data, create_model_fn, from_y, to_y):
        super().__init__(data, create_model_fn)
    
    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(self.data['train']['true X'], self.data['train']['true Y'], epochs=config['num_epochs'], steps_per_epoch=config.get("num_steps"), verbose=0)
        return self.model.get_weights(), len(self.data['train']), {k: v[-1] for k, v in history.history.items()}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.data['test']['true X'], self.data['test']['true Y'], verbose=0)
        backdoor_idx = self.data['test']['true Y'] == config['from_y']
        _, asr = self.model.evaluate(self.data['test']['X'][backdoor_idx], self.data['test']['Y'][backdoor_idx], verbose=0)
        return len(self.data['test']), {'loss': loss, 'accuracy': accuracy, 'asr': asr}


class BackdoorLIE(BackdoorClient):
    def __init__(self, data, create_model_fn, corroborator, from_y, to_y):
        super().__init__(data, create_model_fn, from_y, to_y)
        self.corroborator = corroborator
        self.corroborator.register(self)
    
    def fit(self, parameters, config):  # Note: the loss function is slightly wrong
        z_max = self.corroborator.z_max
        history, mu, sigma = self.corroborator.calc_grad_stats(parameters, config)
        self.model.set_weights(parameters)
        self.model.fit(self.data['train']['X'], self.data['train']['Y'], epochs=config['num_epochs'], steps_per_epoch=config.get("num_steps"), verbose=0)
        update = [np.clip(p, m - z_max * s, m + z_max * s) for p, m, s in zip(self.model.get_weights(), mu, sigma)]
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
        self.history = super().analytics(honest_metrics, honest_samples)
        return self.history, self.mu, self.sigma


def create_model() -> tf.keras.Model:
    inputs = tf.keras.Input((28, 28, 1))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(100, activation="relu")(x)
    x = tf.keras.layers.Dense(50, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def lda(labels, nclients, rng, alpha=0.5):
    """
    Latent Dirichlet allocation defined in https://arxiv.org/abs/1909.06335
    default value from https://arxiv.org/abs/2002.06440
    Optional arguments:
    - alpha: the alpha parameter of the Dirichlet function,
    the distribution is more i.i.d. as alpha approaches infinity and less i.i.d. as alpha approaches 0
    """
    distribution = [[] for _ in range(nclients)]
    nclasses = len(np.unique(labels))
    proportions = rng.dirichlet(np.repeat(alpha, nclients), size=nclasses)
    for c in range(nclasses):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        dists_c = np.split(idx_c, np.round(np.cumsum(proportions[c]) * len(idx_c)).astype(int)[:-1])
        distribution = [distribution[i] + d.tolist() for i, d in enumerate(dists_c)]
    return distribution


def create_clients(data, create_model_fn, network_arch, nadversaries, adversary_type, seed=None):
    Y = data['train']['Y']
    rng = np.random.default_rng(seed)
    idx = iter(lda(Y, count_clients(network_arch), rng, alpha=1000))
    nclients = count_clients(network_arch)
    corroborator = Corroborator(nclients)

    def create_client(client_id: str) -> Client:
        if int(client_id) > (nclients - nadversaries - 1):
            return adversary_type(data.select({"train": next(idx), "test": np.arange(len(data['test']))}), create_model_fn, corroborator)
        return Client(data.select({"train": next(idx), "test": np.arange(len(data['test']))}), create_model_fn)
    return create_client


class Centre(FedAVG):
    def aggregate(
        self, client_parameters: List[Parameters], client_samples: List[int], parameters: Parameters, config: Config
    ) -> Parameters:
        aggregated_parameters = []
        for i in range(len(client_parameters[0])):
            model = skc.KMeans(n_clusters=max(len(client_parameters) // 2 + 1, 1), n_init='auto')
            model.fit([cp[i].reshape(-1) for cp in client_parameters])
            aggregated_parameters.append(np.mean(model.cluster_centers_, axis=0).reshape(parameters[i].shape))
        return aggregated_parameters


class Median(FedAVG):
    def aggregate(
        self, client_parameters: List[Parameters], client_samples: List[int], parameters: Parameters, config: Config
    ) -> Parameters:
        return [np.median(layer, axis=0) for layer in to_attribute_array(client_parameters)]


def experiment(config):
    results = {}
    data = load_mnist()
    data = data.normalise()
    for strategy in [FedAVG, Median, Centre]:
        results[strategy.__name__] = {}
        for adversary_type in [EmptyUpdater, IPM, LIE]:
            train_results = []
            test_results = []
            for i in (pbar := trange(config['repeat'])):
                seed = round(np.pi**i + np.exp(i)) % 2**32
                tf.random.set_seed(seed)
                server = flagon.Server(create_model().get_weights(), config, strategy=strategy())
                network_arch = {"clients": config['num_clients']}
                history = flagon.start_simulation(
                    server,
                    create_clients(data, create_model, network_arch, nadversaries=config['num_adversaries'], adversary_type=adversary_type, seed=seed),
                    network_arch
                )
                train_results.append(history.aggregate_history)
                test_results.append(history.test_history)
            results[strategy.__name__][adversary_type.__name__] = {"train": train_results, "test": test_results}
    return results


def bd_create_clients(data, create_model_fn, network_arch, nadversaries, adversary_type, from_y, to_y, seed=None):
    Y = data['train']['true Y']
    rng = np.random.default_rng(seed)
    idx = iter(lda(Y, count_clients(network_arch), rng, alpha=1000))
    nclients = count_clients(network_arch)
    corroborator = Corroborator(nclients)

    def create_client(client_id: str) -> Client:
        client_idx = next(idx)
        if int(client_id) > (nclients - nadversaries - 1):
            return adversary_type(data.select({"train": np.arange(len(data['train'])), "test": np.arange(len(data['test']))}), create_model_fn, corroborator, from_y, to_y)
        return BackdoorClient(data.select({"train": client_idx, "test": np.arange(len(data['test']))}), create_model_fn, from_y, to_y)
    return create_client

def backdoor_experiment(config, results, strategy=FedAVG):
    data = load_mnist()
    data.map(backdoor_mapping(data, config['from_y'], config['to_y']))
    data.normalise()
    results[strategy.__name__] = {}
    train_results = []
    test_results = []
    for i in (pbar := trange(config['repeat'])):
        seed = round(np.pi**i + np.exp(i)) % 2**32
        tf.random.set_seed(seed)
        server = flagon.Server(create_model().get_weights(), config, strategy=strategy())
        network_arch = {"clients": config['num_clients']}
        history = flagon.start_simulation(
            server,
            bd_create_clients(
                data,
                create_model,
                network_arch,
                nadversaries=config['num_adversaries'],
                adversary_type=BackdoorLIE,
                from_y=config['from_y'],
                to_y=config['to_y'],
                seed=seed
            ),
            network_arch
        )
        train_results.append(history.aggregate_history)
        test_results.append(history.test_history)
        del server
        del network_arch
        gc.collect()
        results[strategy.__name__]["Backdoor LIE"] = {"train": train_results, "test": test_results}
    return results


def reorder_results(results):
    new_results = {}
    for strategy, strategy_results in results.items():
        new_results[strategy] = {}
        for attack, attack_results in strategy_results.items():
            new_results[strategy][attack] = {}
            for train_or_test in ["train", "test"]:
                new_results[strategy][attack][train_or_test] = {outer_k: {k: [] for k in ar.keys()} for outer_k, ar in attack_results[train_or_test][0].items()}
                for i in range(len(attack_results[train_or_test])):
                    for outer_k, vd in attack_results[train_or_test][i].items():
                        for k, v in vd.items():
                            new_results[strategy][attack][train_or_test][outer_k][k].append(v)
    return new_results


def plot_attack_results(results, key, train=True):
    fig, axes = plt.subplots(1, len(results.keys()), figsize=(18, 6))
    axes_iter = iter(axes)
    y_min, y_max = 1.0, 0.0
    for strategy, strategy_results in results.items():
        ax = next(axes_iter)
        line_symbols = itertools.cycle(['-o', '-s', '-^', '-x', '-d', '-p', '-*'])
        for attack, attack_results in strategy_results.items():
            rounds = list(attack_results['train' if train else 'test'].keys())
            y_mean = np.array([np.mean(ar[key]) for ar in attack_results['train' if train else 'test'].values()])
            # y_std = np.array([np.std(ar[key]) for ar in attack_results['train' if train else 'test'].values()])
            y_min = min(y_mean.min(), y_min)
            y_max = max(y_mean.max(), y_max)
            ax.plot(rounds, y_mean, next(line_symbols), label=attack)
            # plt.errorbar(rounds, y_mean, yerr=y_std, label=attack)
        ax.set_xlabel("Round")
        ax.set_ylabel(key.title())
        ax.legend(title="Attack")
        ax.set_title(strategy)
    for ax in axes:
        ax.set_ylim((y_min - 0.01, y_max + 0.01))
    plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    # results = experiment({"num_rounds": 5, "num_epochs": 1, "eval_every": 1, "num_clients": 10, "num_adversaries": 5, "repeat": 10})

    if os.path.exists("results/attack.json"):
        with open("results/attack.json", "r") as f:
            results = json.load(f)
    else:
        results = {}

    results = backdoor_experiment({"num_rounds": 5, "num_epochs": 1, "eval_every": 1, "num_clients": 50, "num_adversaries": 25, "from_y": 1, "to_y": 8, "repeat": 5}, results, Centre)

    with open("results/attack.json", "w") as f:
        json.dump(results, f)

    # results = reorder_results(results)
    # plot_attack_results(results, "accuracy", train=False)
    # plot_attack_results(results, "loss", train=False)
    # plot_attack_results(results, "asr", train=False)