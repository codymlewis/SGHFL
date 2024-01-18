from typing import List, Dict
from functools import partial
import argparse
import pickle
import json
import time
import logging
import os
import numpy as np
from numpy.typing import NDArray
import einops
import sklearn.metrics as skm
import sklearn.cluster as skc
import scipy as sp
import scipy.optimize as sp_opt
from tqdm import tqdm, trange
from safetensors.numpy import load_file

import data_manager


logger = logging.getLogger("solar home experiment")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('| %(name)s %(levelname)s @ %(asctime)s in %(filename)s:%(lineno)d | %(message)s'))
ch.setStream(tqdm)
ch.terminator = ""
logger.addHandler(ch)


class RidgeModel:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.parameters = np.zeros((2, 115))

    def fit(self, X, Y, epochs=1):
        loss = 0.0
        for i in range(Y.shape[1]):
            info = sp_opt.minimize(
                partial(RidgeModel.func, X, Y[:, i], self.alpha),
                x0=self.parameters[i],
                method="L-BFGS-B",
                tol=1e-6,
                bounds=[(0, np.inf)] * X.shape[1],
                jac=True,
                options={"maxiter": epochs}
            )
            self.parameters[i] = info['x']
            loss += info['fun']
        return loss / (X.shape[1] * Y.shape[1])

    def predict(self, X):
        preds = []
        for i in range(self.parameters.shape[0]):
            preds.append(X.dot(self.parameters[i]))
        return np.stack(preds, axis=-1)

    def func(X, Y, alpha, w):
        residual = X.dot(w) - Y
        f = 0.5 * residual.dot(residual) + 0.5 * alpha * w.dot(w)
        grad = X.T @ residual + alpha * w
        return f, grad


class Server:
    def __init__(self, clients, config):
        self.model = RidgeModel()
        if config.get("aggregator") is None:
            self.aggregator = FedAVG()
        else:
            match (config["aggregator"]):
                case "fedavg":
                    self.aggregator = FedAVG()
                case "median":
                    self.aggregator = Median()
                case "centre":
                    self.aggregator = Centre()
        self.clients = clients
        self.config = config
        logger.info("Server initialized with %d clients", len(clients))

    def fit(self):
        start_time = time.time()
        logger.info("Server starting training for %d rounds", self.config['num_rounds'])
        for r in (pbar := trange(self.config['num_rounds'])):
            if self.config.get("drop_round") == r:
                baseline_metrics = self.analytics()
                self.all_clients = self.clients.copy()
                logger.info("Server is dropping 4 clients")
                for _ in range(4):
                    self.clients.pop()

            self.config['round'] = r
            metrics = self.step()
            pbar.set_postfix_str(f"loss: {metrics['loss']:.3f}")
            logger.info("Loss at the end of round %d: %f", r + 1, metrics['loss'])
        logger.info("Server completed training in %f seconds", time.time() - start_time)

        if self.config.get("drop_round"):
            if self.config['drop_round'] < self.config['num_rounds']:
                self.clients = self.all_clients
                return metrics, baseline_metrics
            return metrics, self.analytics()  # If there is no drop then the baseline is the same as the other results

        return metrics

    def step(self):
        all_params, all_losses, all_samples = [], [], []
        for c in self.clients:
            params, loss, num_samples = c.step(self.model.parameters.copy(), self.config)
            all_params.append(params)
            all_losses.append(loss)
            all_samples.append(num_samples)
        results = {}
        if self.config['experiment_type'] == "performance":
            results['cosine similarity'] = cosine_similarity([cp - self.model.parameters for cp in all_params])
        self.model.parameters = self.aggregator.aggregate(all_params, all_samples, self.model.parameters, self.config)
        results['loss'] = np.average(all_losses, weights=all_samples)
        return results

    def analytics(self):
        start_time = time.time()
        logger.info("Server starting analytics")
        all_preds, all_Y_test = [], []
        for c in self.clients:
            client_preds, client_Y_test = c.analytics(self.model.parameters.copy(), self.config)
            all_preds.append(client_preds)
            all_Y_test.append(client_Y_test)
        logger.info("Server completed analytics in %f seconds", time.time() - start_time)

        if self.config['experiment_type'] == "fairness":
            maes, mses, mapes = [], [], []
            for p, yt in zip(all_preds, all_Y_test):
                maes.append(skm.mean_absolute_error(yt, p))
                mses.append(skm.mean_squared_error(yt, p))
                mapes.append(skm.mean_absolute_percentage_error(yt, p))
            return {
                "MAE": np.mean(maes),
                "MAE STD": np.std(maes),
                "MSE": np.mean(mses),
                "MSE STD": np.std(mses),
                "MAPE": np.mean(mapes),
                "MAPE STD": np.std(mapes),
            }

        preds = np.concatenate(all_preds)
        Y_test = np.concatenate(all_Y_test)
        return {
            "MAE": skm.mean_absolute_error(Y_test, preds),
            "RMSE": np.sqrt(skm.mean_squared_error(Y_test, preds)),
            "r2 score": skm.r2_score(Y_test, preds),
            "MAPE": skm.mean_absolute_percentage_error(Y_test, preds),
        }

    def backdoor_analytics(self):
        start_time = time.time()
        logger.info("Server starting backdoor analytics")
        all_preds, all_Y_test = [], []
        for c in self.clients:
            client_preds, client_Y_test = c.backdoor_analytics(self.model.parameters.copy(), self.config)
            all_preds.append(client_preds)
            all_Y_test.append(client_Y_test)
        # Only the energy generation is attacked
        preds = np.concatenate(all_preds)[:, 1]
        Y_test = np.concatenate(all_Y_test)[:, 1]
        logger.info("Server completed analytics in %f seconds", time.time() - start_time)
        return {
            "MAE": skm.mean_absolute_error(Y_test, preds),
            "RMSE": np.sqrt(skm.mean_squared_error(Y_test, preds)),
            "MAPE": skm.mean_absolute_percentage_error(Y_test, preds),
        }

    def evaluate(self, X_test, Y_test):
        preds = self.model.predict(X_test)
        return {
            "MAE": skm.mean_absolute_error(Y_test, preds),
            "RMSE": np.sqrt(skm.mean_squared_error(Y_test, preds)),
            "r2 score": skm.r2_score(Y_test, preds),
            "MAPE": skm.mean_absolute_percentage_error(Y_test, preds),
        }


def cosine_similarity(client_parameters: List[NDArray]) -> float:
    client_parameters = [cp.reshape(-1) for cp in client_parameters]
    similarity_matrix = np.abs(skm.pairwise.cosine_similarity(client_parameters)) - np.eye(len(client_parameters))
    return similarity_matrix.sum() / (len(client_parameters) * (len(client_parameters) - 1))


class MiddleServer:
    def __init__(self, clients, config):
        if config.get("mu1") and config.get("top_k"):
            self.aggregator = TopKKickbackMomentum()
        elif config.get("mu1"):
            self.aggregator = KickbackMomentum()
        elif config.get("top_k"):
            self.aggregator = TopK()
        else:
            self.aggregator = FedAVG()
        self.clients = clients
        logger.info("Middle server initialized with %d clients", len(clients))

    def step(self, parameters, config, finetuning=False):
        start_time = time.time()
        num_episodes = config['num_finetune_episodes'] if finetuning else config['num_episodes']
        logger.info("Middle server starting training for %d episodes", num_episodes)
        for e in range(num_episodes):
            all_params, all_losses, all_samples = [], [], []
            for c in self.clients:
                params, loss, num_samples = c.step(parameters.copy(), config)
                all_params.append(params)
                all_losses.append(loss)
                all_samples.append(num_samples)
            parameters = self.aggregator.aggregate(all_params, all_samples, parameters, config)
        loss = np.average(all_losses, weights=all_samples)
        logger.info("Middle server completed training in %f seconds", time.time() - start_time)
        logger.info("Middle server loss: %f", loss)
        return parameters, loss, sum(all_samples)

    def analytics(self, parameters, config):
        start_time = time.time()
        logger.info("Middle server starting analytics")
        if config.get("num_finetune_episodes"):
            logger.info("Middle server is finetuning")
            parameters, _, _ = self.step(parameters, config, finetuning=True)

        all_preds, all_Y_test = [], []
        for c in self.clients:
            preds, Y_test = c.analytics(parameters.copy(), config)
            all_preds.append(preds)
            all_Y_test.append(Y_test)
        logger.info("Middle server completed analytics in %f seconds", time.time() - start_time)
        return np.concatenate(all_preds), np.concatenate(all_Y_test)


class FedAVG:
    def aggregate(
        self,
        client_parameters: List[NDArray],
        client_samples: List[int],
        parameters: NDArray,
        config: Dict[str, str | int | float]
    ) -> NDArray:
        return np.average(client_parameters, weights=client_samples, axis=0)


class Median:
    def aggregate(
        self,
        client_parameters: List[NDArray],
        client_samples: List[int],
        parameters: NDArray,
        config: Dict[str, str | int | float]
    ) -> NDArray:
        return np.median(client_parameters, axis=0)


class Centre:
    def aggregate(
        self,
        client_parameters: List[NDArray],
        client_samples: List[int],
        parameters: NDArray,
        config: Dict[str, str | int | float]
    ) -> NDArray:
        model = skc.KMeans(n_clusters=len(client_parameters) // 4 + 1, n_init='auto')
        model.fit([cp.reshape(-1) for cp in client_parameters])
        return np.mean(model.cluster_centers_, axis=0).reshape(parameters.shape)


class KickbackMomentum:
    def __init__(self):
        self.momentum = None
        self.prev_parameters = None
        self.episode = 0

    def aggregate(
        self,
        client_parameters: List[NDArray],
        client_samples: List[int],
        parameters: NDArray,
        config: Dict[str, str | int | float]
    ) -> NDArray:
        p = self.episode % config['num_episodes'] == 0
        q = config.get("drop_round") is None or config['round'] < config.get("drop_round")
        if p and q:
            if self.momentum is None:
                self.momentum = np.zeros_like(parameters)
                self.prev_parameters = parameters.copy()
            else:
                self.momentum = config["mu1"] * self.momentum + (parameters - self.prev_parameters)
                self.prev_parameters = parameters.copy()
        self.episode += 1
        grads = np.average([cp - parameters for cp in client_parameters], weights=client_samples, axis=0)
        return self.prev_parameters + config["mu2"] * self.momentum + grads


class TopK:
    def __init__(self):
        self.agg_top_k = None

    def aggregate(
        self,
        client_parameters: List[NDArray],
        client_samples: List[int],
        parameters: NDArray,
        config: Dict[str, str | int | float],
    ) -> NDArray:
        flat_grads = np.average([(p - parameters).reshape(-1) for p in client_parameters], weights=client_samples, axis=0)
        if self.agg_top_k is None:
            self.agg_top_k = np.zeros_like(flat_grads)

        k = round(len(flat_grads) * (1 - config['top_k']))
        if config['round'] < config["drop_round"]:
            idx = np.where(flat_grads >= np.partition(flat_grads, k)[k])[0]
            self.agg_top_k[idx] += 1
            return parameters
        flat_grads = np.where(self.agg_top_k >= np.partition(self.agg_top_k, -k)[-k], flat_grads, 0)
        return parameters + flat_grads.reshape(parameters.shape)


class TopKKickbackMomentum:
    def __init__(self):
        self.momentum = None
        self.prev_parameters = None
        self.episode = 0
        self.agg_top_k = None

    def aggregate(
        self,
        client_parameters: List[NDArray],
        client_samples: List[int],
        parameters: NDArray,
        config: Dict[str, str | int | float],
    ) -> NDArray:
        flat_grads = np.average([(p - parameters).reshape(-1) for p in client_parameters], weights=client_samples, axis=0)
        if self.agg_top_k is None:
            self.agg_top_k = np.zeros_like(flat_grads)

        k = round(len(flat_grads) * (1 - config['top_k']))
        if config['round'] < config["drop_round"]:
            idx = np.where(flat_grads >= np.partition(flat_grads, k)[k])[0]
            self.agg_top_k[idx] += 1
            grads = flat_grads.reshape(parameters.shape)
        else:
            grads = np.where(self.agg_top_k >= np.partition(self.agg_top_k, -k)[-k], flat_grads, 0).reshape(parameters.shape)

        p = self.episode % config['num_episodes'] == 0
        q = config.get("drop_round") is None or config['round'] < config.get("drop_round")
        if p and q:
            if self.momentum is None:
                self.momentum = np.zeros_like(parameters)
                self.prev_parameters = parameters.copy()
            else:
                self.momentum = config["mu1"] * self.momentum + (parameters - self.prev_parameters)
                self.prev_parameters = parameters.copy()
        self.episode += 1
        return self.prev_parameters + config["mu2"] * self.momentum + grads


class Client:
    def __init__(self, data):
        self.data = data

    def step(self, parameters, config):
        model = RidgeModel()
        model.parameters = parameters
        loss = model.fit(self.data['train']['X'], self.data['train']['Y'], epochs=config['num_epochs'])
        return model.parameters, loss, len(self.data['train'])

    def analytics(self, parameters, config):
        model = RidgeModel()
        model.parameters = parameters
        return model.predict(self.data['test']['X']), self.data['test']['Y']

    def backdoor_analytics(self, parameters, config):
        model = RidgeModel()
        model.parameters = parameters
        backdoor_X, backdoor_Y = gen_backdoor_data(self.data['test']['X'], self.data['test']['Y'])
        return model.predict(backdoor_X), backdoor_Y


class EmptyUpdater(Client):
    def step(self, parameters, config):
        _, loss, samples = super().step(parameters, config)
        return parameters, loss, samples


class Adversary(Client):
    def __init__(self, data, corroborator):
        super().__init__(data)
        self.corroborator = corroborator
        self.corroborator.register(self)

    def honest_step(self, parameters, config):
        return super().step(parameters, config)


class LIE(Adversary):
    def step(self, parameters, config):
        mu, sigma, loss = self.corroborator.calc_grad_stats(parameters, config)
        return mu + self.corroborator.z_max * sigma, loss, len(self.data['train'])


class IPM(Adversary):
    def step(self, parameters, config):
        mu, sigma, loss = self.corroborator.calc_grad_stats(parameters, config)
        grads = parameters - mu
        return parameters + (1 / self.corroborator.nadversaries) * grads, loss, len(self.data['train'])


def gen_backdoor_data(X, Y):
    backdoor_X = X.copy().reshape(-1, 23, 5)
    backdoor_X[:, -3:, -1] = 100
    backdoor_Y = Y.copy()
    backdoor_Y[:, 1] = 8
    return backdoor_X.reshape(X.shape), backdoor_Y


class BackdoorLIE(Adversary):
    def __init__(self, data, corroborator):
        super().__init__(data, corroborator)
        self.backdoor_X, self.backdoor_Y = gen_backdoor_data(self.data['train']['X'], self.data['train']['Y'])

    def step(self, parameters, config):
        parameters, loss = self.corroborator.calc_backdoor_parameters(parameters, config)
        return parameters, loss, len(self.data['train'])

    def backdoor_step(self, parameters, config):
        model = RidgeModel()
        model.parameters = parameters
        loss = model.fit(self.backdoor_X, self.backdoor_Y, epochs=config['num_epochs'])
        return model.parameters, loss, len(self.data['train'])


class Corroborator:
    def __init__(self, nclients, nadversaries):
        self.nclients = nclients
        self.adversaries = []
        self.nadversaries = nadversaries
        self.round = -1
        self.mu = None
        self.sigma = None
        self.loss = None
        self.parameters = None
        s = self.nclients // 2 + 1 - self.nadversaries
        self.z_max = sp.stats.norm.ppf((self.nclients - s) / self.nclients)

    def register(self, adversary):
        self.adversaries.append(adversary)

    def calc_grad_stats(self, parameters, config):
        if self.round == config['round']:
            return self.mu, self.sigma, self.loss

        honest_parameters = []
        honest_samples = []
        honest_losses = []
        for a in self.adversaries:
            parameters, loss, samples = a.honest_step(parameters, config)
            honest_parameters.append(parameters)
            honest_samples.append(samples)
            honest_losses.append(loss)

        # Does some aggregation
        self.mu = np.average(honest_parameters, weights=honest_samples, axis=0)
        self.sigma = np.sqrt(np.average((honest_parameters - self.mu)**2, weights=honest_samples, axis=0))
        self.loss = np.average(honest_losses, weights=honest_samples)
        self.round = config['round']
        return self.mu, self.sigma, self.loss

    def calc_backdoor_parameters(self, parameters, config):
        if self.round == config['round']:
            return self.parameters, self.loss

        self.calc_grad_stats(parameters, config)

        backdoor_parameters = []
        backdoor_samples = []
        backdoor_losses = []
        for a in self.adversaries:
            parameters, loss, samples = a.backdoor_step(parameters, config)
            backdoor_parameters.append(parameters)
            backdoor_samples.append(samples)
            backdoor_losses.append(loss)
        logger.info(
            "The backdoor attack attained a loss of %f",
            np.average(backdoor_losses, weights=backdoor_samples)
        )

        self.parameters = np.clip(
            np.average(backdoor_parameters, weights=backdoor_samples, axis=0),
            self.mu - self.z_max * self.sigma,
            self.mu + self.z_max * self.sigma
        )
        return self.parameters, self.loss


def load_data():
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


def load_customer_regions():
    with open("../data/customer_regions.json", 'r') as f:
        customer_regions = json.load(f)
    regions = [[] for _ in np.unique(list(customer_regions.values()))]
    for customer, region_i in customer_regions.items():
        regions[region_i].append(int(customer) - 1)
    return regions


def get_experiment_config(all_exp_configs, exp_id):
    experiment_config = {k: v for k, v in all_exp_configs.items() if k != "experiments"}
    variables = all_exp_configs['experiments'][exp_id - 1]
    experiment_config.update(variables)
    return experiment_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform experiments evaluating the solar home dataset.")
    parser.add_argument("-i", "--id", type=int, default=1,
                        help="Which of the experiments in the config to perform (counts from 1).")
    parser.add_argument("-p", "--performance", action="store_true",
                        help="Perform experiments evaluating the performance.")
    parser.add_argument("-a", "--attack", action="store_true",
                        help="Perform experiments evaluating the vulnerability to and mitigation of attacks.")
    parser.add_argument("-f", "--fairness", action="store_true", help="Perform experiments evaluating the fairness.")
    args = parser.parse_args()

    start_time = time.time()
    keyword = "performance" if args.performance else "attack" if args.attack else "fairness"
    with open(f"configs/{keyword}.json", 'r') as f:
        experiment_config = get_experiment_config(json.load(f), args.id)
    print(f"Performing {keyword} experiment with {experiment_config=}")
    experiment_config['experiment_type'] = keyword

    client_data, X_test, Y_test = load_data()
    if args.performance or args.fairness:
        regions = load_customer_regions()
        network_arch = [
            MiddleServer([Client(client_data[r]) for r in region], experiment_config) for region in regions
        ]
    else:
        if experiment_config["attack"] == "empty":
            adversary_type = EmptyUpdater
        else:
            corroborator = Corroborator(len(client_data), round(len(client_data) * (1 - 0.5)))
            if experiment_config["attack"] == "lie":
                adversary_type = partial(LIE, corroborator=corroborator)
            elif experiment_config["attack"] == "ipm":
                adversary_type = partial(IPM, corroborator=corroborator)
            else:
                adversary_type = partial(BackdoorLIE, corroborator=corroborator)
        network_arch = [
            adversary_type(d) if i + 1 > (len(client_data) * 0.5) else Client(d)
            for i, d in enumerate(client_data)
        ]

    server = Server(
        network_arch,
        experiment_config
    )
    training_metrics = server.fit()

    if args.fairness:
        training_metrics, baseline_metrics = training_metrics

    testing_metrics = server.analytics()
    centralised_metrics = server.evaluate(X_test, Y_test)
    print(f"{training_metrics=}")
    print(f"{testing_metrics=}")
    print(f"{centralised_metrics=}")
    results = {"train": training_metrics, "test": testing_metrics, "centralised": centralised_metrics}

    if args.attack and experiment_config['attack'] == "backdoor_lie":
        backdoor_metrics = server.backdoor_analytics()
        results['backdoor'] = backdoor_metrics
        print(f"{backdoor_metrics=}")
    elif args.fairness:
        results['baseline'] = baseline_metrics
        print(f"{baseline_metrics=}")

    os.makedirs("results", exist_ok=True)
    filename = "results/solar_home_{}.json".format(
        '_'.join([f'{k}={v}' for k, v in experiment_config.items() if k not in ['repeat', 'round']]),
    )
    with open(filename, "w") as f:
        json.dump(results, f)
    print(f"Saved results to {filename}")

    print(f"Experiment took {time.time() - start_time} seconds")
