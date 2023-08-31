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
import scipy.optimize as sp_opt
from tqdm import tqdm, trange

import src.data_manager


logger = logging.getLogger("dylon")
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
        self.clients = clients
        self.config = config
        logger.info("Server initialized with %d clients", len(clients))

    def fit(self):
        start_time = time.time()
        logger.info("Server starting training for %d rounds", self.config['num_rounds'])
        for r in (pbar := trange(self.config['num_rounds'])):
            metrics = self.step()
            pbar.set_postfix_str(f"loss: {metrics['loss']:.3f}")
            logger.info("Loss at the end of round %d: %f", r + 1, metrics['loss'])
        logger.info("Server completed training in %f seconds", time.time() - start_time)
        return metrics

    def step(self):
        all_params, all_losses, all_samples = [], [], []
        for c in self.clients:
            params, loss, num_samples = c.step(self.model.parameters.copy(), self.config)
            all_params.append(params)
            all_losses.append(loss)
            all_samples.append(num_samples)
        all_grads = [cp - self.model.parameters for cp in all_params]
        self.model.parameters += np.average(all_grads, weights=all_samples, axis=0)
        return {
            "loss": np.average(all_losses, weights=all_samples),
            "cosine similarity": cosine_similarity(all_grads),
        }

    def analytics(self):
        start_time = time.time()
        logger.info("Server starting analytics")
        all_preds, all_Y_test = [], []
        for c in self.clients:
            client_preds, client_Y_test = c.analytics(self.model.parameters.copy(), self.config)
            all_preds.append(client_preds)
            all_Y_test.append(client_Y_test)
        preds = np.concatenate(all_preds)
        Y_test = np.concatenate(all_Y_test)
        logger.info("Server completed analytics in %f seconds", time.time() - start_time)
        return {
            "MAE": skm.mean_absolute_error(Y_test, preds),
            "RMSE": np.sqrt(skm.mean_squared_error(Y_test, preds)),
            "r2 score": skm.r2_score(Y_test, preds),
        }

    def evaluate(self, X_test, Y_test):
        preds = self.model.predict(X_test)
        return {
            "MAE": skm.mean_absolute_error(Y_test, preds),
            "RMSE": np.sqrt(skm.mean_squared_error(Y_test, preds)),
            "r2 score": skm.r2_score(Y_test, preds),
        }    


def cosine_similarity(client_parameters: List[NDArray]) -> float:
    client_parameters = [cp.reshape(-1) for cp in client_parameters]
    similarity_matrix = np.abs(skm.pairwise.cosine_similarity(client_parameters)) - np.eye(len(client_parameters))
    return similarity_matrix.sum() / (len(client_parameters) * (len(client_parameters) - 1))


class MiddleServer:
    def __init__(self, clients, config):
        self.aggregator = KickbackMomentum() if config.get("mu1") else FedAVG()
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
        if self.episode % config['num_episodes'] == 0:
            if self.momentum is None:
                self.momentum = np.zeros_like(parameters)
                self.prev_parameters = parameters.copy()
            else:
                self.momentum = config["mu1"] * self.momentum + (parameters - self.prev_parameters)
                self.prev_parameters = parameters.copy()
        self.episode += 1
        grads = np.average([cp - parameters for cp in client_parameters], weights=client_samples, axis=0)
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


def load_data():
    with open("data/solar_home_data.pkl", 'rb') as f:
        data = pickle.load(f)

    client_data = []
    X_test, Y_test = [], []
    for i in range(1, 301):
        idx = np.arange(24, len(data[i]))
        expanded_idx = np.array([np.arange(i - 24, i - 1) for i in idx])
        client_X, client_Y = data[i][expanded_idx], data[i][idx, :2]
        client_X = einops.rearrange(client_X, 'b h s -> b (h s)')
        client_data.append(src.data_manager.Dataset({
            "train": {"X": client_X[:300 * 24], "Y": client_Y[:300 * 24]},
            "test": {"X": client_X[300 * 24:], "Y": client_Y[300 * 24:]}
        }))
        X_test.append(client_data[-1]['test']['X'])
        Y_test.append(client_data[-1]['test']['Y'])
    X_test = np.concatenate(X_test)
    Y_test = np.concatenate(Y_test)

    return client_data, X_test, Y_test


def load_customer_regions():
    with open("data/customer_regions.json", 'r') as f:
        customer_regions = json.load(f)
    regions = [[] for _ in np.unique(list(customer_regions.values()))]
    for customer, region_i in customer_regions.items():
        regions[region_i].append(int(customer) - 1)
    return regions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform experiments evaluating the fairness when clients drop out from colloboration.")
    parser.add_argument("-i", "--id", type=int, default=1, help="Which of the experiments in the config to perform (counts from 1).")
    args = parser.parse_args()
    # TODO: Add attack and fairness evals
    # Maybe try different values for the momentum

    start_time = time.time()
    with open("configs/performance.json", 'r') as f:
        experiment_config = src.common.get_experiment_config(json.load(f), args.id)
    print(f"Performing experiment with {experiment_config=}")

    client_data, X_test, Y_test = load_data()
    regions = load_customer_regions()

    server = Server(
        [MiddleServer([Client(client_data[r]) for r in region], experiment_config) for region in regions],
        experiment_config
    )
    training_metrics = server.fit()

    testing_metrics = server.analytics()
    centralised_metrics = server.evaluate(X_test, Y_test)

    print(f"{training_metrics=}")
    print(f"{testing_metrics=}")
    print(f"{centralised_metrics=}")

    results = {"train": training_metrics, "test": testing_metrics, "centralised": centralised_metrics}
    os.makedirs("results", exist_ok=True)
    filename = "results/solar_home_performance_{}.json".format(
        '_'.join([f'{k}={v}' for k, v in experiment_config.items() if k not in ['repeat']]),
    )
    with open(filename, "w") as f:
        json.dump(results, f)
    print(f"Saved results to {filename}")

    print(f"Experiment took {time.time() - start_time} seconds")