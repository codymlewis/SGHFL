from typing import List, Dict
import time
from functools import partial
import numpy as np
from numpy.typing import NDArray
import sklearn.metrics as skm
import sklearn.cluster as skc
import scipy.optimize as sp_opt
from tqdm import trange

import load_data
from logger import logger


class RidgeModel:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.parameters = np.zeros([])

    def init_params(self, sample_shape):
        self.parameters = np.zeros(sample_shape if len(sample_shape) > 1 else (1,) + sample_shape)

    def fit(self, X, Y, epochs=1):
        loss = 0.0
        for i in range(self.parameters.shape[0]):
            info = sp_opt.minimize(
                partial(RidgeModel.func, X, Y[:, i] if len(Y.shape) > 1 else Y, self.alpha),
                x0=self.parameters[i],
                method="L-BFGS-B",
                tol=1e-6,
                bounds=[(0, np.inf)] * X.shape[1],
                jac=True,
                options={"maxiter": epochs}
            )
            self.parameters[i] = info['x']
            loss += info['fun']
        return loss / np.prod(self.parameters.shape)

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
    def __init__(self, clients, config, sample_shape):
        self.model = RidgeModel()
        self.model.init_params(sample_shape)
        match config.get("aggregator"):
            case "median":
                self.aggregator = Median()
            case "centre":
                self.aggregator = Centre()
            case "krum":
                self.aggregator = Krum()
            case "trimmed_mean":
                self.aggregator = TrimmedMean()
            case _:
                self.aggregator = FedAVG()
        self.clients = clients
        self.config = config
        logger.info("Server initialized with %d clients", len(clients))

    def fit(self):
        start_time = time.time()
        logger.info("Server starting training for %d rounds", self.config['num_rounds'])
        for r in (pbar := trange(self.config['num_rounds'])):
            if self.config.get("drop_round") == r:
                self.all_clients = self.clients.copy()
                n_clients = len(self.all_clients)
                n_dropped_clients = round(n_clients * 0.4)
                logger.info(f"Server is dropping {n_dropped_clients} clients")
                for _ in range(n_dropped_clients):
                    self.clients.pop()

            self.config['round'] = r
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
        preds = np.concatenate(all_preds)
        Y_test = np.concatenate(all_Y_test)

        if self.config['experiment_type'] == "fairness":
            dropped_preds, dropped_Y_test = [], []
            n_dropped_clients = len(self.all_clients) - len(self.clients)
            for c in self.all_clients[-n_dropped_clients:]:
                client_preds, client_Y_test = c.analytics(self.model.parameters.copy(), self.config)
                dropped_preds.append(client_preds)
                dropped_Y_test.append(client_Y_test)
            d_preds = np.concatenate(dropped_preds)
            d_Y_test = np.concatenate(dropped_Y_test)
            return {
                "MAE": skm.mean_absolute_error(Y_test, preds),
                "RMSE": np.sqrt(skm.mean_squared_error(Y_test, preds)),
                "r2 score": skm.r2_score(Y_test, preds),
                "Dropped MAE": skm.mean_absolute_error(d_Y_test, d_preds),
                "Dropped RMSE": np.sqrt(skm.mean_squared_error(d_Y_test, d_preds)),
                "Dropped r2 score": skm.r2_score(d_Y_test, d_preds),
            }

        return {
            "MAE": skm.mean_absolute_error(Y_test, preds),
            "RMSE": np.sqrt(skm.mean_squared_error(Y_test, preds)),
            "r2 score": skm.r2_score(Y_test, preds),
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
        preds = np.concatenate(all_preds)
        Y_test = np.concatenate(all_Y_test)
        if preds.shape[-1] > 1:
            preds = preds[:, -1]
            Y_test = Y_test[:, -1]
        logger.info("Server completed analytics in %f seconds", time.time() - start_time)
        return {
            "MAE": skm.mean_absolute_error(Y_test, preds),
            "RMSE": np.sqrt(skm.mean_squared_error(Y_test, preds)),
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
        match config.get('aggregator'):
            case "mrcs":
                self.aggregator = MRCS()
            case "topk_kickback":
                self.aggregator = TopKKickbackMomentum()
            case "kickback":
                self.aggregator = KickbackMomentum()
            case "topk":
                self.aggregator = TopK()
            case "fedprox":
                self.aggregator = FedProx()
            case _:
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


class TrimmedMean:
    def aggregate(
        self,
        client_parameters: List[NDArray],
        client_samples: List[int],
        parameters: NDArray,
        config: Dict[str, str | int | float]
    ) -> NDArray:
        reject_i = round(0.25 * len(client_parameters))
        sorted_params = np.sort(client_parameters, axis=0)
        return np.mean(sorted_params[reject_i:-reject_i], axis=0)


class Krum:
    def aggregate(
        self,
        client_parameters: List[NDArray],
        client_samples: List[int],
        parameters: NDArray,
        config: Dict[str, str | int | float]
    ) -> NDArray:
        n = len(client_parameters)
        clip = round(0.5 * n)
        X = np.array([p.reshape(-1) for p in client_parameters])
        scores = np.zeros(n)
        distances = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None] - 2 * np.dot(X, X.T)
        for i in range(len(X)):
            scores[i] = np.sum(np.sort(distances[i])[1:((n - clip) - 1)])
        idx = np.argpartition(scores, n - clip)[:(n - clip)]
        return np.mean(X[idx], axis=0).reshape(client_parameters[0].shape)


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


class FedProx:
    def __init__(self):
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
            self.prev_parameters = parameters.copy()
        self.episode += 1
        grads = np.average([cp - parameters for cp in client_parameters], weights=client_samples, axis=0)
        return parameters + grads - config['mu'] * (parameters - self.prev_parameters)


class MRCS:
    def __init__(self):
        self.momentum = None

    def aggregate(
        self,
        client_parameters: List[NDArray],
        client_samples: List[int],
        parameters: NDArray,
        config: Dict[str, str | int | float]
    ) -> NDArray:
        client_grads = [cp - parameters for cp in client_parameters]
        p_vals = []
        for c_grads in client_grads:
            if self.momentum is None:
                p_vals.append(1)
            else:
                sim = np.sum(self.momentum * c_grads) / (np.linalg.norm(self.momentum) * np.linalg.norm(c_grads))
                p_vals.append(max(0, sim))
        if np.sum(p_vals) == 0:
            p_vals = np.ones_like(p_vals)
        p_vals = np.array(p_vals) / np.sum(p_vals)
        agg_updates = np.sum((p_vals * np.array(client_grads).T).T, axis=0)
        if self.momentum is None:
            self.momentum = np.zeros_like(parameters)
        self.momentum = (1 - config["mu"]) * self.momentum + config["mu"] * agg_updates
        return parameters + self.momentum


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
        return parameters + config["mu2"] * self.momentum + grads


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

        k = round(len(flat_grads) * (1 - config['k']))
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

        k = round(len(flat_grads) * (1 - config['k']))
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
        return parameters + config["mu2"] * self.momentum + grads


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
        backdoor_X, backdoor_Y = load_data.gen_backdoor_data(self.data['test']['X'], self.data['test']['Y'])
        return model.predict(backdoor_X), backdoor_Y
