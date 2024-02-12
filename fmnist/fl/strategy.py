from typing import List, Callable, Dict
import itertools
import sklearn.cluster as skc
from .common import Config, Parameters, to_attribute_array, Metrics
import numpy as np


def fedavg(
    client_parameters: List[List[Parameters | int | float]],
    client_samples: List[int]
) -> List[Parameters | int | float]:
    return [np.average(layer, weights=client_samples, axis=0) for layer in to_attribute_array(client_parameters)]


class FedAVG:
    """
    The federated averaging aggregator proposed in https://arxiv.org/abs/1602.05629
    """

    def aggregate(
        self, client_parameters: List[Parameters], client_samples: List[int], parameters: Parameters, config: Config
    ) -> Parameters:
        """
        Aggregate client parameters into a global model.

        Arguments:
        - client_parameters: Copies of the client local models or the gradient updates
        - client_samples: The number of samples each client trained on
        - parameters: The aggregators own current parameters
        """
        return fedavg(client_parameters, client_samples)

    def analytics(
        self,
        client_metrics: List[Metrics],
        client_samples: List[int],
        config: Config,
        aggregate_fn: Callable[[List[List[int | float]], List[int]], List[int | float]] = fedavg
    ) -> Dict[str, int | float]:
        """
        Perform federated analytics on the client local metrics.

        Arguments:
        - client_metrics: Client local metrics
        - client_samples: The number of samples each client trained on
        - aggregate_fn: Function to be used to combine the client metrics
        """
        metrics_skeleton = client_metrics[0]
        distributed_metrics = []
        for cm in client_metrics:
            distributed_metrics.append([v for v in cm.values()])
        aggregated_metrics = aggregate_fn(distributed_metrics, client_samples)

        other_analytics = {}
        if config.get("analytics"):
            for analytic_fn in config['analytics']:
                other_analytics.update(analytic_fn(client_metrics, client_samples, config))

        complete_analytics = {k: fm for k, fm in zip(metrics_skeleton, aggregated_metrics)}
        complete_analytics.update(other_analytics)

        return complete_analytics


class Centre(FedAVG):
    def aggregate(
        self, client_parameters: List[Parameters], client_samples: List[int], parameters: Parameters, config: Config
    ) -> Parameters:
        aggregated_parameters = []
        for i, layer in enumerate(parameters):
            model = skc.KMeans(n_clusters=len(client_parameters) // 2 + 1, n_init='auto')
            model.fit([cp[i].reshape(-1) for cp in client_parameters])
            aggregated_parameters.append(np.mean(model.cluster_centers_, axis=0).reshape(layer.shape))
        return aggregated_parameters


class Median(FedAVG):
    def aggregate(
        self, client_parameters: List[Parameters], client_samples: List[int], parameters: Parameters, config: Config
    ) -> Parameters:
        return [np.median(layer, axis=0) for layer in to_attribute_array(client_parameters)]


class TrimmedMean(FedAVG):
    def aggregate(
        self,
        client_parameters: List[Parameters],
        client_samples: List[int],
        parameters: Parameters,
        config: Config
    ) -> Parameters:
        reject_i = round(0.25 * len(client_parameters))
        sorted_params = [np.sort(layer, axis=0) for layer in to_attribute_array(client_parameters)]
        return [np.mean(sorted_layer[reject_i:-reject_i], axis=0) for sorted_layer in sorted_params]


class Krum(FedAVG):
    def aggregate(
        self,
        client_parameters: List[Parameters],
        client_samples: List[int],
        parameters: Parameters,
        config: Config
    ) -> Parameters:
        aggregated_parameters = []
        n = len(client_parameters)
        clip = round(0.5 * n)
        for client_layers in to_attribute_array(client_parameters):
            X = np.array([l.reshape(-1) for l in client_layers])
            scores = np.zeros(n)
            distances = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None] - 2 * np.dot(X, X.T)
            for i in range(len(X)):
                scores[i] = np.sum(np.sort(distances[i])[1:((n - clip) - 1)])
            idx = np.argpartition(scores, n - clip)[:(n - clip)]
            aggregated_parameters.append(np.mean(X[idx], axis=0).reshape(client_layers[0].shape))
        return aggregated_parameters


class FedProx(FedAVG):
    def __init__(self):
        self.prev_parameters = None
        self.episode = 0

    def aggregate(
        self,
        client_parameters: List[Parameters],
        client_samples: List[int],
        parameters: Parameters,
        config: Config
    ) -> Parameters:
        if self.episode % config['num_episodes'] == 0:
            self.prev_parameters = parameters.copy()
        self.episode += 1
        grads = [
            np.average(clayer, weights=client_samples, axis=0) - slayer
            for clayer, slayer in zip(to_attribute_array(client_parameters), parameters)
        ]
        return [p + g + config['mu'] * (p - pp) for p, pp, g in zip(parameters, self.prev_parameters, grads)]


class MRCS(FedAVG):
    def __init__(self):
        self.momentum = None

    def aggregate(
        self,
        client_parameters: List[Parameters],
        client_samples: List[int],
        parameters: Parameters,
        config: Dict[str, str | int | float]
    ) -> Parameters:
        client_grads = [
            [clayer - slayer for clayer, slayer in zip(client_params, parameters)]
            for client_params in client_parameters
        ]
        p_vals = []
        for c_grads in client_grads:
            if self.momentum is None:
                p_vals.append(1)
            else:
                sim = np.sum([np.sum(m * cg) for m, cg in zip(self.momentum, c_grads)])
                sim = sim / (np.sqrt(np.sum([np.sum(m**2) for m in self.momentum])) * np.sqrt(np.sum([np.sum(cg**2) for cg in c_grads])))
                p_vals.append(max(0, sim))
        if np.sum(p_vals) == 0:
            p_vals = np.ones_like(p_vals)
        p_vals = np.array(p_vals) / np.sum(p_vals)
        agg_updates = [np.average(clayer, weights=p_vals, axis=0) for clayer in to_attribute_array(client_grads)]
        if self.momentum is None:
            self.momentum = [np.zeros_like(p) for p in parameters]
        self.momentum = [(1 - config["mu"]) * m + config["mu"] * au for m, au in zip(self.momentum, agg_updates)]
        return [p + m for p, m in zip(parameters, self.momentum)]


class KickbackMomentum(FedAVG):
    def __init__(self):
        self.momentum = None
        self.prev_parameters = None
        self.episode = 0

    def aggregate(
        self, client_parameters: List[Parameters], client_samples: List[int], parameters: Parameters, config: Config
    ) -> Parameters:
        grads = [
            np.average(clayer, weights=client_samples, axis=0) - slayer
            for clayer, slayer in zip(to_attribute_array(client_parameters), parameters)
        ]
        if self.episode % config['num_episodes'] == 0:
            if self.momentum is None:
                self.momentum = [np.zeros_like(p) for p in parameters]
                self.prev_parameters = parameters
            else:
                self.momentum = [config["mu1"] * m + (p - pp) for m, pp, p in zip(self.momentum, self.prev_parameters, parameters)]
                self.prev_parameters = parameters
        self.episode += 1
        return [p + config["mu2"] * m + g for p, m, g in zip(parameters, self.momentum, grads)]


class TopK(FedAVG):
    def __init__(self):
        self.agg_top_k = None
        self.num_clients = 0

    def aggregate(
        self, client_parameters: List[Parameters], client_samples: List[int], parameters: Parameters, config: Config
    ) -> Parameters:
        grads = [np.average(clayer, weights=client_samples, axis=0) - slayer for clayer, slayer in zip(to_attribute_array(client_parameters), parameters)]
        num_clients = len(client_samples)
        flat_grads = np.concatenate([g.reshape(-1) for g in grads])
        if self.agg_top_k is None:
            self.agg_top_k = np.zeros_like(flat_grads)

        k = round(len(flat_grads) * (1 - config['k']))
        if num_clients >= self.num_clients:
            idx = np.where(flat_grads >= np.partition(flat_grads, k)[k])[0]
            self.agg_top_k[idx] += 1
            self.num_clients = num_clients
        else:
            flat_grads = np.where(self.agg_top_k >= np.partition(self.agg_top_k, -k)[-k], flat_grads, 0)
            grads = [g.reshape(p.shape) for p, g in zip(parameters, np.split(flat_grads, list(itertools.accumulate([np.prod(g.shape) for g in grads]))[:-1]))]

        return [p + g for p, g in zip(parameters, grads)]


class TopKKickbackMomentum(FedAVG):
    def __init__(self):
        self.agg_top_k = None
        self.num_clients = 0
        self.momentum = None
        self.prev_parameters = None
        self.episode = 0
        self.num_clients = 0

    def aggregate(
        self, client_parameters: List[Parameters], client_samples: List[int], parameters: Parameters, config: Config
    ) -> Parameters:
        grads = [np.average(clayer, weights=client_samples, axis=0) - slayer for clayer, slayer in zip(to_attribute_array(client_parameters), parameters)]
        num_clients = len(client_samples)
        flat_grads = np.concatenate([g.reshape(-1) for g in grads])
        if self.agg_top_k is None:
            self.agg_top_k = np.zeros_like(flat_grads)

        k = round(len(flat_grads) * (1 - config['k']))
        if num_clients >= self.num_clients:
            idx = np.where(flat_grads >= np.partition(flat_grads, k)[k])[0]
            self.agg_top_k[idx] += 1
            self.num_clients = num_clients
        else:
            flat_grads = np.where(self.agg_top_k >= np.partition(self.agg_top_k, -k)[-k], flat_grads, 0)
            grads = [g.reshape(p.shape) for p, g in zip(parameters, np.split(flat_grads, list(itertools.accumulate([np.prod(g.shape) for g in grads]))[:-1]))]

        self.num_clients = num_clients
        if self.episode % config['num_episodes'] == 0:
            if self.momentum is None:
                self.momentum = [np.zeros_like(p) for p in parameters]
                self.prev_parameters = parameters
            else:
                self.momentum = [config["mu1"] * m + (p - pp) for m, pp, p in zip(self.momentum, self.prev_parameters, parameters)]
                self.prev_parameters = parameters
        self.episode += 1

        return [p + config["mu2"] * m + g for p, m, g in zip(parameters, self.momentum, grads)]
