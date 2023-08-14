from typing import List
import itertools
import sklearn.cluster as skc
from flagon.common import Config, Parameters, Metrics, count_clients, to_attribute_array
from flagon.strategy import FedAVG
import numpy as np


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


class FreezingMomentum(FedAVG):
    def __init__(self):
        self.momentum = None
        self.prev_parameters = None
        self.episode = 0
        self.num_clients = 0

    def aggregate(
        self, client_parameters: List[Parameters], client_samples: List[int], parameters: Parameters, config: Config
    ) -> Parameters:
        grads = [np.average(clayer, weights=client_samples, axis=0) - slayer for clayer, slayer in zip(to_attribute_array(client_parameters), parameters)]
        num_clients = len(client_samples)
        if num_clients < self.num_clients:
            # Freeze Momomentum
            pass
        else:
            self.num_clients = num_clients
            if self.episode % config['num_episodes'] == 0:
                if self.momentum is None:
                    self.momentum = [np.zeros_like(p) for p in parameters]
                    self.prev_parameters = parameters
                else:
                    self.momentum = [config["mu1"] * m + (p - pp) for m, pp, p in zip(self.momentum, self.prev_parameters, parameters)]
                    self.prev_parameters = parameters
        self.episode += 1
        return [p + config["mu2"] * m + g for p, m, g in zip(self.prev_parameters, self.momentum, grads)]


class BottomK(FedAVG):
    def __init__(self):
        self.avg_bottom_k = None

    def aggregate(
        self, client_parameters: List[Parameters], client_samples: List[int], parameters: Parameters, config: Config
    ) -> Parameters:
        grads = [np.average(clayer, weights=client_samples, axis=0) - slayer for clayer, slayer in zip(to_attribute_array(client_parameters), parameters)]
        flat_grads = np.concatenate([g.reshape(-1) for g in grads])
        if self.avg_bottom_k is None:
            self.avg_bottom_k = np.zeros_like(flat_grads)
        k = len(flat_grads) // config['bottom_k']
        idx = np.where(flat_grads[flat_grads < np.partition(flat_grads, k)[k]])[0]
        self.avg_bottom_k[idx] += 1
        flat_grads = np.where(flat_grads < np.partition(self.avg_bottom_k, k)[k], flat_grads, 0)
        grads = np.split(flat_grads, list(itertools.accumulate([np.prod(g.shape) for g in grads]))[:-1])
        return [p + g for p, g in zip(parameters, grads)]