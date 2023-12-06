from typing import List
import numpy as np

from .strategy import Strategy
from ..common import Config, Parameters, Metrics, to_attribute_array


class FedAVG(Strategy):
    """
    The federated averaging aggregator proposed in https://arxiv.org/abs/1602.05629
    """

    def aggregate(
        self, client_parameters: List[Parameters], client_samples: List[int], parameters: Parameters, config: Config
    ) -> Parameters:
        return fedavg(client_parameters, client_samples)

    def analytics(self, client_metrics: List[Metrics], client_samples: List[int], config: Config) -> Metrics:
        return super().analytics(client_metrics, client_samples, config, aggregate_fn=fedavg)


def fedavg(
    client_parameters: List[List[Parameters | int | float]],
    client_samples: List[int]
) -> List[Parameters | int | float]:
    return [np.average(layer, weights=client_samples, axis=0) for layer in to_attribute_array(client_parameters)]
