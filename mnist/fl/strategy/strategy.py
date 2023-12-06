from typing import List, Dict, Callable
from abc import ABC, abstractmethod

from ..common import Parameters, Metrics, Config


class Strategy(ABC):
    "An aggregation strategy for combining local models into a global model."

    @abstractmethod
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
        pass

    @abstractmethod
    def analytics(
        self,
        client_metrics: List[Metrics],
        client_samples: List[int],
        config: Config,
        aggregate_fn: Callable[[List[List[int | float]], List[int]], List[int | float]]
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
